
import torch
import torch.nn as nn
from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class PiSSAConfig:
    """PiSSA initialization configuration"""
    enabled: bool = False
    niter: int = 4
    modify_base_weight: bool = True
    target_modules: Optional[List[str]] = None


def compute_pissa_init(
    weight: torch.Tensor,
    rank: int,
    niter: int = 4,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    weight_float = weight.float().to(device)
    out_dim, in_dim = weight_float.shape

    actual_rank = min(rank, min(out_dim, in_dim))

    U, S, V = torch.svd_lowrank(weight_float, q=actual_rank, niter=niter)

    # U: [out_dim, rank], S: [rank], V: [in_dim, rank]
    # Note: torch.svd_lowrank returns V, not Vh (V transposed)!

    sqrt_S = torch.sqrt(S)  # [rank]

    # A = diag(sqrt(S)) @ V^T = [rank, in_dim]
    lora_A = sqrt_S.unsqueeze(1) * V.T  # broadcast: [rank, 1] * [rank, in_dim]

    # B = U @ diag(sqrt(S)) = [out_dim, rank]
    lora_B = U * sqrt_S.unsqueeze(0)  # broadcast: [out_dim, rank] * [1, rank]

    # Residual weight: W - B @ A
    low_rank_approx = lora_B @ lora_A  # [out_dim, in_dim]
    residual = weight_float - low_rank_approx

    return {
        'lora_A': lora_A.float().contiguous(),
        'lora_B': lora_B.float().contiguous(),
        'residual': residual.to(weight.dtype).contiguous()
    }


def apply_pissa_init(
    model: nn.Module,
    niter: int = 4,
    modify_base_weight: bool = True,
    target_modules: Optional[List[str]] = None,
    verbose: bool = True
) -> int:
    if verbose:
        print("\n" + "=" * 70)
        print("[PiSSA] Applying PiSSA (Principal Singular values and Singular vectors Adaptation)")
        print("=" * 70)
        print(f"[PiSSA] niter: {niter}")
        print(f"[PiSSA] modify_base_weight: {modify_base_weight}")

    init_count = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Default target modules for LLaMA
    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'dense', 'gate_proj', 'up_proj', 'down_proj', 'fc1', 'fc2']

    for name, module in model.named_modules():
        if not (hasattr(module, 'lora_A') and hasattr(module, 'lora_B')):
            continue

        is_target = any(target in name for target in target_modules)
        if not is_target:
            continue

        # Get active adapter
        active_adapter = getattr(module, 'active_adapter', None)
        if active_adapter is None:
            continue

        if isinstance(active_adapter, (list, tuple, set)):
            active_adapter = list(active_adapter)[0]

        try:
            # Get LoRA modules
            lora_A_module = module.lora_A[active_adapter]
            lora_B_module = module.lora_B[active_adapter]

            # Get base layer and original weight
            base_layer = getattr(module, 'base_layer', None)
            if base_layer is None:
                if verbose:
                    print(f"  [Skip] {name}: No base_layer found")
                continue

            original_weight = base_layer.weight.data

            # Get LoRA rank from existing A matrix
            lora_rank = lora_A_module.weight.shape[0]

            # Compute PiSSA initialization
            pissa_result = compute_pissa_init(
                weight=original_weight,
                rank=lora_rank,
                niter=niter,
                device=device
            )

            # Apply PiSSA initialization to LoRA weights (contiguous for safetensors compatibility)
            # lora_A: [rank, in_dim], lora_B: [out_dim, rank]
            lora_A_module.weight.data = pissa_result['lora_A'].contiguous().to(lora_A_module.weight.device)
            lora_B_module.weight.data = pissa_result['lora_B'].contiguous().to(lora_B_module.weight.device)

            # Modify base weight to residual (optional)
            if modify_base_weight:
                base_layer.weight.data = pissa_result['residual'].contiguous().to(base_layer.weight.device)

            init_count += 1

            if verbose:
                print(f"  [Init] {name}: rank={lora_rank}, "
                      f"A={tuple(lora_A_module.weight.shape)}, "
                      f"B={tuple(lora_B_module.weight.shape)}")

        except Exception as e:
            if verbose:
                print(f"  [Error] {name}: {e}")
            continue

    # Memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if verbose:
        print(f"\n[PiSSA] Initialized {init_count} LoRA layers with PiSSA")
        if modify_base_weight:
            print("[PiSSA] Base weights modified to residual (W - B @ A)")
        else:
            print("[PiSSA] Base weights unchanged (standard LoRA behavior)")
        print("=" * 70)

    return init_count


def apply_pissa_init_to_momentum_lora(
    model: nn.Module,
    niter: int = 4,
    modify_base_weight: bool = True,
    target_modules: Optional[List[str]] = None,
    verbose: bool = True
) -> int:
    from models.momentum_lora import MomentumLoRALinear

    if verbose:
        print("\n" + "=" * 70)
        print("[PiSSA] Applying PiSSA to Momentum LoRA (velocity only)")
        print("=" * 70)
        print(f"[PiSSA] niter: {niter}")
        print(f"[PiSSA] modify_base_weight: {modify_base_weight}")

    init_count = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Default target modules
    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'dense', 'gate_proj', 'up_proj', 'down_proj', 'fc1', 'fc2']

    for name, module in model.named_modules():
        if not isinstance(module, MomentumLoRALinear):
            continue

        is_target = any(target in name for target in target_modules)
        if not is_target:
            continue

        try:
            # Get original weight
            original_weight = module.weight.data

            # Velocity LoRA rank (A1, B1)
            lora_rank = module.A1.shape[0]

            # Compute PiSSA initialization
            pissa_result = compute_pissa_init(
                weight=original_weight,
                rank=lora_rank,
                niter=niter,
                device=device
            )

            # Apply PiSSA initialization to velocity LoRA (A1, B1) (contiguous for safetensors compatibility)
            module.A1.data = pissa_result['lora_A'].contiguous().to(module.A1.device)
            module.B1.data = pissa_result['lora_B'].contiguous().to(module.B1.device)

            # Modify base weight to residual (optional)
            if modify_base_weight:
                module.weight.data = pissa_result['residual'].contiguous().to(module.weight.device)

            init_count += 1

            if verbose:
                print(f"  [Init] {name}: velocity rank={lora_rank}, "
                      f"A1={tuple(module.A1.shape)}, B1={tuple(module.B1.shape)}")

        except Exception as e:
            if verbose:
                print(f"  [Error] {name}: {e}")
            continue

    # Memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if verbose:
        print(f"\n[PiSSA] Initialized {init_count} Momentum LoRA layers with PiSSA")
        if modify_base_weight:
            print("[PiSSA] Base weights modified to residual (W - B @ A)")
        print("=" * 70)

    return init_count


def validate_pissa_init(model: nn.Module, verbose: bool = True) -> Dict[str, float]:
    if verbose:
        print("\n[PiSSA Validation] Checking LoRA initialization...")

    norms = []

    for name, module in model.named_modules():
        if not (hasattr(module, 'lora_A') and hasattr(module, 'lora_B')):
            continue

        active_adapter = getattr(module, 'active_adapter', None)
        if active_adapter is None:
            continue

        if isinstance(active_adapter, (list, tuple, set)):
            active_adapter = list(active_adapter)[0]

        try:
            lora_A = module.lora_A[active_adapter].weight.data
            lora_B = module.lora_B[active_adapter].weight.data

            # Compute B @ A norm
            ba_product = lora_B @ lora_A
            norm = torch.norm(ba_product).item()
            norms.append(norm)

        except Exception:
            continue

    if not norms:
        if verbose:
            print("[PiSSA Validation] No LoRA layers found")
        return {}

    stats = {
        'avg_lora_norm': sum(norms) / len(norms),
        'max_lora_norm': max(norms),
        'min_lora_norm': min(norms),
        'num_layers': len(norms)
    }

    if verbose:
        print(f"[PiSSA Validation] Checked {stats['num_layers']} layers")
        print(f"[PiSSA Validation] ||B @ A|| stats:")
        print(f"  - Average: {stats['avg_lora_norm']:.6f}")
        print(f"  - Max: {stats['max_lora_norm']:.6f}")
        print(f"  - Min: {stats['min_lora_norm']:.6f}")

        if stats['avg_lora_norm'] > 1e-6:
            print("[PiSSA Validation] ✓ PiSSA initialization confirmed (B @ A != 0)")
        else:
            print("[PiSSA Validation] ⚠ LoRA appears to be standard initialized (B @ A ≈ 0)")

    return stats


# ==================== Orthogonal SVD-based LoRA Initialization ====================

@dataclass
class OrthogonalInitConfig:
    """Orthogonal initialization configuration"""
    enabled: bool = False
    scale: float = 0.01
    target_modules: Optional[List[str]] = None


def compute_orthogonal_init(
    weight: torch.Tensor,
    rank: int,
    scale: float = 0.01,
    device: str = "cuda",
    niter: int = 4
) -> Dict[str, torch.Tensor]:
    weight_float = weight.float().to(device)
    out_dim, in_dim = weight_float.shape

    actual_rank = min(rank, min(out_dim, in_dim))

    U_top, S, V_top = torch.svd_lowrank(weight_float, q=actual_rank, niter=niter)
    # U_top: [out_dim, rank], V_top: [in_dim, rank]

    lora_A = torch.randn(actual_rank, in_dim, device=device) * scale
    lora_B = torch.randn(out_dim, actual_rank, device=device) * scale

    lora_A = lora_A - (lora_A @ V_top) @ V_top.T  # [rank, in_dim]

    lora_B = lora_B - U_top @ (U_top.T @ lora_B)  # [out_dim, rank]

    return {
        'lora_A': lora_A.float().contiguous(),
        'lora_B': lora_B.float().contiguous(),
    }


def apply_orthogonal_init(
    model: nn.Module,
    scale: float = 0.01,
    target_modules: Optional[List[str]] = None,
    verbose: bool = True
) -> int:
    if verbose:
        print("\n" + "=" * 70)
        print("[Orthogonal] Applying Orthogonal LoRA Initialization (Top-r Projection)")
        print("=" * 70)
        print(f"[Orthogonal] scale: {scale}")
        print(f"[Orthogonal] Method: Random init + Top-r projection removal")
        print(f"[Orthogonal] Base weight: unchanged")

    init_count = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Default target modules for LLaMA
    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'dense', 'gate_proj', 'up_proj', 'down_proj', 'fc1', 'fc2']

    for name, module in model.named_modules():
        if not (hasattr(module, 'lora_A') and hasattr(module, 'lora_B')):
            continue

        is_target = any(target in name for target in target_modules)
        if not is_target:
            continue

        # Get active adapter
        active_adapter = getattr(module, 'active_adapter', None)
        if active_adapter is None:
            continue

        if isinstance(active_adapter, (list, tuple, set)):
            active_adapter = list(active_adapter)[0]

        try:
            # Get LoRA modules
            lora_A_module = module.lora_A[active_adapter]
            lora_B_module = module.lora_B[active_adapter]

            # Get base layer and original weight
            base_layer = getattr(module, 'base_layer', None)
            if base_layer is None:
                if verbose:
                    print(f"  [Skip] {name}: No base_layer found")
                continue

            original_weight = base_layer.weight.data

            # Get LoRA rank from existing A matrix
            lora_rank = lora_A_module.weight.shape[0]

            # Compute Orthogonal initialization
            orth_result = compute_orthogonal_init(
                weight=original_weight,
                rank=lora_rank,
                scale=scale,
                device=device
            )

            # Apply Orthogonal initialization to LoRA weights (contiguous for safetensors compatibility)
            lora_A_module.weight.data = orth_result['lora_A'].contiguous().to(lora_A_module.weight.device)
            lora_B_module.weight.data = orth_result['lora_B'].contiguous().to(lora_B_module.weight.device)


            init_count += 1

            if verbose:
                print(f"  [Init] {name}: rank={lora_rank}, "
                      f"A={tuple(lora_A_module.weight.shape)}, "
                      f"B={tuple(lora_B_module.weight.shape)}")

        except Exception as e:
            if verbose:
                print(f"  [Error] {name}: {e}")
            continue

    # Memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if verbose:
        print(f"\n[Orthogonal] Initialized {init_count} LoRA layers with Orthogonal SVD")
        print("[Orthogonal] Base weights unchanged")
        print("=" * 70)

    return init_count


def apply_orthogonal_init_to_momentum_lora(
    model: nn.Module,
    scale: float = 0.01,
    target_modules: Optional[List[str]] = None,
    verbose: bool = True
) -> int:
    from models.momentum_lora import MomentumLoRALinear

    if verbose:
        print("\n" + "=" * 70)
        print("[Orthogonal] Applying Orthogonal to Momentum LoRA (velocity only)")
        print("=" * 70)
        print(f"[Orthogonal] scale: {scale}")
        print(f"[Orthogonal] Method: Random init + Top-r projection removal")

    init_count = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Default target modules
    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'dense', 'gate_proj', 'up_proj', 'down_proj', 'fc1', 'fc2']

    for name, module in model.named_modules():
        if not isinstance(module, MomentumLoRALinear):
            continue

        is_target = any(target in name for target in target_modules)
        if not is_target:
            continue

        try:
            # Get original weight
            original_weight = module.weight.data

            # Velocity LoRA rank (A1, B1)
            lora_rank = module.A1.shape[0]

            # Compute Orthogonal initialization
            orth_result = compute_orthogonal_init(
                weight=original_weight,
                rank=lora_rank,
                scale=scale,
                device=device
            )

            # Apply Orthogonal initialization to velocity LoRA (A1, B1) (contiguous for safetensors compatibility)
            module.A1.data = orth_result['lora_A'].contiguous().to(module.A1.device)
            module.B1.data = orth_result['lora_B'].contiguous().to(module.B1.device)


            init_count += 1

            if verbose:
                print(f"  [Init] {name}: velocity rank={lora_rank}, "
                      f"A1={tuple(module.A1.shape)}, B1={tuple(module.B1.shape)}")

        except Exception as e:
            if verbose:
                print(f"  [Error] {name}: {e}")
            continue

    # Memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if verbose:
        print(f"\n[Orthogonal] Initialized {init_count} Momentum LoRA layers with Orthogonal SVD")
        print("[Orthogonal] Base weights unchanged")
        print("=" * 70)

    return init_count


def validate_orthogonal_init(model: nn.Module, verbose: bool = True) -> Dict[str, float]:
    if verbose:
        print("\n[Orthogonal Validation] Checking LoRA initialization...")

    norms = []

    for name, module in model.named_modules():
        if not (hasattr(module, 'lora_A') and hasattr(module, 'lora_B')):
            continue

        active_adapter = getattr(module, 'active_adapter', None)
        if active_adapter is None:
            continue

        if isinstance(active_adapter, (list, tuple, set)):
            active_adapter = list(active_adapter)[0]

        try:
            lora_A = module.lora_A[active_adapter].weight.data
            lora_B = module.lora_B[active_adapter].weight.data

            # Compute B @ A norm
            ba_product = lora_B @ lora_A
            norm = torch.norm(ba_product).item()
            norms.append(norm)

        except Exception:
            continue

    if not norms:
        if verbose:
            print("[Orthogonal Validation] No LoRA layers found")
        return {}

    stats = {
        'avg_lora_norm': sum(norms) / len(norms),
        'max_lora_norm': max(norms),
        'min_lora_norm': min(norms),
        'num_layers': len(norms)
    }

    if verbose:
        print(f"[Orthogonal Validation] Checked {stats['num_layers']} layers")
        print(f"[Orthogonal Validation] ||B @ A|| stats:")
        print(f"  - Average: {stats['avg_lora_norm']:.6f}")
        print(f"  - Max: {stats['max_lora_norm']:.6f}")
        print(f"  - Min: {stats['min_lora_norm']:.6f}")
        print("[Orthogonal Validation] Using bottom-r singular vectors (weak directions)")

    return stats


# ==================== Cache-based Initialization ====================

def _find_cache_entry_for_module(
    svd_cache: Dict,
    module_name: str
) -> Optional[Dict]:
    for cache_name, cache_data in svd_cache.items():
        # Clean up names for matching
        module_parts = module_name.split('.')
        clean_parts = [p for p in module_parts if p not in ('base_model', 'model')]
        clean_name = '.'.join(clean_parts)

        if clean_name in cache_name or cache_name.endswith(clean_name + '.weight'):
            return cache_data
    return None


def apply_pissa_init_with_cache(
    model: nn.Module,
    svd_cache: Dict[str, Dict[str, torch.Tensor]],
    modify_base_weight: bool = True,
    target_modules: Optional[List[str]] = None,
    verbose: bool = True
) -> int:
    if verbose:
        print("\n" + "=" * 70)
        print("[PiSSA+Cache] Applying PiSSA using unified SVD cache")
        print("=" * 70)
        print(f"[PiSSA+Cache] modify_base_weight: {modify_base_weight}")
        print(f"[PiSSA+Cache] Cache entries: {len(svd_cache)}")

    init_count = 0

    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'dense', 'gate_proj', 'up_proj', 'down_proj', 'fc1', 'fc2']

    for name, module in model.named_modules():
        if not (hasattr(module, 'lora_A') and hasattr(module, 'lora_B')):
            continue

        is_target = any(target in name for target in target_modules)
        if not is_target:
            continue

        active_adapter = getattr(module, 'active_adapter', None)
        if active_adapter is None:
            continue

        if isinstance(active_adapter, (list, tuple, set)):
            active_adapter = list(active_adapter)[0]

        try:
            lora_A_module = module.lora_A[active_adapter]
            lora_B_module = module.lora_B[active_adapter]

            base_layer = getattr(module, 'base_layer', None)
            if base_layer is None:
                continue

            original_weight = base_layer.weight.data
            lora_rank = lora_A_module.weight.shape[0]

            # Find cache entry
            cache_entry = _find_cache_entry_for_module(svd_cache, name)
            if cache_entry is None:
                if verbose:
                    print(f"  [Skip] {name}: No cache entry found")
                continue

            # Extract from cache
            device = original_weight.device
            U = cache_entry['U'].to(device)      # [out_dim, cached_rank]
            S = cache_entry['S'].to(device)      # [cached_rank]
            Vt = cache_entry['Vt'].to(device)    # [cached_rank, in_dim]

            # Use top-r
            actual_rank = min(lora_rank, len(S))
            U_r = U[:, :actual_rank]
            S_r = S[:actual_rank]
            Vt_r = Vt[:actual_rank, :]

            # sqrt(S) scaling
            sqrt_S = torch.sqrt(S_r)

            # LoRA initialization
            lora_A = sqrt_S.unsqueeze(1) * Vt_r  # [rank, in_dim]
            lora_B = U_r * sqrt_S.unsqueeze(0)   # [out_dim, rank]

            # Apply (contiguous for safetensors compatibility)
            lora_A_module.weight.data = lora_A.float().contiguous().to(lora_A_module.weight.device)
            lora_B_module.weight.data = lora_B.float().contiguous().to(lora_B_module.weight.device)

            # Modify base weight to residual
            if modify_base_weight:
                low_rank_approx = lora_B @ lora_A
                residual = original_weight.float() - low_rank_approx
                base_layer.weight.data = residual.to(original_weight.dtype).contiguous().to(base_layer.weight.device)

            init_count += 1

            if verbose:
                print(f"  [Init] {name}: rank={lora_rank} (from cache)")

        except Exception as e:
            if verbose:
                print(f"  [Error] {name}: {e}")
            continue

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if verbose:
        print(f"\n[PiSSA+Cache] Initialized {init_count} LoRA layers")
        print("=" * 70)

    return init_count


def apply_orthogonal_init_with_cache(
    model: nn.Module,
    svd_cache: Dict[str, Dict[str, torch.Tensor]],
    scale: float = 0.01,
    target_modules: Optional[List[str]] = None,
    verbose: bool = True
) -> int:
    if verbose:
        print("\n" + "=" * 70)
        print("[Orthogonal+Cache] Applying Orthogonal init using Top-r projection")
        print("=" * 70)
        print(f"[Orthogonal+Cache] scale: {scale}")
        print(f"[Orthogonal+Cache] Cache entries: {len(svd_cache)}")
        print("[Orthogonal+Cache] Method: Random init + Top-r projection removal")

    init_count = 0
    fallback_count = 0

    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'dense', 'gate_proj', 'up_proj', 'down_proj', 'fc1', 'fc2']

    for name, module in model.named_modules():
        if not (hasattr(module, 'lora_A') and hasattr(module, 'lora_B')):
            continue

        is_target = any(target in name for target in target_modules)
        if not is_target:
            continue

        active_adapter = getattr(module, 'active_adapter', None)
        if active_adapter is None:
            continue

        if isinstance(active_adapter, (list, tuple, set)):
            active_adapter = list(active_adapter)[0]

        try:
            lora_A_module = module.lora_A[active_adapter]
            lora_B_module = module.lora_B[active_adapter]

            base_layer = getattr(module, 'base_layer', None)
            if base_layer is None:
                continue

            original_weight = base_layer.weight.data
            lora_rank = lora_A_module.weight.shape[0]
            out_dim, in_dim = original_weight.shape

            # Find cache entry
            cache_entry = _find_cache_entry_for_module(svd_cache, name)

            if cache_entry is not None:
                device = original_weight.device

                U_top = cache_entry['U'].to(device)      # [out_dim, cached_rank]
                Vt = cache_entry['Vt'].to(device)        # [cached_rank, in_dim]
                V_top = Vt.T                              # [in_dim, cached_rank]
                cached_rank = cache_entry['full_rank']

                # Use min of lora_rank and cached_rank for projection
                proj_rank = min(lora_rank, cached_rank)
                U_proj = U_top[:, :proj_rank]    # [out_dim, proj_rank]
                V_proj = V_top[:, :proj_rank]    # [in_dim, proj_rank]

                lora_A = torch.randn(lora_rank, in_dim, device=device) * scale
                lora_B = torch.randn(out_dim, lora_rank, device=device) * scale

                lora_A = lora_A - (lora_A @ V_proj) @ V_proj.T
                lora_B = lora_B - U_proj @ (U_proj.T @ lora_B)

            else:
                # No cache entry, compute directly
                if verbose:
                    print(f"  [Fallback] {name}: Computing SVD (no cache entry)")
                device = original_weight.device
                orth_result = compute_orthogonal_init(original_weight, lora_rank, scale, str(device))
                lora_A = orth_result['lora_A']
                lora_B = orth_result['lora_B']
                fallback_count += 1

            # Apply (contiguous for safetensors compatibility)
            lora_A_module.weight.data = lora_A.float().contiguous().to(lora_A_module.weight.device)
            lora_B_module.weight.data = lora_B.float().contiguous().to(lora_B_module.weight.device)

            init_count += 1

            if verbose and cache_entry is not None:
                print(f"  [Init] {name}: rank={lora_rank} (from cache, projection)")

        except Exception as e:
            if verbose:
                print(f"  [Error] {name}: {e}")
            continue

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if verbose:
        print(f"\n[Orthogonal+Cache] Initialized {init_count} LoRA layers")
        if fallback_count > 0:
            print(f"[Orthogonal+Cache] Fallback to direct computation: {fallback_count} layers")
        print("=" * 70)

    return init_count
