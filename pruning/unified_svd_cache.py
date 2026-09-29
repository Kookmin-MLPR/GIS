
import os
import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class UnifiedSVDConfig:
    """Unified SVD cache configuration"""
    max_rank: int = 128
    use_randomized: bool = True
    target_modules: Optional[List[str]] = None
    cache_path: Optional[str] = None


def compute_unified_svd_cache(
    model: nn.Module,
    max_rank: int = 128,
    use_randomized: bool = True,
    target_modules: Optional[List[str]] = None,
    device: str = "cuda",
    verbose: bool = True
) -> Dict[str, Dict[str, torch.Tensor]]:
    if verbose:
        print("\n" + "=" * 70)
        print("[Unified SVD Cache] Computing SVD for all target modules...")
        print("=" * 70)
        print(f"[Config] max_rank: {max_rank}")
        print(f"[Config] use_randomized: {use_randomized}")
        if not use_randomized:
            print("[Config] Using Full SVD (required for Orthogonal init)")

    svd_cache = {}
    cached_count = 0
    total_memory = 0

    # Default target modules for LLaMA-style models
    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                         'gate_proj', 'up_proj', 'down_proj']

    with torch.no_grad():
        for name, param in model.named_parameters():
            # Check if this is a target weight matrix
            if 'weight' not in name:
                continue
            if param.dim() != 2:
                continue

            # Check if it matches target modules
            is_target = any(target in name for target in target_modules)
            if not is_target:
                continue

            # Get weight data
            weight = param.data.float().to(device)
            out_dim, in_dim = weight.shape

            # Determine actual rank
            min_dim = min(out_dim, in_dim)

            try:
                if use_randomized:
                    # Randomized SVD (faster, but only top-k)
                    actual_rank = min(max_rank, min_dim)
                    U, S, V = torch.svd_lowrank(weight, q=actual_rank, niter=4)
                    # V: [in_dim, rank], need Vt: [rank, in_dim]
                    Vt = V.T
                else:
                    # Full SVD (slower, but complete)
                    U_full, S_full, Vt_full = torch.linalg.svd(weight, full_matrices=False)
                    # U_full: [out_dim, min_dim]
                    # S_full: [min_dim]
                    # Vt_full: [min_dim, in_dim]

                    # Truncate to max_rank
                    actual_rank = min(max_rank, min_dim)
                    U = U_full[:, :actual_rank]
                    S = S_full[:actual_rank]
                    Vt = Vt_full[:actual_rank, :]

                    # Clean up full matrices
                    del U_full, S_full, Vt_full

                # Store on CPU to save GPU memory
                cache_entry = {
                    'U': U.cpu().clone(),      # [out_dim, rank]
                    'S': S.cpu().clone(),      # [rank]
                    'Vt': Vt.cpu().clone(),    # [rank, in_dim]
                    'full_rank': actual_rank,
                    'weight_shape': (out_dim, in_dim),
                    'is_full_svd': not use_randomized
                }

                svd_cache[name] = cache_entry
                cached_count += 1

                # Memory tracking
                entry_memory = (U.numel() + S.numel() + Vt.numel()) * 4  # FP32
                total_memory += entry_memory

                if verbose:
                    print(f"  Cached: {name}")
                    print(f"    Shape: {weight.shape} -> U:{U.shape}, S:{S.shape}, Vt:{Vt.shape}")

                # Clean up
                del weight, U, S, Vt

            except Exception as e:
                if verbose:
                    print(f"  [Warning] Failed to compute SVD for {name}: {e}")
                continue

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if verbose:
        print(f"\n[Unified SVD Cache] Cached {cached_count} weight matrices")
        print(f"[Unified SVD Cache] Total memory: ~{total_memory / 1024 / 1024:.1f} MB")
        print("=" * 70)

    return svd_cache


def save_unified_svd_cache(svd_cache: Dict, save_path: str, verbose: bool = True):
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    if verbose:
        print(f"[Unified SVD Cache] Saving to {save_path}...")

    torch.save(svd_cache, save_path)

    if verbose:
        file_size = os.path.getsize(save_path) / 1024 / 1024
        print(f"[Unified SVD Cache] Saved {len(svd_cache)} matrices ({file_size:.1f} MB)")


def load_unified_svd_cache(load_path: str, verbose: bool = True) -> Dict:
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"SVD cache not found at {load_path}")

    if verbose:
        print(f"[Unified SVD Cache] Loading from {load_path}...")

    svd_cache = torch.load(load_path, map_location='cpu')

    if verbose:
        print(f"[Unified SVD Cache] Loaded {len(svd_cache)} matrices")

        # Check if it's full SVD
        sample_entry = next(iter(svd_cache.values()))
        is_full = sample_entry.get('is_full_svd', False)
        max_rank = sample_entry.get('full_rank', 0)
        print(f"[Unified SVD Cache] Full SVD: {is_full}, Max rank: {max_rank}")

    return svd_cache


# ==================== PiSSA Integration ====================

def get_pissa_init_from_cache(
    svd_cache: Dict,
    layer_name: str,
    lora_rank: int,
    device: str = "cuda"
) -> Optional[Dict[str, torch.Tensor]]:
    # Find matching cache entry
    cache_entry = _find_cache_entry(svd_cache, layer_name)
    if cache_entry is None:
        return None

    U = cache_entry['U'].to(device)      # [out_dim, cached_rank]
    S = cache_entry['S'].to(device)      # [cached_rank]
    Vt = cache_entry['Vt'].to(device)    # [cached_rank, in_dim]

    # Use top-r (PiSSA uses dominant directions)
    actual_rank = min(lora_rank, len(S))
    U_r = U[:, :actual_rank]       # [out_dim, r]
    S_r = S[:actual_rank]          # [r]
    Vt_r = Vt[:actual_rank, :]     # [r, in_dim]

    # sqrt(S) scaling
    sqrt_S = torch.sqrt(S_r)  # [r]

    # LoRA initialization
    # A = diag(sqrt(S)) @ Vt = [rank, in_dim]
    lora_A = sqrt_S.unsqueeze(1) * Vt_r  # broadcast: [r, 1] * [r, in_dim]

    # B = U @ diag(sqrt(S)) = [out_dim, rank]
    lora_B = U_r * sqrt_S.unsqueeze(0)   # broadcast: [out_dim, r] * [1, r]

    return {
        'lora_A': lora_A.float(),
        'lora_B': lora_B.float(),
        'U_r': U_r,
        'S_r': S_r,
        'Vt_r': Vt_r
    }


def compute_pissa_residual(
    original_weight: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor
) -> torch.Tensor:
    low_rank_approx = lora_B @ lora_A
    residual = original_weight.float() - low_rank_approx
    return residual.to(original_weight.dtype)


# ==================== Orthogonal Init Integration ====================

def get_orthogonal_init_from_cache(
    svd_cache: Dict,
    layer_name: str,
    lora_rank: int,
    scale: float = 0.01,
    device: str = "cuda"
) -> Optional[Dict[str, torch.Tensor]]:
    cache_entry = _find_cache_entry(svd_cache, layer_name)
    if cache_entry is None:
        return None

    cached_rank = cache_entry['full_rank']
    out_dim, in_dim = cache_entry['weight_shape']

    U_top = cache_entry['U'].to(device)      # [out_dim, cached_rank]
    Vt = cache_entry['Vt'].to(device)        # [cached_rank, in_dim]
    V_top = Vt.T                              # [in_dim, cached_rank]

    # Use min of lora_rank and cached_rank for projection
    proj_rank = min(lora_rank, cached_rank)
    U_proj = U_top[:, :proj_rank]    # [out_dim, proj_rank]
    V_proj = V_top[:, :proj_rank]    # [in_dim, proj_rank]

    lora_A = torch.randn(lora_rank, in_dim, device=device) * scale
    lora_B = torch.randn(out_dim, lora_rank, device=device) * scale

    lora_A = lora_A - (lora_A @ V_proj) @ V_proj.T
    lora_B = lora_B - U_proj @ (U_proj.T @ lora_B)

    return {
        'lora_A': lora_A.float(),
        'lora_B': lora_B.float(),
    }


# ==================== Soft Orthogonal Integration ====================

def get_soft_orth_vectors_from_cache(
    svd_cache: Dict,
    layer_name: str,
    k: int,
    device: str = "cuda"
) -> Optional[Dict[str, torch.Tensor]]:
    cache_entry = _find_cache_entry(svd_cache, layer_name)
    if cache_entry is None:
        return None

    U = cache_entry['U']       # [out_dim, cached_rank]
    Vt = cache_entry['Vt']     # [cached_rank, in_dim]
    cached_rank = cache_entry['full_rank']

    actual_k = min(k, cached_rank)

    # Top-k (dominant directions)
    U_k = U[:, :actual_k].to(device)     # [out_dim, k]
    V_k = Vt[:actual_k, :].T.to(device)  # [in_dim, k] (transpose Vt to get V)

    return {
        'U_k': U_k,
        'V_k': V_k,
    }


def register_svd_buffers_from_unified_cache(
    model: nn.Module,
    svd_cache: Dict[str, Dict[str, torch.Tensor]],
    svd_rank: int = 64,
    verbose: bool = True
) -> int:
    if verbose:
        print("\n" + "=" * 70)
        print("[Unified SVD] Registering SVD buffers from unified cache...")
        print(f"[Unified SVD] Using top-{svd_rank} singular vectors for soft orth")
        print("=" * 70)

    num_registered = 0

    for name, module in model.named_modules():
        # Check if module has LoRA components
        if not (hasattr(module, 'lora_A') and hasattr(module, 'lora_B')):
            continue

        # Get base_layer to register buffers
        base_layer = getattr(module, 'base_layer', None)
        if base_layer is None:
            continue

        # Find matching SVD cache entry
        cache_entry = _find_cache_entry_for_module(svd_cache, name)
        if cache_entry is None:
            continue

        U = cache_entry['U']       # [out_dim, cached_rank]
        Vt = cache_entry['Vt']     # [cached_rank, in_dim]
        cached_rank = cache_entry['full_rank']

        actual_k = min(svd_rank, cached_rank)

        # Top-k (dominant directions)
        U_k = U[:, :actual_k]            # [out_dim, k]
        V_k = Vt[:actual_k, :].T         # [in_dim, k]

        # Register as buffers
        base_layer.register_buffer('svd_U_k', U_k.clone(), persistent=False)
        base_layer.register_buffer('svd_V_k', V_k.clone(), persistent=False)

        num_registered += 1

        if verbose:
            print(f"  Registered: {name} | U_k: {U_k.shape}, V_k: {V_k.shape}")

    if verbose:
        print(f"\n[Unified SVD] Registered {num_registered} modules")
        print("=" * 70)

    return num_registered


# ==================== Helper Functions ====================

def _find_cache_entry(
    svd_cache: Dict,
    layer_name: str
) -> Optional[Dict]:
    # Direct match
    if layer_name in svd_cache:
        return svd_cache[layer_name]

    # Try with .weight suffix
    if not layer_name.endswith('.weight'):
        if layer_name + '.weight' in svd_cache:
            return svd_cache[layer_name + '.weight']

    # Partial match (module name within cache name)
    for cache_name, cache_data in svd_cache.items():
        # Clean up PEFT-style names
        clean_layer = layer_name.replace('base_model.model.', '').replace('.base_layer', '')
        clean_cache = cache_name.replace('base_model.model.', '')

        if clean_layer in clean_cache or clean_cache.endswith(clean_layer):
            return cache_data

    return None


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


def get_cache_info(svd_cache: Dict) -> Dict:
    if not svd_cache:
        return {'num_layers': 0}

    sample = next(iter(svd_cache.values()))
    total_memory = 0

    for entry in svd_cache.values():
        total_memory += (entry['U'].numel() + entry['S'].numel() + entry['Vt'].numel()) * 4

    return {
        'num_layers': len(svd_cache),
        'max_rank': sample.get('full_rank', 0),
        'is_full_svd': sample.get('is_full_svd', False),
        'total_memory_mb': total_memory / 1024 / 1024,
        'layers': list(svd_cache.keys())
    }


def validate_cache_compatibility(
    svd_cache: Dict,
    lora_rank: int,
    soft_orth_rank: int,
    verbose: bool = True
) -> bool:
    if not svd_cache:
        if verbose:
            print("[Validation] Empty cache")
        return False

    info = get_cache_info(svd_cache)
    issues = []

    # Check rank sufficiency
    # Orthogonal now uses projection, so max(lora_rank, soft_orth_rank) is enough
    max_needed = max(lora_rank, soft_orth_rank)
    if info['max_rank'] < max_needed:
        issues.append(f"Cache rank ({info['max_rank']}) < needed ({max_needed})")

    if verbose:
        print("\n[Cache Validation]")
        print(f"  Cache layers: {info['num_layers']}")
        print(f"  Cache max rank: {info['max_rank']}")
        print(f"  Needed: LoRA rank={lora_rank}, Soft orth rank={soft_orth_rank}")

        if issues:
            print("\n[Issues]")
            for issue in issues:
                print(f"  - {issue}")
            print("\n[Result] NOT COMPATIBLE")
        else:
            print("\n[Result] COMPATIBLE")

    return len(issues) == 0
