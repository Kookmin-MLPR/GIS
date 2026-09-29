
import os
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


# Strategy presets
STRATEGIES = {
    'recover': {
        'lambda_U': 0.001,
        'lambda_V': 0.001,
        'description': 'Align with original W - good for high pruning rates'
    },
    'balanced': {
        'lambda_U': 0.01,
        'lambda_V': 0.01,
        'description': 'Balance recovery and new learning'
    },
    'orthogonal': {
        'lambda_U': 0.1,
        'lambda_V': 0.1,
        'description': 'Learn orthogonal directions - good for new tasks'
    }
}


def compute_svd_cache(
    model: nn.Module,
    rank_k: int = 64,
    target_modules: Optional[list] = None,
    device: str = "cuda"
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Compute and cache top-k singular vectors for each weight matrix.

    This should be called BEFORE pruning/training with the original model.

    Args:
        model: Original model before pruning
        rank_k: Number of top singular vectors to keep (default: 64)
        target_modules: List of module name patterns to target (e.g., ['q_proj', 'k_proj'])
                       If None, targets all 2D weight matrices
        device: Device for computation

    Returns:
        svd_cache: Dictionary mapping layer names to {'U_k': U_k, 'V_k': V_k}
    """
    print("\n" + "=" * 70)
    print("[SVD Cache] Computing SVD for soft orthogonal regularization...")
    print(f"[SVD Cache] Using top-{rank_k} singular vectors")
    print("=" * 70)

    svd_cache = {}
    cached_count = 0

    # Default target modules for LLaMA-style models
    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

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
            weight = param.data.float()  # Convert to float32 for SVD stability

            # Determine actual rank (can't exceed matrix dimensions)
            actual_rank = min(rank_k, min(weight.shape))

            # Compute truncated SVD using torch.svd_lowrank for efficiency
            try:
                # Move to device if needed
                weight_device = weight.to(device)

                # Use randomized SVD for efficiency (faster for large matrices)
                U, S, Vh = torch.svd_lowrank(weight_device, q=actual_rank)

                # U: [out_dim, k], Vh: [k, in_dim]
                U_k = U.cpu().clone()  # Store on CPU to save GPU memory
                V_k = Vh.t().cpu().clone()  # Transpose to [in_dim, k] and store on CPU

                # Clean up to save GPU memory
                del weight_device, U, S, Vh

                svd_cache[name] = {
                    'U_k': U_k,
                    'V_k': V_k
                }

                cached_count += 1
                print(f"  Cached: {name} | U_k: {U_k.shape}, V_k: {V_k.shape}")

            except Exception as e:
                print(f"  [Warning] Failed to compute SVD for {name}: {e}")
                continue

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n[SVD Cache] Cached {cached_count} weight matrices")
    print("=" * 70)

    return svd_cache


def save_svd_cache(svd_cache: Dict, save_path: str):
    """
    Save SVD cache to disk for later use.

    Args:
        svd_cache: SVD cache dictionary
        save_path: Path to save the cache (e.g., './svd_cache.pt')
    """
    print(f"[SVD Cache] Saving to {save_path}...")
    torch.save(svd_cache, save_path)
    print(f"[SVD Cache] Saved {len(svd_cache)} matrices")


def load_svd_cache(load_path: str) -> Dict:
    """
    Load SVD cache from disk.

    Args:
        load_path: Path to the cache file

    Returns:
        svd_cache: Loaded SVD cache dictionary
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"SVD cache not found at {load_path}")

    print(f"[SVD Cache] Loading from {load_path}...")
    svd_cache = torch.load(load_path)
    print(f"[SVD Cache] Loaded {len(svd_cache)} matrices")

    return svd_cache


def register_svd_buffers_to_model(
    model: nn.Module,
    svd_cache: Dict[str, Dict[str, torch.Tensor]]
) -> int:
    """
    Register SVD tensors as buffers on LoRA modules.

    This moves SVD tensors to GPU and registers them as non-trainable buffers,
    avoiding CPU->GPU transfer overhead during each training step.

    Args:
        model: Model with LoRA adapters (PEFT model)
        svd_cache: Pre-computed SVD components (on CPU)

    Returns:
        num_registered: Number of modules with registered SVD buffers
    """
    print("\n" + "=" * 70)
    print("[SVD Buffer] Registering SVD tensors as model buffers...")
    print("=" * 70)

    num_registered = 0
    total_memory = 0

    for name, module in model.named_modules():
        # Check if module has LoRA components
        if not (hasattr(module, 'lora_A') and hasattr(module, 'lora_B')):
            continue

        # Get base_layer to register buffers
        base_layer = getattr(module, 'base_layer', None)
        if base_layer is None:
            continue

        # Find matching SVD cache entry
        svd_entry = None
        matched_cache_name = None
        for cache_name, cache_data in svd_cache.items():
            # Match by checking if the module name is in the cache name
            module_parts = name.split('.')
            clean_parts = [p for p in module_parts if p not in ('base_model', 'model')]
            clean_name = '.'.join(clean_parts)

            if clean_name in cache_name or cache_name.endswith(clean_name + '.weight'):
                svd_entry = cache_data
                matched_cache_name = cache_name
                break

        if svd_entry is None:
            continue

        # Register U_k and V_k as buffers on the base_layer
        # These will automatically move to the correct device with the model
        U_k = svd_entry['U_k']
        V_k = svd_entry['V_k']

        # Register as buffers (persistent=True means they're saved with state_dict)
        base_layer.register_buffer('svd_U_k', U_k.clone(), persistent=False)
        base_layer.register_buffer('svd_V_k', V_k.clone(), persistent=False)

        num_registered += 1
        total_memory += U_k.numel() * 4 + V_k.numel() * 4  # FP32

        print(f"  Registered: {name} | U_k: {U_k.shape}, V_k: {V_k.shape}")

    print(f"\n[SVD Buffer] Registered {num_registered} modules")
    print(f"[SVD Buffer] Total memory: ~{total_memory / 1024 / 1024:.1f} MB")
    print("[SVD Buffer] Buffers will move to GPU with model automatically")
    print("=" * 70)

    return num_registered


def soft_orthogonal_loss(
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    U_k: torch.Tensor,
    V_k: torch.Tensor,
    lambda_U: float,
    lambda_V: float,
    mask_A: Optional[torch.Tensor] = None,
    mask_B: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute soft orthogonal regularization loss for a single LoRA layer.

    Args:
        lora_A: LoRA A matrix [rank, in_dim]
        lora_B: LoRA B matrix [out_dim, rank]
        U_k: Left singular vectors [out_dim, k]
        V_k: Right singular vectors [in_dim, k]
        lambda_U: Regularization strength for output subspace
        lambda_V: Regularization strength for input subspace
        mask_A: Optional pruning mask for A [rank, in_dim]
        mask_B: Optional pruning mask for B [out_dim, rank]

    Returns:
        loss: Scalar regularization loss
    """
    # Apply masks if provided
    if mask_A is not None:
        masked_A = mask_A * lora_A
    else:
        masked_A = lora_A

    if mask_B is not None:
        masked_B = mask_B * lora_B
    else:
        masked_B = lora_B

    # Move U_k, V_k to same device as LoRA weights
    U_k = U_k.to(lora_B.device)
    V_k = V_k.to(lora_A.device)

    # Output subspace projection: ||U_k^T (M_B ⊙ B)||²_F
    # U_k: [out_dim, k], masked_B: [out_dim, rank]
    # proj_B: [k, rank]
    proj_B = U_k.t() @ masked_B.float()
    loss_U = lambda_U * torch.sum(proj_B ** 2)

    # Input subspace projection: ||(M_A ⊙ A) V_k||²_F
    # masked_A: [rank, in_dim], V_k: [in_dim, k]
    # proj_A: [rank, k]
    proj_A = masked_A.float() @ V_k
    loss_V = lambda_V * torch.sum(proj_A ** 2)

    return loss_U + loss_V


def compute_soft_orthogonal_loss_for_model(
    model: nn.Module,
    lambda_U: float = 0.01,
    lambda_V: float = 0.01,
    svd_cache: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
) -> Tuple[torch.Tensor, int]:
    """
    Compute total soft orthogonal regularization loss for the entire model.

    This function iterates through all LoRA modules in the model and computes
    the soft orthogonal loss for each one that has SVD buffers registered.

    Uses registered buffers (svd_U_k, svd_V_k) on base_layer for efficient GPU computation.
    Falls back to svd_cache dictionary if buffers are not registered.

    Args:
        model: Model with LoRA adapters (PEFT model)
        lambda_U: Regularization strength for output subspace
        lambda_V: Regularization strength for input subspace
        svd_cache: Optional fallback SVD cache (deprecated, use register_svd_buffers_to_model instead)

    Returns:
        total_loss: Total soft orthogonal regularization loss
        num_layers: Number of LoRA layers that contributed to the loss
    """
    total_loss = torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')
    num_layers = 0

    for name, module in model.named_modules():
        # Check if module has LoRA components
        if not (hasattr(module, 'lora_A') and hasattr(module, 'lora_B')):
            continue

        # Get active adapter
        active_adapter = getattr(module, 'active_adapter', None)
        if active_adapter is None:
            continue

        if isinstance(active_adapter, (list, tuple, set)):
            active_adapter = list(active_adapter)[0]

        try:
            # Get LoRA weights
            lora_A_module = module.lora_A[active_adapter]
            lora_B_module = module.lora_B[active_adapter]
            lora_A = lora_A_module.weight  # [rank, in_dim]
            lora_B = lora_B_module.weight  # [out_dim, rank]

            # Get base_layer
            base_layer = getattr(module, 'base_layer', None)
            if base_layer is None:
                continue

            # Try to get SVD from registered buffers first (fast path - already on GPU)
            U_k = getattr(base_layer, 'svd_U_k', None)
            V_k = getattr(base_layer, 'svd_V_k', None)

            # Fallback to svd_cache dictionary if buffers not registered
            if U_k is None or V_k is None:
                if svd_cache is None:
                    continue

                # Find matching SVD cache entry
                svd_entry = None
                for cache_name, cache_data in svd_cache.items():
                    module_parts = name.split('.')
                    clean_parts = [p for p in module_parts if p not in ('base_model', 'model')]
                    clean_name = '.'.join(clean_parts)

                    if clean_name in cache_name or cache_name.endswith(clean_name + '.weight'):
                        svd_entry = cache_data
                        break

                if svd_entry is None:
                    continue

                U_k = svd_entry['U_k'].to(lora_B.device)
                V_k = svd_entry['V_k'].to(lora_A.device)

            # Get masks if they exist
            mask_A = None
            mask_B = None

            if hasattr(base_layer, 'weight_input_mask'):
                mask_A = base_layer.weight_input_mask
                if mask_A is not None:
                    mask_A = mask_A.unsqueeze(0).expand(lora_A.shape[0], -1)

            if hasattr(base_layer, 'weight_output_mask'):
                mask_B = base_layer.weight_output_mask
                if mask_B is not None:
                    mask_B = mask_B.unsqueeze(1).expand(-1, lora_B.shape[1])

            # Compute loss for this layer (U_k, V_k already on correct device)
            layer_loss = soft_orthogonal_loss(
                lora_A, lora_B, U_k, V_k,
                lambda_U, lambda_V,
                mask_A, mask_B
            )

            total_loss = total_loss + layer_loss
            num_layers += 1

        except Exception as e:
            # Skip layers that fail
            continue

    return total_loss, num_layers


def validate_soft_orth_setup(model: nn.Module, svd_cache: Dict) -> bool:
    """
    Validate that soft orthogonal setup is correct.

    Args:
        model: Model with LoRA adapters
        svd_cache: Pre-computed SVD cache

    Returns:
        is_valid: True if validation passed
    """
    print("\n" + "=" * 70)
    print("[Validation] Validating soft orthogonal regularization setup...")
    print("=" * 70)

    issues = []
    matched_layers = 0

    for name, module in model.named_modules():
        if not (hasattr(module, 'lora_A') and hasattr(module, 'lora_B')):
            continue

        active_adapter = getattr(module, 'active_adapter', None)
        if active_adapter is None:
            continue

        if isinstance(active_adapter, (list, tuple, set)):
            active_adapter = list(active_adapter)[0]

        try:
            lora_A = module.lora_A[active_adapter].weight
            lora_B = module.lora_B[active_adapter].weight

            # Find matching SVD cache
            found_match = False
            for cache_name, cache_data in svd_cache.items():
                module_parts = name.split('.')
                clean_parts = [p for p in module_parts if p not in ('base_model', 'model')]
                clean_name = '.'.join(clean_parts)

                if clean_name in cache_name or cache_name.endswith(clean_name + '.weight'):
                    U_k = cache_data['U_k']
                    V_k = cache_data['V_k']

                    # Check dimensions
                    if U_k.shape[0] != lora_B.shape[0]:
                        issues.append(f"Dimension mismatch in {name}: U_k[0]={U_k.shape[0]} vs lora_B[0]={lora_B.shape[0]}")
                    elif V_k.shape[0] != lora_A.shape[1]:
                        issues.append(f"Dimension mismatch in {name}: V_k[0]={V_k.shape[0]} vs lora_A[1]={lora_A.shape[1]}")
                    else:
                        found_match = True
                        matched_layers += 1
                    break

            if not found_match:
                issues.append(f"No matching SVD cache for {name}")

        except Exception as e:
            issues.append(f"Error checking {name}: {e}")

    print(f"\n[Validation] Found {matched_layers} LoRA layers with matching SVD cache")

    if issues:
        print("\n[Validation] Issues found:")
        for issue in issues[:10]:  # Show first 10 issues
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more issues")
        print("\n[Validation] Setup validation FAILED")
        return False
    else:
        print("\n[Validation] All validations passed!")
        return True


class SoftOrthogonalConfig:
    """Configuration class for soft orthogonal regularization."""

    # Valid apply_timing options
    TIMING_BEFORE_WARMUP = "before_warmup"
    TIMING_AFTER_WARMUP = "after_warmup"
    TIMING_ALWAYS = "always"
    VALID_TIMINGS = [TIMING_BEFORE_WARMUP, TIMING_AFTER_WARMUP, TIMING_ALWAYS]

    def __init__(
        self,
        use_soft_orthogonal: bool = False,
        svd_rank: int = 64,
        lambda_U: float = 0.01,
        lambda_V: float = 0.01,
        strategy: Optional[str] = None,
        svd_cache_path: Optional[str] = None,
        apply_timing: str = "always"
    ):
        """
        Initialize soft orthogonal configuration.

        Args:
            use_soft_orthogonal: Whether to use soft orthogonal regularization
            svd_rank: Number of singular vectors to use (default: 64)
            lambda_U: Regularization strength for output subspace
            lambda_V: Regularization strength for input subspace
            strategy: Preset strategy name ('recover', 'balanced', 'orthogonal')
                     If set, overrides lambda_U and lambda_V
            svd_cache_path: Path to pre-computed SVD cache file
            apply_timing: When to apply soft orthogonal regularization
                - "before_warmup": Apply only during warmup (step < warmup_steps)
                - "after_warmup": Apply only after warmup (step >= warmup_steps)
                - "always": Apply throughout training (default)
        """
        self.use_soft_orthogonal = use_soft_orthogonal
        self.svd_rank = svd_rank
        self.svd_cache_path = svd_cache_path

        # Validate and set apply_timing
        if apply_timing not in self.VALID_TIMINGS:
            raise ValueError(f"apply_timing must be one of {self.VALID_TIMINGS}, got '{apply_timing}'")
        self.apply_timing = apply_timing

        # Apply strategy if specified
        if strategy is not None and strategy in STRATEGIES:
            self.lambda_U = STRATEGIES[strategy]['lambda_U']
            self.lambda_V = STRATEGIES[strategy]['lambda_V']
            self.strategy = strategy
        else:
            self.lambda_U = lambda_U
            self.lambda_V = lambda_V
            self.strategy = 'custom'

    def should_apply_at_step(self, current_step: int, warmup_steps: int) -> bool:
        """
        Check if soft orthogonal should be applied at the current step.

        Args:
            current_step: Current training step (global_step)
            warmup_steps: Total warmup steps

        Returns:
            True if soft orthogonal should be applied
        """
        if self.apply_timing == self.TIMING_ALWAYS:
            return True
        elif self.apply_timing == self.TIMING_BEFORE_WARMUP:
            return current_step < warmup_steps
        elif self.apply_timing == self.TIMING_AFTER_WARMUP:
            return current_step >= warmup_steps
        return True  # Fallback to always

    def __repr__(self):
        return (f"SoftOrthogonalConfig("
                f"use_soft_orthogonal={self.use_soft_orthogonal}, "
                f"svd_rank={self.svd_rank}, "
                f"lambda_U={self.lambda_U}, "
                f"lambda_V={self.lambda_V}, "
                f"strategy={self.strategy}, "
                f"apply_timing={self.apply_timing})")


# ==================== Unified Cache Integration ====================

def convert_unified_cache_to_soft_orth(
    unified_cache: Dict[str, Dict[str, torch.Tensor]],
    rank_k: int = 64,
    verbose: bool = True
) -> Dict[str, Dict[str, torch.Tensor]]:
    if verbose:
        print("\n" + "=" * 70)
        print("[Unified->SoftOrth] Converting unified cache to soft orth format")
        print(f"[Unified->SoftOrth] Using top-{rank_k} singular vectors")
        print("=" * 70)

    soft_orth_cache = {}

    for name, entry in unified_cache.items():
        U = entry['U']       # [out_dim, cached_rank]
        Vt = entry['Vt']     # [cached_rank, in_dim]
        cached_rank = entry.get('full_rank', U.shape[1])

        actual_k = min(rank_k, cached_rank)

        # Top-k (dominant directions)
        U_k = U[:, :actual_k]            # [out_dim, k]
        V_k = Vt[:actual_k, :].T         # [in_dim, k] (transpose Vt to get V)

        soft_orth_cache[name] = {
            'U_k': U_k.clone(),
            'V_k': V_k.clone()
        }

    if verbose:
        print(f"[Unified->SoftOrth] Converted {len(soft_orth_cache)} entries")
        print("=" * 70)

    return soft_orth_cache


def is_unified_cache_format(cache: Dict) -> bool:
    if not cache:
        return False
    sample = next(iter(cache.values()))
    return 'U' in sample and 'S' in sample and 'Vt' in sample


def get_or_convert_svd_cache(
    cache: Dict,
    rank_k: int = 64,
    verbose: bool = False
) -> Dict[str, Dict[str, torch.Tensor]]:
    if is_unified_cache_format(cache):
        return convert_unified_cache_to_soft_orth(cache, rank_k, verbose)
    else:
        # Already soft orth format
        return cache
