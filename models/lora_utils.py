"""
LoRA Utility Functions for LLM Pruning Recovery

This module provides utility functions for adding LoRA (standard or momentum)
to language models. It supports both PEFT-based standard LoRA and custom
Momentum LoRA implementations.

Usage:
    # Standard LoRA (using PEFT)
    config = {'type': 'standard', 'rank': 16, ...}
    model = add_lora_to_model(model, config)

    # Momentum LoRA (custom implementation)
    config = {'type': 'momentum', 'momentum': {'rank1': 16, 'rank2': 32, ...}, ...}
    model = add_lora_to_model(model, config)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

from .momentum_lora import (
    MomentumLoRALinear,
    SVDMomentumLoRALayer,
    collect_momentum_layers,
    update_all_momentum_buffers,
    compute_total_soft_orth_loss,
)


@dataclass
class MomentumLoRAConfig:
    """Configuration for Momentum LoRA."""

    # LoRA type: 'standard' (PEFT) or 'momentum' (custom)
    lora_type: str = "standard"

    # Standard LoRA parameters (for PEFT)
    rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # Momentum LoRA parameters (for custom implementation)
    rank1: int = 16  # Velocity LoRA rank
    rank2: int = 32  # Momentum LoRA rank
    beta: float = 0.9  # Momentum decay rate
    reset_interval: int = 100  # Steps between velocity resets

    # Soft orthogonal regularization
    use_soft_orthogonal: bool = True
    lambda_U: float = 0.01
    lambda_V: float = 0.01
    svd_rank: int = 64

    # Target modules to apply LoRA
    target_modules: List[str] = None

    def __post_init__(self):
        if self.target_modules is None:
            # Default target modules for LLaMA-style models
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MomentumLoRAConfig":
        """Create config from dictionary."""
        lora_type = config_dict.get('type', 'standard')

        if lora_type == 'standard':
            return cls(
                lora_type='standard',
                rank=config_dict.get('rank', 16),
                lora_alpha=config_dict.get('lora_alpha', 16),
                lora_dropout=config_dict.get('dropout', 0.0),
                target_modules=config_dict.get('target_modules'),
                use_soft_orthogonal=config_dict.get('soft_orthogonal', {}).get('enabled', False),
                lambda_U=config_dict.get('soft_orthogonal', {}).get('lambda_U', 0.01),
                lambda_V=config_dict.get('soft_orthogonal', {}).get('lambda_V', 0.01),
                svd_rank=config_dict.get('soft_orthogonal', {}).get('svd_rank', 64),
            )
        else:
            momentum_config = config_dict.get('momentum', {})
            return cls(
                lora_type='momentum',
                rank1=momentum_config.get('rank1', 16),
                rank2=momentum_config.get('rank2', 32),
                beta=momentum_config.get('beta', 0.9),
                reset_interval=momentum_config.get('reset_interval', 100),
                target_modules=config_dict.get('target_modules'),
                use_soft_orthogonal=config_dict.get('soft_orthogonal', {}).get('enabled', True),
                lambda_U=config_dict.get('soft_orthogonal', {}).get('lambda_U', 0.01),
                lambda_V=config_dict.get('soft_orthogonal', {}).get('lambda_V', 0.01),
                svd_rank=config_dict.get('soft_orthogonal', {}).get('svd_rank', 64),
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        if self.lora_type == 'standard':
            return {
                'type': 'standard',
                'rank': self.rank,
                'lora_alpha': self.lora_alpha,
                'dropout': self.lora_dropout,
                'target_modules': self.target_modules,
                'soft_orthogonal': {
                    'enabled': self.use_soft_orthogonal,
                    'lambda_U': self.lambda_U,
                    'lambda_V': self.lambda_V,
                    'svd_rank': self.svd_rank,
                },
            }
        else:
            return {
                'type': 'momentum',
                'momentum': {
                    'rank1': self.rank1,
                    'rank2': self.rank2,
                    'beta': self.beta,
                    'reset_interval': self.reset_interval,
                },
                'target_modules': self.target_modules,
                'soft_orthogonal': {
                    'enabled': self.use_soft_orthogonal,
                    'lambda_U': self.lambda_U,
                    'lambda_V': self.lambda_V,
                    'svd_rank': self.svd_rank,
                },
            }


def add_lora_to_model(
    model: nn.Module,
    config: Union[Dict[str, Any], MomentumLoRAConfig],
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """
    Add LoRA (standard or momentum) to model.

    This is the main entry point for adding LoRA adapters to a model.
    Supports both PEFT-based standard LoRA and custom Momentum LoRA.

    Args:
        model: Model to modify
        config: LoRA configuration (dict or MomentumLoRAConfig)
        target_modules: List of module names to replace (e.g., ["q_proj", "v_proj"])
                       Overrides config.target_modules if provided

    Returns:
        Modified model with LoRA adapters

    Example:
        # Standard LoRA (using PEFT)
        config = {
            'type': 'standard',
            'rank': 16,
            'lora_alpha': 32,
            'dropout': 0.1,
        }
        model = add_lora_to_model(model, config)

        # Momentum LoRA (custom)
        config = {
            'type': 'momentum',
            'momentum': {
                'rank1': 16,
                'rank2': 32,
                'beta': 0.9,
                'reset_interval': 100,
            },
            'soft_orthogonal': {
                'enabled': True,
                'lambda_U': 0.01,
                'lambda_V': 0.01,
                'svd_rank': 64,
            },
        }
        model = add_lora_to_model(model, config)
    """
    # Convert dict to config if needed
    if isinstance(config, dict):
        config = MomentumLoRAConfig.from_dict(config)

    # Override target modules if provided
    if target_modules is not None:
        config.target_modules = target_modules

    lora_type = config.lora_type

    if lora_type == 'standard':
        # Use PEFT for standard LoRA
        model = _add_standard_lora(model, config)
    elif lora_type == 'momentum':
        # Use custom Momentum LoRA
        model = _add_momentum_lora(model, config)
    else:
        raise ValueError(f"Unknown LoRA type: {lora_type}. Use 'standard' or 'momentum'.")

    return model


def _add_standard_lora(
    model: nn.Module,
    config: MomentumLoRAConfig,
) -> nn.Module:
    """
    Add standard LoRA using PEFT library.

    Args:
        model: Model to modify
        config: LoRA configuration

    Returns:
        PEFT model with LoRA adapters
    """
    try:
        from peft import get_peft_model, LoraConfig, TaskType
    except ImportError:
        raise ImportError(
            "PEFT is required for standard LoRA. Install with: pip install peft"
        )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
    )

    model = get_peft_model(model, peft_config)

    print(f"[LoRA] Added standard LoRA with rank={config.rank}")
    print(f"[LoRA] Target modules: {config.target_modules}")

    return model


def _add_momentum_lora(
    model: nn.Module,
    config: MomentumLoRAConfig,
) -> nn.Module:
    """
    Add Momentum LoRA by replacing target Linear layers.

    Args:
        model: Model to modify
        config: Momentum LoRA configuration

    Returns:
        Modified model with Momentum LoRA layers
    """
    return replace_with_momentum_lora(
        model=model,
        rank1=config.rank1,
        rank2=config.rank2,
        beta=config.beta,
        reset_interval=config.reset_interval,
        alpha=config.rank1,  # LoRA alpha = rank for standard scaling
        target_modules=config.target_modules,
        use_soft_orth=config.use_soft_orthogonal,
        svd_rank=config.svd_rank,
    )


def replace_with_momentum_lora(
    model: nn.Module,
    rank1: int,
    rank2: int,
    beta: float = 0.9,
    reset_interval: int = 100,
    alpha: float = None,
    target_modules: Optional[List[str]] = None,
    use_soft_orth: bool = True,
    svd_rank: int = 64,
) -> nn.Module:
    """
    Replace Linear layers with MomentumLoRALinear.

    Args:
        model: Model to modify
        rank1: Velocity LoRA rank
        rank2: Momentum LoRA rank
        beta: Momentum decay rate
        reset_interval: Steps between resets
        alpha: LoRA scaling factor (default: rank1)
        target_modules: Module names to replace
        use_soft_orth: Enable soft orthogonal regularization
        svd_rank: SVD rank for regularization

    Returns:
        Modified model
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "dense", "gate_proj", "up_proj", "down_proj", "fc1", "fc2"]

    if alpha is None:
        alpha = rank1

    replaced_count = 0
    modules_to_replace = []

    # First pass: find modules to replace
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this module should be replaced
            should_replace = any(target in name for target in target_modules)
            if should_replace:
                modules_to_replace.append((name, module))

    # Second pass: replace modules
    for name, module in modules_to_replace:
        # Create Momentum LoRA wrapper
        momentum_layer = MomentumLoRALinear(
            base_layer=module,
            rank1=rank1,
            rank2=rank2,
            beta=beta,
            reset_interval=reset_interval,
            alpha=alpha,
            use_soft_orth=use_soft_orth,
            svd_rank=svd_rank,
        )

        # Replace in parent module
        _set_module_by_name(model, name, momentum_layer)
        replaced_count += 1

    print(f"\n[MomentumLoRA] Replaced {replaced_count} Linear layers")
    print(f"[MomentumLoRA] Configuration:")
    print(f"  - rank1 (velocity): {rank1}")
    print(f"  - rank2 (momentum): {rank2}")
    print(f"  - beta: {beta}")
    print(f"  - reset_interval: {reset_interval}")
    print(f"  - alpha: {alpha}")
    print(f"  - use_soft_orth: {use_soft_orth}")

    # Freeze base weights (W_0 is already a buffer, but make sure no gradients flow)
    _freeze_base_weights(model)

    return model


def _set_module_by_name(model: nn.Module, name: str, new_module: nn.Module):
    """
    Set a module by its name path.

    Args:
        model: Root model
        name: Dot-separated module path (e.g., "layers.0.self_attn.q_proj")
        new_module: New module to set
    """
    parts = name.split('.')
    parent = model

    # Navigate to parent module
    for part in parts[:-1]:
        parent = getattr(parent, part)

    # Set the new module
    setattr(parent, parts[-1], new_module)


def _freeze_base_weights(model: nn.Module):
    """
    Ensure base weights in Momentum LoRA layers are frozen.

    In Momentum LoRA, only A1, B1, A2, B2 should be trainable.
    W_0 and K are buffers (already non-trainable).
    """
    for module in model.modules():
        if isinstance(module, MomentumLoRALinear):
            # The MomentumLoRALinear only has A1, B1, A2, B2 as parameters
            # W_0 and K are registered as buffers
            pass
        elif isinstance(module, SVDMomentumLoRALayer):
            pass


def apply_svd_cache_to_momentum_layers(
    model: nn.Module,
    svd_cache: Dict[str, Dict[str, torch.Tensor]],
):
    """
    Apply pre-computed SVD cache to Momentum LoRA layers.

    Args:
        model: Model with Momentum LoRA layers
        svd_cache: Pre-computed SVD components {param_name: {'U_k': tensor, 'V_k': tensor}}
    """
    print("\n" + "=" * 70)
    print("[SVD Cache] Applying to Momentum LoRA layers...")
    print("=" * 70)

    applied_count = 0

    for name, module in model.named_modules():
        if isinstance(module, MomentumLoRALinear):
            # Find matching SVD cache entry
            svd_entry = _find_matching_svd_entry(name, svd_cache)

            if svd_entry is not None:
                module.set_svd_cache(svd_entry['U_k'], svd_entry['V_k'])
                applied_count += 1
                print(f"  Applied SVD to: {name}")

        elif isinstance(module, SVDMomentumLoRALayer):
            svd_entry = _find_matching_svd_entry(name, svd_cache)

            if svd_entry is not None:
                module.set_svd_cache(svd_entry['U_k'], svd_entry['V_k'])
                applied_count += 1
                print(f"  Applied SVD to: {name}")

    print(f"\n[SVD Cache] Applied to {applied_count} Momentum LoRA layers")
    print("=" * 70)


def _find_matching_svd_entry(
    module_name: str,
    svd_cache: Dict[str, Dict[str, torch.Tensor]],
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Find matching SVD cache entry for a module.

    Args:
        module_name: Name of the module (e.g., "model.layers.0.self_attn.q_proj")
        svd_cache: SVD cache dictionary

    Returns:
        Matching SVD entry or None
    """
    # Clean up module name (remove common prefixes)
    clean_parts = []
    for part in module_name.split('.'):
        if part not in ('base_model', 'model', 'momentum_lora'):
            clean_parts.append(part)
    clean_name = '.'.join(clean_parts)

    # Try to find matching cache entry
    for cache_name, cache_data in svd_cache.items():
        # Check if cache name contains the module name
        if clean_name in cache_name:
            return cache_data
        # Check if cache name ends with module name + .weight
        if cache_name.endswith(clean_name + '.weight'):
            return cache_data
        # Check if module name ends with the core part of cache name
        cache_clean = cache_name.replace('.weight', '')
        cache_parts = [p for p in cache_clean.split('.') if p not in ('base_model', 'model')]
        if '.'.join(cache_parts) == clean_name:
            return cache_data

    return None


def get_trainable_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Get count of trainable parameters in the model.

    Args:
        model: Model to analyze

    Returns:
        Dictionary with parameter counts
    """
    total_params = 0
    trainable_params = 0
    momentum_lora_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            # Check if this is a Momentum LoRA parameter
            if any(x in name for x in ['A1', 'B1', 'A2', 'B2']):
                momentum_lora_params += param.numel()

    return {
        'total': total_params,
        'trainable': trainable_params,
        'momentum_lora': momentum_lora_params,
        'frozen': total_params - trainable_params,
        'trainable_percent': 100 * trainable_params / total_params if total_params > 0 else 0,
    }


def print_trainable_parameters(model: nn.Module):
    """Print trainable parameter statistics."""
    stats = get_trainable_parameters(model)

    print("\n" + "=" * 50)
    print("Trainable Parameters Summary")
    print("=" * 50)
    print(f"Total parameters:     {stats['total']:,}")
    print(f"Trainable parameters: {stats['trainable']:,}")
    print(f"Frozen parameters:    {stats['frozen']:,}")
    print(f"Trainable %:          {stats['trainable_percent']:.2f}%")
    if stats['momentum_lora'] > 0:
        print(f"Momentum LoRA params: {stats['momentum_lora']:,}")
    print("=" * 50)


class MomentumLoRACallback:
    """
    Callback for updating momentum buffers during training.

    This callback should be called after each optimizer.step() to update
    the momentum buffers in all Momentum LoRA layers.

    Usage with HuggingFace Trainer:
        from transformers import TrainerCallback

        class MomentumLoRATrainerCallback(TrainerCallback):
            def __init__(self, model):
                self.model = model

            def on_step_end(self, args, state, control, **kwargs):
                update_all_momentum_buffers(self.model)

        trainer = Trainer(
            model=model,
            callbacks=[MomentumLoRATrainerCallback(model)],
            ...
        )
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.momentum_layers = collect_momentum_layers(model)
        print(f"[MomentumLoRACallback] Found {len(self.momentum_layers)} Momentum LoRA layers")

    def on_step_end(self):
        """Update momentum buffers after optimizer step."""
        update_all_momentum_buffers(self.model)

    def get_soft_orth_loss(
        self,
        lambda_U: float = 0.01,
        lambda_V: float = 0.01,
    ) -> torch.Tensor:
        """Compute soft orthogonal loss for all layers."""
        loss, num_layers = compute_total_soft_orth_loss(
            self.model, lambda_U, lambda_V
        )
        return loss
