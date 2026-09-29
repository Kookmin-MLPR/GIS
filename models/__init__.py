"""
Model utilities for Student LoRA

Includes:
- Standard LoRA (PEFT-based)
- Momentum LoRA (custom implementation for improved pruning recovery)
"""

from .student_lora import (
    create_student_model,
    apply_forward_hooks_for_masks,
    update_student_masks
)

# Momentum LoRA components
from .momentum_lora import (
    MomentumLoRALayer,
    SVDMomentumLoRALayer,
    MomentumLoRALinear,
    collect_momentum_layers,
    update_all_momentum_buffers,
    compute_total_soft_orth_loss,
    get_momentum_lora_state_dict,
    load_momentum_lora_state_dict,
    merge_momentum_lora_to_base,
)

# LoRA utilities
from .lora_utils import (
    MomentumLoRAConfig,
    add_lora_to_model,
    replace_with_momentum_lora,
    apply_svd_cache_to_momentum_layers,
    get_trainable_parameters,
    print_trainable_parameters,
    MomentumLoRACallback,
)

__all__ = [
    # Student LoRA
    'create_student_model',
    'apply_forward_hooks_for_masks',
    'update_student_masks',

    # Momentum LoRA layers
    'MomentumLoRALayer',
    'SVDMomentumLoRALayer',
    'MomentumLoRALinear',
    'collect_momentum_layers',
    'update_all_momentum_buffers',
    'compute_total_soft_orth_loss',
    'get_momentum_lora_state_dict',
    'load_momentum_lora_state_dict',
    'merge_momentum_lora_to_base',

    # LoRA utilities
    'MomentumLoRAConfig',
    'add_lora_to_model',
    'replace_with_momentum_lora',
    'apply_svd_cache_to_momentum_layers',
    'get_trainable_parameters',
    'print_trainable_parameters',
    'MomentumLoRACallback',

    # Heterogeneous models
    'heterogeneous_llama',
    'heterogeneous_phi',
]
