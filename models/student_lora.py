
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from typing import Dict, Optional
import threading
from contextlib import contextmanager

# Momentum LoRA imports
from models.lora_utils import add_lora_to_model, MomentumLoRAConfig
from models.momentum_lora import MomentumLoRALinear

# LLaMA uses RMSNorm instead of LayerNorm
try:
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    HAS_LLAMA_RMSNORM = True
except ImportError:
    HAS_LLAMA_RMSNORM = False
    LlamaRMSNorm = None

# Thread-local storage for tracking Student vs Teacher forward
_thread_local = threading.local()

@contextmanager
def _set_student_forward_context(is_student: bool):
    """
    Context manager to set whether current forward pass is Student or Teacher

    Args:
        is_student: True for Student forward, False for Teacher forward

    Usage:
        with _set_student_forward_context(True):
            student_outputs = student_model(inputs)  # Masks applied

        with _set_student_forward_context(False):
            teacher_outputs = teacher_model(inputs)  # Masks NOT applied
    """
    # Save previous state
    previous = getattr(_thread_local, 'is_student_forward', None)

    # Set new state
    _thread_local.is_student_forward = is_student

    try:
        yield
    finally:
        # Restore previous state
        if previous is None:
            if hasattr(_thread_local, 'is_student_forward'):
                delattr(_thread_local, 'is_student_forward')
        else:
            _thread_local.is_student_forward = previous

def create_student_model(
    base_model,
    lora_config,
    masks=None,
    momentum_lora_config: Optional[MomentumLoRAConfig] = None
):
    # Momentum LoRA vs Standard PEFT LoRA
    use_momentum_lora = (
        momentum_lora_config is not None and
        getattr(momentum_lora_config, 'lora_type', 'standard') == 'momentum'
    )

    if use_momentum_lora:
        print("[Student] Using Momentum LoRA")
        student_model = add_lora_to_model(
            model=base_model,
            config=momentum_lora_config,
            target_modules=lora_config.target_modules,
        )

        # Debug info
        from models.momentum_lora import MomentumLoRALinear
        momentum_count = sum(1 for _, m in student_model.named_modules() if isinstance(m, MomentumLoRALinear))
        print(f"[Student][Debug] Momentum LoRA layers: {momentum_count}")

    else:
        peft_config = LoraConfig(
            r=lora_config.lora_r,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
            target_modules=lora_config.target_modules,
            bias="none",
            task_type="CAUSAL_LM"
        )

        student_model = get_peft_model(base_model, peft_config)

        # Debug: inspect adapter configuration
        debug_print_done = False
        for name, module in student_model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                adapter_names = list(module.lora_A.keys())
                active_adapter = getattr(module, "active_adapter", None)
                requires_grad = []
                for adapter in adapter_names:
                    try:
                        requires_grad.append((adapter, module.lora_A[adapter].weight.requires_grad))
                    except Exception:
                        continue
                print(f"[Student][Debug] Module: {name}, adapters={adapter_names}, active={active_adapter}, req_grad={requires_grad}")
                debug_print_done = True
                break
        if not debug_print_done:
            print("[Student][Debug] No LoRA adapters found for inspection")

    student_model._is_student_with_masks = True
    student_model._uses_momentum_lora = use_momentum_lora

    if hasattr(student_model, 'base_model'):
        if hasattr(student_model.base_model, 'model'):
            student_model.base_model.model._is_student_with_masks = True
        else:
            student_model.base_model._is_student_with_masks = True

    if masks is not None:
        apply_masks_to_student(student_model, masks)

    lora_type = "Momentum LoRA" if use_momentum_lora else "PEFT LoRA"
    print(f"[Student] {lora_type} initialized")
    print("[Student] IMPORTANT: Masks will be applied to BOTH Frozen LLM and LoRA")
    print("[Student] Student flag set: _is_student_with_masks = True")

    return student_model

def apply_masks_to_student(model, masks: Dict[str, torch.Tensor], mask_mode: str = "static"):
    print("\n[Mask] Applying masks to Student model...")
    applied_count = 0
    debug_first_10 = []
    skip_lora = (mask_mode == "dynamic")

    for name, mask in masks.items():
        try:
            parts = name.split('.')
            mask_type = parts[-1]  # 'head_mask', 'ffn_mask', etc.
            module_path_from_mask = '.'.join(parts[:-1])

            #
            # - PEFT LoRA: "base_model.model.model.layers.X..."

            path_candidates = []

            if module_path_from_mask.startswith("base_model.model.layers"):
                peft_path = module_path_from_mask.replace(
                    "base_model.model.layers",
                    "base_model.model.model.layers",
                    1
                )
                path_candidates.append(peft_path)
            elif module_path_from_mask.startswith("base_model.model.") and not module_path_from_mask.startswith("base_model.model.model."):
                peft_path = module_path_from_mask.replace(
                    "base_model.model.",
                    "base_model.model.model.",
                    1
                )
                path_candidates.append(peft_path)

            if module_path_from_mask.startswith("base_model.model."):
                momentum_path = module_path_from_mask.replace("base_model.model.", "model.", 1)
                path_candidates.append(momentum_path)

            path_candidates.append(module_path_from_mask)

            module = None
            found = False
            for module_path in path_candidates:
                module = model
                found = True
                for attr in module_path.split('.'):
                    if not hasattr(module, attr):
                        found = False
                        break
                    module = getattr(module, attr)
                if found:
                    break

            if not found:
                continue

            if True:
                if isinstance(module, MomentumLoRALinear):
                    if skip_lora:
                        continue

                    momentum_lora = module.momentum_lora
                    if mask_type == 'input':
                        momentum_lora.register_buffer('weight_input_mask', mask.clone())
                    else:
                        momentum_lora.register_buffer('weight_output_mask', mask.clone())

                    applied_count += 1

                    if len(debug_first_10) < 10:
                        debug_first_10.append((name, module_path, 'MomentumLoRALinear', mask_type))

                elif hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    if skip_lora:
                        continue

                    base_layer = module.base_layer
                    if hasattr(base_layer, 'weight'):

                        if mask_type == 'input':
                            base_layer.register_buffer('weight_input_mask', mask.clone())
                        else:
                            base_layer.register_buffer('weight_output_mask', mask.clone())

                        applied_count += 1

                        if len(debug_first_10) < 10:
                            debug_first_10.append((name, module_path, str(type(module).__name__), mask_type))

                elif isinstance(module, nn.LayerNorm) or (HAS_LLAMA_RMSNORM and isinstance(module, LlamaRMSNorm)):
                    module.register_buffer('layernorm_output_mask', mask.clone())
                    applied_count += 1

                    if len(debug_first_10) < 10:
                        debug_first_10.append((name, module_path, str(type(module).__name__), mask_type))

                elif isinstance(module, nn.Linear):
                    if mask_type == 'input':
                        module.register_buffer('weight_input_mask', mask.clone())
                    else:
                        module.register_buffer('weight_output_mask', mask.clone())
                    applied_count += 1

                    if len(debug_first_10) < 10:
                        debug_first_10.append((name, module_path, str(type(module).__name__), mask_type))

                elif isinstance(module, nn.Embedding):
                    module.register_buffer('embedding_output_mask', mask.clone())
                    applied_count += 1

                    if len(debug_first_10) < 10:
                        debug_first_10.append((name, module_path, str(type(module).__name__), mask_type))

        except Exception as e:
            continue

    print(f"[Mask] Applied {applied_count} masks to Student")

    if len(debug_first_10) > 0:
        print(f"[Mask] First {len(debug_first_10)} applied masks:")
        for mask_name, module_path, module_type, m_type in debug_first_10:
            print(f"  - {mask_name}")
            print(f"    → Module: {module_path} ({module_type}), Type: {m_type}")

def _is_student_forward(module):
    if hasattr(_thread_local, 'is_student_forward'):
        return _thread_local.is_student_forward

    if hasattr(module, '_is_student_with_masks'):
        return getattr(module, '_is_student_with_masks', False)

    current = module
    visited = set()

    for _ in range(15):
        if id(current) in visited:
            break
        visited.add(id(current))

        parent = None
        for attr in ['_parent', 'parent', '_forward_module']:
            if hasattr(current, attr):
                parent = getattr(current, attr, None)
                if parent is not None:
                    break

        if parent is None:
            break

        if hasattr(parent, '_is_student_with_masks'):
            return getattr(parent, '_is_student_with_masks', False)

        current = parent

    return False

def apply_forward_hooks_for_masks(model, mask_mode: str = "static", timing: str = "after"):
    print("\n[Hook] Registering forward hooks for masking...")
    print(f"[Hook] Timing: {timing}")
    hook_count = 0
    skip_lora = (mask_mode == "dynamic")

    for name, module in model.named_modules():
        if isinstance(module, MomentumLoRALinear):
            momentum_lora = module.momentum_lora

            if skip_lora:
                continue

            # Output mask
            if hasattr(momentum_lora, 'weight_output_mask'):
                if timing == "before":
                    def create_momentum_output_hook_before(m):
                        def hook(module, input, output):
                            if _is_student_forward(module):
                                if hasattr(module.momentum_lora, 'weight_output_mask'):
                                    import torch.nn.functional as F
                                    ml = module.momentum_lora
                                    mask = ml.weight_output_mask

                                    velocity = ml.B1 @ ml.A1
                                    auxiliary = ml.B2 @ ml.A2
                                    effective_weight = ml.W_0 + ml.K.to(ml.W_0.dtype) + ml.scaling1 * velocity + ml.scaling2 * auxiliary

                                    masked_weight = effective_weight * mask.unsqueeze(1).to(effective_weight.device, dtype=effective_weight.dtype)
                                    return F.linear(input[0], masked_weight, ml.bias)
                            return output
                        return hook
                    module.register_forward_hook(create_momentum_output_hook_before(momentum_lora))
                    hook_count += 1
                else:
                    def create_momentum_output_hook_after(m):
                        def hook(module, input, output):
                            if _is_student_forward(module):
                                if hasattr(module.momentum_lora, 'weight_output_mask'):
                                    mask = module.momentum_lora.weight_output_mask.to(output.device, dtype=output.dtype)
                                    return output * mask
                            return output
                        return hook
                    module.register_forward_hook(create_momentum_output_hook_after(momentum_lora))
                    hook_count += 1

            # Input mask
            if hasattr(momentum_lora, 'weight_input_mask'):
                if timing == "before":
                    def create_momentum_input_hook_before(m):
                        def hook(module, input, output):
                            if _is_student_forward(module):
                                if hasattr(module.momentum_lora, 'weight_input_mask'):
                                    import torch.nn.functional as F
                                    ml = module.momentum_lora
                                    mask = ml.weight_input_mask

                                    base_output = F.linear(input[0], ml.W_0 + ml.K.to(ml.W_0.dtype), None)

                                    # A1: [rank1, in_features], mask: [in_features]
                                    masked_A1 = ml.A1 * mask.unsqueeze(0).to(ml.A1.device, dtype=ml.A1.dtype)
                                    masked_A2 = ml.A2 * mask.unsqueeze(0).to(ml.A2.device, dtype=ml.A2.dtype)

                                    velocity_output = F.linear(F.linear(input[0], masked_A1), ml.B1) * ml.scaling1
                                    auxiliary_output = F.linear(F.linear(input[0], masked_A2), ml.B2) * ml.scaling2

                                    output = base_output + velocity_output + auxiliary_output
                                    if ml.bias is not None:
                                        output = output + ml.bias
                                    return output
                            return output
                        return hook
                    module.register_forward_hook(create_momentum_input_hook_before(momentum_lora))
                    hook_count += 1
                else:
                    def create_momentum_input_hook_after(m):
                        def hook(module, input):
                            if _is_student_forward(module):
                                if hasattr(module.momentum_lora, 'weight_input_mask'):
                                    mask = module.momentum_lora.weight_input_mask.to(input[0].device, dtype=input[0].dtype)
                                    masked_input = input[0] * mask
                                    return (masked_input,)
                            return input
                        return hook
                    module.register_forward_pre_hook(create_momentum_input_hook_after(momentum_lora))
                    hook_count += 1

            continue

        elif hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            base_layer = module.base_layer

            if skip_lora:
                if hasattr(base_layer, 'weight_output_mask'):
                    mask = base_layer.weight_output_mask

                    if timing == "before":
                        def create_base_hook_before(m):
                            def hook(module, input, output):
                                if hasattr(module, 'weight_output_mask') and _is_student_forward(module):
                                    import torch.nn.functional as F
                                    mask = module.weight_output_mask
                                    masked_weight = module.weight * mask.unsqueeze(1).to(module.weight.device, dtype=module.weight.dtype)
                                    return F.linear(input[0], masked_weight, module.bias)
                                return output
                            return hook
                        base_layer.register_forward_hook(create_base_hook_before(mask))
                        hook_count += 1
                    else:
                        def create_base_hook_after(m):
                            def hook(module, input, output):
                                if hasattr(module, 'weight_output_mask') and _is_student_forward(module):
                                    mask = module.weight_output_mask.to(output.device, dtype=output.dtype)
                                    return output * mask
                                return output
                            return hook
                        base_layer.register_forward_hook(create_base_hook_after(mask))
                        hook_count += 1

                if hasattr(base_layer, 'weight_input_mask'):
                    mask = base_layer.weight_input_mask

                    def create_base_input_hook(m):
                        def hook(module, input):
                            if hasattr(module, 'weight_input_mask') and _is_student_forward(module):
                                mask = module.weight_input_mask.to(input[0].device, dtype=input[0].dtype)
                                masked_input = input[0] * mask
                                return (masked_input,)
                            return input
                        return hook
                    base_layer.register_forward_pre_hook(create_base_input_hook(mask))
                    hook_count += 1

                continue


            if hasattr(base_layer, 'weight_output_mask'):
                mask = base_layer.weight_output_mask

                if timing == "before":
                    def create_output_hook_before(m):
                        def hook(module, input, output):
                            if _is_student_forward(module):
                                if hasattr(module.base_layer, 'weight_output_mask'):
                                    import torch.nn.functional as F
                                    mask = module.base_layer.weight_output_mask
                                    base_layer = module.base_layer

                                    masked_base_weight = base_layer.weight * mask.unsqueeze(1).to(base_layer.weight.device, dtype=base_layer.weight.dtype)
                                    base_output = F.linear(input[0], masked_base_weight, base_layer.bias)

                                    lora_A_dict = module.lora_A
                                    lora_B_dict = module.lora_B
                                    lora_dropout_dict = module.lora_dropout
                                    scaling_dict = module.scaling

                                    active_adapter = getattr(module, 'active_adapter', 'default')
                                    if isinstance(active_adapter, list):
                                        active_adapter = active_adapter[0]

                                    if active_adapter in lora_A_dict and active_adapter in lora_B_dict:
                                        lora_A = lora_A_dict[active_adapter]
                                        lora_B = lora_B_dict[active_adapter]
                                        lora_dropout = lora_dropout_dict.get(active_adapter, nn.Identity())
                                        scaling = scaling_dict[active_adapter]

                                        # LoRA forward with masked lora_B weight
                                        # lora_B.weight: (out_features, r)
                                        # mask: (out_features,) → (out_features, 1)
                                        masked_lora_B_weight = lora_B.weight * mask.unsqueeze(1).to(lora_B.weight.device, dtype=lora_B.weight.dtype)

                                        x = lora_dropout(input[0])
                                        x = F.linear(x, lora_A.weight)  # (batch, r)
                                        x = F.linear(x, masked_lora_B_weight)  # (batch, out) - masked!
                                        lora_output = x * scaling

                                        return base_output + lora_output
                                    else:
                                        return base_output
                            return output
                        return hook

                    module.register_forward_hook(create_output_hook_before(mask))
                    hook_count += 1
                else:
                    def create_output_hook_after(m):
                        def hook(module, input, output):
                            if _is_student_forward(module):
                                if hasattr(module.base_layer, 'weight_output_mask'):
                                    mask = module.base_layer.weight_output_mask.to(output.device, dtype=output.dtype)
                                    return output * mask
                            return output
                        return hook

                    module.register_forward_hook(create_output_hook_after(mask))
                    hook_count += 1

            if hasattr(base_layer, 'weight_input_mask'):
                mask = base_layer.weight_input_mask

                if timing == "before":
                    def create_input_hook_before(m):
                        def hook(module, input, output):
                            if _is_student_forward(module):
                                if hasattr(module.base_layer, 'weight_input_mask'):
                                    import torch.nn.functional as F
                                    mask = module.base_layer.weight_input_mask
                                    base_layer = module.base_layer

                                    base_output = base_layer(input[0])

                                    lora_A_dict = module.lora_A
                                    lora_B_dict = module.lora_B
                                    lora_dropout_dict = module.lora_dropout
                                    scaling_dict = module.scaling

                                    active_adapter = getattr(module, 'active_adapter', 'default')
                                    if isinstance(active_adapter, list):
                                        active_adapter = active_adapter[0]

                                    if active_adapter in lora_A_dict and active_adapter in lora_B_dict:
                                        lora_A = lora_A_dict[active_adapter]
                                        lora_B = lora_B_dict[active_adapter]
                                        lora_dropout = lora_dropout_dict.get(active_adapter, nn.Identity())
                                        scaling = scaling_dict[active_adapter]

                                        # lora_A.weight: (r, in_features)
                                        # mask: (in_features,) → (1, in_features)
                                        masked_lora_A_weight = lora_A.weight * mask.unsqueeze(0).to(lora_A.weight.device, dtype=lora_A.weight.dtype)

                                        x = lora_dropout(input[0])
                                        x = F.linear(x, masked_lora_A_weight)  # (batch, r) - masked!
                                        x = F.linear(x, lora_B.weight)  # (batch, out)
                                        lora_output = x * scaling

                                        return base_output + lora_output
                                    else:
                                        return base_output
                            return output
                        return hook

                    module.register_forward_hook(create_input_hook_before(mask))
                    hook_count += 1
                else:
                    def create_input_hook_after(m):
                        def hook(module, input):
                            if _is_student_forward(module):
                                if hasattr(module.base_layer, 'weight_input_mask'):
                                    mask = module.base_layer.weight_input_mask.to(input[0].device, dtype=input[0].dtype)
                                    masked_input = input[0] * mask
                                    return (masked_input,)
                            return input
                        return hook

                    module.register_forward_pre_hook(create_input_hook_after(mask))
                    hook_count += 1

        elif isinstance(module, nn.LayerNorm) or (HAS_LLAMA_RMSNORM and isinstance(module, LlamaRMSNorm)):
            if hasattr(module, 'layernorm_output_mask'):
                mask = module.layernorm_output_mask

                def create_layernorm_hook(m):
                    def hook(module, input, output):
                        if hasattr(module, 'layernorm_output_mask') and _is_student_forward(module):
                            mask = module.layernorm_output_mask.to(output.device, dtype=output.dtype)
                            return output * mask
                        return output
                    return hook

                module.register_forward_hook(create_layernorm_hook(mask))
                hook_count += 1

        elif isinstance(module, nn.Linear):
            if hasattr(module, 'weight_output_mask'):
                mask = module.weight_output_mask

                if timing == "before":
                    # Before: Weight masking
                    def create_output_hook_before(m):
                        def hook(module, input, output):
                            if hasattr(module, 'weight_output_mask') and _is_student_forward(module):
                                import torch.nn.functional as F
                                mask = module.weight_output_mask
                                masked_weight = module.weight * mask.unsqueeze(1).to(module.weight.device, dtype=module.weight.dtype)
                                return F.linear(input[0], masked_weight, module.bias)
                            return output
                        return hook
                    module.register_forward_hook(create_output_hook_before(mask))
                    hook_count += 1
                else:
                    # After: Output masking
                    def create_output_hook_after(m):
                        def hook(module, input, output):
                            if hasattr(module, 'weight_output_mask') and _is_student_forward(module):
                                mask = module.weight_output_mask.to(output.device, dtype=output.dtype)
                                return output * mask
                            return output
                        return hook

                    module.register_forward_hook(create_output_hook_after(mask))
                    hook_count += 1

            if hasattr(module, 'weight_input_mask'):
                mask = module.weight_input_mask

                def create_input_hook(m):
                    def hook(module, input):
                        if hasattr(module, 'weight_input_mask') and _is_student_forward(module):
                            mask = module.weight_input_mask.to(input[0].device, dtype=input[0].dtype)
                            masked_input = input[0] * mask
                            return (masked_input,)
                        return input
                    return hook

                module.register_forward_pre_hook(create_input_hook(mask))
                hook_count += 1

        # Embedding layer (nn.Embedding)
        elif isinstance(module, nn.Embedding):
            if hasattr(module, 'embedding_output_mask'):
                mask = module.embedding_output_mask

                def create_embedding_hook(m):
                    def hook(module, input, output):
                        if hasattr(module, 'embedding_output_mask') and _is_student_forward(module):
                            mask = module.embedding_output_mask.to(output.device, dtype=output.dtype)
                            return output * mask
                        return output
                    return hook

                module.register_forward_hook(create_embedding_hook(mask))
                hook_count += 1

    print(f"[Hook] Registered {hook_count} forward hooks")
    print("[Hook] Masks will be applied ONLY for Student forward (_is_student_with_masks=True)")
    print("[Hook] Teacher forward will NOT be affected by masks (shared base is safe!)")

def apply_static_masks_to_lora(model, masks: Dict[str, torch.Tensor]):
    print("\n[Warning] apply_static_masks_to_lora() is deprecated")
    applied_count = 0

    for name, mask in masks.items():
        try:
            parts = name.split('.')
            mask_type = parts[-1]
            module_path_from_mask = '.'.join(parts[:-1])

            if module_path_from_mask.startswith("base_model.model.layers"):
                module_path = module_path_from_mask.replace(
                    "base_model.model.layers",
                    "base_model.model.model.layers",
                    1
                )
            elif module_path_from_mask.startswith("base_model.model.") and not module_path_from_mask.startswith("base_model.model.model."):
                module_path = module_path_from_mask.replace(
                    "base_model.model.",
                    "base_model.model.model.",
                    1
                )
            else:
                module_path = module_path_from_mask

            module = model
            found = True
            for attr in module_path.split('.'):
                if not hasattr(module, attr):
                    found = False
                    break
                module = getattr(module, attr)

            if not found:
                continue

            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                lora_A_dict = module.lora_A
                lora_B_dict = module.lora_B

                for adapter_name in lora_A_dict.keys():
                    lora_A = lora_A_dict[adapter_name]
                    lora_B = lora_B_dict[adapter_name]

                    bool_mask = mask.bool()

                    if mask_type == 'input':
                        # lora_A: (r, in_features)
                        # mask: (in_features,)
                        # lora_A.weight[:, ~mask] = 0
                        with torch.no_grad():
                            lora_A.weight.data[:, ~bool_mask] = 0.0
                            applied_count += 1
                    else:
                        # lora_B: (out_features, r)
                        # mask: (out_features,)
                        # lora_B.weight[~mask, :] = 0
                        with torch.no_grad():
                            lora_B.weight.data[~bool_mask, :] = 0.0
                            applied_count += 1

        except Exception as e:
            continue

    print(f"[Static Mask] Applied static masks to {applied_count} LoRA modules")

def update_student_masks(model, new_masks: Dict[str, torch.Tensor], mask_mode: str = "dynamic"):
    updated_count = 0

    for name, new_mask in new_masks.items():
        try:
            parts = name.split('.')
            mask_type = parts[-1]
            module_path_from_mask = '.'.join(parts[:-1])

            # - PEFT LoRA: "base_model.model.model.layers.X..."

            path_candidates = []

            if module_path_from_mask.startswith("base_model.model.layers"):
                peft_path = module_path_from_mask.replace(
                    "base_model.model.layers",
                    "base_model.model.model.layers",
                    1
                )
                path_candidates.append(peft_path)
            elif module_path_from_mask.startswith("base_model.model.") and not module_path_from_mask.startswith("base_model.model.model."):
                peft_path = module_path_from_mask.replace(
                    "base_model.model.",
                    "base_model.model.model.",
                    1
                )
                path_candidates.append(peft_path)

            if module_path_from_mask.startswith("base_model.model."):
                momentum_path = module_path_from_mask.replace("base_model.model.", "model.", 1)
                path_candidates.append(momentum_path)

            path_candidates.append(module_path_from_mask)

            module = None
            found = False
            for module_path in path_candidates:
                module = model
                found = True
                for attr in module_path.split('.'):
                    if not hasattr(module, attr):
                        found = False
                        break
                    module = getattr(module, attr)
                if found:
                    break

            if not found:
                continue

            if True:
                if isinstance(module, MomentumLoRALinear):
                    if mask_mode == "dynamic":
                        continue

                    momentum_lora = module.momentum_lora
                    if mask_type == 'input':
                        if hasattr(momentum_lora, 'weight_input_mask'):
                            momentum_lora.weight_input_mask.copy_(new_mask)
                            updated_count += 1
                    else:
                        if hasattr(momentum_lora, 'weight_output_mask'):
                            momentum_lora.weight_output_mask.copy_(new_mask)
                            updated_count += 1

                elif hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    if mask_mode == "dynamic":
                        continue

                    base_layer = module.base_layer
                    if mask_type == 'input':
                        if hasattr(base_layer, 'weight_input_mask'):
                            base_layer.weight_input_mask.copy_(new_mask)
                            updated_count += 1
                    else:
                        if hasattr(base_layer, 'weight_output_mask'):
                            base_layer.weight_output_mask.copy_(new_mask)
                            updated_count += 1

                elif isinstance(module, nn.LayerNorm) or (HAS_LLAMA_RMSNORM and isinstance(module, LlamaRMSNorm)):
                    if hasattr(module, 'layernorm_output_mask'):
                        module.layernorm_output_mask.copy_(new_mask)
                        updated_count += 1

                elif isinstance(module, nn.Linear):
                    if mask_type == 'input':
                        if hasattr(module, 'weight_input_mask'):
                            module.weight_input_mask.copy_(new_mask)
                            updated_count += 1
                    else:
                        if hasattr(module, 'weight_output_mask'):
                            module.weight_output_mask.copy_(new_mask)
                            updated_count += 1

        except Exception:
            continue

    return updated_count
