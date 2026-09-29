
import gc
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn

from pruning.model_descriptor import get_down_proj_module, get_attn_output_proj


@dataclass
class ScaleCalibrationConfig:
    enabled: bool = False
    method: str = "flap"  # "flap", "ntk", or "ntk_pure"
    target: str = "both"  # "ffn", "attn", "both"
    add_bias: bool = False
    ntk_alpha: float = 0.3
    verbose: bool = True
    handle_dim_mask: bool = True
    sequential: bool = False
    ntk_unit_mode: str = "group"
    dimension_multiple: int = 16


class _EarlyExitException(Exception):
    pass


def collect_sublayer_outputs(
    model: nn.Module,
    dataloader,
    device: str = "cuda",
    ffn_masks: Dict[int, torch.Tensor] = None,
    head_masks: Dict[int, torch.Tensor] = None,
    apply_masks: bool = False,
    target_layers: list = None
) -> Dict[int, Dict[str, torch.Tensor]]:
    outputs = {}
    hooks = []

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    capture_layers = set(target_layers) if target_layers is not None else set(range(num_layers))
    for l in capture_layers:
        outputs[l] = {"attn": [], "ffn": []}

    def make_attn_hook(layer_idx):
        def hook(module, input, output):
            # output: [batch, seq, hidden]
            out = output.detach().float()
            outputs[layer_idx]["attn"].append(out.reshape(-1, out.shape[-1]))
        return hook

    def make_o_proj_input_hook(layer_idx, mask):
        def hook(module, input):
            x = input[0]
            batch_size, seq_len, _ = x.shape
            x_reshaped = x.view(batch_size, seq_len, num_heads, head_dim)
            mask_expanded = mask.to(device=x.device, dtype=x.dtype).view(1, 1, num_heads, 1)
            x_masked = x_reshaped * mask_expanded
            return (x_masked.view(batch_size, seq_len, -1),)
        return hook

    def make_ffn_hook(layer_idx):
        def hook(module, input, output):
            # output: [batch, seq, hidden]
            out = output.detach().float()
            outputs[layer_idx]["ffn"].append(out.reshape(-1, out.shape[-1]))
        return hook

    def make_down_proj_input_hook(layer_idx, mask):
        def hook(module, input):
            x = input[0]
            mask_expanded = mask.to(device=x.device, dtype=x.dtype).view(1, 1, -1)
            return (x * mask_expanded,)
        return hook

    max_target_layer = max(capture_layers) if target_layers is not None else num_layers - 1

    for layer_idx, layer in enumerate(model.model.layers):
        if layer_idx > max_target_layer:
            break

        if layer_idx in capture_layers:
            o_proj_module = get_attn_output_proj(layer.self_attn)
            if o_proj_module is not None:
                hooks.append(o_proj_module.register_forward_hook(make_attn_hook(layer_idx)))

            down_proj_module = get_down_proj_module(layer.mlp)
            if down_proj_module is not None:
                hooks.append(down_proj_module.register_forward_hook(make_ffn_hook(layer_idx)))

        if apply_masks:
            if head_masks and layer_idx in head_masks:
                o_proj_module = get_attn_output_proj(layer.self_attn)
                if o_proj_module is not None:
                    hooks.append(o_proj_module.register_forward_pre_hook(
                        make_o_proj_input_hook(layer_idx, head_masks[layer_idx])
                    ))

            if ffn_masks and layer_idx in ffn_masks:
                down_proj_module = get_down_proj_module(layer.mlp)
                if down_proj_module is not None:
                    hooks.append(down_proj_module.register_forward_pre_hook(
                        make_down_proj_input_hook(layer_idx, ffn_masks[layer_idx])
                    ))

    early_exit = target_layers is not None and max_target_layer < num_layers - 1
    if early_exit:
        def _early_exit_hook(module, input, output):
            raise _EarlyExitException()
        hooks.append(model.model.layers[max_target_layer].register_forward_hook(_early_exit_hook))

    # Forward pass
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            try:
                model(input_ids=input_ids, attention_mask=attention_mask)
            except _EarlyExitException:
                pass

    for hook in hooks:
        hook.remove()

    for l in capture_layers:
        if outputs[l]["attn"]:
            outputs[l]["attn"] = torch.cat(outputs[l]["attn"], dim=0)
        if outputs[l]["ffn"]:
            outputs[l]["ffn"] = torch.cat(outputs[l]["ffn"], dim=0)

    return outputs


def collect_sublayer_gradients(
    model: nn.Module,
    dataloader,
    device: str = "cuda",
    ffn_masks: Dict[int, torch.Tensor] = None,
    head_masks: Dict[int, torch.Tensor] = None,
) -> Dict[int, Dict[str, torch.Tensor]]:
    gradients = {}
    hooks = []
    activations = {}

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    for l in range(num_layers):
        gradients[l] = {"attn": [], "ffn": []}
        activations[l] = {"attn": None, "ffn": None}

    def make_save_activation_hook(layer_idx, key):
        def hook(module, input, output):
            activations[layer_idx][key] = output
        return hook

    for layer_idx, layer in enumerate(model.model.layers):
        o_proj_module = get_attn_output_proj(layer.self_attn)
        if o_proj_module is not None:
            hooks.append(o_proj_module.register_forward_hook(
                make_save_activation_hook(layer_idx, "attn")
            ))
        down_proj_module = get_down_proj_module(layer.mlp)
        if down_proj_module is not None:
            hooks.append(down_proj_module.register_forward_hook(
                make_save_activation_hook(layer_idx, "ffn")
            ))

    # Forward + Backward
    model.eval()
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Forward with gradient
        model.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Backward
        loss.backward()

        for layer_idx in range(num_layers):
            # Attention output gradient
            if activations[layer_idx]["attn"] is not None:
                attn_out = activations[layer_idx]["attn"]
                if attn_out.grad is not None:
                    grad_mag = attn_out.grad.abs().mean(dim=(0, 1))  # [hidden_dim]
                    gradients[layer_idx]["attn"].append(grad_mag.detach())

            # FFN output gradient
            if activations[layer_idx]["ffn"] is not None:
                ffn_out = activations[layer_idx]["ffn"]
                if ffn_out.grad is not None:
                    grad_mag = ffn_out.grad.abs().mean(dim=(0, 1))  # [hidden_dim]
                    gradients[layer_idx]["ffn"].append(grad_mag.detach())

    for hook in hooks:
        hook.remove()

    for l in range(num_layers):
        if gradients[l]["attn"]:
            gradients[l]["attn"] = torch.stack(gradients[l]["attn"]).mean(dim=0)
        else:
            gradients[l]["attn"] = None
        if gradients[l]["ffn"]:
            gradients[l]["ffn"] = torch.stack(gradients[l]["ffn"]).mean(dim=0)
        else:
            gradients[l]["ffn"] = None

    return gradients


def compute_ntk_sensitivity_simple(
    model: nn.Module,
    dataloader,
    ffn_masks: Dict[int, torch.Tensor] = None,
    head_masks: Dict[int, torch.Tensor] = None,
    device: str = "cuda",
    verbose: bool = False
) -> Dict[int, Dict[str, torch.Tensor]]:
    if verbose:
        print("[NTK] Computing output-based sensitivity...")

    original_outputs = collect_sublayer_outputs(
        model, dataloader, device,
        ffn_masks=None, head_masks=None, apply_masks=False
    )

    pruned_outputs = collect_sublayer_outputs(
        model, dataloader, device,
        ffn_masks=ffn_masks, head_masks=head_masks, apply_masks=True
    )

    num_layers = model.config.num_hidden_layers
    sensitivities = {}

    for layer_idx in range(num_layers):
        sensitivities[layer_idx] = {}

        # Attention sensitivity
        orig_attn = original_outputs[layer_idx].get("attn")
        prun_attn = pruned_outputs[layer_idx].get("attn")
        if orig_attn is not None and prun_attn is not None:
            diff = (orig_attn - prun_attn) ** 2
            sensitivity = diff.mean(dim=0).sqrt()  # [hidden_dim]
            sensitivity = sensitivity / (sensitivity.mean() + 1e-8)
            sensitivities[layer_idx]["attn"] = sensitivity

        # FFN sensitivity
        orig_ffn = original_outputs[layer_idx].get("ffn")
        prun_ffn = pruned_outputs[layer_idx].get("ffn")
        if orig_ffn is not None and prun_ffn is not None:
            diff = (orig_ffn - prun_ffn) ** 2
            sensitivity = diff.mean(dim=0).sqrt()  # [hidden_dim]
            sensitivity = sensitivity / (sensitivity.mean() + 1e-8)
            sensitivities[layer_idx]["ffn"] = sensitivity

    if verbose:
        print(f"[NTK] Sensitivity computed for {num_layers} layers")
        if 0 in sensitivities and "ffn" in sensitivities[0]:
            s = sensitivities[0]["ffn"]
            print(f"  Layer 0 FFN: min={s.min():.4f}, max={s.max():.4f}, mean={s.mean():.4f}")

    del original_outputs, pruned_outputs
    gc.collect()

    return sensitivities


def compute_regression_coefficients(
    original: torch.Tensor,
    pruned: torch.Tensor,
    sensitivity: torch.Tensor = None,
    ntk_alpha: float = 0.3,
    verbose: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    # [N, D]
    x = pruned.float()
    y = original.float()

    N = x.shape[0]

    mean_x = x.mean(dim=0)  # [D]
    mean_y = y.mean(dim=0)  # [D]

    # Var(x) = E[x^2] - E[x]^2
    # Cov(x, y) = E[xy] - E[x]E[y]
    var_x = (x ** 2).mean(dim=0) - mean_x ** 2  # [D]
    cov_xy = (x * y).mean(dim=0) - mean_x * mean_y  # [D]

    # A = Cov(x, y) / Var(x)
    eps = 1e-8
    A = cov_xy / (var_x + eps)

    A = torch.where(var_x < eps, torch.ones_like(A), A)

    A = A.clamp(0.5, 2.0)

    # B = mean(y) - A * mean(x)
    B = mean_y - A * mean_x

    if sensitivity is not None:
        # A_adjusted = A * (1 + alpha * (sensitivity - 1))
        sensitivity = sensitivity.to(A.device)
        adjustment = 1 + ntk_alpha * (sensitivity - 1)
        # clamp to prevent extreme values
        adjustment = adjustment.clamp(0.5, 2.0)
        A = A * adjustment

        if verbose:
            print(f"    [NTK] Adjustment: min={adjustment.min():.4f}, max={adjustment.max():.4f}, mean={adjustment.mean():.4f}")

    if verbose:
        print(f"    A: min={A.min():.4f}, max={A.max():.4f}, mean={A.mean():.4f}")
        print(f"    B: min={B.min():.4f}, max={B.max():.4f}, mean={B.mean():.4f}")

    return A, B


def apply_regression_to_weights(
    model: nn.Module,
    regression_coeffs: Dict[int, Dict[str, tuple]],
    config: ScaleCalibrationConfig
):
    num_layers = model.config.num_hidden_layers

    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]

        if layer_idx not in regression_coeffs:
            continue

        coeffs = regression_coeffs[layer_idx]

        # Attention (o_proj / dense)
        if config.target in ["attn", "both"] and "attn" in coeffs:
            A, B = coeffs["attn"]
            o_proj = get_attn_output_proj(layer.self_attn)

            # output[i] = sum_j(W[i,j] * input[j]) → output[i] * A[i] = sum_j(W[i,j] * A[i] * input[j])
            with torch.no_grad():
                A_device = A.to(o_proj.weight.device, dtype=o_proj.weight.dtype)
                o_proj.weight.data *= A_device.view(-1, 1)

                if config.add_bias:
                    B_device = B.to(o_proj.weight.device, dtype=o_proj.weight.dtype)
                    if o_proj.bias is None:
                        o_proj.bias = nn.Parameter(B_device)
                    else:
                        o_proj.bias.data += B_device

            if config.verbose and layer_idx < 3:
                print(f"  Layer {layer_idx} Attn: A_mean={A.mean():.4f}, B_mean={B.mean():.4f}")

        # FFN (down_proj / fc2)
        if config.target in ["ffn", "both"] and "ffn" in coeffs:
            A, B = coeffs["ffn"]
            down_proj = get_down_proj_module(layer.mlp)

            with torch.no_grad():
                A_device = A.to(down_proj.weight.device, dtype=down_proj.weight.dtype)
                down_proj.weight.data *= A_device.view(-1, 1)

                if config.add_bias:
                    B_device = B.to(down_proj.weight.device, dtype=down_proj.weight.dtype)
                    if down_proj.bias is None:
                        down_proj.bias = nn.Parameter(B_device)
                    else:
                        down_proj.bias.data += B_device

            if config.verbose and layer_idx < 3:
                print(f"  Layer {layer_idx} FFN: A_mean={A.mean():.4f}, B_mean={B.mean():.4f}")


def calibrate_output_scale(
    model: nn.Module,
    calibration_dataloader,
    ffn_masks: Dict[int, torch.Tensor] = None,
    head_masks: Dict[int, torch.Tensor] = None,
    config: ScaleCalibrationConfig = None,
    device: str = "cuda",
    **kwargs
) -> Dict[int, Dict[str, tuple]]:
    if config is None:
        config = ScaleCalibrationConfig(enabled=True)

    if not config.enabled:
        return {}

    num_layers = model.config.num_hidden_layers
    method_names = {
        "flap": "FLAP (Dimension-wise Regression)",
        "ntk": "NTK (Sensitivity-weighted Regression)",
        "ntk_pure": "NTK Pure (Unit NTK → Output Dimension Projection)"
    }
    method_name = method_names.get(config.method, config.method)

    if config.verbose:
        print("\n" + "=" * 60)
        print(f"Output Scale Calibration ({method_name})")
        print("=" * 60)
        print(f"Method: {config.method}")
        print(f"Target: {config.target}")
        print(f"Add bias: {config.add_bias}")
        if config.method == "ntk":
            print(f"NTK alpha: {config.ntk_alpha}")
        if config.method == "ntk_pure":
            print(f"NTK unit mode: {config.ntk_unit_mode}")
            print(f"Dimension multiple: {config.dimension_multiple}")
        print(f"Sequential: {config.sequential}")
        print(f"Layers: {num_layers}")

    if config.method == "ntk_pure":
        if config.verbose:
            print("\n[NTK Pure] Using unit NTK → output dimension projection...")

        regression_coeffs = compute_ntk_pure_scale_factors(
            model=model,
            calibration_dataloader=calibration_dataloader,
            ffn_masks=ffn_masks,
            head_masks=head_masks,
            config=config,
            device=device
        )

        if config.verbose:
            print("[NTK Pure] Applying scale factors to weights...")

        apply_regression_to_weights(model, regression_coeffs, config)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if config.verbose:
            print(f"\n[Scale Calibration] ({config.method.upper()}) Completed!")
            print("=" * 60)

        return regression_coeffs

    sensitivities = None
    if config.method == "ntk":
        if config.verbose:
            print("\n[Step 0] Computing NTK sensitivities...")
        sensitivities = compute_ntk_sensitivity_simple(
            model, calibration_dataloader,
            ffn_masks=ffn_masks, head_masks=head_masks,
            device=device, verbose=config.verbose
        )

    if config.sequential:
        if config.verbose:
            print(f"\n[Sequential Mode] Layer-by-layer calibration ({num_layers} layers)...")

        regression_coeffs = {}

        for layer_idx in range(num_layers):
            if config.verbose:
                print(f"\n  [Layer {layer_idx}/{num_layers-1}]", end="")

            original_outputs = collect_sublayer_outputs(
                model, calibration_dataloader, device,
                ffn_masks=None, head_masks=None, apply_masks=False,
                target_layers=[layer_idx]
            )

            pruned_outputs = collect_sublayer_outputs(
                model, calibration_dataloader, device,
                ffn_masks=ffn_masks, head_masks=head_masks, apply_masks=True,
                target_layers=[layer_idx]
            )

            regression_coeffs[layer_idx] = {}
            layer_sens = sensitivities[layer_idx] if sensitivities else None

            # Attention
            if config.target in ["attn", "both"]:
                orig_attn = original_outputs[layer_idx].get("attn")
                prun_attn = pruned_outputs[layer_idx].get("attn")
                attn_sens = layer_sens.get("attn") if layer_sens else None

                if orig_attn is not None and prun_attn is not None and len(orig_attn) > 0:
                    A, B = compute_regression_coefficients(
                        orig_attn, prun_attn,
                        sensitivity=attn_sens if config.method == "ntk" else None,
                        ntk_alpha=config.ntk_alpha,
                        verbose=(config.verbose and layer_idx < 2)
                    )
                    regression_coeffs[layer_idx]["attn"] = (A, B)

            # FFN
            if config.target in ["ffn", "both"]:
                orig_ffn = original_outputs[layer_idx].get("ffn")
                prun_ffn = pruned_outputs[layer_idx].get("ffn")
                ffn_sens = layer_sens.get("ffn") if layer_sens else None

                if orig_ffn is not None and prun_ffn is not None and len(orig_ffn) > 0:
                    A, B = compute_regression_coefficients(
                        orig_ffn, prun_ffn,
                        sensitivity=ffn_sens if config.method == "ntk" else None,
                        ntk_alpha=config.ntk_alpha,
                        verbose=(config.verbose and layer_idx < 2)
                    )
                    regression_coeffs[layer_idx]["ffn"] = (A, B)

            apply_regression_to_weights(model, {layer_idx: regression_coeffs[layer_idx]}, config)

            del original_outputs, pruned_outputs
            gc.collect()

        if sensitivities:
            del sensitivities
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if config.verbose:
            print(f"\n\n[Scale Calibration] ({config.method.upper()}, Sequential) Completed!")
            print("=" * 60)

        return regression_coeffs

    if config.verbose:
        print("\n[Step 1] Collecting original model outputs...")

    original_outputs = collect_sublayer_outputs(
        model, calibration_dataloader, device,
        ffn_masks=None, head_masks=None, apply_masks=False
    )

    if config.verbose:
        print("[Step 2] Collecting pruned model outputs (with masks)...")

    pruned_outputs = collect_sublayer_outputs(
        model, calibration_dataloader, device,
        ffn_masks=ffn_masks, head_masks=head_masks, apply_masks=True
    )

    if config.verbose:
        print("[Step 3] Computing regression coefficients...")

    regression_coeffs = {}

    for layer_idx in range(num_layers):
        regression_coeffs[layer_idx] = {}

        layer_sens = sensitivities[layer_idx] if sensitivities else None

        # Attention
        if config.target in ["attn", "both"]:
            orig_attn = original_outputs[layer_idx].get("attn")
            prun_attn = pruned_outputs[layer_idx].get("attn")
            attn_sens = layer_sens.get("attn") if layer_sens else None

            if orig_attn is not None and prun_attn is not None and len(orig_attn) > 0:
                A, B = compute_regression_coefficients(
                    orig_attn, prun_attn,
                    sensitivity=attn_sens if config.method == "ntk" else None,
                    ntk_alpha=config.ntk_alpha,
                    verbose=(config.verbose and layer_idx < 2)
                )
                regression_coeffs[layer_idx]["attn"] = (A, B)

        # FFN
        if config.target in ["ffn", "both"]:
            orig_ffn = original_outputs[layer_idx].get("ffn")
            prun_ffn = pruned_outputs[layer_idx].get("ffn")
            ffn_sens = layer_sens.get("ffn") if layer_sens else None

            if orig_ffn is not None and prun_ffn is not None and len(orig_ffn) > 0:
                A, B = compute_regression_coefficients(
                    orig_ffn, prun_ffn,
                    sensitivity=ffn_sens if config.method == "ntk" else None,
                    ntk_alpha=config.ntk_alpha,
                    verbose=(config.verbose and layer_idx < 2)
                )
                regression_coeffs[layer_idx]["ffn"] = (A, B)

    if config.verbose:
        print("[Step 4] Applying regression coefficients to weights...")

    apply_regression_to_weights(model, regression_coeffs, config)

    del original_outputs, pruned_outputs
    if sensitivities:
        del sensitivities
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if config.verbose:
        print(f"\n[Scale Calibration] ({config.method.upper()}) Completed!")
        print("=" * 60)

    return regression_coeffs


def compute_ntk_pure_scale_factors(
    model: nn.Module,
    calibration_dataloader,
    ffn_masks: Dict[int, torch.Tensor],
    head_masks: Dict[int, torch.Tensor],
    config: ScaleCalibrationConfig,
    device: str = "cuda"
) -> Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
    from pruning.cett_importance import _compute_ntk_sensitivity_per_unit

    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    intermediate_size = model.config.intermediate_size
    num_heads = model.config.num_attention_heads
    head_dim = hidden_size // num_heads

    scale_factors = {}

    if config.verbose:
        print("\n[NTK Pure] Computing unit-level NTK sensitivities...")
        print(f"  Unit mode: {config.ntk_unit_mode}")
        print(f"  Dimension multiple: {config.dimension_multiple}")

    # ===== FFN Scale Factors =====
    if config.target in ["ffn", "both"] and ffn_masks:
        if config.verbose:
            print("\n[NTK Pure] Computing FFN scale factors...")

        ffn_ntk = _compute_ntk_sensitivity_per_unit(
            model=model,
            dataloader=calibration_dataloader,
            current_masks=ffn_masks,
            prev_masks=None,
            mask_type="ffn",
            num_samples=10,
            device=device,
            verbose=config.verbose,
            dimension_multiple=config.dimension_multiple,
            ntk_target_mode="default",
            ntk_unit_mode=config.ntk_unit_mode
        )

        for layer_idx in range(num_layers):
            if layer_idx not in scale_factors:
                scale_factors[layer_idx] = {}

            # down_proj weight: [hidden_size, intermediate_size]
            down_proj = get_down_proj_module(model.model.layers[layer_idx].mlp)
            W = down_proj.weight.data.float()  # [4096, 11008]

            if layer_idx not in ffn_ntk:
                A = torch.ones(hidden_size, device='cpu')
                B = torch.zeros(hidden_size, device='cpu')
                scale_factors[layer_idx]["ffn"] = (A, B)
                continue

            ntk_units = ffn_ntk[layer_idx].to(W.device)  # [num_units]

            if config.ntk_unit_mode == "group":
                # [688] → [11008]
                ntk_neuron = ntk_units.repeat_interleave(config.dimension_multiple)
            else:
                ntk_neuron = ntk_units

            if len(ntk_neuron) < intermediate_size:
                padding = torch.ones(intermediate_size - len(ntk_neuron), device=ntk_neuron.device)
                ntk_neuron = torch.cat([ntk_neuron, padding])
            ntk_neuron = ntk_neuron[:intermediate_size]

            if layer_idx in ffn_masks:
                mask = ffn_masks[layer_idx].to(W.device).float()  # [11008]
            else:
                mask = torch.ones(intermediate_size, device=W.device)

            # W: [hidden_size, intermediate_size] = [4096, 11008]
            W_squared = W ** 2  # [4096, 11008]

            # orig_contrib[d] = Σ_j W_dj² × ntk_j
            orig_contrib = W_squared @ ntk_neuron  # [4096]

            # pruned_contrib[d] = Σ_j W_dj² × ntk_j × mask_j
            pruned_contrib = W_squared @ (ntk_neuron * mask)  # [4096]

            # Scale factor: A[d] = sqrt(orig_contrib[d] / pruned_contrib[d])
            eps = 1e-8
            A = torch.sqrt(orig_contrib / (pruned_contrib + eps))

            A = A.clamp(0.5, 2.0)

            A = torch.where(pruned_contrib < eps, torch.ones_like(A), A)

            B = torch.zeros_like(A)

            scale_factors[layer_idx]["ffn"] = (A.cpu(), B.cpu())

            if config.verbose and layer_idx < 3:
                print(f"  Layer {layer_idx} FFN: A_mean={A.mean():.4f}, A_min={A.min():.4f}, A_max={A.max():.4f}")

    # ===== Attention Scale Factors =====
    if config.target in ["attn", "both"] and head_masks:
        if config.verbose:
            print("\n[NTK Pure] Computing Attention scale factors...")

        head_ntk = _compute_ntk_sensitivity_per_unit(
            model=model,
            dataloader=calibration_dataloader,
            current_masks=head_masks,
            prev_masks=None,
            mask_type="head",
            num_samples=10,
            device=device,
            verbose=config.verbose,
            dimension_multiple=1,
            num_heads=num_heads,
            ntk_target_mode="default",
            ntk_unit_mode="neuron"
        )

        for layer_idx in range(num_layers):
            if layer_idx not in scale_factors:
                scale_factors[layer_idx] = {}

            # o_proj weight: [hidden_size, hidden_size]
            o_proj = get_attn_output_proj(model.model.layers[layer_idx].self_attn)
            W = o_proj.weight.data.float()  # [4096, 4096]

            if layer_idx not in head_ntk:
                A = torch.ones(hidden_size, device='cpu')
                B = torch.zeros(hidden_size, device='cpu')
                scale_factors[layer_idx]["attn"] = (A, B)
                continue

            ntk_heads = head_ntk[layer_idx].to(W.device)  # [num_heads]

            ntk_dim = ntk_heads.repeat_interleave(head_dim)  # [4096]

            if layer_idx in head_masks:
                mask_heads = head_masks[layer_idx].to(W.device).float()  # [32]
                mask = mask_heads.repeat_interleave(head_dim)  # [4096]
            else:
                mask = torch.ones(hidden_size, device=W.device)

            W_squared = W ** 2  # [4096, 4096]

            # orig_contrib[d] = Σ_j W_dj² × ntk_j
            orig_contrib = W_squared @ ntk_dim  # [4096]

            # pruned_contrib[d] = Σ_j W_dj² × ntk_j × mask_j
            pruned_contrib = W_squared @ (ntk_dim * mask)  # [4096]

            # Scale factor
            eps = 1e-8
            A = torch.sqrt(orig_contrib / (pruned_contrib + eps))
            A = A.clamp(0.5, 2.0)
            A = torch.where(pruned_contrib < eps, torch.ones_like(A), A)

            B = torch.zeros_like(A)

            scale_factors[layer_idx]["attn"] = (A.cpu(), B.cpu())

            if config.verbose and layer_idx < 3:
                print(f"  Layer {layer_idx} Attn: A_mean={A.mean():.4f}, A_min={A.min():.4f}, A_max={A.max():.4f}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if config.verbose:
        print("\n[NTK Pure] Scale factors computed successfully!")

    return scale_factors


def iterative_calibrate_step(
    model: nn.Module,
    dataloader,
    prev_ffn_masks: Dict[int, torch.Tensor],
    prev_head_masks: Dict[int, torch.Tensor],
    curr_ffn_masks: Dict[int, torch.Tensor],
    curr_head_masks: Dict[int, torch.Tensor],
    step_idx: int = 0,
    device: str = "cuda",
    verbose: bool = True
) -> Dict[int, Dict[str, tuple]]:
    if verbose:
        print(f"\n  [Iter-FLAP] Step {step_idx}: Collecting teacher outputs (prev masks)...")

    teacher_outputs = collect_sublayer_outputs(
        model, dataloader, device,
        ffn_masks=prev_ffn_masks, head_masks=prev_head_masks,
        apply_masks=True
    )

    if verbose:
        print(f"  [Iter-FLAP] Step {step_idx}: Collecting student outputs (curr masks)...")

    student_outputs = collect_sublayer_outputs(
        model, dataloader, device,
        ffn_masks=curr_ffn_masks, head_masks=curr_head_masks,
        apply_masks=True
    )

    regression_coeffs = {}
    for layer_idx in teacher_outputs:
        regression_coeffs[layer_idx] = {}
        for sublayer in ["attn", "ffn"]:
            if sublayer in teacher_outputs[layer_idx] and sublayer in student_outputs[layer_idx]:
                teacher_out = teacher_outputs[layer_idx][sublayer]
                student_out = student_outputs[layer_idx][sublayer]

                if teacher_out.numel() == 0 or student_out.numel() == 0:
                    continue

                A, B = compute_regression_coefficients(
                    original=teacher_out,
                    pruned=student_out,
                    verbose=False
                )
                regression_coeffs[layer_idx][sublayer] = (A, B)

    # 4. Apply to model weights
    config = ScaleCalibrationConfig(
        enabled=True, method="flap", target="both",
        add_bias=False, verbose=False
    )
    apply_regression_to_weights(model, regression_coeffs, config)

    if verbose:
        a_means = []
        for layer_idx in regression_coeffs:
            for sublayer in regression_coeffs[layer_idx]:
                A, B = regression_coeffs[layer_idx][sublayer]
                a_means.append(A.mean().item())
        if a_means:
            avg_a = sum(a_means) / len(a_means)
            print(f"  [Iter-FLAP] Step {step_idx}: Applied. Avg A={avg_a:.4f} "
                  f"(range: {min(a_means):.4f}~{max(a_means):.4f})")

    del teacher_outputs, student_outputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return regression_coeffs


def save_regression_coeffs(
    regression_coeffs: Dict[int, Dict[str, tuple]],
    save_path: str
):
    import json

    serializable = {}
    for layer_idx, coeffs in regression_coeffs.items():
        serializable[str(layer_idx)] = {}
        for key, (A, B) in coeffs.items():
            serializable[str(layer_idx)][key] = {
                "A_mean": float(A.mean()),
                "A_std": float(A.std()),
                "B_mean": float(B.mean()),
                "B_std": float(B.std()),
            }

    with open(save_path, 'w') as f:
        json.dump(serializable, f, indent=2)
