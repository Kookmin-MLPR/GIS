
import os
import gc
import math
import copy
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import get_model_name, LLAMA_MODELS, is_vlm_key
from models.qwen_vl_pruning import VLMCausalLMFacade, sync_vlm_config_after_pruning
from pruning.cett_importance import (
    TaylorImportanceCalculator,
    create_calibration_dataloader,
    _compute_ntk_sensitivity,
    _compute_ntk_sensitivity_per_unit,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Option C: GandA-based Automatic Block Pruning")

    # Model
    parser.add_argument("--model", type=str, default="llama1-7b",
                        choices=list(LLAMA_MODELS.keys()) + ["custom"],
                        help="Model to use")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Local model path (overrides HuggingFace)")

    parser.add_argument("--pruning_ratio", type=float, default=0.3,
                        help="Target total model pruning ratio (e.g., 0.3 = 30%% of total params)")
    parser.add_argument("--pruning_target_mode", type=str, default="total",
                        choices=["total", "ffn"],
                        help="Pruning target mode: "
                             "total = ratio based on total model params (default), "
                             "ffn = ratio based on FFN neuron count directly")

    # Block (depth) pruning
    parser.add_argument("--block_search_ratio", type=float, default=None,
                        help="Separate ratio for block candidate search (global mask). "
                             "If set, uses this higher ratio to identify block candidates, "
                             "while --pruning_ratio remains the actual target. "
                             "Useful for models with uniform importance (e.g., LLaMA3). "
                             "Default: None (same as pruning_ratio)")
    parser.add_argument("--block_remove_threshold", type=float, default=0.80,
                        help="Remove block if simulated_ratio > this (default: 0.80)")
    parser.add_argument("--block_importance_mode", type=str, default="neuron",
                        choices=["neuron", "output"],
                        help="Block removal criterion: "
                             "neuron = per-neuron simulated ratio (down_proj input), "
                             "output = block output importance (down_proj output)")
    parser.add_argument("--protect_first", type=int, default=2,
                        help="Number of first layers to protect from block removal")
    parser.add_argument("--protect_last", type=int, default=1,
                        help="Number of last layers to protect from block removal")

    # Calibration
    parser.add_argument("--calibration_samples", type=int, default=128,
                        help="Number of calibration samples")
    parser.add_argument("--calibration_batch_size", type=int, default=4,
                        help="Calibration batch size")
    parser.add_argument("--calibration_max_length", type=int, default=512,
                        help="Calibration max sequence length")
    parser.add_argument("--calibration_dataset", type=str, default="bookcorpus_local",
                        help="Calibration dataset (c4_local, wikitext2_val, bookcorpus_local)")

    # Importance
    parser.add_argument("--importance_method", type=str, default="ganda",
                        help="Importance method (ganda, taylor, wanda, etc.)")
    parser.add_argument("--ffn_mode", type=str, default="down",
                        help="FFN importance mode (down, up, all)")

    parser.add_argument("--use_ntk_layer", action="store_true",
                        help="Compute layer-wise NTK sensitivity and compare block removal")
    parser.add_argument("--ntk_method", type=str, default="frobenius",
                        choices=["frobenius", "eigenvalue", "delta", "quadform"],
                        help="NTK sensitivity method")
    parser.add_argument("--ntk_samples", type=int, default=16,
                        help="Number of samples for NTK computation")
    parser.add_argument("--ntk_target_mode", type=str, default="all",
                        choices=["default", "all"],
                        help="NTK target params: default=down_proj only, all=up+gate+down")
    parser.add_argument("--ntk_alpha", type=float, default=0.3,
                        help="NTK adjustment strength (0=no effect, higher=stronger)")
    parser.add_argument("--ntk_alpha_layer", type=float, default=None,
                        help="Override NTK alpha for depth(layer) adjustment. If None, uses --ntk_alpha.")
    parser.add_argument("--ntk_alpha_neuron", type=float, default=None,
                        help="Override NTK alpha for width(neuron) adjustment. If None, uses --ntk_alpha.")
    parser.add_argument("--ntk_norm_method", type=str, default="legacy",
                        choices=["legacy", "robust"],
                        help="NTK sensitivity normalization: "
                             "legacy=mean-based (sensitive to outliers), "
                             "robust=log+MAD z-score (stable)")
    parser.add_argument("--depth_score_mode", type=str, default="legacy",
                        choices=["legacy", "neuron_only", "neuron_plus_layer"],
                        help="How depth importance is computed:\n"
                             "  legacy            = sum(ffn_scores) × layer-NTK adjustment (current)\n"
                             "  neuron_only       = sum(ffn_scores × neuron-NTK adj) only (no layer-NTK)\n"
                             "  neuron_plus_layer = sum(ffn_scores × neuron-NTK adj) × layer-NTK adj")

    # Unified pruning mode
    parser.add_argument("--unified_pruning", action="store_true",
                        help="Unified depth+width pruning: auto depth/width ratio via greedy ranking")
    parser.add_argument("--unified_fill_width", action="store_true",
                        help="Unified mode: fill remaining budget with width pruning (default: stop at last fit)")
    parser.add_argument("--unified_stop_mode", type=str, default="before",
                        choices=["before", "after"],
                        help="Unified greedy stop policy relative to target ratio: "
                             "'before' = stop right before exceeding target (under-prune, default), "
                             "'after'  = stop right after crossing target (over-prune)")
    parser.add_argument("--unified_iterative", action="store_true",
                        help="Unified mode: iteratively re-compute GandA importance and NTK sensitivities "
                             "after removing each group (1 depth or 1 width super-group per iteration). "
                             "Much slower but adapts scores to the evolving (masked) model state.")
    parser.add_argument("--unified_disable_depth", action="store_true",
                        help="Unified mode: disable depth pruning (width-only).")
    parser.add_argument("--unified_disable_width", action="store_true",
                        help="Unified mode: disable width pruning (depth-only).")
    parser.add_argument("--max_depth_remove", type=int, default=0,
                        help="Maximum number of depth layers to remove (0=unlimited). "
                             "Remaining budget is filled with width pruning. "
                             "Useful at high pruning ratios to prevent excessive consecutive layer removal.")
    parser.add_argument("--force_removed_layers", type=str, default=None,
                        help="Force-remove specified layers (comma-separated, e.g. '7,8,11,12,14,24'). "
                             "Sets their depth_scores to 0 so greedy picks them first. "
                             "Width budget is filled afterwards by greedy on remaining layers.")

    parser.add_argument("--pruned_model_path", type=str, default=None,
                        help="Path to already-pruned model. Skips pruning, runs finetune+eval only.")

    # Width pruning
    parser.add_argument("--dimension_multiple", type=int, default=128,
                        help="FFN dimension grouping multiple")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing for memory savings")

    # Evaluation
    parser.add_argument("--eval", action="store_true",
                        help="Run lm_eval benchmarks after pruning")
    parser.add_argument("--eval_batch_size", type=int, default=16,
                        help="Evaluation batch size")

    # Fine-tuning (Stage 2)
    parser.add_argument("--finetune", action="store_true",
                        help="Run fine-tuning after pruning")
    parser.add_argument("--s2_num_epochs", type=int, default=2)
    parser.add_argument("--s2_batch_size", type=int, default=64)
    parser.add_argument("--s2_micro_batch_size", type=int, default=16)
    parser.add_argument("--s2_learning_rate", type=float, default=5e-4)
    parser.add_argument("--s2_warmup_steps", type=int, default=100)
    parser.add_argument("--s2_lr_scheduler", type=str, default="cosine")
    parser.add_argument("--s2_max_seq_length", type=int, default=128)
    parser.add_argument("--s2_lora_r", type=int, default=8)
    parser.add_argument("--s2_lora_alpha", type=int, default=16)
    parser.add_argument("--s2_lora_dropout", type=float, default=0.0)
    parser.add_argument("--s2_dataset", type=str, default="alpaca_cleaned",
                        choices=["alpaca", "alpaca_cleaned", "c4"])
    parser.add_argument("--s2_regularization", type=str, default="none",
                        choices=["none", "l1", "l2"],
                        help="Regularization type for LoRA parameters (default: none)")
    parser.add_argument("--s2_reg_lambda", type=float, default=1e-4,
                        help="Regularization strength (default: 1e-4)")
    parser.add_argument("--eval_after_finetune", action="store_true",
                        help="Run lm_eval benchmarks after fine-tuning")

    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs/depth_pruning",
                        help="Output directory for results and pruned model")
    parser.add_argument("--result_dir", type=str, default="./result",
                        help="Result directory for summary (e.g., ./result → ./result/depth_pruning/)")

    return parser.parse_args()


def compute_ganda_scores(model, tokenizer, args, device):
    print("\n" + "=" * 70)
    print("Step 1: Computing GandA Importance Scores")
    print("=" * 70)

    calib_dataloader = create_calibration_dataloader(
        dataset_name=args.calibration_dataset,
        num_samples=args.calibration_samples,
        batch_size=args.calibration_batch_size,
        max_length=args.calibration_max_length,
        seed=args.seed,
        tokenizer=tokenizer,
    )

    calculator = TaylorImportanceCalculator(
        model=model,
        importance_method=args.importance_method,
        device=device,
        ffn_mode=args.ffn_mode,
        verbose=True,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
    )

    ffn_scores = calculator.compute_taylor_scores(
        dataloader=calib_dataloader,
        num_samples=args.calibration_samples,
    )
    # ffn_scores: {layer_idx: tensor[intermediate_size]}

    del calculator
    gc.collect()
    torch.cuda.empty_cache()

    return ffn_scores


def compute_block_output_importance(model, tokenizer, args, device):
    from pruning.cett_importance import create_calibration_dataloader
    from pruning.model_descriptor import get_down_proj_module

    print("\n" + "=" * 70)
    print("Computing Block Output Importance (down_proj output)")
    print("=" * 70)

    calib_dataloader = create_calibration_dataloader(
        dataset_name=args.calibration_dataset,
        num_samples=args.calibration_samples,
        batch_size=args.calibration_batch_size,
        max_length=args.calibration_max_length,
        seed=args.seed,
        tokenizer=tokenizer,
    )

    num_layers = model.config.num_hidden_layers
    current_activations = {}  # layer_idx -> tensor (1 batch only)
    current_gradients = {}    # layer_idx -> tensor (1 batch only)
    block_importance = {i: 0.0 for i in range(num_layers)}
    hooks = []

    for layer_idx in range(num_layers):
        mlp = model.model.layers[layer_idx].mlp
        down_proj = get_down_proj_module(mlp)
        if down_proj is None:
            continue
        target = down_proj.base_layer if hasattr(down_proj, 'base_layer') else down_proj

        def make_fwd(idx):
            def hook(module, inp, out):
                current_activations[idx] = out.detach()
            return hook

        def make_bwd(idx):
            def hook(module, grad_in, grad_out):
                if grad_out and grad_out[0] is not None:
                    current_gradients[idx] = grad_out[0].detach()
            return hook

        hooks.append(target.register_forward_hook(make_fwd(layer_idx)))
        hooks.append(target.register_full_backward_hook(make_bwd(layer_idx)))

    # Forward + backward pass
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    samples_done = 0
    for batch in calib_dataloader:
        if samples_done >= args.calibration_samples:
            break

        if isinstance(batch, dict):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            input_ids = batch['input_ids']
            attention_mask = batch.get('attention_mask')
        else:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device) if len(batch) > 1 else None

        model.zero_grad()
        for param in model.parameters():
            param.requires_grad = True

        kwargs = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
        else:
            labels = input_ids.clone()
        kwargs["labels"] = labels

        outputs = model(**kwargs)
        outputs.loss.backward()

        for param in model.parameters():
            param.requires_grad = False

        for layer_idx in range(num_layers):
            act = current_activations.get(layer_idx)
            grad = current_gradients.get(layer_idx)
            if act is not None and grad is not None:
                block_importance[layer_idx] += (grad.float() * act.float()).abs().sum().item()

        current_activations.clear()
        current_gradients.clear()
        del outputs
        torch.cuda.empty_cache()

        samples_done += input_ids.size(0)

    for h in hooks:
        h.remove()

    for layer_idx in range(num_layers):
        if samples_done > 0:
            block_importance[layer_idx] /= samples_done

    max_imp = max(block_importance.values()) if block_importance else 1.0
    if max_imp > 0:
        for k in block_importance:
            block_importance[k] /= max_imp

    print(f"\n  {'Layer':>6} | {'Importance':>12} | {'Bar':>20}")
    print("  " + "-" * 46)
    for layer_idx in range(num_layers):
        imp = block_importance[layer_idx]
        bar = "#" * int(imp * 20)
        print(f"  {layer_idx:>6} | {imp:>12.6f} | {bar}")

    gc.collect()
    torch.cuda.empty_cache()

    return block_importance


def compute_layer_ntk_sensitivity(model, tokenizer, args, device, ffn_scores):
    from pruning.model_descriptor import get_down_proj_module

    print("\n" + "=" * 70)
    print("Computing Layer-wise NTK Sensitivity for Depth Pruning")
    print("=" * 70)

    ntk_method = getattr(args, 'ntk_method', 'frobenius')
    ntk_samples = getattr(args, 'ntk_samples', 16)
    ntk_target_mode = getattr(args, 'ntk_target_mode', 'default')

    print(f"  NTK method:       {ntk_method}")
    print(f"  NTK samples:      {ntk_samples}")
    print(f"  NTK target mode:  {ntk_target_mode}")

    # Calibration dataloader
    calib_dataloader = create_calibration_dataloader(
        dataset_name=args.calibration_dataset,
        num_samples=ntk_samples,
        batch_size=args.calibration_batch_size,
        max_length=args.calibration_max_length,
        seed=args.seed,
        tokenizer=tokenizer,
    )

    num_layers = model.config.num_hidden_layers
    model.eval()

    target_params = {}
    for layer_idx in range(num_layers):
        mlp = model.model.layers[layer_idx].mlp
        if ntk_target_mode == "all":
            params = []
            for name in ['up_proj', 'gate_proj', 'down_proj']:
                if hasattr(mlp, name):
                    params.append(getattr(mlp, name).weight)
            if not params and hasattr(mlp, 'fc1'):
                params.append(mlp.fc1.weight)
                if hasattr(mlp, 'fc2'):
                    params.append(mlp.fc2.weight)
            target_params[layer_idx] = params
        else:
            down_proj = get_down_proj_module(mlp)
            if down_proj is not None:
                w = down_proj.base_layer.weight if hasattr(down_proj, 'base_layer') else down_proj.weight
                target_params[layer_idx] = [w]

    print(f"  Target params per layer: {[p.shape for p in target_params.get(0, [])]}")

    cached_batches = []
    data_iter = iter(calib_dataloader)
    for _ in range(ntk_samples):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(calib_dataloader)
            batch = next(data_iter)
        cached_batches.append(batch)

    original_requires_grad = {}
    for layer_idx, params in target_params.items():
        for param in params:
            pid = id(param)
            if pid not in original_requires_grad:
                original_requires_grad[pid] = param.requires_grad
                param.requires_grad_(True)

    layer_sensitivities = {}

    for layer_idx in range(num_layers):
        if layer_idx not in target_params or not target_params[layer_idx]:
            layer_sensitivities[layer_idx] = 1.0
            continue

        layer_grads = []

        for batch in cached_batches:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(device)

            model.zero_grad(set_to_none=True)
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            outputs.loss.backward()

            grads = []
            for param in target_params[layer_idx]:
                if param.grad is not None:
                    grads.append(param.grad.detach().float().flatten())
            if grads:
                grad_vec = torch.cat(grads, dim=0)
                grad_vec = grad_vec.clamp(-1e4, 1e4)
                layer_grads.append(grad_vec)

            del outputs
            model.zero_grad(set_to_none=True)

        if len(layer_grads) == 0:
            layer_sensitivities[layer_idx] = 1.0
        else:
            G = torch.stack(layer_grads)  # [num_samples, param_size]
            K = torch.mm(G, G.t())  # [num_samples, num_samples]

            if ntk_method == "eigenvalue":
                try:
                    eigenvalues = torch.linalg.eigvalsh(K)
                    k = min(getattr(args, 'ntk_eigenvalue_k', 5), len(eigenvalues))
                    sensitivity = eigenvalues[-k:].sum().item()
                except Exception:
                    sensitivity = torch.norm(K, p='fro').item()
            elif ntk_method == "delta":
                sensitivity = torch.trace(K).item()
            elif ntk_method == "quadform":
                sensitivity = K.sum().item()
            else:  # frobenius
                sensitivity = torch.norm(K, p='fro').item()

            layer_sensitivities[layer_idx] = sensitivity

            if layer_idx < 3:
                print(f"    [NTK] Layer {layer_idx}: sens={sensitivity:.6f} ({ntk_method})")

        del layer_grads
        gc.collect()
        torch.cuda.empty_cache()

    ntk_norm = getattr(args, 'ntk_norm_method', 'legacy')
    values = list(layer_sensitivities.values())

    if ntk_norm == "robust" and len(values) > 1:
        import math
        log_values = [math.log(v + 1e-10) for v in values]
        median_val = sorted(log_values)[len(log_values) // 2]
        mad = sorted([abs(v - median_val) for v in log_values])[len(log_values) // 2]
        mad = max(mad, 1e-10)
        for k in layer_sensitivities:
            log_s = math.log(layer_sensitivities[k] + 1e-10)
            z = (log_s - median_val) / mad
            layer_sensitivities[k] = math.exp(z * 0.2)  # 0.2 = moderate scaling
        vals = list(layer_sensitivities.values())
        m = sum(vals) / len(vals)
        if m > 0:
            for k in layer_sensitivities:
                layer_sensitivities[k] /= m
        print(f"  [Norm] robust (log+MAD z-score)")
    else:
        mean_sens = sum(values) / len(values) if values else 1.0
        if mean_sens > 0:
            for k in layer_sensitivities:
                layer_sensitivities[k] /= mean_sens
        print(f"  [Norm] legacy (mean-based)")

    for layer_idx, params in target_params.items():
        for param in params:
            pid = id(param)
            if pid in original_requires_grad:
                param.requires_grad_(original_requires_grad[pid])

    print(f"\n  {'Layer':>6} | {'NTK Sensitivity':>16} | {'Bar':>20}")
    print("  " + "-" * 50)
    for layer_idx in range(num_layers):
        sens = layer_sensitivities.get(layer_idx, 1.0)
        bar_len = min(20, max(0, int(sens * 10)))
        bar = "#" * bar_len
        print(f"  {layer_idx:>6} | {sens:>16.4f} | {bar}")

    gc.collect()
    torch.cuda.empty_cache()

    return layer_sensitivities


def compute_neuron_ntk_sensitivity(model, tokenizer, args, device):
    print("\n" + "=" * 70)
    print("Computing Neuron-level NTK Sensitivity for Width Groups")
    print("=" * 70)

    ntk_method = getattr(args, 'ntk_method', 'frobenius')
    ntk_samples = getattr(args, 'ntk_samples', 16)
    ntk_target_mode = getattr(args, 'ntk_target_mode', 'default')

    print(f"  NTK method:       {ntk_method}")
    print(f"  NTK samples:      {ntk_samples}")
    print(f"  NTK target mode:  {ntk_target_mode}")
    print(f"  Unit mode:        neuron")

    calib_dataloader = create_calibration_dataloader(
        dataset_name=args.calibration_dataset,
        num_samples=ntk_samples,
        batch_size=args.calibration_batch_size,
        max_length=args.calibration_max_length,
        seed=args.seed,
        tokenizer=tokenizer,
    )

    num_layers = model.config.num_hidden_layers
    intermediate_size = getattr(model.config, 'intermediate_size', model.config.hidden_size * 4)

    current_masks = {}
    for layer_idx in range(num_layers):
        current_masks[layer_idx] = torch.ones(intermediate_size)

    neuron_sensitivities = _compute_ntk_sensitivity_per_unit(
        model=model,
        dataloader=calib_dataloader,
        current_masks=current_masks,
        prev_masks=None,
        mask_type="ffn",
        num_samples=ntk_samples,
        device=device,
        verbose=True,
        dimension_multiple=1,
        ntk_target_mode=ntk_target_mode,
        ntk_unit_mode="neuron",
        ntk_method=ntk_method,
        normalize_per_layer=True,
    )

    print(f"\n  {'Layer':>6} | {'Min':>10} | {'Max':>10} | {'Mean':>10} | {'Std':>10}")
    print("  " + "-" * 50)
    for layer_idx in range(min(num_layers, 5)):
        if layer_idx in neuron_sensitivities:
            s = neuron_sensitivities[layer_idx]
            print(f"  {layer_idx:>6} | {s.min():.4f} | {s.max():.4f} | {s.mean():.4f} | {s.std():.4f}")
    if num_layers > 5:
        print(f"  ... ({num_layers - 5} more layers)")

    gc.collect()
    torch.cuda.empty_cache()

    return neuron_sensitivities


def compare_block_removal_with_ntk(simulated_ratios, ntk_sensitivities,
                                    num_layers, block_remove_threshold,
                                    protect_first, protect_last,
                                    target_ratio, model_info,
                                    ntk_alpha=0.3,
                                    pruning_target_mode="total"):
    print("\n" + "=" * 70)
    print("Comparison: GandA-only vs NTK-adjusted Block Removal")
    print("=" * 70)
    print(f"  NTK adjustment alpha: {ntk_alpha}")

    protected_layers = set()
    for i in range(protect_first):
        protected_layers.add(i)
    for i in range(num_layers - protect_last, num_layers):
        protected_layers.add(i)

    if pruning_target_mode == "ffn":
        max_removable = int(target_ratio * num_layers)
    else:
        max_removable_params = target_ratio * model_info["total_params"]
        max_removable = int(max_removable_params / model_info["block_params"])

    adjusted_ratios = {}
    for layer_idx in range(num_layers):
        sim_ratio = simulated_ratios[layer_idx]
        ntk_sens = ntk_sensitivities.get(layer_idx, 1.0)
        factor = max(0.01, 2.0 - ntk_sens)
        adjusted_ratios[layer_idx] = sim_ratio * (factor ** ntk_alpha)

    ganda_candidates = []
    for layer_idx in range(num_layers):
        if layer_idx in protected_layers:
            continue
        if simulated_ratios[layer_idx] > block_remove_threshold:
            ganda_candidates.append((layer_idx, simulated_ratios[layer_idx]))
    ganda_candidates.sort(key=lambda x: x[1], reverse=True)
    ganda_removed = set()
    for layer_idx, _ in ganda_candidates[:max_removable]:
        ganda_removed.add(layer_idx)

    ntk_candidates = []
    for layer_idx in range(num_layers):
        if layer_idx in protected_layers:
            continue
        if adjusted_ratios[layer_idx] > block_remove_threshold:
            ntk_candidates.append((layer_idx, adjusted_ratios[layer_idx]))
    ntk_candidates.sort(key=lambda x: x[1], reverse=True)
    ntk_removed = set()
    for layer_idx, _ in ntk_candidates[:max_removable]:
        ntk_removed.add(layer_idx)

    print(f"\n  {'Layer':>6} | {'Sim Ratio':>10} | {'NTK Sens':>10} | {'Adj Ratio':>10} | {'GandA':>8} | {'NTK-Adj':>8} | {'Diff':>6}")
    print("  " + "-" * 80)

    diff_count = 0
    for layer_idx in range(num_layers):
        sim = simulated_ratios[layer_idx]
        ntk_s = ntk_sensitivities.get(layer_idx, 1.0)
        adj = adjusted_ratios[layer_idx]
        is_protected = layer_idx in protected_layers

        if is_protected:
            g_dec = "PROT"
            n_dec = "PROT"
        else:
            g_dec = "REMOVE" if layer_idx in ganda_removed else "KEEP"
            n_dec = "REMOVE" if layer_idx in ntk_removed else "KEEP"

        diff = ""
        if not is_protected and g_dec != n_dec:
            diff = " <<<<<"
            diff_count += 1

        print(f"  {layer_idx:>6} | {sim:>9.2%} | {ntk_s:>10.4f} | {adj:>9.2%} | {g_dec:>8} | {n_dec:>8} |{diff}")

    print(f"\n  Summary:")
    print(f"    GandA-only removed:    {sorted(ganda_removed)}")
    print(f"    NTK-adjusted removed:  {sorted(ntk_removed)}")
    print(f"    Differences:           {diff_count} layers changed")

    only_ganda = ganda_removed - ntk_removed
    only_ntk = ntk_removed - ganda_removed
    if only_ganda:
        print(f"    Saved by NTK (high sensitivity):  {sorted(only_ganda)}")
    if only_ntk:
        print(f"    Newly removed by NTK (low sens):  {sorted(only_ntk)}")

    return {
        "ganda_removed": sorted(ganda_removed),
        "ntk_removed": sorted(ntk_removed),
        "adjusted_ratios": adjusted_ratios,
        "ntk_sensitivities": ntk_sensitivities,
        "diff_count": diff_count,
    }


def compute_model_param_info(model):
    total_params = sum(p.numel() for p in model.parameters())

    layer0 = model.model.layers[0]
    block_params = sum(p.numel() for p in layer0.parameters())
    ffn_params_per_layer = sum(p.numel() for p in layer0.mlp.parameters())
    attn_params_per_layer = sum(p.numel() for p in layer0.self_attn.parameters())

    num_layers = model.config.num_hidden_layers
    total_block_params = block_params * num_layers
    embed_params = total_params - total_block_params

    return {
        "total_params": total_params,
        "block_params": block_params,
        "ffn_params_per_layer": ffn_params_per_layer,
        "attn_params_per_layer": attn_params_per_layer,
        "embed_params": embed_params,
        "total_ffn_params": ffn_params_per_layer * num_layers,
        "total_attn_params": attn_params_per_layer * num_layers,
        "num_layers": num_layers,
    }


def compute_global_threshold(ffn_scores, target_ratio, model_info, pruning_target_mode="total"):
    print("\n" + "=" * 70)
    print("Step 2: Computing Global Threshold")
    print("=" * 70)

    all_scores = torch.cat([scores for scores in ffn_scores.values()])

    if pruning_target_mode == "ffn":
        ffn_equiv_ratio = target_ratio
        ffn_equiv_ratio = min(ffn_equiv_ratio, 0.99)

        print(f"  Total neurons: {all_scores.numel()}")
        print(f"  Pruning target mode:        ffn (direct FFN neuron ratio)")
        print(f"  Target FFN pruning ratio:   {target_ratio:.2%}")
        print(f"  FFN equivalent ratio:       {ffn_equiv_ratio:.2%} (no conversion)")
    else:
        ffn_equiv_ratio = target_ratio * model_info["total_params"] / model_info["total_ffn_params"]
        ffn_equiv_ratio = min(ffn_equiv_ratio, 0.99)

        print(f"  Total neurons: {all_scores.numel()}")
        print(f"  Pruning target mode:        total (model-wide param ratio)")
        print(f"  Target model pruning ratio: {target_ratio:.2%}")
        print(f"  FFN equivalent ratio:       {ffn_equiv_ratio:.2%}")
        print(f"    (= {target_ratio:.2f} × {model_info['total_params']:,} / {model_info['total_ffn_params']:,})")

    print(f"  Score stats: min={all_scores.min():.6f}, max={all_scores.max():.6f}, "
          f"mean={all_scores.mean():.6f}, std={all_scores.std():.6f}")

    threshold = torch.quantile(all_scores.float(), ffn_equiv_ratio)
    print(f"  Global threshold (percentile {ffn_equiv_ratio:.0%}): {threshold:.6f}")

    return threshold, all_scores, ffn_equiv_ratio


def simulate_per_block_ratios(ffn_scores, global_threshold, num_layers):
    print("\n" + "=" * 70)
    print("Step 3: Per-Block Simulated Pruning Ratios")
    print("=" * 70)

    simulated_ratios = {}
    for layer_idx in range(num_layers):
        scores = ffn_scores[layer_idx]
        total_neurons = scores.numel()
        pruned_count = (scores <= global_threshold).sum().item()
        sim_ratio = pruned_count / total_neurons
        simulated_ratios[layer_idx] = sim_ratio

    print(f"\n  {'Layer':>6} | {'Simulated Ratio':>16} | {'Pruned/Total':>14} | {'Status':>10}")
    print("  " + "-" * 58)
    for layer_idx in range(num_layers):
        scores = ffn_scores[layer_idx]
        total = scores.numel()
        pruned = int(total * simulated_ratios[layer_idx])
        ratio = simulated_ratios[layer_idx]
        bar = "#" * int(ratio * 20)
        print(f"  {layer_idx:>6} | {ratio:>15.2%} | {pruned:>6}/{total:<6} | {bar}")

    return simulated_ratios


def decide_block_removal(simulated_ratios, num_layers, block_remove_threshold,
                         protect_first, protect_last, target_ratio, model_info,
                         block_importance_mode="neuron", block_output_importance=None,
                         pruning_target_mode="total"):
    print("\n" + "=" * 70)
    print("Step 4: Block Removal Decision")
    print("=" * 70)
    print(f"  Block importance mode:  {block_importance_mode}")
    print(f"  Block remove threshold: {block_remove_threshold:.2%}")
    print(f"  Protected first layers: {protect_first}")
    print(f"  Protected last layers:  {protect_last}")

    protected_layers = set()
    for i in range(protect_first):
        protected_layers.add(i)
    for i in range(num_layers - protect_last, num_layers):
        protected_layers.add(i)

    if pruning_target_mode == "ffn":
        max_removable = int(target_ratio * num_layers)
        print(f"  Pruning target mode:    ffn")
        print(f"  Max removable blocks:   {max_removable}")
        print(f"    (= floor({target_ratio:.2f} × {num_layers}))")
    else:
        max_removable_params = target_ratio * model_info["total_params"]
        max_removable = int(max_removable_params / model_info["block_params"])
        print(f"  Pruning target mode:    total")
        print(f"  Max removable blocks:   {max_removable}")
        print(f"    (= floor({target_ratio:.2f} × {model_info['total_params']:,} / {model_info['block_params']:,}))")

    if block_importance_mode == "output":
        if block_output_importance is None:
            raise ValueError("block_output_importance required for output mode")

        candidates = []
        for layer_idx in range(num_layers):
            if layer_idx in protected_layers:
                continue
            candidates.append((layer_idx, block_output_importance[layer_idx]))

        candidates.sort(key=lambda x: x[1])

        output_threshold = 1.0 - block_remove_threshold
        selected_for_removal = set()
        for layer_idx, imp in candidates:
            if len(selected_for_removal) >= max_removable:
                break
            if imp <= output_threshold:
                selected_for_removal.add(layer_idx)

        removed_layers = []
        kept_layers = []

        print(f"  Output importance threshold: <= {output_threshold:.2f} (= 1 - {block_remove_threshold})")
        print(f"\n  {'Layer':>6} | {'Output Imp':>10} | {'Threshold':>10} | {'Protected':>10} | {'Decision':>10}")
        print("  " + "-" * 68)

        for layer_idx in range(num_layers):
            imp = block_output_importance[layer_idx]
            is_protected = layer_idx in protected_layers
            below_threshold = imp <= output_threshold

            if is_protected:
                decision = "KEEP (protected)"
                kept_layers.append(layer_idx)
            elif layer_idx in selected_for_removal:
                decision = "REMOVE"
                removed_layers.append(layer_idx)
            elif below_threshold:
                decision = "KEEP (cap limit)"
                kept_layers.append(layer_idx)
            else:
                decision = "KEEP"
                kept_layers.append(layer_idx)

            print(f"  {layer_idx:>6} | {imp:>10.6f} | {'<=' + f'{output_threshold:.2f}':>10} | "
                  f"{'Yes' if is_protected else 'No':>10} | {decision}")

    else:
        candidates = []
        for layer_idx in range(num_layers):
            if layer_idx in protected_layers:
                continue
            if simulated_ratios[layer_idx] > block_remove_threshold:
                candidates.append((layer_idx, simulated_ratios[layer_idx]))

        candidates.sort(key=lambda x: x[1], reverse=True)

        selected_for_removal = set()
        for layer_idx, sim_ratio in candidates:
            if len(selected_for_removal) >= max_removable:
                break
            selected_for_removal.add(layer_idx)

        if len(candidates) > max_removable:
            print(f"  Candidates exceeding threshold: {len(candidates)}, capped to {max_removable}")

        removed_layers = []
        kept_layers = []

        print(f"\n  {'Layer':>6} | {'Sim Ratio':>10} | {'Threshold':>10} | {'Protected':>10} | {'Decision':>10}")
        print("  " + "-" * 68)

        for layer_idx in range(num_layers):
            sim_ratio = simulated_ratios[layer_idx]
            is_protected = layer_idx in protected_layers
            exceeds = sim_ratio > block_remove_threshold

            if is_protected:
                decision = "KEEP (protected)"
                kept_layers.append(layer_idx)
            elif layer_idx in selected_for_removal:
                decision = "REMOVE"
                removed_layers.append(layer_idx)
            elif exceeds:
                decision = "KEEP (cap limit)"
                kept_layers.append(layer_idx)
            else:
                decision = "KEEP"
                kept_layers.append(layer_idx)

            print(f"  {layer_idx:>6} | {sim_ratio:>9.2%} | {block_remove_threshold:>9.2%} | "
                  f"{'Yes' if is_protected else 'No':>10} | {decision}")

    print(f"\n  Summary:")
    print(f"    Total blocks:   {num_layers}")
    print(f"    Removed blocks: {len(removed_layers)} {removed_layers}")
    print(f"    Kept blocks:    {len(kept_layers)}")
    print(f"    Block pruning ratio: {len(removed_layers) / num_layers:.2%}")

    return removed_layers, kept_layers


def recalculate_width_ratio(target_ratio, removed_layers, num_layers, model_info,
                            pruning_target_mode="total"):
    print("\n" + "=" * 70)
    print("Step 5: Recalculate Width Pruning Ratio (Parameter-based)")
    print("=" * 70)

    num_removed = len(removed_layers)
    total_params = model_info["total_params"]
    block_params = model_info["block_params"]
    ffn_params_per_layer = model_info["ffn_params_per_layer"]

    if pruning_target_mode == "ffn":
        total_ffn_params = model_info["total_ffn_params"]
        target_ffn_to_remove = target_ratio * total_ffn_params
        removed_ffn_by_blocks = num_removed * ffn_params_per_layer
        block_ffn_reduction = removed_ffn_by_blocks / total_ffn_params

        print(f"  Pruning target mode:        ffn")
        print(f"  Target FFN pruning ratio:   {target_ratio:.2%}")
        print(f"  Target FFN params to remove:{int(target_ffn_to_remove):,}")
        print(f"  FFN removed by blocks:      {removed_ffn_by_blocks:,} ({block_ffn_reduction:.2%} of total FFN)")

        if removed_ffn_by_blocks >= target_ffn_to_remove:
            remaining_width_ratio = 0.0
            print(f"  Block pruning alone meets FFN target! No FFN width pruning needed.")
        else:
            needed_additional = target_ffn_to_remove - removed_ffn_by_blocks
            remaining_ffn_total = (num_layers - num_removed) * ffn_params_per_layer
            remaining_width_ratio = needed_additional / remaining_ffn_total
            remaining_width_ratio = min(remaining_width_ratio, 0.95)

            print(f"  Additional FFN needed:      {int(needed_additional):,}")
            print(f"  Remaining FFN params:       {remaining_ffn_total:,} ({num_layers - num_removed} layers)")
            print(f"  FFN width pruning ratio:    {remaining_width_ratio:.2%}")
            print(f"    (= {int(needed_additional):,} / {remaining_ffn_total:,})")

        ffn_width_removed = (num_layers - num_removed) * ffn_params_per_layer * remaining_width_ratio
        actual_ffn_removed = removed_ffn_by_blocks + ffn_width_removed
        actual_ffn_ratio = actual_ffn_removed / total_ffn_params
        removed_by_blocks_total = num_removed * block_params
        actual_total_removed = removed_by_blocks_total + ffn_width_removed
        actual_total_ratio = actual_total_removed / total_params

        print(f"\n  Verification (FFN basis):")
        print(f"    FFN block removal:  {removed_ffn_by_blocks:,} params ({block_ffn_reduction:.2%} of FFN)")
        print(f"    FFN width removal:  {int(ffn_width_removed):,} params ({ffn_width_removed / total_ffn_params:.2%} of FFN)")
        print(f"    Total FFN removed:  {int(actual_ffn_removed):,} params ({actual_ffn_ratio:.2%} of FFN)")
        print(f"    Target FFN:         {int(target_ffn_to_remove):,} params ({target_ratio:.2%} of FFN)")
        print(f"  Verification (total model):")
        print(f"    Total removed:      {int(actual_total_removed):,} params ({actual_total_ratio:.2%} of total)")
    else:
        removed_by_blocks = num_removed * block_params
        block_reduction = removed_by_blocks / total_params

        target_params_to_remove = target_ratio * total_params

        print(f"  Pruning target mode:        total")
        print(f"  Target model pruning ratio: {target_ratio:.2%}")
        print(f"  Target params to remove:    {int(target_params_to_remove):,}")
        print(f"  Removed by blocks:          {removed_by_blocks:,} ({block_reduction:.2%} of total)")

        if removed_by_blocks >= target_params_to_remove:
            remaining_width_ratio = 0.0
            print(f"  Block pruning alone meets target! No FFN width pruning needed.")
        else:
            needed_additional = target_params_to_remove - removed_by_blocks
            remaining_ffn_total = (num_layers - num_removed) * ffn_params_per_layer
            remaining_width_ratio = needed_additional / remaining_ffn_total
            remaining_width_ratio = min(remaining_width_ratio, 0.95)

            print(f"  Additional params needed:   {int(needed_additional):,}")
            print(f"  Remaining FFN params:       {remaining_ffn_total:,} ({num_layers - num_removed} layers)")
            print(f"  FFN width pruning ratio:    {remaining_width_ratio:.2%}")
            print(f"    (= {int(needed_additional):,} / {remaining_ffn_total:,})")

        ffn_width_removed = (num_layers - num_removed) * ffn_params_per_layer * remaining_width_ratio
        actual_total_removed = removed_by_blocks + ffn_width_removed
        actual_total_ratio = actual_total_removed / total_params

        print(f"\n  Verification:")
        print(f"    Block removal:    {removed_by_blocks:,} params ({block_reduction:.2%})")
        print(f"    FFN width:        {int(ffn_width_removed):,} params ({ffn_width_removed / total_params:.2%})")
        print(f"    Total removed:    {int(actual_total_removed):,} params ({actual_total_ratio:.2%})")
        print(f"    Target:           {int(target_params_to_remove):,} params ({target_ratio:.2%})")

    return remaining_width_ratio


###############################################################################
#              Unified Depth+Width Pruning (Auto Ratio)
###############################################################################

def compute_unified_scores(ffn_scores, model_info, num_layers, dimension_multiple,
                           protect_first, protect_last, intermediate_size,
                           ntk_sensitivities=None, ntk_alpha=0.0,
                           neuron_sensitivities=None,
                           depth_score_mode="legacy",
                           ntk_alpha_layer=None, ntk_alpha_neuron=None):
    print("\n" + "=" * 70)
    print("Unified Pruning: Computing Depth/Width Scores")
    print("=" * 70)

    block_params = model_info["block_params"]
    ffn_params_per_layer = model_info["ffn_params_per_layer"]
    params_per_neuron = ffn_params_per_layer / intermediate_size
    num_groups = intermediate_size // dimension_multiple

    protected = set(range(protect_first)) | set(range(num_layers - protect_last, num_layers))

    a_layer = ntk_alpha_layer if ntk_alpha_layer is not None else ntk_alpha
    a_neuron = ntk_alpha_neuron if ntk_alpha_neuron is not None else ntk_alpha

    # --- Depth scores ---
    depth_cost = ffn_params_per_layer
    depth_scores = {}
    for layer_idx in range(num_layers):
        if depth_score_mode in ("neuron_only", "neuron_plus_layer") \
                and neuron_sensitivities is not None and a_neuron != 0 \
                and layer_idx in neuron_sensitivities:
            neuron_ntk = neuron_sensitivities[layer_idx].float()
            adjusted = ffn_scores[layer_idx].float() * (1.0 + a_neuron * (neuron_ntk - 1.0))
            score = adjusted.sum().item()
        else:
            score = ffn_scores[layer_idx].sum().item()
        if depth_score_mode in ("legacy", "neuron_plus_layer") \
                and ntk_sensitivities is not None and a_layer != 0:
            ntk_s = ntk_sensitivities.get(layer_idx, 1.0)
            score *= (1.0 + a_layer * (ntk_s - 1.0))
        depth_scores[layer_idx] = score

    # --- Width group scores (RANK-BASED) ---
    group_scores = {}
    layer_rank_indices = {}
    for g in range(num_groups):
        group_scores[g] = {}

    for layer_idx in range(num_layers):
        neuron_scores = ffn_scores[layer_idx].float().clone()
        if neuron_sensitivities is not None and a_neuron != 0 and layer_idx in neuron_sensitivities:
            neuron_ntk = neuron_sensitivities[layer_idx].float()
            neuron_scores = neuron_scores * (1.0 + a_neuron * (neuron_ntk - 1.0))
        elif ntk_sensitivities is not None and a_neuron != 0:
            ntk_s = ntk_sensitivities.get(layer_idx, 1.0)
            neuron_scores = neuron_scores * (1.0 + a_neuron * (ntk_s - 1.0))

        sorted_vals, sorted_idx = torch.sort(neuron_scores, descending=False)

        layer_rank_indices[layer_idx] = {}
        for g in range(num_groups):
            start = g * dimension_multiple
            end = start + dimension_multiple
            group_score = sorted_vals[start:end].sum().item()
            group_scores[g][layer_idx] = group_score
            layer_rank_indices[layer_idx][g] = sorted_idx[start:end].tolist()

    if depth_score_mode == "neuron_only" and neuron_sensitivities is not None and a_neuron != 0:
        depth_ntk = "neuron-NTK→sum"
    elif depth_score_mode == "neuron_plus_layer" and neuron_sensitivities is not None and a_neuron != 0:
        depth_ntk = "neuron-NTK→sum × layer-NTK"
    elif ntk_sensitivities is not None and a_layer != 0:
        depth_ntk = "layer-NTK"
    else:
        depth_ntk = "OFF"
    width_ntk = "neuron-NTK" if (neuron_sensitivities is not None and a_neuron != 0) else (
        "layer-NTK" if (ntk_sensitivities is not None and a_neuron != 0) else "OFF")
    print(f"  NTK adjustment:   depth={depth_ntk} (α_layer={a_layer}), width={width_ntk} (α_neuron={a_neuron}, mode={depth_score_mode})")
    print(f"  Depth candidates: {num_layers - len(protected)} layers (protected: {sorted(protected)})")
    print(f"  Width candidates: {num_groups} groups × {dimension_multiple} neurons")
    print(f"  Block params:     {block_params:,}")
    print(f"  Width step cost:  {num_layers} layers × {dimension_multiple} neurons × {params_per_neuron:.0f} = "
          f"{int(num_layers * dimension_multiple * params_per_neuron):,}")

    print(f"  Depth cost basis: FFN params = {depth_cost:,} (not block_params)")
    print(f"\n  {'Layer':>6} | {'Importance':>14} | {'Eff (imp/cost)':>14} | {'Protected':>10}")
    print("  " + "-" * 60)
    for layer_idx in sorted(depth_scores.keys()):
        imp = depth_scores[layer_idx]
        eff = imp / depth_cost
        prot = "Yes" if layer_idx in protected else ""
        print(f"  {layer_idx:>6} | {imp:>14.6e} | {eff:>14.6e} | {prot:>10}")

    return depth_scores, group_scores, protected, layer_rank_indices


def greedy_unified_pruning(depth_scores, group_scores, model_info,
                           target_ratio, num_layers, dimension_multiple,
                           protected, intermediate_size, fill_width=False,
                           stop_mode="before",
                           disable_depth=False, disable_width=False):
    print("\n" + "=" * 70)
    print("Unified Pruning: Greedy Selection")
    print("=" * 70)

    block_params = model_info["block_params"]
    ffn_params_per_layer = model_info["ffn_params_per_layer"]
    total_params = model_info["total_params"]
    params_per_neuron = ffn_params_per_layer / intermediate_size
    num_base_groups = intermediate_size // dimension_multiple  # 86 (128-neuron)

    bases_per_super = 4
    super_neuron_size = dimension_multiple * bases_per_super  # 512
    num_super_groups = num_base_groups // bases_per_super     # 21
    num_remainder = num_base_groups % bases_per_super         # 2

    target_remove = target_ratio * total_params

    surviving_layers = set(range(num_layers))
    surviving_supers = {}
    for s in range(num_super_groups):
        surviving_supers[s] = list(range(s * bases_per_super, (s + 1) * bases_per_super))
    if num_remainder > 0:
        surviving_supers[num_super_groups] = list(range(num_super_groups * bases_per_super, num_base_groups))

    full_super_ids = [s for s in surviving_supers
                      if len(surviving_supers[s]) == bases_per_super]
    remainder_super_ids = [s for s in surviving_supers
                           if len(surviving_supers[s]) != bases_per_super]

    total_width_candidates = len(full_super_ids)
    super_cost_base = int(num_layers * super_neuron_size * params_per_neuron)

    params_removed = 0
    removed_layers = []
    removed_base_groups = []

    print(f"  Target: remove {int(target_remove):,} params ({target_ratio:.2%} of {total_params:,})")
    print(f"  Candidates: {num_layers} depth + {total_width_candidates} width super-groups "
          f"(remainder {len(remainder_super_ids)} excluded from greedy) = {num_layers + total_width_candidates}")
    print(f"  Depth 1 step:  {block_params:,} params")
    print(f"  Width 1 super: {super_cost_base:,} params ({super_neuron_size} neurons × {num_layers} layers)")
    print(f"  Base groups:   {num_base_groups} ({dimension_multiple} neurons each)")
    print(f"  Super groups:  {num_super_groups} × {bases_per_super} bases" +
          (f" + 1 × {num_remainder} bases (excluded)" if num_remainder > 0 else ""))
    print(f"  Selection:     efficiency (imp/cost) based, lowest first")

    step = 0
    print(f"\n  {'Step':>5} | {'Type':>6} | {'ID':>8} | {'Importance':>14} | {'Efficiency':>14} | {'Params':>16} | {'Cumul':>14}")
    print("  " + "-" * 100)

    while params_removed < target_remove:
        n_surviving = len(surviving_layers)
        if n_surviving == 0:
            break

        # --- Best depth candidate ---
        best_depth_layer = None
        best_depth_imp = float('inf')
        if not disable_depth:
            for i in surviving_layers - protected:
                if depth_scores[i] < best_depth_imp:
                    best_depth_imp = depth_scores[i]
                    best_depth_layer = i

        # --- Best width super-group candidate (full super-groups only) ---
        best_super_id = None
        best_super_imp = float('inf')
        if disable_width:
            full_super_ids_for_pick = []
        else:
            full_super_ids_for_pick = full_super_ids
        for s_id in full_super_ids_for_pick:
            if s_id not in surviving_supers:
                continue
            base_list = surviving_supers[s_id]
            total_imp = 0
            for bg in base_list:
                total_imp += sum(group_scores[bg][i] for i in surviving_layers)
            if total_imp < best_super_imp:
                best_super_imp = total_imp
                best_super_id = s_id

        if best_depth_layer is None and best_super_id is None:
            break

        if best_super_id is not None:
            super_bases = surviving_supers[best_super_id]
            super_neurons = len(super_bases) * dimension_multiple
            super_cost = int(n_surviving * super_neurons * params_per_neuron)
        else:
            super_cost = float('inf')

        width_neurons_removed = len(removed_base_groups) * dimension_multiple
        depth_overlap = int(width_neurons_removed * params_per_neuron)
        effective_depth_cost = max(block_params - depth_overlap, 0)

        depth_ok = (best_depth_layer is not None and
                    params_removed + effective_depth_cost <= target_remove)
        width_ok = (best_super_id is not None and
                    params_removed + super_cost <= target_remove)

        if not depth_ok and not width_ok:
            if stop_mode == "after":
                depth_eff = (best_depth_imp / block_params
                             if best_depth_layer is not None and block_params > 0 else float('inf'))
                width_eff = (best_super_imp / super_cost
                             if best_super_id is not None and super_cost > 0 else float('inf'))
                step += 1
                if depth_eff <= width_eff and best_depth_layer is not None:
                    removed_layers.append(best_depth_layer)
                    surviving_layers.remove(best_depth_layer)
                    params_removed += effective_depth_cost
                    print(f"  {step:>5} | {'DEPTH*':>6} | L{best_depth_layer:<6} | {best_depth_imp:>14.6e} | "
                          f"{depth_eff:>14.6e} | {effective_depth_cost:>16,} | {params_removed:>14,} "
                          f"({params_removed/total_params:.2%})  [over-shoot]")
                elif best_super_id is not None:
                    super_bases_pick = surviving_supers.pop(best_super_id)
                    if best_super_id in full_super_ids:
                        full_super_ids.remove(best_super_id)
                    removed_base_groups.extend(super_bases_pick)
                    params_removed += super_cost
                    neurons = len(super_bases_pick) * dimension_multiple
                    print(f"  {step:>5} | {'WIDTH*':>6} | S{best_super_id:<2}({neurons:>3}) | {best_super_imp:>14.6e} | "
                          f"{width_eff:>14.6e} | {super_cost:>16,} | {params_removed:>14,} "
                          f"({params_removed/total_params:.2%})  [over-shoot]")
                break

            if fill_width and not disable_width:
                remaining = target_remove - params_removed
                base_cost = int(n_surviving * dimension_multiple * params_per_neuron)
                if base_cost > 0:
                    all_surviving_bases = set()
                    for base_list in surviving_supers.values():
                        all_surviving_bases.update(base_list)

                    fill_count = int(remaining / base_cost)
                    fill_count = min(fill_count, len(all_surviving_bases))
                    if fill_count > 0:
                        base_imps = []
                        for bg in all_surviving_bases:
                            imp = sum(group_scores[bg][i] for i in surviving_layers)
                            base_imps.append((bg, imp))
                        base_imps.sort(key=lambda x: x[1])

                        filled = []
                        for bg, _ in base_imps[:fill_count]:
                            removed_base_groups.append(bg)
                            filled.append(bg)
                        for bg in filled:
                            for s_id in list(surviving_supers.keys()):
                                if bg in surviving_supers[s_id]:
                                    surviving_supers[s_id].remove(bg)
                                    if not surviving_supers[s_id]:
                                        del surviving_supers[s_id]
                                    break

                        fill_cost = fill_count * base_cost
                        params_removed += fill_cost
                        step += 1
                        print(f"  {step:>5} | {'FILL':>6} | {fill_count}×128  | {'(budget)':>14} | "
                              f"{fill_cost:>16,} | {params_removed:>14,} ({params_removed/total_params:.2%})")
            break

        choose_depth = False
        if depth_ok and width_ok:
            depth_eff = best_depth_imp / block_params if block_params > 0 else float('inf')
            width_eff = best_super_imp / super_cost if super_cost > 0 else float('inf')
            choose_depth = (depth_eff <= width_eff)
        elif depth_ok:
            choose_depth = True

        step += 1

        if choose_depth:
            removed_layers.append(best_depth_layer)
            surviving_layers.remove(best_depth_layer)
            params_removed += effective_depth_cost
            eff = best_depth_imp / block_params
            print(f"  {step:>5} | {'DEPTH':>6} | L{best_depth_layer:<6} | {best_depth_imp:>14.6e} | "
                  f"{eff:>14.6e} | {effective_depth_cost:>16,} | {params_removed:>14,} ({params_removed/total_params:.2%})")
        else:
            super_bases = surviving_supers.pop(best_super_id)
            if best_super_id in full_super_ids:
                full_super_ids.remove(best_super_id)
            removed_base_groups.extend(super_bases)
            params_removed += super_cost
            neurons = len(super_bases) * dimension_multiple
            eff = best_super_imp / super_cost
            print(f"  {step:>5} | {'WIDTH':>6} | S{best_super_id:<2}({neurons:>3}) | {best_super_imp:>14.6e} | "
                  f"{eff:>14.6e} | {super_cost:>16,} | {params_removed:>14,} ({params_removed/total_params:.2%})")

    # Summary
    kept_layers = sorted(surviving_layers)
    actual_ratio = params_removed / total_params
    depth_params = len(removed_layers) * block_params
    width_params = params_removed - depth_params

    print(f"\n  Result:")
    print(f"    Depth removed: {len(removed_layers)} layers → {sorted(removed_layers)}")
    print(f"    Width removed: {len(removed_base_groups)} base groups ({len(removed_base_groups) * dimension_multiple} neurons/layer)")
    print(f"    Depth params:  {depth_params:,} ({depth_params / total_params:.2%})")
    print(f"    Width params:  {int(width_params):,} ({width_params / total_params:.2%})")
    print(f"    Total removed: {int(params_removed):,} ({actual_ratio:.2%}, target: {target_ratio:.2%})")
    print(f"    Kept layers:   {len(kept_layers)} → {kept_layers}")

    return removed_layers, kept_layers, removed_base_groups, params_removed


def _apply_removal_masks_to_model(model, removed_layers_set, layer_removed_neurons):
    """Zero out weights for removed depth layers and removed FFN neurons (in-place).

    - removed_layers_set: set of layer indices to fully remove (attn.o_proj + mlp.down_proj zeroed).
    - layer_removed_neurons: dict[layer_idx -> set(neuron_idx)] for partial width removals.

    After this call, forward passes through the model behave as if the masked parts are absent.
    Used by iterative pruning to recompute importance/NTK on the evolving masked state.
    """
    with torch.no_grad():
        for L in removed_layers_set:
            layer = model.model.layers[L]
            layer.self_attn.o_proj.weight.zero_()
            if getattr(layer.self_attn.o_proj, 'bias', None) is not None:
                layer.self_attn.o_proj.bias.zero_()
            layer.mlp.down_proj.weight.zero_()
            if getattr(layer.mlp.down_proj, 'bias', None) is not None:
                layer.mlp.down_proj.bias.zero_()

        for L, neurons in layer_removed_neurons.items():
            if L in removed_layers_set or not neurons:
                continue
            layer = model.model.layers[L]
            idx = torch.tensor(sorted(neurons), dtype=torch.long,
                               device=layer.mlp.down_proj.weight.device)
            # gate_proj/up_proj: [intermediate, hidden] → zero rows
            layer.mlp.gate_proj.weight[idx, :] = 0
            layer.mlp.up_proj.weight[idx, :] = 0
            if getattr(layer.mlp.gate_proj, 'bias', None) is not None:
                layer.mlp.gate_proj.bias[idx] = 0
            if getattr(layer.mlp.up_proj, 'bias', None) is not None:
                layer.mlp.up_proj.bias[idx] = 0
            # down_proj: [hidden, intermediate] → zero columns
            layer.mlp.down_proj.weight[:, idx] = 0


def iterative_unified_pruning(model, tokenizer, args, device, num_layers, intermediate_size,
                              model_info):
    print("\n" + "=" * 70)
    print("Iterative Unified Pruning (recompute GandA + NTK every step)")
    print("=" * 70)

    block_params = model_info["block_params"]
    ffn_params_per_layer = model_info["ffn_params_per_layer"]
    total_params = model_info["total_params"]
    params_per_neuron = ffn_params_per_layer / intermediate_size
    dimension_multiple = args.dimension_multiple
    bases_per_super = 4
    super_neurons = dimension_multiple * bases_per_super  # 512

    target_remove = args.pruning_ratio * total_params
    protected = set(range(args.protect_first)) | set(range(num_layers - args.protect_last, num_layers))

    removed_layers = []  # ordered
    removed_layers_set = set()
    layer_removed_neurons = {i: set() for i in range(num_layers)}  # per-layer removed neuron indices
    iter_picks = []  # log
    params_removed = 0

    print(f"  Target: remove {int(target_remove):,} params ({args.pruning_ratio:.2%} of {total_params:,})")
    print(f"  Protected layers: {sorted(protected)}")
    print(f"  Depth step:  {block_params:,} params")
    print(f"  Width step:  {int(num_layers * super_neurons * params_per_neuron):,} params "
          f"@ t=0 ({super_neurons} neurons × {num_layers} layers)")
    print(f"  Stop mode:   {args.unified_stop_mode}")

    iteration = 0
    while params_removed < target_remove:
        iteration += 1

        # 1. Apply current masks to the model
        _apply_removal_masks_to_model(model, removed_layers_set, layer_removed_neurons)

        # 2. Recompute GandA on masked model
        print(f"\n  ── Iteration {iteration} ──  (cum removed: {params_removed/total_params:.2%})")
        ffn_scores = compute_ganda_scores(model, tokenizer, args, device)

        # 3. Recompute NTK on masked model (optional)
        ntk_sensitivities = None
        neuron_sensitivities = None
        if args.use_ntk_layer:
            gc.collect()
            torch.cuda.empty_cache()
            ntk_sensitivities = compute_layer_ntk_sensitivity(
                model, tokenizer, args, device, ffn_scores
            )
            gc.collect()
            torch.cuda.empty_cache()
            neuron_sensitivities = compute_neuron_ntk_sensitivity(
                model, tokenizer, args, device
            )

        # 4. Compute depth_scores and rank-based group assignment
        #    IMPORTANT: exclude already-removed neurons from both the depth sum and
        #    the width sort, so previously-removed elements don't poison the scores.
        depth_scores = {}
        layer_rank_indices = {}  # L → {rank_g: [neuron_idx_list]}
        use_ntk = args.use_ntk_layer and args.ntk_alpha != 0

        for L in range(num_layers):
            layer_rank_indices[L] = {}
            if L in removed_layers_set:
                depth_scores[L] = float('inf')
                continue

            # Active neurons mask for this layer
            removed_set_L = layer_removed_neurons[L]
            if removed_set_L:
                active_mask = torch.ones(intermediate_size, dtype=torch.bool,
                                         device=ffn_scores[L].device)
                active_idx_list = sorted(removed_set_L)
                active_mask[torch.tensor(active_idx_list, dtype=torch.long,
                                         device=ffn_scores[L].device)] = False
            else:
                active_mask = torch.ones(intermediate_size, dtype=torch.bool,
                                         device=ffn_scores[L].device)

            active_idx = active_mask.nonzero().flatten()  # absolute neuron indices that are active

            # Raw (NTK-adjusted) per-neuron scores over ACTIVE neurons
            neuron_scores = ffn_scores[L].float().clone()
            if use_ntk:
                if neuron_sensitivities is not None and L in neuron_sensitivities:
                    ntk_vec = neuron_sensitivities[L].float().to(neuron_scores.device)
                    neuron_scores = neuron_scores * (1.0 + args.ntk_alpha * (ntk_vec - 1.0))
                elif ntk_sensitivities is not None:
                    ntk_s = ntk_sensitivities.get(L, 1.0)
                    neuron_scores = neuron_scores * (1.0 + args.ntk_alpha * (ntk_s - 1.0))

            active_scores = neuron_scores[active_idx]  # shape = [num_active]

            # depth_score = sum over ACTIVE neurons only
            depth_scores[L] = float(active_scores.sum().item())

            # rank-based base groups over ACTIVE neurons only
            if active_idx.numel() >= dimension_multiple:
                sorted_vals, sorted_rank = torch.sort(active_scores, descending=False)
                sorted_abs = active_idx[sorted_rank]  # absolute indices in ascending importance
                num_active = int(active_idx.numel())
                num_full_groups = num_active // dimension_multiple
                for g in range(num_full_groups):
                    start = g * dimension_multiple
                    end = start + dimension_multiple
                    layer_rank_indices[L][g] = sorted_abs[start:end].tolist()

        # (Optional) verbose depth ranking print for each iteration
        print(f"\n  Depth scores (active-masked, NTK={use_ntk}):")
        print(f"    {'Layer':>5} | {'Importance':>14} | {'Eff':>14} | {'Prot':>5}")
        print("    " + "-" * 50)
        for L in sorted(depth_scores.keys()):
            prot = "Yes" if L in protected else ""
            imp = depth_scores[L]
            eff = imp / block_params if math.isfinite(imp) else float('inf')
            if math.isfinite(imp):
                print(f"    {L:>5} | {imp:>14.6e} | {eff:>14.6e} | {prot:>5}")
            else:
                print(f"    {L:>5} | {'removed':>14} | {'-':>14} | {prot:>5}")

        n_surviving = num_layers - len(removed_layers_set)

        # 6a. Best depth candidate (layer with lowest finite depth score, not protected, not removed)
        best_depth = None
        best_depth_imp = float('inf')
        for L in range(num_layers):
            if L in protected or L in removed_layers_set:
                continue
            s = depth_scores[L]
            if s < best_depth_imp and math.isfinite(s):
                best_depth_imp = s
                best_depth = L

        # 6b. Best width super-group candidate @ current state = ranks 0..3 of each surviving layer
        # (ranks 0..3 = 4 lowest-importance base groups, 128 neurons each = 512 neurons/layer)
        width_imp_sum = 0.0
        width_neurons_per_layer = {}  # L → list of neuron idx to remove
        width_valid = True
        for L in range(num_layers):
            if L in removed_layers_set:
                continue
            lri = layer_rank_indices.get(L, {})
            if any(g not in lri for g in range(bases_per_super)):
                width_valid = False
                break
            picked_neurons = []
            for g in range(bases_per_super):
                picked_neurons.extend(lri[g])
            width_neurons_per_layer[L] = picked_neurons

            neuron_scores_L = ffn_scores[L].float()
            if use_ntk:
                if neuron_sensitivities is not None and L in neuron_sensitivities:
                    ntk_vec = neuron_sensitivities[L].float().to(neuron_scores_L.device)
                    neuron_scores_L = neuron_scores_L * (1.0 + args.ntk_alpha * (ntk_vec - 1.0))
                elif ntk_sensitivities is not None:
                    ntk_s = ntk_sensitivities.get(L, 1.0)
                    neuron_scores_L = neuron_scores_L * (1.0 + args.ntk_alpha * (ntk_s - 1.0))
            idx_t = torch.tensor(picked_neurons, dtype=torch.long, device=neuron_scores_L.device)
            width_imp_sum += float(neuron_scores_L[idx_t].sum().item())

        super_cost = int(n_surviving * super_neurons * params_per_neuron) if width_valid else 0
        depth_eff = (best_depth_imp / block_params) if best_depth is not None else float('inf')
        width_eff = (width_imp_sum / super_cost) if (width_valid and super_cost > 0) else float('inf')

        depth_ok = best_depth is not None and params_removed + block_params <= target_remove
        width_ok = width_valid and super_cost > 0 and params_removed + super_cost <= target_remove

        # 7. Budget / stop_mode handling
        if not depth_ok and not width_ok:
            if args.unified_stop_mode == "after":
                # one more step that overshoots, then stop
                if depth_eff <= width_eff and best_depth is not None:
                    removed_layers.append(best_depth)
                    removed_layers_set.add(best_depth)
                    params_removed += block_params
                    iter_picks.append(("DEPTH*", best_depth, best_depth_imp, block_params, True))
                    print(f"    → DEPTH* L{best_depth}  imp={best_depth_imp:.3e}  "
                          f"cum={params_removed/total_params:.2%}  [over-shoot]")
                elif width_valid:
                    for L, ns in width_neurons_per_layer.items():
                        layer_removed_neurons[L].update(ns)
                    params_removed += super_cost
                    iter_picks.append(("WIDTH*", -1, width_imp_sum, super_cost, True))
                    print(f"    → WIDTH* 512n/layer  imp={width_imp_sum:.3e}  "
                          f"cum={params_removed/total_params:.2%}  [over-shoot]")
            else:
                print(f"    (stop: both depth/width would overshoot, stop_mode=before)")
            break

        # Pick the better (lower efficiency) feasible candidate
        choose_depth = False
        if depth_ok and width_ok:
            choose_depth = (depth_eff <= width_eff)
        elif depth_ok:
            choose_depth = True
        else:
            choose_depth = False

        if choose_depth:
            removed_layers.append(best_depth)
            removed_layers_set.add(best_depth)
            params_removed += block_params
            iter_picks.append(("DEPTH", best_depth, best_depth_imp, block_params, False))
            print(f"    → DEPTH L{best_depth}  eff={depth_eff:.3e}  "
                  f"imp={best_depth_imp:.3e}  cum={params_removed/total_params:.2%}")
        else:
            for L, ns in width_neurons_per_layer.items():
                layer_removed_neurons[L].update(ns)
            params_removed += super_cost
            iter_picks.append(("WIDTH", -1, width_imp_sum, super_cost, False))
            print(f"    → WIDTH 512n/layer  eff={width_eff:.3e}  "
                  f"imp={width_imp_sum:.3e}  cum={params_removed/total_params:.2%}")

    kept_layers = sorted(set(range(num_layers)) - removed_layers_set)

    # Build per-layer ffn masks (True=keep)
    ffn_masks = {}
    uniform_removed_count = None
    for L in kept_layers:
        mask = torch.ones(intermediate_size, dtype=torch.bool)
        for n in layer_removed_neurons[L]:
            mask[n] = False
        ffn_masks[L] = mask
        cnt = int((~mask).sum().item())
        if uniform_removed_count is None:
            uniform_removed_count = cnt
        elif cnt != uniform_removed_count:
            uniform_removed_count = -1  # non-uniform

    actual_ratio = params_removed / total_params
    print(f"\n  Iterative Result:")
    print(f"    Iterations:    {iteration}")
    print(f"    Depth removed: {len(removed_layers)} layers → {sorted(removed_layers)}")
    print(f"    Width removed: "
          + (f"{uniform_removed_count} neurons/layer (uniform)"
             if uniform_removed_count and uniform_removed_count != -1
             else "non-uniform per layer"))
    print(f"    Total removed: {int(params_removed):,} ({actual_ratio:.2%}, target: {args.pruning_ratio:.2%})")
    print(f"    Kept layers:   {len(kept_layers)} → {kept_layers}")

    return removed_layers, kept_layers, ffn_masks, params_removed, iter_picks


def create_unified_width_masks(kept_layers, removed_groups, dimension_multiple, intermediate_size,
                               layer_rank_indices=None):
    print("\n" + "=" * 70)
    print("Unified Pruning: Creating Width Masks (rank-based)")
    print("=" * 70)

    removed_set = set(removed_groups)
    pruned_per_layer = len(removed_set) * dimension_multiple
    kept_per_layer = intermediate_size - pruned_per_layer
    width_ratio = pruned_per_layer / intermediate_size

    print(f"  Removed rank groups: {len(removed_set)} ({pruned_per_layer} neurons/layer)")
    print(f"  Kept neurons:   {kept_per_layer} / {intermediate_size} ({1 - width_ratio:.2%})")
    print(f"  Width ratio:    {width_ratio:.2%}")
    print(f"  Dimension check: {kept_per_layer} % {dimension_multiple} = "
          f"{kept_per_layer % dimension_multiple} "
          f"({'OK' if kept_per_layer % dimension_multiple == 0 else 'WARN'})")

    ffn_masks = {}
    if layer_rank_indices is not None:
        for layer_idx in kept_layers:
            mask = torch.ones(intermediate_size)
            if layer_idx in layer_rank_indices:
                rank_map = layer_rank_indices[layer_idx]
                for g in removed_set:
                    if g in rank_map:
                        for nidx in rank_map[g]:
                            mask[nidx] = 0.0
            ffn_masks[layer_idx] = mask

        sample_layers = list(kept_layers)[:3]
        print(f"\n  [Sample] Removed indices per layer (first 3 kept layers):")
        for L in sample_layers:
            removed_idx = [i for i, v in enumerate(ffn_masks[L].tolist()) if v == 0.0]
            print(f"    Layer {L}: {len(removed_idx)} indices, first 8 = {removed_idx[:8]}, "
                  f"last 4 = {removed_idx[-4:]}")
    else:
        mask_template = torch.ones(intermediate_size)
        for g in removed_set:
            start = g * dimension_multiple
            end = start + dimension_multiple
            mask_template[start:end] = 0.0
        for layer_idx in kept_layers:
            ffn_masks[layer_idx] = mask_template.clone()

    return ffn_masks


def create_width_masks(ffn_scores, kept_layers, remaining_width_ratio,
                       dimension_multiple, intermediate_size):
    print("\n" + "=" * 70)
    print("Step 6: Create Width Masks for Remaining Blocks")
    print("=" * 70)

    num_groups = intermediate_size // dimension_multiple
    num_groups_to_keep = max(1, int(num_groups * (1 - remaining_width_ratio)))
    num_keep = num_groups_to_keep * dimension_multiple

    print(f"  Intermediate size:     {intermediate_size}")
    print(f"  Dimension multiple:    {dimension_multiple}")
    print(f"  Total groups:          {num_groups}")
    print(f"  Groups to keep:        {num_groups_to_keep}")
    print(f"  Neurons to keep:       {num_keep} / {intermediate_size}")
    print(f"  Width pruning ratio:   {remaining_width_ratio:.2%}")

    ffn_masks = {}

    for layer_idx in kept_layers:
        scores = ffn_scores[layer_idx]

        if remaining_width_ratio <= 0:
            ffn_masks[layer_idx] = torch.ones(intermediate_size)
        else:
            group_importance = torch.zeros(num_groups)
            for g in range(num_groups):
                start = g * dimension_multiple
                end = start + dimension_multiple
                group_importance[g] = scores[start:end].sum()

            _, top_group_indices = torch.topk(group_importance, num_groups_to_keep)

            mask = torch.zeros(intermediate_size)
            for g_idx in top_group_indices:
                start = g_idx.item() * dimension_multiple
                end = start + dimension_multiple
                mask[start:end] = 1.0

            ffn_masks[layer_idx] = mask

    print(f"\n  {'Layer':>6} | {'Kept Neurons':>14} | {'Pruned Neurons':>15} | {'Layer Ratio':>12}")
    print("  " + "-" * 58)

    total_kept = 0
    total_neurons = 0
    for layer_idx in sorted(ffn_masks.keys()):
        mask = ffn_masks[layer_idx]
        kept = int(mask.sum().item())
        total = mask.numel()
        ratio = 1.0 - kept / total
        total_kept += kept
        total_neurons += total
        print(f"  {layer_idx:>6} | {kept:>14} | {total - kept:>15} | {ratio:>11.2%}")

    return ffn_masks


def print_final_summary(num_layers, removed_layers, kept_layers, ffn_masks,
                        intermediate_size, target_ratio, model_info):
    print("\n" + "=" * 70)
    print("FINAL SUMMARY: Depth + Width Pruning Result")
    print("=" * 70)

    total_params = model_info["total_params"]
    block_params = model_info["block_params"]
    ffn_params_per_layer = model_info["ffn_params_per_layer"]

    print(f"\n  [Depth Pruning]")
    print(f"    Total blocks:    {num_layers}")
    print(f"    Removed blocks:  {len(removed_layers)} → {removed_layers}")
    print(f"    Remaining blocks: {len(kept_layers)}")
    removed_block_params = len(removed_layers) * block_params
    print(f"    Params removed:  {removed_block_params:,} ({removed_block_params / total_params:.2%} of total)")

    total_original_neurons = num_layers * intermediate_size
    total_remaining_neurons = 0
    for layer_idx in kept_layers:
        if layer_idx in ffn_masks:
            total_remaining_neurons += int(ffn_masks[layer_idx].sum().item())

    print(f"\n  [Width Pruning] (remaining blocks only)")
    ffn_width_removed_params = 0
    for layer_idx in kept_layers:
        if layer_idx in ffn_masks:
            mask = ffn_masks[layer_idx]
            kept = int(mask.sum().item())
            pruned_neurons = intermediate_size - kept
            if pruned_neurons > 0:
                print(f"    Layer {layer_idx:>2}: {kept}/{intermediate_size} neurons kept ({pruned_neurons} pruned)")
            ffn_width_removed_params += pruned_neurons * (ffn_params_per_layer / intermediate_size)

    if ffn_width_removed_params == 0:
        print(f"    (No width pruning applied)")

    total_removed_params = removed_block_params + ffn_width_removed_params
    actual_ratio = total_removed_params / total_params

    print(f"\n  [Overall - Total Model Parameters]")
    print(f"    Original params:     {total_params:,}")
    print(f"    Block removal:       -{removed_block_params:,} ({removed_block_params / total_params:.2%})")
    print(f"    FFN width removal:   -{int(ffn_width_removed_params):,} ({ffn_width_removed_params / total_params:.2%})")
    print(f"    Total removed:       -{int(total_removed_params):,}")
    print(f"    Remaining params:    {int(total_params - total_removed_params):,}")
    print(f"    Effective pruning:   {actual_ratio:.2%}  (target: {target_ratio:.2%})")


def physically_prune_model(model, tokenizer, removed_layers, kept_layers, ffn_masks,
                           output_dir):
    print("\n" + "=" * 70)
    print("Step 7: Physical Pruning (Depth + Width)")
    print("=" * 70)

    import copy

    pruned_model = copy.deepcopy(model)
    pruned_model.eval()

    if removed_layers:
        print(f"\n  [Depth] Removing {len(removed_layers)} blocks: {removed_layers}")
        for layer_idx in sorted(removed_layers, reverse=True):
            del pruned_model.model.layers[layer_idx]
        pruned_model.config.num_hidden_layers = len(pruned_model.model.layers)
        if hasattr(pruned_model.config, 'layer_types') and pruned_model.config.layer_types is not None:
            kept_indices = [i for i in range(model.config.num_hidden_layers) if i not in removed_layers]
            pruned_model.config.layer_types = [pruned_model.config.layer_types[i] for i in kept_indices]
        print(f"  [Depth] Remaining layers: {pruned_model.config.num_hidden_layers}")

    old_to_new = {}
    new_idx = 0
    num_original = model.config.num_hidden_layers
    for old_idx in range(num_original):
        if old_idx not in removed_layers:
            old_to_new[old_idx] = new_idx
            new_idx += 1

    width_pruned_any = False
    for old_layer_idx in kept_layers:
        if old_layer_idx not in ffn_masks:
            continue
        mask = ffn_masks[old_layer_idx]
        if mask.sum().item() == mask.numel():
            continue

        width_pruned_any = True
        new_layer_idx = old_to_new[old_layer_idx]
        layer = pruned_model.model.layers[new_layer_idx]
        keep_indices = torch.where(mask > 0)[0]
        device = next(layer.parameters()).device

        # LLaMA-style: gate_proj, up_proj, down_proj
        # gate_proj: [intermediate, hidden] → output pruning
        if hasattr(layer.mlp, 'gate_proj'):
            w = layer.mlp.gate_proj.weight.data
            layer.mlp.gate_proj = nn.Linear(w.size(1), len(keep_indices),
                                            bias=layer.mlp.gate_proj.bias is not None,
                                            device=device, dtype=w.dtype)
            layer.mlp.gate_proj.weight.data = w[keep_indices.to(device)]

        # up_proj: [intermediate, hidden] → output pruning
        if hasattr(layer.mlp, 'up_proj'):
            w = layer.mlp.up_proj.weight.data
            layer.mlp.up_proj = nn.Linear(w.size(1), len(keep_indices),
                                          bias=layer.mlp.up_proj.bias is not None,
                                          device=device, dtype=w.dtype)
            layer.mlp.up_proj.weight.data = w[keep_indices.to(device)]

        # down_proj: [hidden, intermediate] → input pruning
        if hasattr(layer.mlp, 'down_proj'):
            w = layer.mlp.down_proj.weight.data
            layer.mlp.down_proj = nn.Linear(len(keep_indices), w.size(0),
                                            bias=layer.mlp.down_proj.bias is not None,
                                            device=device, dtype=w.dtype)
            layer.mlp.down_proj.weight.data = w[:, keep_indices.to(device)]

        # Phi-style: fc1, fc2
        # fc1: [intermediate, hidden] → output pruning
        if hasattr(layer.mlp, 'fc1'):
            w = layer.mlp.fc1.weight.data
            b = layer.mlp.fc1.bias
            layer.mlp.fc1 = nn.Linear(w.size(1), len(keep_indices),
                                      bias=b is not None,
                                      device=device, dtype=w.dtype)
            layer.mlp.fc1.weight.data = w[keep_indices.to(device)]
            if b is not None:
                layer.mlp.fc1.bias.data = b.data[keep_indices.to(device)]

        # fc2: [hidden, intermediate] → input pruning
        if hasattr(layer.mlp, 'fc2'):
            w = layer.mlp.fc2.weight.data
            b = layer.mlp.fc2.bias
            layer.mlp.fc2 = nn.Linear(len(keep_indices), w.size(0),
                                      bias=b is not None,
                                      device=device, dtype=w.dtype)
            layer.mlp.fc2.weight.data = w[:, keep_indices.to(device)]
            if b is not None:
                layer.mlp.fc2.bias.data = b.data

    if width_pruned_any:
        first_layer = pruned_model.model.layers[0]
        if hasattr(first_layer.mlp, 'gate_proj'):
            pruned_model.config.intermediate_size = first_layer.mlp.gate_proj.out_features
        elif hasattr(first_layer.mlp, 'fc1'):
            pruned_model.config.intermediate_size = first_layer.mlp.fc1.out_features
        print(f"  [Width] FFN intermediate_size: {pruned_model.config.intermediate_size}")

    original_params = sum(p.numel() for p in model.parameters())
    pruned_params = sum(p.numel() for p in pruned_model.parameters())
    reduction = (original_params - pruned_params) / original_params

    print(f"\n  [Parameters]")
    print(f"    Original: {original_params:,}")
    print(f"    Pruned:   {pruned_params:,}")
    print(f"    Reduction: {reduction:.2%}")

    save_path = os.path.join(output_dir, "pruned_model")
    os.makedirs(save_path, exist_ok=True)

    print(f"\n  Saving pruned model to: {save_path}")
    pruned_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    layer_sizes = []
    for i, layer in enumerate(pruned_model.model.layers):
        info = {
            "layer_idx": i,
            "num_heads": pruned_model.config.num_attention_heads,
        }
        if hasattr(layer.mlp, 'gate_proj'):
            info["intermediate_size"] = layer.mlp.gate_proj.out_features
        elif hasattr(layer.mlp, 'fc1'):
            info["intermediate_size"] = layer.mlp.fc1.out_features
        layer_sizes.append(info)

    with open(os.path.join(save_path, "layer_sizes.json"), "w") as f:
        json.dump(layer_sizes, f, indent=2)

    print(f"  Model saved!")
    return pruned_model, save_path


def run_evaluation(model_path, batch_size=16, device="cuda:0", label="Pruned"):
    print("\n" + "=" * 70)
    print(f"Evaluation [{label}]: Running lm_eval Benchmarks")
    print("=" * 70)

    from evaluation.lm_eval_utils import evaluate_merged_model

    results = evaluate_merged_model(
        model_path=model_path,
        output_dir=os.path.join(model_path, "eval_results"),
        batch_size=batch_size,
        device=device,
    )

    return results


def run_finetuning(pruned_model_path, args):
    """Stage 2: Fine-tuning using accelerate + stage2_finetune.py"""
    print("\n" + "=" * 70)
    print("Stage 2: Fine-tuning Pruned Model")
    print("=" * 70)

    import subprocess

    finetuned_output = os.path.join(args.output_dir, "finetuned_model")

    s2_args = []
    s2_args += ["--model", args.model]
    s2_args += ["--input_model_path", pruned_model_path]
    s2_args += ["--output_dir", finetuned_output]
    s2_args += ["--num_epochs", str(args.s2_num_epochs)]
    s2_args += ["--batch_size", str(args.s2_batch_size)]
    s2_args += ["--micro_batch_size", str(args.s2_micro_batch_size)]
    s2_args += ["--learning_rate", str(args.s2_learning_rate)]
    s2_args += ["--warmup_steps", str(args.s2_warmup_steps)]
    s2_args += ["--lr_scheduler_type", args.s2_lr_scheduler]
    s2_args += ["--max_seq_length", str(args.s2_max_seq_length)]
    s2_args += ["--lora_r", str(args.s2_lora_r)]
    s2_args += ["--lora_alpha", str(args.s2_lora_alpha)]
    s2_args += ["--lora_dropout", str(args.s2_lora_dropout)]
    s2_args += ["--dataset", args.s2_dataset]
    s2_args += ["--seed", str(args.seed)]

    if args.s2_regularization != "none":
        s2_args += ["--regularization", args.s2_regularization]
        s2_args += ["--reg_lambda", str(args.s2_reg_lambda)]

    if args.model_path:
        s2_args += ["--model_path", args.model_path]

    cmd = ["accelerate", "launch", "--mixed_precision", "fp16",
           "stage2_finetune.py"] + s2_args

    print(f"\n  Config:")
    print(f"    Input:    {pruned_model_path}")
    print(f"    Output:   {finetuned_output}")
    print(f"    Epochs:   {args.s2_num_epochs}")
    print(f"    LR:       {args.s2_learning_rate} ({args.s2_lr_scheduler})")
    print(f"    Batch:    {args.s2_batch_size} (micro={args.s2_micro_batch_size})")
    print(f"    LoRA:     r={args.s2_lora_r}, alpha={args.s2_lora_alpha}")
    print(f"    Dataset:  {args.s2_dataset}")
    print(f"\n  Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("Fine-tuning failed!")
        return None

    print("Fine-tuning completed!")
    return finetuned_output


def extract_eval_metrics(eval_results):
    metrics = {
        "tasks": {},
        "avg_accuracy": 0.0,
        "avg_best_accuracy": 0.0,
    }
    if not eval_results:
        return metrics

    results = eval_results.get("results", {})
    accuracy_tasks = ["winogrande", "hellaswag", "arc_easy", "arc_challenge", "piqa"]

    accuracies = []
    best_accuracies = []

    for task in accuracy_tasks:
        if task in results:
            acc = results[task].get("acc,none", results[task].get("acc", None))
            acc_norm = results[task].get("acc_norm,none", results[task].get("acc_norm", None))
            metrics["tasks"][task] = {"acc": acc, "acc_norm": acc_norm}

            if acc is not None:
                accuracies.append(acc)
            if acc_norm is not None:
                best_accuracies.append(max(acc, acc_norm) if acc else acc_norm)
            elif acc is not None:
                best_accuracies.append(acc)

    if accuracies:
        metrics["avg_accuracy"] = sum(accuracies) / len(accuracies)
    if best_accuracies:
        metrics["avg_best_accuracy"] = sum(best_accuracies) / len(best_accuracies)

    return metrics


def save_depth_pruning_result(result_dir, args, pruning_info, eval_metrics_pruned=None,
                               eval_metrics_finetuned=None):
    depth_dir = os.path.join(result_dir, "depth_pruning")

    model_ratio_dir = os.path.join(depth_dir, args.model, f"pr_{args.pruning_ratio}")
    os.makedirs(model_ratio_dir, exist_ok=True)

    parts = [args.model]
    parts.append(f"seed_{args.seed}")
    parts.append(f"pr{args.pruning_ratio}")
    parts.append(f"ptm_{args.pruning_target_mode}")
    if args.block_search_ratio is not None:
        parts.append(f"bsr{args.block_search_ratio}")
    parts.append(f"thr{args.block_remove_threshold}")
    parts.append(f"pf{args.protect_first}_pl{args.protect_last}")
    parts.append(f"bim_{args.block_importance_mode}")
    parts.append(f"{args.importance_method}_{args.ffn_mode}")
    parts.append(f"cal{args.calibration_samples}")
    ds_short = args.calibration_dataset[:3] if len(args.calibration_dataset) > 3 else args.calibration_dataset
    parts.append(f"ds_{ds_short}")
    parts.append(f"blk{len(pruning_info['removed_layers'])}")
    if args.use_ntk_layer:
        parts.append(f"ntk_{args.ntk_method}_{args.ntk_norm_method}_a{args.ntk_alpha}")
        ns_val = getattr(args, 'ntk_samples', None)
        if ns_val is not None:
            parts.append(f"ns{ns_val}")
        aL = getattr(args, 'ntk_alpha_layer', None)
        aN = getattr(args, 'ntk_alpha_neuron', None)
        if aL is not None:
            parts.append(f"aL{aL}")
        if aN is not None:
            parts.append(f"aN{aN}")
    if args.unified_pruning:
        parts.append("unified")
    if hasattr(args, 's2_lora_alpha') and args.s2_lora_alpha is not None:
        parts.append(f"la{args.s2_lora_alpha}")
    if hasattr(args, 's2_max_seq_length') and args.s2_max_seq_length is not None:
        parts.append(f"seq{args.s2_max_seq_length}")

    # pruned eval
    if eval_metrics_pruned:
        pruned_filename = "_".join(parts) + "_pruned.json"
        pruned_data = {
            "seed": args.seed,
            "stage": "pruned",
            "timestamp": datetime.now().isoformat(),
            "metrics": eval_metrics_pruned,
            "pruning_config": _build_pruning_config(args, pruning_info),
        }
        pruned_path = os.path.join(model_ratio_dir, pruned_filename)
        with open(pruned_path, "w") as f:
            json.dump(pruned_data, f, indent=2)
        print(f"  [Result] Pruned eval saved: {pruned_path}")

    # finetuned eval
    if eval_metrics_finetuned:
        ft_filename = "_".join(parts) + "_finetuned.json"
        ft_data = {
            "seed": args.seed,
            "stage": "finetuned",
            "timestamp": datetime.now().isoformat(),
            "metrics": eval_metrics_finetuned,
            "pruning_config": _build_pruning_config(args, pruning_info),
            "finetune_config": {
                "num_epochs": args.s2_num_epochs,
                "batch_size": args.s2_batch_size,
                "learning_rate": args.s2_learning_rate,
                "lr_scheduler": args.s2_lr_scheduler,
                "lora_r": args.s2_lora_r,
                "lora_alpha": args.s2_lora_alpha,
                "dataset": args.s2_dataset,
                "max_seq_length": getattr(args, 's2_max_seq_length', None),
            },
        }
        ft_path = os.path.join(model_ratio_dir, ft_filename)
        with open(ft_path, "w") as f:
            json.dump(ft_data, f, indent=2)
        print(f"  [Result] Finetuned eval saved: {ft_path}")

    _update_depth_summary(depth_dir)


def _build_pruning_config(args, pruning_info):
    config = {
        "model": args.model,
        "pruning_ratio": args.pruning_ratio,
        "pruning_target_mode": getattr(args, 'pruning_target_mode', 'total'),
        "block_search_ratio": getattr(args, 'block_search_ratio', None),
        "block_remove_threshold": getattr(args, 'block_remove_threshold', 0),
        "block_importance_mode": getattr(args, 'block_importance_mode', 'neuron'),
        "protect_first": args.protect_first,
        "protect_last": args.protect_last,
        "importance_method": args.importance_method,
        "ffn_mode": args.ffn_mode,
        "calibration_samples": args.calibration_samples,
        "calibration_dataset": args.calibration_dataset,
        "dimension_multiple": args.dimension_multiple,
        "removed_layers": pruning_info["removed_layers"],
        "kept_layers": pruning_info["kept_layers"],
        "num_removed": len(pruning_info["removed_layers"]),
        "num_kept": len(pruning_info["kept_layers"]),
        "num_total_layers": pruning_info.get("num_total_layers", 0),
        "original_params": pruning_info.get("original_params", 0),
        "pruned_params": pruning_info.get("pruned_params", 0),
        "param_reduction": pruning_info.get("param_reduction", 0),
        "width_pruning_ratio": pruning_info.get("width_pruning_ratio", 0),
        "intermediate_size_original": pruning_info.get("intermediate_size_original", 0),
        "intermediate_size_pruned": pruning_info.get("intermediate_size_pruned", 0),
        "unified_pruning": getattr(args, 'unified_pruning', False),
        "unified_fill_width": getattr(args, 'unified_fill_width', False),
        "unified_stop_mode": getattr(args, 'unified_stop_mode', 'before'),
        "unified_iterative": getattr(args, 'unified_iterative', False),
        "use_ntk": args.use_ntk_layer,
        "ntk_method": args.ntk_method if args.use_ntk_layer else "none",
        "ntk_norm": getattr(args, 'ntk_norm_method', 'none') if args.use_ntk_layer else "none",
        "ntk_alpha": args.ntk_alpha if args.use_ntk_layer else 0,
        "ntk_alpha_layer": getattr(args, 'ntk_alpha_layer', None) if args.use_ntk_layer else None,
        "ntk_alpha_neuron": getattr(args, 'ntk_alpha_neuron', None) if args.use_ntk_layer else None,
        "ntk_samples": getattr(args, 'ntk_samples', None) if args.use_ntk_layer else None,
    }
    return config


def _update_depth_summary(depth_dir):
    all_results = []
    for root, dirs, files in os.walk(depth_dir):
        for f in files:
            if f.endswith(".json") and f not in ("summary.json",):
                filepath = os.path.join(root, f)
                with open(filepath, "r") as fh:
                    data = json.load(fh)
                    data["_filename"] = f
                    all_results.append(data)

    if not all_results:
        return

    all_results.sort(key=lambda x: x["metrics"].get("avg_best_accuracy", 0), reverse=True)

    best = all_results[0]
    best_acc = best["metrics"].get("avg_best_accuracy", 0)

    lines = []
    lines.append("=" * 120)
    lines.append("Depth Pruning Results Summary")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 120)
    lines.append("")
    lines.append("*" * 120)
    lines.append(f"  OVERALL BEST: {best.get('pruning_config', {}).get('model', '?')}, "
                 f"Stage {best.get('stage', '?')}, Seed {best.get('seed', '?')}  "
                 f"(Avg Best Acc: {best_acc:.4f})")
    lines.append("*" * 120)
    lines.append("")

    accuracy_tasks = ["winogrande", "hellaswag", "arc_easy", "arc_challenge", "piqa"]

    header = (f"{'Model':<12s} {'Seed':>4s} {'Stage':<10s} "
              f"{'Avg Acc':>8s} {'AvgBest':>8s} "
              f"{'PR':>6s} {'ACT%':>6s} "
              f"{'STOP':>4s} {'FILL':>4s} {'ITER':>4s} "
              f"{'NTK_M':>8s} {'NTK_N':>6s} {'NS':>4s} {'α':>4s} {'αL':>4s} {'αN':>4s} "
              f"{'Lr':>4s} {'Lα':>4s} {'LR':>7s} {'SEQ':>4s} "
              f"{'PF':>3s} {'PL':>3s} "
              f"{'BLK_RM':>6s} {'KEPT':>4s} {'LYRS':>4s} "
              f"{'FFN_O':>6s} {'FFN_P':>6s} "
              f"{'W_R':>6s} "
              f"{'IMP':>6s} {'FM':>4s} {'CAL':>4s} {'DS':>5s} ")
    for t in accuracy_tasks:
        short = t[:5] if len(t) > 5 else t
        header += f" {short:>7s}"
    header += f"  {'Rank':>4s}"

    lines.append("=" * len(header))
    lines.append("All Results Comparison")
    lines.append("=" * len(header))
    lines.append(header)
    lines.append("-" * len(header))

    for rank, result in enumerate(all_results, 1):
        pc = result.get("pruning_config", {})
        m = result.get("metrics", {})

        model = pc.get("model", "-")
        if len(model) > 12:
            model = model[:10] + ".."
        seed = result.get("seed", 0)
        stage = result.get("stage", "-")

        avg_acc = m.get("avg_accuracy", 0)
        avg_best = m.get("avg_best_accuracy", 0)
        ffn_r = pc.get("pruning_ratio", 0)
        par_r = pc.get("param_reduction", 0)
        ntk_m = pc.get("ntk_method", "none")
        ntk_n = pc.get("ntk_norm", "none")
        ntk_a = pc.get("ntk_alpha", 0)
        ntk_aL = pc.get("ntk_alpha_layer", None)
        ntk_aN = pc.get("ntk_alpha_neuron", None)
        ntk_aL_str = f"{ntk_aL:.1f}" if ntk_aL is not None else "-"
        ntk_aN_str = f"{ntk_aN:.1f}" if ntk_aN is not None else "-"
        ntk_ns = pc.get("ntk_samples", None)
        ntk_ns_str = str(ntk_ns) if ntk_ns is not None else "-"
        pf = pc.get("protect_first", 0)
        pl = pc.get("protect_last", 0)
        blk_rm = pc.get("num_removed", 0)
        kept = pc.get("num_kept", 0)
        lyrs = pc.get("num_total_layers", 0)
        ffn_o = pc.get("intermediate_size_original", 0)
        ffn_p = pc.get("intermediate_size_pruned", 0)
        w_r = pc.get("width_pruning_ratio", 0)
        imp = pc.get("importance_method", "-")
        fm = pc.get("ffn_mode", "-")
        cal = pc.get("calibration_samples", 0)
        ds = pc.get("calibration_dataset", "-")
        ds_short = ds[:5] if len(ds) > 5 else ds
        ntk_m_short = ntk_m[:4] if ntk_m != "none" else "-"
        ntk_n_short = ntk_n[:3] if ntk_n != "none" else "-"
        stop_mode = pc.get("unified_stop_mode", "before")
        stop_short = "bef" if stop_mode == "before" else "aft"
        fill_width = pc.get("unified_fill_width", False)
        fill_short = "on" if fill_width else "off"
        iterative = pc.get("unified_iterative", False)
        iter_short = "on" if iterative else "off"
        ft_cfg = result.get("finetune_config", {})
        lora_r = ft_cfg.get("lora_r", "-")
        lora_a = ft_cfg.get("lora_alpha", "-")
        lr_val = ft_cfg.get("learning_rate", None)
        if isinstance(lr_val, (int, float)):
            lr_str = f"{lr_val:.0e}"
        elif lr_val is not None:
            lr_str = str(lr_val)
        else:
            lr_str = "-"
        seq_len = ft_cfg.get("max_seq_length", None)
        seq_str = str(seq_len) if seq_len is not None else "-"

        row = (f"{model:<12s} {seed:>4d} {stage:<10s} "
               f"{avg_acc:>8.4f} {avg_best:>8.4f} "
               f"{ffn_r:>6.2f} {par_r:>5.1%} "
               f"{stop_short:>4s} {fill_short:>4s} {iter_short:>4s} "
               f"{ntk_m_short:>8s} {ntk_n_short:>6s} {ntk_ns_str:>4s} {ntk_a:>4.1f} {ntk_aL_str:>4s} {ntk_aN_str:>4s} "
               f"{str(lora_r):>4s} {str(lora_a):>4s} {lr_str:>7s} {seq_str:>4s} "
               f"{pf:>3d} {pl:>3d} "
               f"{blk_rm:>6d} {kept:>4d} {lyrs:>4d} "
               f"{ffn_o:>6d} {ffn_p:>6d} "
               f"{w_r:>5.1%} "
               f"{imp:>6s} {fm:>4s} {cal:>4d} {ds_short:>5s} ")

        tasks_data = m.get("tasks", {})
        for t in accuracy_tasks:
            if t in tasks_data:
                acc = tasks_data[t].get("acc")
                acc_n = tasks_data[t].get("acc_norm")
                val = max(acc or 0, acc_n or 0)
                row += f" {val:>7.4f}"
            else:
                row += f" {'N/A':>7s}"

        marker = " <-- BEST" if rank == 1 else ""
        row += f"  {rank:>4d}{marker}"
        lines.append(row)

    lines.append("=" * len(header))
    lines.append("")

    lines.append("=" * 80)
    lines.append("Detailed Results")
    lines.append("=" * 80)

    for result in all_results:
        pc = result.get("pruning_config", {})
        m = result.get("metrics", {})
        lines.append("")
        lines.append(f"--- {pc.get('model', '?')}, Stage: {result.get('stage', '?')}, "
                     f"Seed: {result.get('seed', '?')} ---")
        orig_p = pc.get('original_params', 0)
        prun_p = pc.get('pruned_params', 0)
        lines.append(f"  Pruning Ratio (target): {pc.get('pruning_ratio', 0):.2%}")
        lines.append(f"  Pruning Target Mode:    {pc.get('pruning_target_mode', 'total')}")
        bsr_val = pc.get('block_search_ratio')
        lines.append(f"  Block Search Ratio:     {f'{bsr_val:.2%}' if bsr_val is not None else 'None (same as pruning_ratio)'}")
        lines.append(f"  Actual Pruning:         {pc.get('param_reduction', 0):.2%} ({orig_p:,} → {prun_p:,})")
        lines.append(f"  Block Remove Threshold: {pc.get('block_remove_threshold', 0):.2%}")
        lines.append(f"  Block Importance Mode:  {pc.get('block_importance_mode', 'neuron')}")
        lines.append(f"  Protected First/Last:   {pc.get('protect_first', 0)} / {pc.get('protect_last', 0)}")
        lines.append(f"  Total Layers (orig):    {pc.get('num_total_layers', 0)}")
        lines.append(f"  Removed Blocks:         {pc.get('num_removed', 0)} {pc.get('removed_layers', [])}")
        lines.append(f"  Kept Blocks:            {pc.get('num_kept', 0)}")
        lines.append(f"  FFN Dimension:          {pc.get('intermediate_size_original', 0)} -> {pc.get('intermediate_size_pruned', 0)}")
        lines.append(f"  Width Pruning Ratio:    {pc.get('width_pruning_ratio', 0):.2%}")
        lines.append(f"  Stop Mode / Fill Width: {pc.get('unified_stop_mode', 'before')} / "
                     f"{pc.get('unified_fill_width', False)}")
        lines.append(f"  Iterative:              {pc.get('unified_iterative', False)}")
        lines.append(f"  Importance:             {pc.get('importance_method', '-')} ({pc.get('ffn_mode', '-')})")
        lines.append(f"  Calibration:            {pc.get('calibration_samples', 0)} samples, {pc.get('calibration_dataset', '-')}")

        ft_cfg = result.get("finetune_config", {})
        if ft_cfg:
            seq_len_detail = ft_cfg.get('max_seq_length', '-')
            lines.append(f"  Fine-tune:              epochs={ft_cfg.get('num_epochs')}, "
                         f"lr={ft_cfg.get('learning_rate')}, "
                         f"LoRA r={ft_cfg.get('lora_r')}/alpha={ft_cfg.get('lora_alpha')}, "
                         f"seq_len={seq_len_detail}, "
                         f"dataset={ft_cfg.get('dataset')}")

        lines.append(f"\n  {'Task':<20s} {'acc':>10s} {'acc_norm':>10s}")
        lines.append(f"  {'-' * 42}")
        for t in accuracy_tasks:
            if t in m.get("tasks", {}):
                acc = m["tasks"][t].get("acc")
                acc_n = m["tasks"][t].get("acc_norm")
                acc_s = f"{acc:.4f}" if acc is not None else "N/A"
                acc_n_s = f"{acc_n:.4f}" if acc_n is not None else "N/A"
                lines.append(f"  {t:<20s} {acc_s:>10s} {acc_n_s:>10s}")
        lines.append(f"  {'-' * 42}")
        lines.append(f"  {'Avg Accuracy:':<20s} {m.get('avg_accuracy', 0):>10.4f}")
        lines.append(f"  {'Avg Best Acc:':<20s} {m.get('avg_best_accuracy', 0):>10.4f}")

    summary_path = os.path.join(depth_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))

    summary_json_path = os.path.join(depth_dir, "summary.json")
    summary_json = {
        "best_model": best.get("pruning_config", {}).get("model", "?"),
        "best_stage": best.get("stage", "?"),
        "best_seed": best.get("seed", 0),
        "best_accuracy": best_acc,
        "num_results": len(all_results),
        "all_results": all_results,
    }
    with open(summary_json_path, "w") as f:
        json.dump(summary_json, f, indent=2, default=str)

    print(f"  [Summary] Updated: {summary_path}")
    print(f"  [Summary] Best: {best.get('pruning_config', {}).get('model', '?')}, "
          f"Stage {best.get('stage', '?')}, Acc {best_acc:.4f}")


def main():
    args = parse_args()

    # ==================== Skip Pruning Mode ====================
    if args.pruned_model_path is not None:
        print("=" * 70)
        print("SKIP PRUNING MODE: Using existing pruned model")
        print("=" * 70)
        print(f"  Pruned model: {args.pruned_model_path}")
        print(f"  LoRA alpha:   {args.s2_lora_alpha}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pruned_model_path = args.pruned_model_path

        summary_dir = os.path.dirname(pruned_model_path)
        masks_path = os.path.join(summary_dir, "depth_pruning_masks.pt")
        pruning_info = {"removed_layers": [], "kept_layers": [], "num_total_layers": 0,
                        "original_params": 0, "pruned_params": 0, "param_reduction": 0,
                        "width_pruning_ratio": 0, "intermediate_size_original": 0,
                        "intermediate_size_pruned": 0}
        if os.path.exists(masks_path):
            mask_data = torch.load(masks_path, map_location="cpu")
            pruning_info["removed_layers"] = mask_data.get("removed_layers", [])
            pruning_info["kept_layers"] = mask_data.get("kept_layers", [])
            pruning_info["num_total_layers"] = mask_data.get("num_total_layers", 0)
            pruning_info["original_params"] = mask_data.get("original_params", 0)
            pruning_info["intermediate_size_original"] = mask_data.get("intermediate_size_original", 0)

        _is_vlm = is_vlm_key(args.model)
        if _is_vlm:
            from transformers import AutoModelForImageTextToText
            _tmp = AutoModelForImageTextToText.from_pretrained(
                pruned_model_path, torch_dtype=torch.float16, trust_remote_code=True)
            pruning_info["pruned_params"] = sum(
                p.numel() for p in _tmp.model.language_model.parameters()
            ) + sum(p.numel() for p in _tmp.lm_head.parameters())
            pruning_info["intermediate_size_pruned"] = getattr(
                _tmp.config.text_config, "intermediate_size", 0)
        else:
            _tmp = AutoModelForCausalLM.from_pretrained(pruned_model_path, dtype=torch.float16)
            pruning_info["pruned_params"] = sum(p.numel() for p in _tmp.parameters())
            pruning_info["intermediate_size_pruned"] = getattr(_tmp.config, 'intermediate_size', 0)
        del _tmp
        gc.collect()

        if pruning_info["original_params"] == 0:
            pruning_info["original_params"] = pruning_info["pruned_params"]
        if pruning_info["num_total_layers"] == 0:
            n_removed = len(pruning_info["removed_layers"])
            n_kept = len(pruning_info["kept_layers"])
            if n_removed + n_kept > 0:
                pruning_info["num_total_layers"] = n_removed + n_kept
            else:
                if _is_vlm:
                    from transformers import AutoModelForImageTextToText
                    _cfg = AutoModelForImageTextToText.from_pretrained(
                        pruned_model_path, torch_dtype=torch.float16,
                        trust_remote_code=True).config.text_config
                else:
                    _cfg = AutoModelForCausalLM.from_pretrained(
                        pruned_model_path, dtype=torch.float16).config
                pruning_info["num_total_layers"] = getattr(_cfg, 'num_hidden_layers', 0)

        orig_p = pruning_info["original_params"]
        prun_p = pruning_info["pruned_params"]
        pruning_info["param_reduction"] = (orig_p - prun_p) / orig_p if orig_p > 0 else 0
        orig_ffn = pruning_info["intermediate_size_original"]
        prun_ffn = pruning_info["intermediate_size_pruned"]
        pruning_info["width_pruning_ratio"] = 1.0 - prun_ffn / orig_ffn if orig_ffn > 0 else 0

        # Fine-tuning
        finetuned_path = None
        if args.finetune:
            finetuned_path = run_finetuning(pruned_model_path, args)

        # Evaluation
        eval_metrics_finetuned = None
        if args.eval_after_finetune and finetuned_path:
            gc.collect()
            torch.cuda.empty_cache()
            eval_results = run_evaluation(
                model_path=finetuned_path,
                batch_size=args.eval_batch_size,
                device=str(device),
                label="Fine-tuned",
            )
            eval_metrics_finetuned = extract_eval_metrics(eval_results)

        # Save results
        if eval_metrics_finetuned:
            save_depth_pruning_result(
                result_dir=args.result_dir,
                args=args,
                pruning_info=pruning_info,
                eval_metrics_pruned=None,
                eval_metrics_finetuned=eval_metrics_finetuned,
            )

        print("\nDone!")
        return

    print("=" * 70)
    print("Option C: GandA-based Automatic Block Pruning")
    print("=" * 70)
    print(f"  Model:                  {args.model}")
    ptm_label = "total (model-wide)" if args.pruning_target_mode == "total" else "ffn (FFN neuron)"
    print(f"  Pruning Ratio:          {args.pruning_ratio:.2%}")
    print(f"  Pruning Target Mode:    {args.pruning_target_mode} ({ptm_label})")
    if args.block_search_ratio is not None:
        print(f"  Block Search Ratio:     {args.block_search_ratio:.2%} (separate ratio for block identification)")
    else:
        print(f"  Block Search Ratio:     None (same as pruning_ratio)")
    print(f"  Block Remove Threshold: {args.block_remove_threshold:.2%}")
    print(f"  Protected First/Last:   {args.protect_first} / {args.protect_last}")
    print(f"  Calibration Samples:    {args.calibration_samples}")
    print(f"  Dimension Multiple:     {args.dimension_multiple}")
    print(f"  Importance Method:      {args.importance_method}")
    print(f"  FFN Mode:               {args.ffn_mode}")
    print(f"  Block Importance Mode:  {args.block_importance_mode}")

    # ==================== Model Loading ====================
    print("\n" + "=" * 70)
    print("Loading Model and Tokenizer")
    print("=" * 70)

    model_name = args.model_path if args.model_path else get_model_name(args.model)
    print(f"  Model path: {model_name}")

    _offline = os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=_offline,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # VLM (Qwen2.5-VL / Qwen3-VL) branch: load full multimodal model, then
    # expose its language_model + lm_head as a CausalLM facade for the
    # existing pruning code. The facade shares tensors with the underlying
    # VLM so all pruning mutations propagate to it.
    _is_vlm = is_vlm_key(args.model)
    if _is_vlm:
        from transformers import AutoModelForImageTextToText
        print(f"  [VLM] Loading via AutoModelForImageTextToText")
        _vlm = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=_offline,
        )
        _vlm.eval()
        for param in _vlm.parameters():
            param.requires_grad = False
        model = VLMCausalLMFacade(_vlm)
        print(f"  [VLM] Facade exposes language_model "
              f"({len(model.model.layers)} decoder layers)")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=_offline,
        )
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    device = next(model.parameters()).device
    num_layers = model.config.num_hidden_layers
    intermediate_size = getattr(model.config, 'intermediate_size', model.config.hidden_size * 4)

    print(f"  Loaded: {num_layers} layers, intermediate_size={intermediate_size}")
    print(f"  Device: {device}")

    # ==================== Model Parameter Analysis ====================
    model_info = compute_model_param_info(model)

    print(f"\n  [Model Parameter Breakdown]")
    print(f"    Total params:    {model_info['total_params']:,}")
    print(f"    Per block:       {model_info['block_params']:,}")
    print(f"      - Attention:   {model_info['attn_params_per_layer']:,}")
    print(f"      - FFN:         {model_info['ffn_params_per_layer']:,}")
    print(f"    Total FFN:       {model_info['total_ffn_params']:,} ({model_info['total_ffn_params'] / model_info['total_params']:.1%})")
    print(f"    Total Attention: {model_info['total_attn_params']:,} ({model_info['total_attn_params'] / model_info['total_params']:.1%})")
    print(f"    Embedding+Other: {model_info['embed_params']:,} ({model_info['embed_params'] / model_info['total_params']:.1%})")

    # ==================== Iterative Unified Pruning (early branch) ====================
    if args.unified_pruning and args.unified_iterative:
        print("\n" + "=" * 70)
        print("MODE: Iterative Unified Depth+Width Pruning (per-step recompute)")
        print("=" * 70)

        removed_layers, kept_layers, ffn_masks, params_removed, iter_picks = \
            iterative_unified_pruning(
                model, tokenizer, args, device,
                num_layers, intermediate_size, model_info,
            )

        # Final Summary
        print_final_summary(
            num_layers, removed_layers, kept_layers, ffn_masks,
            intermediate_size, args.pruning_ratio, model_info
        )
        ffn_scores = None  # not used in iterative path

    elif args.unified_pruning:
        # ==================== Step 1: Compute GandA Scores ====================
        ffn_scores = compute_ganda_scores(model, tokenizer, args, device)

        # ==================== Unified Pruning Mode (non-iterative) ====================
        print("\n" + "=" * 70)
        print("MODE: Unified Depth+Width Pruning (Auto Ratio)")
        print("=" * 70)

        # Optional NTK
        ntk_sensitivities = None
        neuron_sensitivities = None
        if args.use_ntk_layer:
            gc.collect()
            torch.cuda.empty_cache()
            ntk_sensitivities = compute_layer_ntk_sensitivity(
                model, tokenizer, args, device, ffn_scores
            )
            gc.collect()
            torch.cuda.empty_cache()
            neuron_sensitivities = compute_neuron_ntk_sensitivity(
                model, tokenizer, args, device
            )

        depth_scores_pre, group_scores_pre, _, _ = compute_unified_scores(
            ffn_scores, model_info, num_layers, args.dimension_multiple,
            args.protect_first, args.protect_last, intermediate_size,
            ntk_sensitivities=None,
            ntk_alpha=0.0,
            neuron_sensitivities=None,
        )

        depth_scores, group_scores, protected, layer_rank_indices = compute_unified_scores(
            ffn_scores, model_info, num_layers, args.dimension_multiple,
            args.protect_first, args.protect_last, intermediate_size,
            ntk_sensitivities=ntk_sensitivities,
            ntk_alpha=args.ntk_alpha if args.use_ntk_layer else 0.0,
            neuron_sensitivities=neuron_sensitivities,
            depth_score_mode=getattr(args, 'depth_score_mode', 'legacy'),
            ntk_alpha_layer=getattr(args, 'ntk_alpha_layer', None) if args.use_ntk_layer else None,
            ntk_alpha_neuron=getattr(args, 'ntk_alpha_neuron', None) if args.use_ntk_layer else None,
        )

        # ==================== Force depth removal override ====================
        if args.force_removed_layers:
            forced = [int(x.strip()) for x in args.force_removed_layers.split(',') if x.strip()]
            print("\n" + "=" * 70)
            print(f"[FORCE] Override depth_scores=0 for layers: {forced}")
            print(f"[FORCE] Greedy will pick these first (lowest score = first removed)")
            print("=" * 70)
            for L in forced:
                if L in depth_scores:
                    depth_scores[L] = 0.0
                else:
                    print(f"[FORCE] WARNING: layer {L} is protected/missing -> ignored")

        if args.use_ntk_layer and args.ntk_alpha != 0:
            print("\n" + "=" * 70)
            print("Depth Score: Pre-NTK vs Post-NTK Comparison")
            print("=" * 70)
            print(f"  {'Layer':>5} | {'NTK_sens':>10} | {'Pre-NTK':>14} | {'Post-NTK':>14} | {'ΔScore%':>8} | {'Prot':>5}")
            print("  " + "-" * 74)
            for layer_idx in sorted(depth_scores.keys()):
                pre = depth_scores_pre[layer_idx]
                post = depth_scores[layer_idx]
                ntk_s = ntk_sensitivities.get(layer_idx, 1.0) if ntk_sensitivities else 1.0
                delta_pct = ((post - pre) / pre * 100) if pre > 0 else 0.0
                prot = "Yes" if layer_idx in protected else ""
                print(f"  {layer_idx:>5} | {ntk_s:>10.4f} | {pre:>14.6e} | {post:>14.6e} | {delta_pct:>+7.2f}% | {prot:>5}")

            print("\n  Width Group Importance (sum over all layers)")
            print(f"  {'Group':>5} | {'Pre-NTK':>14} | {'Post-NTK':>14} | {'ΔScore%':>8}")
            print("  " + "-" * 54)
            num_groups = intermediate_size // args.dimension_multiple
            for g in range(num_groups):
                pre_sum = sum(group_scores_pre[g].values())
                post_sum = sum(group_scores[g].values())
                delta_pct = ((post_sum - pre_sum) / pre_sum * 100) if pre_sum > 0 else 0.0
                print(f"  {g:>5} | {pre_sum:>14.6e} | {post_sum:>14.6e} | {delta_pct:>+7.2f}%")

        scores_save = {
            "depth_scores_pre_ntk": {int(k): float(v) for k, v in depth_scores_pre.items()},
            "depth_scores_post_ntk": {int(k): float(v) for k, v in depth_scores.items()},
            "ntk_sensitivities": ({int(k): float(v) for k, v in ntk_sensitivities.items()}
                                  if ntk_sensitivities else None),
            "ntk_alpha": args.ntk_alpha if args.use_ntk_layer else 0.0,
            "ntk_method": args.ntk_method if args.use_ntk_layer else "none",
            "ntk_norm_method": getattr(args, 'ntk_norm_method', 'none') if args.use_ntk_layer else "none",
            "protected_layers": sorted(protected),
            "group_scores_pre_ntk_sum": {int(g): float(sum(group_scores_pre[g].values()))
                                         for g in group_scores_pre},
            "group_scores_post_ntk_sum": {int(g): float(sum(group_scores[g].values()))
                                          for g in group_scores},
        }
        os.makedirs(args.output_dir, exist_ok=True)
        scores_save_path = os.path.join(args.output_dir, "pruning_scores.pt")
        torch.save(scores_save, scores_save_path)
        print(f"\n  [Scores] Saved to: {scores_save_path}")

        removed_layers, kept_layers, removed_groups, params_removed = greedy_unified_pruning(
            depth_scores, group_scores, model_info,
            args.pruning_ratio, num_layers, args.dimension_multiple,
            protected, intermediate_size,
            fill_width=args.unified_fill_width,
            stop_mode=args.unified_stop_mode,
            disable_depth=getattr(args, 'unified_disable_depth', False),
            disable_width=getattr(args, 'unified_disable_width', False),
        )

        ffn_masks = create_unified_width_masks(
            kept_layers, removed_groups, args.dimension_multiple, intermediate_size,
            layer_rank_indices=layer_rank_indices,
        )

        # Final Summary
        print_final_summary(
            num_layers, removed_layers, kept_layers, ffn_masks,
            intermediate_size, args.pruning_ratio, model_info
        )

    else:
        # ==================== Original Mode (Threshold-based) ====================
        ffn_scores = compute_ganda_scores(model, tokenizer, args, device)

        # Step 2: Global Threshold
        global_threshold, all_scores, ffn_equiv_ratio = compute_global_threshold(
            ffn_scores, args.pruning_ratio, model_info,
            pruning_target_mode=args.pruning_target_mode,
        )

        # Step 2.5: Block Search Threshold (optional)
        if args.block_search_ratio is not None:
            print("\n" + "=" * 70)
            print("Step 2.5: Computing Block Search Threshold (separate ratio)")
            print("=" * 70)
            print(f"  Block search ratio: {args.block_search_ratio:.2%} (vs pruning_ratio: {args.pruning_ratio:.2%})")
            block_search_threshold, _, block_search_ffn_equiv = compute_global_threshold(
                ffn_scores, args.block_search_ratio, model_info,
                pruning_target_mode=args.pruning_target_mode,
            )
        else:
            block_search_threshold = global_threshold

        # Step 3: Per-Block Simulated Ratios
        simulated_ratios = simulate_per_block_ratios(ffn_scores, block_search_threshold, num_layers)

        # Step 3.5: Block Output Importance (optional)
        block_output_importance = None
        if args.block_importance_mode == "output":
            gc.collect()
            torch.cuda.empty_cache()
            block_output_importance = compute_block_output_importance(
                model, tokenizer, args, device
            )

        # Step 4: Block Removal Decision
        removed_layers, kept_layers = decide_block_removal(
            simulated_ratios, num_layers,
            args.block_remove_threshold,
            args.protect_first, args.protect_last,
            args.pruning_ratio, model_info,
            block_importance_mode=args.block_importance_mode,
            block_output_importance=block_output_importance,
            pruning_target_mode=args.pruning_target_mode,
        )

        # Step 4.5: NTK Layer Sensitivity Comparison (optional)
        ntk_comparison = None
        if args.use_ntk_layer:
            gc.collect()
            torch.cuda.empty_cache()
            ntk_sensitivities = compute_layer_ntk_sensitivity(
                model, tokenizer, args, device, ffn_scores
            )
            ntk_comparison = compare_block_removal_with_ntk(
                simulated_ratios, ntk_sensitivities,
                num_layers, args.block_remove_threshold,
                args.protect_first, args.protect_last,
                args.pruning_ratio, model_info,
                ntk_alpha=args.ntk_alpha,
                pruning_target_mode=args.pruning_target_mode,
            )

            ntk_result_path = os.path.join(args.output_dir, "ntk_layer_comparison.json")
            os.makedirs(args.output_dir, exist_ok=True)
            with open(ntk_result_path, "w") as f:
                json.dump(ntk_comparison, f, indent=2)
            print(f"  NTK comparison saved to: {ntk_result_path}")

        # Step 5: Recalculate Width Ratio
        remaining_width_ratio = recalculate_width_ratio(
            args.pruning_ratio, removed_layers, num_layers, model_info,
            pruning_target_mode=args.pruning_target_mode,
        )

        # Step 6: Create Width Masks
        ffn_masks = create_width_masks(
            ffn_scores, kept_layers, remaining_width_ratio,
            args.dimension_multiple, intermediate_size
        )

        # Final Summary
        print_final_summary(
            num_layers, removed_layers, kept_layers, ffn_masks,
            intermediate_size, args.pruning_ratio, model_info
        )

    # ==================== Common Post-Processing (all modes) ====================

    # Save Results
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Save masks
    mask_data = {
        "removed_layers": removed_layers,
        "kept_layers": kept_layers,
        "ffn_masks": {str(k): v for k, v in ffn_masks.items()},
    }
    save_path = os.path.join(output_dir, "depth_pruning_masks.pt")
    torch.save(mask_data, save_path)
    print(f"\n  Masks saved to: {save_path}")

    # ==================== Step 7: Physical Pruning ====================
    original_params = sum(p.numel() for p in model.parameters())

    pruned_model, pruned_model_path = physically_prune_model(
        model, tokenizer, removed_layers, kept_layers, ffn_masks, output_dir
    )

    pruned_params = sum(p.numel() for p in pruned_model.parameters())
    param_reduction = (original_params - pruned_params) / original_params

    if kept_layers and kept_layers[0] in ffn_masks:
        ffn_pruned_size = int(ffn_masks[kept_layers[0]].sum().item())
    else:
        ffn_pruned_size = intermediate_size

    pruning_info = {
        "removed_layers": removed_layers,
        "kept_layers": kept_layers,
        "original_params": original_params,
        "pruned_params": pruned_params,
        "param_reduction": param_reduction,
        "width_pruning_ratio": (1.0 - ffn_pruned_size / intermediate_size) if args.unified_pruning else remaining_width_ratio,
        "num_total_layers": num_layers,
        "intermediate_size_original": intermediate_size,
        "intermediate_size_pruned": ffn_pruned_size,
    }

    # Save summary text
    summary_path = os.path.join(output_dir, "depth_pruning_summary.txt")
    with open(summary_path, "w") as f:
        f.write("GandA-based Automatic Block Pruning Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model:                  {args.model}\n")
        f.write(f"Pruning Ratio (target): {args.pruning_ratio:.2%}\n")
        f.write(f"Actual Pruning:         {param_reduction:.2%} ({original_params:,} → {pruned_params:,})\n")
        f.write(f"Protected First/Last:   {args.protect_first} / {args.protect_last}\n\n")
        f.write(f"Layers:          {num_layers} → {len(kept_layers)} ({len(removed_layers)} blocks removed)\n")
        f.write(f"Removed Layers:  {removed_layers}\n")
        f.write(f"Kept Layers:     {kept_layers}\n")
        f.write(f"FFN Dimension:   {intermediate_size} → {ffn_pruned_size}\n\n")

    print(f"  Summary saved to: {summary_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ==================== Step 8: Evaluation (Pruned) ====================
    eval_results_pruned = None
    eval_metrics_pruned = None
    if args.eval:
        del pruned_model
        gc.collect()
        torch.cuda.empty_cache()

        eval_results_pruned = run_evaluation(
            model_path=pruned_model_path,
            batch_size=args.eval_batch_size,
            device=str(device),
            label="Pruned (before fine-tuning)",
        )
        eval_metrics_pruned = extract_eval_metrics(eval_results_pruned)

        gc.collect()
        torch.cuda.empty_cache()
    else:
        del pruned_model
        gc.collect()
        torch.cuda.empty_cache()
        print("\n  (Pruned eval skipped. Use --eval to run lm_eval benchmarks)")

    # ==================== Step 9: Fine-tuning ====================
    finetuned_path = None
    if args.finetune:
        finetuned_path = run_finetuning(pruned_model_path, args)

        if finetuned_path is None:
            print("\n  Fine-tuning failed. Skipping fine-tuned evaluation.")

    # ==================== Step 10: Evaluation (Fine-tuned) ====================
    eval_results_finetuned = None
    eval_metrics_finetuned = None
    if args.eval_after_finetune and finetuned_path:
        gc.collect()
        torch.cuda.empty_cache()

        eval_results_finetuned = run_evaluation(
            model_path=finetuned_path,
            batch_size=args.eval_batch_size,
            device=str(device),
            label="Fine-tuned",
        )
        eval_metrics_finetuned = extract_eval_metrics(eval_results_finetuned)

    # ==================== Step 11: Save Results ====================
    if eval_metrics_pruned or eval_metrics_finetuned:
        print("\n" + "=" * 70)
        print(f"Step 11: Saving Results to {args.result_dir}/depth_pruning/")
        print("=" * 70)

        save_depth_pruning_result(
            result_dir=args.result_dir,
            args=args,
            pruning_info=pruning_info,
            eval_metrics_pruned=eval_metrics_pruned,
            eval_metrics_finetuned=eval_metrics_finetuned,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
        