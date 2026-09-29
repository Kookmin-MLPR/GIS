
import os
import gc
import random
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from pruning.model_descriptor import (
    get_down_proj_module, get_up_proj_modules, get_attn_output_proj,
    is_swiglu_ffn
)


class CETTImportanceCalculator:

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        verbose: bool = True
    ):
        self.model = model
        self.device = device
        self.verbose = verbose

        self.config = model.config
        self.num_layers = self.config.num_hidden_layers
        self.hidden_size = self.config.hidden_size
        self.intermediate_size = getattr(self.config, 'intermediate_size', self.hidden_size * 4)

        self.activations = {}  # {layer_idx: [batch_activations]}
        self.hooks = []

    def _get_mlp_module(self, layer_idx: int):
        try:
            if hasattr(self.model, 'model'):
                base = self.model.model
            else:
                base = self.model

            if hasattr(base, 'model'):
                base = base.model

            if hasattr(base, 'layers'):
                return base.layers[layer_idx].mlp
            elif hasattr(base, 'decoder'):
                return base.decoder.layers[layer_idx].mlp
        except (AttributeError, IndexError):
            pass

        try:
            return self.model.base_model.model.model.layers[layer_idx].mlp
        except (AttributeError, IndexError):
            pass

        return None

    def _get_down_proj_weight(self, layer_idx: int) -> Optional[torch.Tensor]:
        mlp = self._get_mlp_module(layer_idx)
        if mlp is None:
            return None

        down_proj = get_down_proj_module(mlp)
        if down_proj is None:
            return None

        if hasattr(down_proj, 'base_layer'):
            base_weight = down_proj.base_layer.weight.data

            active_adapter = getattr(down_proj, 'active_adapter', None)
            if active_adapter and hasattr(down_proj, 'lora_A') and hasattr(down_proj, 'lora_B'):
                if isinstance(active_adapter, (list, tuple, set)):
                    active_adapter = list(active_adapter)[0]
                try:
                    lora_A = down_proj.lora_A[active_adapter].weight.data
                    lora_B = down_proj.lora_B[active_adapter].weight.data
                    scaling = down_proj.scaling.get(active_adapter, 1.0)
                    lora_delta = (lora_B @ lora_A) * scaling
                    return base_weight + lora_delta
                except (KeyError, AttributeError):
                    pass
            return base_weight

        elif hasattr(down_proj, 'momentum_lora'):
            mom = down_proj.momentum_lora
            base_weight = mom.W_0.data
            lora_delta = mom.B1.data @ mom.A1.data + mom.B2.data @ mom.A2.data
            return base_weight + lora_delta

        elif hasattr(down_proj, 'weight'):
            return down_proj.weight.data

        return None

    def _register_activation_hooks(self):
        self.activations = {i: [] for i in range(self.num_layers)}

        for layer_idx in range(self.num_layers):
            mlp = self._get_mlp_module(layer_idx)
            if mlp is None:
                continue

            down_proj = get_down_proj_module(mlp)
            if down_proj is None:
                continue

            if hasattr(down_proj, 'base_layer'):
                target_module = down_proj.base_layer
            elif hasattr(down_proj, 'momentum_lora'):
                target_module = down_proj
            else:
                target_module = down_proj

            def make_hook(idx):
                def hook(module, input, output):
                    if len(input) > 0 and input[0] is not None:
                        # input[0] shape: [batch, seq_len, intermediate_size]
                        act = input[0].detach()
                        self.activations[idx].append(act.cpu())
                return hook

            h = target_module.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(h)

    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    @torch.no_grad()
    def collect_activations(
        self,
        dataloader,
        num_samples: int = 128
    ):
        self._register_activation_hooks()

        self.model.eval()
        samples_collected = 0

        if self.verbose:
            print(f"\n[CETT] Collecting activations from {num_samples} samples...")

        pbar = tqdm(dataloader, desc="[CETT] Calibration", disable=not self.verbose)

        for batch in pbar:
            if samples_collected >= num_samples:
                break

            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                input_ids = batch.get('input_ids')
                attention_mask = batch.get('attention_mask')
            else:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device) if len(batch) > 1 else None

            try:
                if attention_mask is not None:
                    self.model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    self.model(input_ids=input_ids)
            except Exception as e:
                if self.verbose:
                    print(f"[CETT] Forward error: {e}")
                continue

            samples_collected += input_ids.size(0)
            pbar.set_postfix({"samples": samples_collected})

        self._remove_hooks()

        if self.verbose:
            print(f"[CETT] Collected activations for {samples_collected} samples, {self.num_layers} layers")

    def compute_cett_scores(self) -> Dict[int, torch.Tensor]:
        if not self.activations or all(len(v) == 0 for v in self.activations.values()):
            raise ValueError("[CETT] No activations collected. Call collect_activations first.")

        cett_scores = {}

        if self.verbose:
            print("\n[CETT] Computing CETT scores...")

        for layer_idx in tqdm(range(self.num_layers), desc="[CETT] Layers", disable=not self.verbose):
            if not self.activations[layer_idx]:
                if self.verbose:
                    print(f"  Layer {layer_idx}: No activations, using uniform scores")
                cett_scores[layer_idx] = torch.ones(self.intermediate_size)
                continue

            W_down = self._get_down_proj_weight(layer_idx)
            if W_down is None:
                if self.verbose:
                    print(f"  Layer {layer_idx}: No down_proj weight found, using uniform scores")
                cett_scores[layer_idx] = torch.ones(self.intermediate_size)
                continue

            W_down = W_down.float().to(self.device)  # [hidden_size, intermediate_size]

            all_activations = torch.cat(self.activations[layer_idx], dim=0)  # [total_tokens, seq_len, intermediate_size]

            # Flatten batch and sequence
            if all_activations.dim() == 3:
                all_activations = all_activations.view(-1, all_activations.size(-1))  # [total_tokens * seq_len, intermediate_size]

            all_activations = all_activations.float().to(self.device)

            # h_j = W_down[:, j] * activation[j] for each token
            # CETT_j = mean(||h_j|| / ||h_total||)

            num_tokens = all_activations.size(0)
            neuron_cett = torch.zeros(self.intermediate_size, device=self.device)

            batch_size = min(512, num_tokens)

            for start_idx in range(0, num_tokens, batch_size):
                end_idx = min(start_idx + batch_size, num_tokens)
                act_batch = all_activations[start_idx:end_idx]  # [batch, intermediate_size]
                current_batch_size = act_batch.size(0)

                # h_total shape: [hidden_size, batch]
                h_total = W_down @ act_batch.T
                h_total_norm = torch.norm(h_total, dim=0)  # [batch]
                h_total_norm = h_total_norm.clamp(min=1e-8)

                # h_j[hidden, batch, neuron] = W_down[hidden, neuron] * act[batch, neuron]
                # W_down: [hidden_size, intermediate_size]
                # act_batch: [batch, intermediate_size]

                # W_down[:, j] * act_batch[:, j] for all j
                # = element-wise: W_down[h, j] * act_batch[b, j] → need [h, b, j]
                # W_down.unsqueeze(1): [hidden, 1, inter]
                # act_batch.T.unsqueeze(0): [1, inter, batch] → permute to [1, batch, inter]

                # h_j_norm = ||W_down[:, j]|| * |act_batch[:, j]| (column-wise)

                W_col_norms = torch.norm(W_down, dim=0)  # [intermediate_size]
                act_abs = torch.abs(act_batch)  # [batch, intermediate_size]

                # h_j_norm for each neuron and each token
                h_j_norms = W_col_norms.unsqueeze(0) * act_abs  # [batch, intermediate_size]

                contributions = h_j_norms / h_total_norm.unsqueeze(1)  # [batch, intermediate_size]

                neuron_cett += contributions.sum(dim=0)  # [intermediate_size]

            neuron_cett /= num_tokens

            cett_scores[layer_idx] = neuron_cett.cpu()

            del all_activations
            if self.device == "cuda":
                torch.cuda.empty_cache()

        self.activations = {}

        if self.verbose:
            print("[CETT] CETT scores computed successfully")
            for layer_idx in range(min(3, self.num_layers)):
                scores = cett_scores[layer_idx]
                print(f"  Layer {layer_idx}: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

        return cett_scores

    def compute_ffn_importance(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self, '_cett_scores'):
            raise ValueError("Call compute_cett_scores first or use compute_all_ffn_importance")

        scores = self._cett_scores.get(layer_idx)
        if scores is None:
            return torch.ones(self.intermediate_size), torch.ones(self.intermediate_size)

        return scores, scores

    def compute_all_ffn_importance(
        self,
        dataloader,
        num_samples: int = 128
    ) -> Dict[int, torch.Tensor]:
        self.collect_activations(dataloader, num_samples)
        self._cett_scores = self.compute_cett_scores()
        return self._cett_scores

    # ==================== Attention Head CETT ====================

    def _get_attention_module(self, layer_idx: int):
        try:
            if hasattr(self.model, 'model'):
                base = self.model.model
            else:
                base = self.model

            if hasattr(base, 'model'):
                base = base.model

            if hasattr(base, 'layers'):
                return base.layers[layer_idx].self_attn
        except (AttributeError, IndexError):
            pass

        try:
            return self.model.base_model.model.model.layers[layer_idx].self_attn
        except (AttributeError, IndexError):
            pass

        return None

    def _get_o_proj_weight(self, layer_idx: int) -> Optional[torch.Tensor]:
        attn = self._get_attention_module(layer_idx)
        if attn is None:
            return None

        o_proj = get_attn_output_proj(attn)
        if o_proj is None:
            return None

        if hasattr(o_proj, 'base_layer'):
            return o_proj.base_layer.weight.data
        elif hasattr(o_proj, 'weight'):
            return o_proj.weight.data

        return None

    def _register_attention_hooks(self):
        self.attention_activations = {i: [] for i in range(self.num_layers)}

        for layer_idx in range(self.num_layers):
            attn = self._get_attention_module(layer_idx)
            if attn is None:
                continue

            o_proj = get_attn_output_proj(attn)
            if o_proj is None:
                continue

            if hasattr(o_proj, 'base_layer'):
                target_module = o_proj.base_layer
            else:
                target_module = o_proj

            def make_hook(idx):
                def hook(module, input, output):
                    if len(input) > 0 and input[0] is not None:
                        # input[0] shape: [batch, seq_len, num_heads * head_dim]
                        act = input[0].detach()
                        self.attention_activations[idx].append(act.cpu())
                return hook

            h = target_module.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(h)

    @torch.no_grad()
    def collect_attention_activations(
        self,
        dataloader,
        num_samples: int = 128
    ):
        self._register_attention_hooks()

        self.model.eval()
        samples_collected = 0

        if self.verbose:
            print(f"\n[CETT-Head] Collecting attention outputs from {num_samples} samples...")

        pbar = tqdm(dataloader, desc="[CETT-Head] Calibration", disable=not self.verbose)

        for batch in pbar:
            if samples_collected >= num_samples:
                break

            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                input_ids = batch.get('input_ids')
                attention_mask = batch.get('attention_mask')
            else:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device) if len(batch) > 1 else None

            try:
                if attention_mask is not None:
                    self.model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    self.model(input_ids=input_ids)
            except Exception as e:
                if self.verbose:
                    print(f"[CETT-Head] Forward error: {e}")
                continue

            samples_collected += input_ids.size(0)
            pbar.set_postfix({"samples": samples_collected})

        self._remove_hooks()

        if self.verbose:
            print(f"[CETT-Head] Collected attention outputs for {samples_collected} samples")

    def compute_head_cett_scores(self) -> Dict[int, torch.Tensor]:
        if not hasattr(self, 'attention_activations') or not self.attention_activations:
            raise ValueError("[CETT-Head] No attention activations collected.")

        num_heads = self.config.num_attention_heads
        head_dim = self.hidden_size // num_heads
        head_cett_scores = {}

        if self.verbose:
            print("\n[CETT-Head] Computing head CETT scores...")

        for layer_idx in tqdm(range(self.num_layers), desc="[CETT-Head] Layers", disable=not self.verbose):
            if not self.attention_activations[layer_idx]:
                head_cett_scores[layer_idx] = torch.ones(num_heads)
                continue

            W_o = self._get_o_proj_weight(layer_idx)
            if W_o is None:
                head_cett_scores[layer_idx] = torch.ones(num_heads)
                continue

            W_o = W_o.float().to(self.device)  # [hidden_size, num_heads * head_dim]

            all_attn_outputs = torch.cat(self.attention_activations[layer_idx], dim=0)
            if all_attn_outputs.dim() == 3:
                all_attn_outputs = all_attn_outputs.view(-1, all_attn_outputs.size(-1))
            all_attn_outputs = all_attn_outputs.float().to(self.device)

            num_tokens = all_attn_outputs.size(0)
            head_cett = torch.zeros(num_heads, device=self.device)

            batch_size = min(512, num_tokens)

            for start_idx in range(0, num_tokens, batch_size):
                end_idx = min(start_idx + batch_size, num_tokens)
                attn_batch = all_attn_outputs[start_idx:end_idx]  # [batch, num_heads * head_dim]

                h_total = W_o @ attn_batch.T  # [hidden_size, batch]
                h_total_norm = torch.norm(h_total, dim=0).clamp(min=1e-8)  # [batch]

                for head_i in range(num_heads):
                    head_start = head_i * head_dim
                    head_end = (head_i + 1) * head_dim

                    W_head = W_o[:, head_start:head_end]  # [hidden_size, head_dim]
                    attn_head = attn_batch[:, head_start:head_end]  # [batch, head_dim]

                    h_head = W_head @ attn_head.T  # [hidden_size, batch]
                    h_head_norm = torch.norm(h_head, dim=0)  # [batch]

                    contribution = (h_head_norm / h_total_norm).sum().item()
                    head_cett[head_i] += contribution

            head_cett /= num_tokens
            head_cett_scores[layer_idx] = head_cett.cpu()

            del all_attn_outputs
            if self.device == "cuda":
                torch.cuda.empty_cache()

        self.attention_activations = {}

        if self.verbose:
            print("[CETT-Head] Head CETT scores computed successfully")
            for layer_idx in range(min(3, self.num_layers)):
                scores = head_cett_scores[layer_idx]
                print(f"  Layer {layer_idx}: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

        return head_cett_scores

    def compute_all_head_importance(
        self,
        dataloader,
        num_samples: int = 128
    ) -> Dict[int, torch.Tensor]:
        self.collect_attention_activations(dataloader, num_samples)
        return self.compute_head_cett_scores()


def compute_cett_dimension_masks(
    model: nn.Module,
    dataloader,
    dimension_pruning_ratio: float,
    num_samples: int = 128,
    dimension_group_size: int = 16,
    global_pruning: bool = True,
    device: str = "cuda",
    verbose: bool = True
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    calculator = CETTImportanceCalculator(model, device, verbose)

    calculator.collect_attention_activations(dataloader, num_samples)

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    num_groups_per_layer = head_dim // dimension_group_size

    if verbose:
        print(f"\n[CETT Dim] Computing dimension importance...")
        print(f"[CETT Dim] Dimension pruning ratio: {dimension_pruning_ratio:.2%}")
        print(f"[CETT Dim] Head dim: {head_dim}, Group size: {dimension_group_size}")
        print(f"[CETT Dim] Groups per layer: {num_groups_per_layer}")
        print(f"[CETT Dim] Pruning mode: {'Global' if global_pruning else 'Layer-wise'}")

    layer_dim_importance = {}

    for layer_idx in tqdm(range(num_layers), desc="[CETT Dim] Computing importance", disable=not verbose):
        if not calculator.attention_activations.get(layer_idx):
            layer_dim_importance[layer_idx] = torch.ones(head_dim, device=device)
            continue

        # o_proj weight
        W_o = calculator._get_o_proj_weight(layer_idx)
        if W_o is None:
            layer_dim_importance[layer_idx] = torch.ones(head_dim, device=device)
            continue

        W_o = W_o.float().to(device)  # [hidden_size, num_heads * head_dim]

        # Attention outputs
        all_attn = torch.cat(calculator.attention_activations[layer_idx], dim=0)
        if all_attn.dim() == 3:
            all_attn = all_attn.view(-1, all_attn.size(-1))
        all_attn = all_attn.float().to(device)

        num_tokens = all_attn.size(0)

        dim_importance = torch.zeros(head_dim, device=device)

        batch_size = min(512, num_tokens)

        for start_idx in range(0, num_tokens, batch_size):
            end_idx = min(start_idx + batch_size, num_tokens)
            attn_batch = all_attn[start_idx:end_idx]

            h_total = W_o @ attn_batch.T
            h_total_norm = torch.norm(h_total, dim=0).clamp(min=1e-8)

            for dim_i in range(head_dim):
                dim_indices = [h * head_dim + dim_i for h in range(num_heads)]

                W_dim = W_o[:, dim_indices]  # [hidden_size, num_heads]
                attn_dim = attn_batch[:, dim_indices]  # [batch, num_heads]

                h_dim = W_dim @ attn_dim.T  # [hidden_size, batch]
                h_dim_norm = torch.norm(h_dim, dim=0)

                contribution = (h_dim_norm / h_total_norm).sum().item()
                dim_importance[dim_i] += contribution

        dim_importance /= num_tokens
        layer_dim_importance[layer_idx] = dim_importance

        del all_attn

    calculator.attention_activations = {}
    if device == "cuda":
        torch.cuda.empty_cache()

    qk_dim_masks = {}
    v_dim_masks = {}

    if global_pruning:
        all_groups = []  # (importance, layer_idx, group_indices)

        for layer_idx in range(num_layers):
            dim_importance = layer_dim_importance[layer_idx]

            sorted_scores, sorted_indices = torch.sort(dim_importance, descending=True)

            for group_idx in range(num_groups_per_layer):
                start_idx = group_idx * dimension_group_size
                end_idx = start_idx + dimension_group_size

                group_indices = sorted_indices[start_idx:end_idx]
                group_importance = sorted_scores[start_idx:end_idx].mean().item()

                all_groups.append((group_importance, layer_idx, group_indices.cpu()))

        all_groups.sort(key=lambda x: x[0], reverse=True)

        total_groups = len(all_groups)
        num_groups_to_keep = int(total_groups * (1 - dimension_pruning_ratio))
        num_groups_to_keep = max(num_layers, num_groups_to_keep)

        selected_groups = all_groups[:num_groups_to_keep]

        if verbose:
            print(f"\n[CETT Dim] Global Dimension Pruning:")
            print(f"  Total groups: {total_groups} → Keeping: {num_groups_to_keep}")
            print(f"  Dimensions per layer: varies (global selection)")

        layer_groups = {i: [] for i in range(num_layers)}
        for _, layer_idx, group_indices in selected_groups:
            layer_groups[layer_idx].append(group_indices)

        for layer_idx in range(num_layers):
            groups = layer_groups[layer_idx]

            mask = torch.zeros(head_dim)

            for group_indices in groups:
                mask[group_indices] = 1.0

            qk_dim_masks[layer_idx] = mask.clone()
            v_dim_masks[layer_idx] = mask.clone()

            num_kept = int(mask.sum().item())
            if verbose and (layer_idx < 3 or layer_idx >= num_layers - 1):
                num_groups_kept = len(groups)
                print(f"  Layer {layer_idx}: {num_kept}/{head_dim} dims ({num_groups_kept} groups)")

    else:
        for layer_idx in range(num_layers):
            dim_importance = layer_dim_importance[layer_idx]

            group_importance = []
            sorted_scores, sorted_indices = torch.sort(dim_importance, descending=True)

            for group_idx in range(num_groups_per_layer):
                start_idx = group_idx * dimension_group_size
                end_idx = start_idx + dimension_group_size
                group_indices = sorted_indices[start_idx:end_idx]
                group_imp = sorted_scores[start_idx:end_idx].mean().item()
                group_importance.append((group_imp, group_indices.cpu()))

            group_importance.sort(key=lambda x: x[0], reverse=True)

            num_groups_to_keep = int(num_groups_per_layer * (1 - dimension_pruning_ratio))
            num_groups_to_keep = max(1, num_groups_to_keep)

            mask = torch.zeros(head_dim)
            for _, group_indices in group_importance[:num_groups_to_keep]:
                mask[group_indices] = 1.0

            qk_dim_masks[layer_idx] = mask.clone()
            v_dim_masks[layer_idx] = mask.clone()

            num_kept = int(mask.sum().item())
            if verbose and (layer_idx < 3 or layer_idx >= num_layers - 1):
                print(f"  Layer {layer_idx}: {num_kept}/{head_dim} dims ({num_groups_to_keep} groups)")

    return qk_dim_masks, v_dim_masks


def compute_cett_head_masks(
    model: nn.Module,
    dataloader,
    head_pruning_ratio: float,
    num_samples: int = 128,
    global_pruning: bool = True,
    device: str = "cuda",
    verbose: bool = True
) -> Dict[int, torch.Tensor]:
    calculator = CETTImportanceCalculator(model, device, verbose)
    head_scores = calculator.compute_all_head_importance(dataloader, num_samples)

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    if global_pruning:
        score_tensors = [head_scores[layer_idx] for layer_idx in range(num_layers)]
        all_scores = torch.cat(score_tensors)  # [num_layers * num_heads]

        all_indices = [(layer_idx, head_idx)
                       for layer_idx in range(num_layers)
                       for head_idx in range(num_heads)]
        total_heads = len(all_scores)
        num_to_prune = int(total_heads * head_pruning_ratio)

        if verbose:
            print(f"\n[CETT Head Mask] Total heads: {total_heads}")
            print(f"[CETT Head Mask] Pruning ratio: {head_pruning_ratio:.2%}")
            print(f"[CETT Head Mask] Heads to prune: {num_to_prune}")

        _, sorted_indices = torch.sort(all_scores)
        prune_indices = set(sorted_indices[:num_to_prune].tolist())

        head_masks = {}
        for layer_idx in range(num_layers):
            head_masks[layer_idx] = torch.ones(num_heads)

        for flat_idx in prune_indices:
            layer_idx, head_idx = all_indices[flat_idx]
            head_masks[layer_idx][head_idx] = 0.0

        if not allow_full_block_pruning:
            for layer_idx in range(num_layers):
                if head_masks[layer_idx].sum() == 0:
                    scores = head_scores[layer_idx]
                    best_head = scores.argmax().item()
                    head_masks[layer_idx][best_head] = 1.0

    else:
        # Layer-wise pruning
        head_masks = {}
        min_heads = 0 if allow_full_block_pruning else 1
        for layer_idx in range(num_layers):
            scores = head_scores[layer_idx]
            num_to_keep = max(min_heads, int(num_heads * (1 - head_pruning_ratio)))
            threshold = torch.topk(scores, num_to_keep).values[-1].item() if num_to_keep > 0 else float('inf')
            mask = (scores >= threshold).float() if num_to_keep > 0 else torch.zeros(num_heads)
            head_masks[layer_idx] = mask

    if verbose:
        for layer_idx in range(num_layers):
            kept = int(head_masks[layer_idx].sum().item())
            print(f"  Layer {layer_idx}: {kept}/{num_heads} heads kept")

    return head_masks


def compute_cett_masks(
    model: nn.Module,
    dataloader,
    ffn_pruning_ratio: float,
    num_samples: int = 128,
    dimension_multiple: int = 1,
    device: str = "cuda",
    verbose: bool = True,
    allow_full_block_pruning: bool = False
) -> Dict[int, torch.Tensor]:
    calculator = CETTImportanceCalculator(model, device, verbose)
    cett_scores = calculator.compute_all_ffn_importance(dataloader, num_samples)

    num_layers = model.config.num_hidden_layers
    intermediate_size = getattr(model.config, 'intermediate_size', model.config.hidden_size * 4)

    score_tensors = [cett_scores[layer_idx] for layer_idx in range(num_layers)]
    all_scores = torch.cat(score_tensors)  # [num_layers * intermediate_size]

    all_indices = [(layer_idx, neuron_idx)
                   for layer_idx in range(num_layers)
                   for neuron_idx in range(intermediate_size)]

    total_neurons = len(all_scores)
    num_to_prune = int(total_neurons * ffn_pruning_ratio)

    if dimension_multiple > 1:
        num_to_prune = (num_to_prune // dimension_multiple) * dimension_multiple

    if verbose:
        print(f"\n[CETT Mask] Total neurons: {total_neurons}")
        print(f"[CETT Mask] Pruning ratio: {ffn_pruning_ratio:.2%}")
        print(f"[CETT Mask] Neurons to prune: {num_to_prune}")

    _, sorted_indices = torch.sort(all_scores)
    prune_indices = set(sorted_indices[:num_to_prune].tolist())

    ffn_masks = {}
    for layer_idx in range(num_layers):
        mask = torch.ones(intermediate_size)
        ffn_masks[layer_idx] = mask

    for flat_idx in prune_indices:
        layer_idx, neuron_idx = all_indices[flat_idx]
        ffn_masks[layer_idx][neuron_idx] = 0.0

    if verbose:
        for layer_idx in range(num_layers):
            kept = ffn_masks[layer_idx].sum().item()
            total = ffn_masks[layer_idx].numel()
            print(f"  Layer {layer_idx}: {int(kept)}/{total} neurons kept ({kept/total:.1%})")

    return ffn_masks


def create_calibration_dataloader(
    dataset_name: str = "c4_local",
    num_samples: int = 128,
    batch_size: int = 4,
    max_length: int = 512,
    seed: int = 42,
    tokenizer = None,
    distributed: bool = False
):
    import random
    from torch.utils.data import DataLoader

    random.seed(seed)
    torch.manual_seed(seed)

    if dataset_name == "c4_local":
        from utils import load_c4_local_dataset
        raw_dataset, _ = load_c4_local_dataset(
            config=None,
            num_samples=num_samples * 2,
            max_length=max_length
        )
        def tokenize_fn(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors=None
            )
        dataset = raw_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    elif dataset_name == "wikitext2_val":
        from datasets import load_dataset
        raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        raw_dataset = raw_dataset.filter(lambda x: len(x["text"].strip()) > 0)
        def tokenize_fn(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors=None
            )
        dataset = raw_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
        if len(dataset) > num_samples * 2:
            dataset = dataset.select(range(num_samples * 2))

    elif dataset_name == "bookcorpus_local":
        from datasets import Dataset
        import json
        print(f"[BookCorpus-Local] Loading {num_samples} samples from local cache...")

        local_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "bookcorpus_samples")

        if not os.path.exists(local_dir):
            raise FileNotFoundError(
                f"BookCorpus local cache not found at {local_dir}\n"
                f"Run: python scripts/download_bookcorpus_samples.py"
            )

        sample_dir = os.path.join(local_dir, f"n{num_samples}")
        seed_file = os.path.join(sample_dir, f"seed_{seed}.json")

        if os.path.exists(seed_file):
            print(f"[BookCorpus-Local] Found cached file: n{num_samples}/seed_{seed}.json")
            with open(seed_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            samples = data["samples"][:num_samples]
        elif os.path.exists(sample_dir):
            fallback_file = os.path.join(sample_dir, "seed_42.json")
            if os.path.exists(fallback_file):
                print(f"[BookCorpus-Local] No file for seed {seed}, using n{num_samples}/seed_42.json")
                with open(fallback_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                random.seed(seed)
                samples = data["samples"].copy()
                random.shuffle(samples)
                samples = samples[:num_samples]
            else:
                import glob
                json_files = glob.glob(os.path.join(sample_dir, "seed_*.json"))
                if json_files:
                    print(f"[BookCorpus-Local] Using {json_files[0]}")
                    with open(json_files[0], "r", encoding="utf-8") as f:
                        data = json.load(f)
                    samples = data["samples"][:num_samples]
                else:
                    raise FileNotFoundError(f"No cached samples found in {sample_dir}")
        else:
            old_seed_file = os.path.join(local_dir, f"seed_{seed}.json")
            if os.path.exists(old_seed_file):
                print(f"[BookCorpus-Local] Using legacy format: seed_{seed}.json")
                with open(old_seed_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                samples = data["samples"][:num_samples]
            else:
                fallback_file = os.path.join(local_dir, "seed_42.json")
                if os.path.exists(fallback_file):
                    print(f"[BookCorpus-Local] Using legacy fallback: seed_42.json")
                    with open(fallback_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    random.seed(seed)
                    samples = data["samples"].copy()
                    random.shuffle(samples)
                    samples = samples[:num_samples]
                else:
                    raise FileNotFoundError(
                        f"No cached samples found. Expected:\n"
                        f"  - {seed_file} (new format)\n"
                        f"  - {old_seed_file} (legacy format)\n"
                        f"Run: python scripts/download_bookcorpus_samples.py --num_samples {num_samples}"
                    )

        print(f"[BookCorpus-Local] Loaded {len(samples)} samples")
        raw_dataset = Dataset.from_list(samples)

        def tokenize_fn(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors=None
            )
        dataset = raw_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    elif dataset_name == "bookcorpus":
        from datasets import load_dataset, Dataset
        print(f"[BookCorpus] Loading {num_samples} random samples...")
        raw_dataset = None

        dataset_attempts = [
            ("bookcorpusopen", {"trust_remote_code": True}),
            ("bookcorpusopen", {}),
            ("bookcorpus", {"trust_remote_code": True}),
            ("bookcorpus", {}),
            ("wikitext", {"name": "wikitext-103-raw-v1"}),
            ("wikitext", {"name": "wikitext-2-raw-v1"}),
        ]

        pool_size = num_samples * 10
        samples = []

        for ds_name, extra_kwargs in dataset_attempts:
            try:
                raw_dataset = load_dataset(ds_name, split="train", streaming=True, **extra_kwargs)

                error_count = 0
                max_errors = 100
                max_attempts = pool_size * 3
                dataset_iter = iter(raw_dataset)
                attempt = 0

                while len(samples) < pool_size and attempt < max_attempts:
                    attempt += 1
                    try:
                        example = next(dataset_iter)
                        text = example.get("text", "")
                        if len(text.strip()) > 50:
                            samples.append({"text": text[:2000]})
                    except StopIteration:
                        break
                    except UnicodeDecodeError:
                        error_count += 1
                        if error_count > max_errors:
                            print(f"[BookCorpus] {ds_name}: too many UnicodeDecodeError ({error_count}), trying next...")
                            break
                        continue
                    except Exception as inner_e:
                        error_count += 1
                        if error_count > max_errors:
                            raise inner_e
                        continue

                if len(samples) >= num_samples:
                    print(f"[BookCorpus] Successfully loaded: {ds_name} ({len(samples)} samples)")
                    break
                else:
                    print(f"[BookCorpus] {ds_name}: only got {len(samples)} samples, trying next...")
                    samples = []

            except Exception as e:
                print(f"[BookCorpus] {ds_name} failed ({type(e).__name__}), trying next...")
                samples = []
                continue

        if len(samples) < num_samples:
            raise RuntimeError(f"Failed to load enough samples (got {len(samples)}, need {num_samples})")

        if len(samples) > num_samples:
            indices = random.sample(range(len(samples)), num_samples)
            samples = [samples[i] for i in indices]

        raw_dataset = Dataset.from_list(samples)
        print(f"[BookCorpus] Selected {len(raw_dataset)} samples")

        def tokenize_fn(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors=None
            )
        dataset = raw_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if len(dataset) > num_samples:
        indices = random.sample(range(len(dataset)), num_samples)
        dataset = dataset.select(indices)

    def collate_fn(batch):
        input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
        attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch])
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    sampler = None
    if distributed:
        from torch.utils.data import DistributedSampler
        import torch.distributed as _dist
        sampler = DistributedSampler(
            dataset,
            num_replicas=_dist.get_world_size(),
            rank=_dist.get_rank(),
            shuffle=False,
            seed=seed
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        collate_fn=collate_fn
    )

    return dataloader


# ==================== Importance Calculator (Gradient-based) ====================

class TaylorImportanceCalculator:

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        verbose: bool = True,
        ffn_mode: str = "down",
        importance_method: str = "ganda",  # "ganda", "taylor", "wanda"
        head_masks: Dict[int, torch.Tensor] = None,
        ffn_masks: Dict[int, torch.Tensor] = None,
        dim_masks: Dict[int, torch.Tensor] = None,
        embedding_mask: torch.Tensor = None,
        use_gradient_checkpointing: bool = False,
        use_correct_normalization: bool = False  # True: sum-based (correct), False: mean-based (legacy)
    ):
        self.model = model
        self.device = device
        self.verbose = verbose
        self.ffn_mode = ffn_mode  # "down", "up", "all"
        self.importance_method = importance_method  # "ganda", "taylor", "wanda"
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_correct_normalization = use_correct_normalization

        if use_gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                if verbose:
                    print(f"[Importance] Gradient checkpointing enabled for memory savings")

        self.head_masks = head_masks  # {layer_idx: [num_heads] tensor}
        self.ffn_masks = ffn_masks    # {layer_idx: [intermediate_size] tensor}
        self.dim_masks = dim_masks    # {layer_idx: [head_dim] tensor}
        self.embedding_mask = embedding_mask  # [hidden_size] tensor

        self.config = model.config
        self.num_layers = self.config.num_hidden_layers
        self.hidden_size = self.config.hidden_size
        self.intermediate_size = getattr(self.config, 'intermediate_size', self.hidden_size * 4)

        self.activations = {}
        self.gradients = {}
        self.weights = {}
        self.up_activations = {}
        self.up_gradients = {}
        self.up_weights = {}
        self.gate_activations = {}
        self.gate_gradients = {}
        self.gate_weights = {}
        self.q_activations = {}
        self.k_activations = {}
        self.hooks = []
        self.premasking_hooks = []

        if self.verbose:
            print(f"[Importance] Method: {importance_method} ({self._get_method_formula()})")
            if head_masks is not None:
                total_heads = sum(m.numel() for m in head_masks.values())
                pruned_heads = sum((m == 0).sum().item() for m in head_masks.values())
                print(f"[Importance] Head pre-masking enabled: {pruned_heads}/{total_heads} heads masked")
            if ffn_masks is not None:
                total_neurons = sum(m.numel() for m in ffn_masks.values())
                pruned_neurons = sum((m == 0).sum().item() for m in ffn_masks.values())
                print(f"[Importance] FFN pre-masking enabled: {pruned_neurons}/{total_neurons} neurons masked")
            if dim_masks is not None:
                total_dims = sum(m.numel() for m in dim_masks.values())
                pruned_dims = sum((m == 0).sum().item() for m in dim_masks.values())
                print(f"[Importance] Dimension pre-masking enabled: {pruned_dims}/{total_dims} dims masked")
            if embedding_mask is not None:
                total_hidden = embedding_mask.numel()
                pruned_hidden = (embedding_mask == 0).sum().item()
                print(f"[Importance] Embedding pre-masking enabled: {pruned_hidden}/{total_hidden} hidden dims masked")

    def _get_method_formula(self) -> str:
        formulas = {
            "ganda": "|grad × activation|",
            "taylor": "|grad × weight|",
            "wanda": "|weight × activation|",
            "dynamics": "Σ ||∇_{θ_g} L(x)||² (gradient norm²)",
            "coupling": "||K^(g) - diag(K^(g))||_F² (sample interaction)",
            "ganda_ntk_dyn": "|grad × activation| × dynamics",
            "ganda_ntk_coupling": "|grad × activation| × coupling",
            "ganda_ntk_combined": "|grad × activation| × dynamics × coupling",
            # STI methods
            "sti": "SVD(|grad × act|) → spectral importance",
            "sti_magnitude": "STI × magnitude (ganda)"
        }
        return formulas.get(self.importance_method, "unknown")

    def _is_ntk_method(self) -> bool:
        return self.importance_method in ["ganda_ntk_dyn", "ganda_ntk_coupling", "ganda_ntk_combined"]

    def _is_standalone_ntk_method(self) -> bool:
        return self.importance_method in ["dynamics", "coupling"]

    def _is_sti_method(self) -> bool:
        return self.importance_method in ["sti", "sti_magnitude"]

    def _register_premasking_hooks(self):
        self._remove_premasking_hooks()

        if self.head_masks is not None:
            num_heads = self.config.num_attention_heads
            head_dim = self.hidden_size // num_heads

            for layer_idx in range(self.num_layers):
                if layer_idx not in self.head_masks:
                    continue

                mask = self.head_masks[layer_idx]  # [num_heads]
                if mask.sum() == num_heads:
                    continue

                attn = self._get_attention_module(layer_idx)
                if attn is None:
                    continue

                o_proj = get_attn_output_proj(attn)
                if o_proj is None:
                    continue
                if hasattr(o_proj, 'base_layer'):
                    target_module = o_proj.base_layer
                else:
                    target_module = o_proj

                def make_head_mask_hook(layer_mask, n_heads, h_dim):
                    def hook(module, input, output):
                        # output shape: [batch, seq, hidden_size]
                        # mask shape: [num_heads] -> expand to [1, 1, hidden_size]
                        batch_size, seq_len, hidden = output.shape
                        # [num_heads] -> [num_heads, head_dim] -> [hidden_size]
                        expanded_mask = layer_mask.unsqueeze(-1).expand(-1, h_dim).reshape(-1)
                        expanded_mask = expanded_mask.to(output.device, dtype=output.dtype)
                        return output * expanded_mask.unsqueeze(0).unsqueeze(0)
                    return hook

                h = target_module.register_forward_hook(
                    make_head_mask_hook(mask, num_heads, head_dim)
                )
                self.premasking_hooks.append(h)

        if self.ffn_masks is not None:
            for layer_idx in range(self.num_layers):
                if layer_idx not in self.ffn_masks:
                    continue

                mask = self.ffn_masks[layer_idx]  # [intermediate_size]
                if mask.sum() == self.intermediate_size:
                    continue

                mlp = self._get_mlp_module(layer_idx)
                if mlp is None:
                    continue

                def make_ffn_mask_hook(layer_mask):
                    def hook(module, input, output):
                        # output shape: [batch, seq, intermediate_size]
                        m = layer_mask.to(output.device, dtype=output.dtype)
                        return output * m.unsqueeze(0).unsqueeze(0)
                    return hook

                for up_module in get_up_proj_modules(mlp):
                    if hasattr(up_module, 'base_layer'):
                        target = up_module.base_layer
                    else:
                        target = up_module
                    h = target.register_forward_hook(make_ffn_mask_hook(mask))
                    self.premasking_hooks.append(h)

        if self.dim_masks is not None:
            num_heads = self.config.num_attention_heads
            head_dim = self.hidden_size // num_heads

            for layer_idx in range(self.num_layers):
                if layer_idx not in self.dim_masks:
                    continue

                mask = self.dim_masks[layer_idx]  # [head_dim]
                if mask.sum() == head_dim:
                    continue

                attn = self._get_attention_module(layer_idx)
                if attn is None:
                    continue

                def make_dim_mask_hook(layer_mask, n_heads, h_dim):
                    def hook(module, input, output):
                        # output shape: [batch, seq, num_heads * head_dim]
                        batch_size, seq_len, total_dim = output.shape
                        # [head_dim] -> [num_heads, head_dim] -> [num_heads * head_dim]
                        expanded_mask = layer_mask.unsqueeze(0).expand(n_heads, -1).reshape(-1)
                        expanded_mask = expanded_mask.to(output.device, dtype=output.dtype)
                        return output * expanded_mask.unsqueeze(0).unsqueeze(0)
                    return hook

                for proj_name in ['q_proj', 'k_proj', 'v_proj']:
                    proj = getattr(attn, proj_name, None)
                    if proj is None:
                        continue
                    if hasattr(proj, 'base_layer'):
                        target_module = proj.base_layer
                    else:
                        target_module = proj

                    h = target_module.register_forward_hook(
                        make_dim_mask_hook(mask, num_heads, head_dim)
                    )
                    self.premasking_hooks.append(h)

        if self.embedding_mask is not None:
            embed_mask = self.embedding_mask  # [hidden_size]
            if embed_mask.sum() < self.hidden_size:

                def make_embedding_mask_hook(emb_mask):
                    def hook(module, input, output):
                        # output shape: [batch, seq, hidden_size]
                        m = emb_mask.to(output.device, dtype=output.dtype)
                        return output * m.unsqueeze(0).unsqueeze(0)
                    return hook

                embed_tokens = self.model.model.embed_tokens
                if hasattr(embed_tokens, 'base_layer'):
                    embed_target = embed_tokens.base_layer
                else:
                    embed_target = embed_tokens

                h = embed_target.register_forward_hook(make_embedding_mask_hook(embed_mask))
                self.premasking_hooks.append(h)

                for layer_idx in range(self.num_layers):
                    layer = self.model.model.layers[layer_idx]


                    def make_layer_output_hook(emb_mask):
                        def hook(module, input, output):
                            # LlamaDecoderLayer output: (hidden_states, ...) tuple
                            if isinstance(output, tuple):
                                hidden = output[0]
                                m = emb_mask.to(hidden.device, dtype=hidden.dtype)
                                masked_hidden = hidden * m.unsqueeze(0).unsqueeze(0)
                                return (masked_hidden,) + output[1:]
                            else:
                                m = emb_mask.to(output.device, dtype=output.dtype)
                                return output * m.unsqueeze(0).unsqueeze(0)
                        return hook

                    h = layer.register_forward_hook(make_layer_output_hook(embed_mask))
                    self.premasking_hooks.append(h)

        if self.verbose and (self.head_masks is not None or self.ffn_masks is not None or
                             self.dim_masks is not None or self.embedding_mask is not None):
            print(f"[Pre-masking] Registered {len(self.premasking_hooks)} pre-masking hooks")

    def _remove_premasking_hooks(self):
        for h in self.premasking_hooks:
            h.remove()
        self.premasking_hooks = []

    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations.clear()
        self.gradients.clear()
        self.weights.clear()
        self.up_activations.clear()
        self.up_gradients.clear()
        self.up_weights.clear()
        self.gate_activations.clear()
        self.gate_gradients.clear()
        self.gate_weights.clear()
        self.q_activations.clear()
        self.k_activations.clear()

    def _get_base_method(self) -> str:
        if self._is_ntk_method():
            return "ganda"
        if self._is_sti_method():
            return "ganda"
        return self.importance_method

    def _compute_importance(self, act, grad, weight) -> torch.Tensor:
        method = self._get_base_method()

        if method == "ganda":
            # |gradient × activation|
            importance = (act * grad).abs()
        elif method == "taylor":
            # |gradient × weight|
            if weight is not None and weight.dim() == 2:
                w_norm = weight.abs().mean(dim=0)  # [intermediate]
            elif weight is not None:
                w_norm = weight.abs()
            else:
                w_norm = torch.ones(grad.shape[-1], device=grad.device)
            importance = (grad.abs() * w_norm.unsqueeze(0).unsqueeze(0))
        elif method == "wanda":
            # |weight × activation|
            if weight is not None and weight.dim() == 2:
                w_norm = weight.abs().mean(dim=0)  # [intermediate]
            elif weight is not None:
                w_norm = weight.abs()
            else:
                w_norm = torch.ones(act.shape[-1], device=act.device)
            importance = (act.abs() * w_norm.unsqueeze(0).unsqueeze(0))
        elif method == "wag":
            if weight is not None and weight.dim() == 2:
                w_norm = weight.abs().mean(dim=0)  # [intermediate]
            elif weight is not None:
                w_norm = weight.abs()
            else:
                w_norm = torch.ones(act.shape[-1], device=act.device)
            # grad × act × weight
            importance = (act * grad).abs() * w_norm.unsqueeze(0).unsqueeze(0)
        else:
            importance = (act * grad).abs()

        return importance

    def _normalize_scores(self, scores: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        all_scores = torch.cat([s.flatten() for s in scores.values()])
        min_val = all_scores.min()
        max_val = all_scores.max()

        if max_val - min_val < 1e-8:
            return {k: torch.ones_like(v) for k, v in scores.items()}

        normalized = {}
        for layer_idx, s in scores.items():
            normalized[layer_idx] = (s - min_val) / (max_val - min_val)

        return normalized

    # ==================== STI (Spectral Taylor Importance) ====================

    def _compute_sti_scores_from_matrix(self, taylor_matrix: torch.Tensor) -> torch.Tensor:
        T = taylor_matrix  # [N, D]

        if T.shape[0] < 2:
            return T.abs().mean(dim=0)

        original_dtype = T.dtype
        T = T.float()

        # T_centered = T - T.mean(dim=0, keepdim=True)

        try:
            # SVD: T = U @ diag(S) @ Vh
            _, S, Vh = torch.linalg.svd(T, full_matrices=False)

            energy = S ** 2
            energy_ratio = energy / (energy.sum() + 1e-8)  # [min(N, D)]

            # STI: energy-weighted importance
            # STI[j] = Σ_k (energy[k] × Vh[k,j]²)
            importance = (energy_ratio.unsqueeze(-1) * (Vh ** 2)).sum(dim=0)  # [D]

            return importance

        except Exception as e:
            if self.verbose:
                print(f"[STI] SVD failed, using fallback: {e}")
            return taylor_matrix.abs().mean(dim=0)

    def _compute_ffn_sti_scores(
        self,
        dataloader,
        num_samples: int = 128
    ) -> Dict[int, torch.Tensor]:
        self._register_hooks()
        self.model.train()

        for param in self.model.parameters():
            param.requires_grad = True

        if self.verbose:
            print(f"\n[STI-FFN] Computing STI scores...")
            print(f"[STI-FFN] Collecting per-sample Taylor values for SVD")

        taylor_per_sample = {i: [] for i in range(self.num_layers)}

        num_processed = 0
        pbar = tqdm(dataloader, desc="[STI-FFN] Collecting samples", disable=not self.verbose)

        for batch in pbar:
            if num_processed >= num_samples:
                break

            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                input_ids = batch.get('input_ids')
                attention_mask = batch.get('attention_mask')
            else:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device) if len(batch) > 1 else None

            labels = input_ids.clone()

            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()

                for layer_idx in range(self.num_layers):
                    if self.activations[layer_idx] and self.gradients[layer_idx]:
                        act = self.activations[layer_idx][-1]
                        grad = self.gradients[layer_idx][-1]

                        if act.shape == grad.shape:
                            taylor = (act * grad).abs()  # [batch, seq, intermediate]
                            taylor_mean = taylor.mean(dim=(0, 1)).cpu()
                            taylor_per_sample[layer_idx].append(taylor_mean)

                num_processed += input_ids.size(0)
                pbar.set_postfix({"samples": num_processed})

            except Exception as e:
                if self.verbose:
                    print(f"[STI-FFN] Error: {e}")
                continue
            finally:
                self.model.zero_grad()
                for i in range(self.num_layers):
                    self.activations[i] = []
                    self.gradients[i] = []

        self._remove_hooks()

        sti_scores = {}
        for layer_idx in range(self.num_layers):
            if taylor_per_sample[layer_idx]:
                # Taylor Matrix: [num_samples, intermediate_size]
                T = torch.stack(taylor_per_sample[layer_idx])
                sti_scores[layer_idx] = self._compute_sti_scores_from_matrix(T)
            else:
                sti_scores[layer_idx] = torch.zeros(self.intermediate_size)

        if self.verbose:
            print(f"[STI-FFN] Processed {num_processed} samples")
            print(f"[STI-FFN] Taylor matrix shape per layer: [{len(taylor_per_sample[0])}, {self.intermediate_size}]")
            for layer_idx in range(min(3, self.num_layers)):
                scores = sti_scores[layer_idx]
                print(f"  Layer {layer_idx}: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

        return sti_scores

    def _compute_head_sti_scores(
        self,
        dataloader,
        num_samples: int = 128,
        head_mode: str = "o"
    ) -> Dict[int, torch.Tensor]:
        self._register_head_hooks(head_mode)
        self.model.train()

        for param in self.model.parameters():
            param.requires_grad = True

        num_heads = self.config.num_attention_heads
        head_dim = self.hidden_size // num_heads

        if self.verbose:
            print(f"\n[STI-Head] Computing STI scores...")
            print(f"[STI-Head] Mode: {head_mode}, Heads: {num_heads}")

        taylor_per_sample = {i: [] for i in range(self.num_layers)}

        num_processed = 0
        pbar = tqdm(dataloader, desc="[STI-Head] Collecting samples", disable=not self.verbose)

        for batch in pbar:
            if num_processed >= num_samples:
                break

            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                input_ids = batch.get('input_ids')
                attention_mask = batch.get('attention_mask')
            else:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device) if len(batch) > 1 else None

            labels = input_ids.clone()

            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()

                for layer_idx in range(self.num_layers):
                    if self.head_activations[layer_idx] and self.head_gradients[layer_idx]:
                        act = self.head_activations[layer_idx][-1]
                        grad = self.head_gradients[layer_idx][-1]

                        if act.shape == grad.shape:
                            # |act × grad|
                            taylor = (act * grad).abs()  # [batch, seq, num_heads * head_dim]
                            # Reshape to [batch, seq, num_heads, head_dim]
                            taylor = taylor.view(taylor.shape[0], taylor.shape[1], num_heads, head_dim)
                            taylor_per_head = taylor.sum(dim=-1).mean(dim=(0, 1)).cpu()
                            taylor_per_sample[layer_idx].append(taylor_per_head)

                num_processed += input_ids.size(0)
                pbar.set_postfix({"samples": num_processed})

            except Exception as e:
                if self.verbose:
                    print(f"[STI-Head] Error: {e}")
                continue
            finally:
                self.model.zero_grad()
                for i in range(self.num_layers):
                    self.head_activations[i] = []
                    self.head_gradients[i] = []

        self._remove_hooks()

        sti_scores = {}
        for layer_idx in range(self.num_layers):
            if taylor_per_sample[layer_idx]:
                T = torch.stack(taylor_per_sample[layer_idx])  # [num_samples, num_heads]
                sti_scores[layer_idx] = self._compute_sti_scores_from_matrix(T)
            else:
                sti_scores[layer_idx] = torch.zeros(num_heads)

        if self.verbose:
            print(f"[STI-Head] Processed {num_processed} samples")
            for layer_idx in range(min(3, self.num_layers)):
                scores = sti_scores[layer_idx]
                print(f"  Layer {layer_idx}: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

        return sti_scores

    def _compute_dimension_sti_scores(
        self,
        dataloader,
        num_samples: int = 128,
        head_mode: str = "o"
    ) -> Dict[int, torch.Tensor]:
        self._register_head_hooks(head_mode)
        self.model.train()

        for param in self.model.parameters():
            param.requires_grad = True

        num_heads = self.config.num_attention_heads
        head_dim = self.hidden_size // num_heads

        if self.verbose:
            print(f"\n[STI-Dim] Computing STI scores...")
            print(f"[STI-Dim] Head dim: {head_dim}")

        taylor_per_sample = {i: [] for i in range(self.num_layers)}

        num_processed = 0
        pbar = tqdm(dataloader, desc="[STI-Dim] Collecting samples", disable=not self.verbose)

        for batch in pbar:
            if num_processed >= num_samples:
                break

            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                input_ids = batch.get('input_ids')
                attention_mask = batch.get('attention_mask')
            else:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device) if len(batch) > 1 else None

            labels = input_ids.clone()

            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()

                for layer_idx in range(self.num_layers):
                    if self.head_activations[layer_idx] and self.head_gradients[layer_idx]:
                        act = self.head_activations[layer_idx][-1]
                        grad = self.head_gradients[layer_idx][-1]

                        if act.shape == grad.shape:
                            # |act × grad|
                            taylor = (act * grad).abs()
                            # Reshape to [batch, seq, num_heads, head_dim]
                            taylor = taylor.view(taylor.shape[0], taylor.shape[1], num_heads, head_dim)
                            taylor_per_dim = taylor.mean(dim=(0, 1, 2)).cpu()
                            taylor_per_sample[layer_idx].append(taylor_per_dim)

                num_processed += input_ids.size(0)
                pbar.set_postfix({"samples": num_processed})

            except Exception as e:
                if self.verbose:
                    print(f"[STI-Dim] Error: {e}")
                continue
            finally:
                self.model.zero_grad()
                for i in range(self.num_layers):
                    self.head_activations[i] = []
                    self.head_gradients[i] = []

        self._remove_hooks()

        sti_scores = {}
        for layer_idx in range(self.num_layers):
            if taylor_per_sample[layer_idx]:
                T = torch.stack(taylor_per_sample[layer_idx])  # [num_samples, head_dim]
                sti_scores[layer_idx] = self._compute_sti_scores_from_matrix(T)
            else:
                sti_scores[layer_idx] = torch.zeros(head_dim)

        if self.verbose:
            print(f"[STI-Dim] Processed {num_processed} samples")
            for layer_idx in range(min(3, self.num_layers)):
                scores = sti_scores[layer_idx]
                print(f"  Layer {layer_idx}: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

        return sti_scores

    # ==================== End STI ====================


    def compute_ffn_diversity_scores(
        self,
        dataloader,
        num_samples: int = 128,
        block_size: int = 64,
        method: str = "dominant"
    ) -> Dict[int, torch.Tensor]:
        self._register_hooks()
        self.model.train()

        for param in self.model.parameters():
            param.requires_grad = True

        if self.verbose:
            print(f"\n[SVD-Diversity] Computing diversity scores...")
            print(f"[SVD-Diversity] Block size: {block_size}, Method: {method}")

        taylor_per_sample = {i: [] for i in range(self.num_layers)}
        num_processed = 0

        pbar = tqdm(dataloader, desc="[SVD-Diversity] Collecting samples", disable=not self.verbose)

        for batch in pbar:
            if num_processed >= num_samples:
                break

            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                input_ids = batch.get('input_ids')
                attention_mask = batch.get('attention_mask')
            else:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device) if len(batch) > 1 else None

            labels = input_ids.clone()

            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()

                for layer_idx in range(self.num_layers):
                    if self.activations[layer_idx] and self.gradients[layer_idx]:
                        act = self.activations[layer_idx][-1]
                        grad = self.gradients[layer_idx][-1]

                        if act.shape == grad.shape:
                            taylor = (act * grad).abs()
                            taylor_mean = taylor.mean(dim=(0, 1)).cpu()
                            taylor_per_sample[layer_idx].append(taylor_mean)

                num_processed += input_ids.size(0)
                pbar.set_postfix({"samples": num_processed})

            except Exception as e:
                if self.verbose:
                    print(f"[SVD-Diversity] Error: {e}")
                continue
            finally:
                self.model.zero_grad()
                for i in range(self.num_layers):
                    self.activations[i] = []
                    self.gradients[i] = []

        self._remove_hooks()

        diversity_scores = {}
        num_blocks = self.intermediate_size // block_size

        for layer_idx in range(self.num_layers):
            if taylor_per_sample[layer_idx]:
                T = torch.stack(taylor_per_sample[layer_idx])  # [N, intermediate_size]
                diversity_scores[layer_idx] = self._compute_block_diversity(
                    T, block_size, method
                )
            else:
                diversity_scores[layer_idx] = torch.ones(num_blocks)

        if self.verbose:
            print(f"[SVD-Diversity] Processed {num_processed} samples")
            print(f"[SVD-Diversity] Num blocks per layer: {num_blocks}")
            for layer_idx in range(min(3, self.num_layers)):
                scores = diversity_scores[layer_idx]
                print(f"  Layer {layer_idx}: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")

        return diversity_scores

    def _compute_block_diversity(
        self,
        taylor_matrix: torch.Tensor,
        block_size: int,
        method: str = "dominant"
    ) -> torch.Tensor:
        T = taylor_matrix.float()  # [N, D]
        N, D = T.shape
        num_blocks = D // block_size

        if N < 2 or num_blocks < 2:
            return torch.ones(num_blocks)

        T_reshaped = T.view(N, num_blocks, block_size)
        block_features = T_reshaped.mean(dim=-1)  # [N, num_blocks]

        try:
            _, S, Vh = torch.linalg.svd(block_features, full_matrices=False)
            # Vh: [min(N, num_blocks), num_blocks]

            if method == "dominant":
                dominant_comp = Vh.abs().argmax(dim=0)  # [num_blocks]

                diversity = torch.zeros(num_blocks)
                for b in range(num_blocks):
                    comp = dominant_comp[b]
                    num_same = (dominant_comp == comp).sum().item()
                    diversity[b] = 1.0 / num_same

            else:  # "soft"
                energy = S ** 2
                energy_ratio = energy / (energy.sum() + 1e-8)  # [min(N, num_blocks)]

                block_contrib = (Vh ** 2)  # [min(N, num_blocks), num_blocks]

                comp_total = block_contrib.sum(dim=1, keepdim=True) + 1e-8  # [min, 1]
                relative_contrib = block_contrib / comp_total  # [min, num_blocks]

                # weighted by energy ratio
                diversity = (energy_ratio.unsqueeze(-1) * relative_contrib).sum(dim=0)

            diversity = diversity / (diversity.max() + 1e-8)

            return diversity

        except Exception as e:
            if self.verbose:
                print(f"[SVD-Diversity] SVD failed, returning uniform diversity: {e}")
            return torch.ones(num_blocks)

    def compute_diversity_adjusted_importance(
        self,
        importance_scores: Dict[int, torch.Tensor],
        diversity_scores: Dict[int, torch.Tensor],
        block_size: int = 64,
        alpha: float = 1.0
    ) -> Dict[int, torch.Tensor]:
        adjusted_scores = {}

        for layer_idx in importance_scores.keys():
            imp = importance_scores[layer_idx]  # [intermediate_size]
            div = diversity_scores.get(layer_idx, None)

            if div is None or alpha == 0:
                adjusted_scores[layer_idx] = imp
                continue

            num_blocks = len(div)


            imp_sorted, sorted_indices = torch.sort(imp, descending=True)
            imp_reshaped = imp_sorted.view(num_blocks, block_size)
            block_importance = imp_reshaped.sum(dim=-1)  # [num_blocks]

            adjusted_block = block_importance * (div ** alpha)

            adjusted = torch.zeros_like(imp)
            for b in range(num_blocks):
                start_idx = b * block_size
                end_idx = start_idx + block_size
                block_indices = sorted_indices[start_idx:end_idx]
                block_vals = imp[block_indices]
                block_sum = block_vals.sum() + 1e-8
                scale = adjusted_block[b] / block_sum
                adjusted[block_indices] = block_vals * scale

            adjusted_scores[layer_idx] = adjusted

        if self.verbose:
            print(f"\n[SVD-Diversity] Applied diversity adjustment (alpha={alpha})")
            for layer_idx in range(min(3, self.num_layers)):
                orig = importance_scores[layer_idx]
                adj = adjusted_scores[layer_idx]
                print(f"  Layer {layer_idx}: orig_max={orig.max():.4f}, adj_max={adj.max():.4f}")

        return adjusted_scores

    # ==================== End SVD Diversity ====================

    def _compute_dynamics_scores(
        self,
        dataloader,
        num_samples: int = 128
    ) -> Dict[int, torch.Tensor]:
        self._register_hooks_for_ntk()
        self.model.train()

        for param in self.model.parameters():
            param.requires_grad = True

        if self.verbose:
            print(f"\n[NTK-Dynamics] Computing dynamics scores (gradient norm²)...")

        dynamics_scores = {i: torch.zeros(self.intermediate_size, device='cpu')
                          for i in range(self.num_layers)}

        num_processed = 0
        pbar = tqdm(dataloader, desc="[NTK-Dynamics]", disable=not self.verbose)

        for batch in pbar:
            if num_processed >= num_samples:
                break

            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                input_ids = batch.get('input_ids')
                attention_mask = batch.get('attention_mask')
            else:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device) if len(batch) > 1 else None

            labels = input_ids.clone()

            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()

                if num_processed == 0 and self.verbose:
                    collected_layers = sum(1 for i in range(self.num_layers) if self.gradients[i])
                    print(f"[NTK-Dynamics] Gradient collected for {collected_layers}/{self.num_layers} layers")

                for layer_idx in range(self.num_layers):
                    if self.gradients[layer_idx]:
                        grad = self.gradients[layer_idx][-1]  # [batch, seq, intermediate_size]
                        # gradient norm² per neuron: sum over batch and seq
                        grad_norm_sq = (grad ** 2).sum(dim=(0, 1))  # [intermediate_size]
                        dynamics_scores[layer_idx] += grad_norm_sq.cpu()

                num_processed += input_ids.size(0)
                pbar.set_postfix({"samples": num_processed})

            except Exception as e:
                if self.verbose:
                    print(f"[NTK-Dynamics] Error: {e}")
                continue
            finally:
                self.model.zero_grad()
                for i in range(self.num_layers):
                    self.activations[i] = []
                    self.gradients[i] = []

        self._remove_hooks()

        from pruning.distributed_utils import all_reduce_dict
        all_reduce_dict(dynamics_scores)

        if self.verbose:
            print(f"[NTK-Dynamics] Processed {num_processed} samples")
            for layer_idx in range(min(3, self.num_layers)):
                scores = dynamics_scores[layer_idx]
                print(f"  Layer {layer_idx}: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

        return dynamics_scores

    def _compute_coupling_scores(
        self,
        dataloader,
        num_samples: int = 128
    ) -> Dict[int, torch.Tensor]:
        self._register_hooks_for_ntk()
        self.model.train()

        for param in self.model.parameters():
            param.requires_grad = True

        if self.verbose:
            print(f"\n[NTK-Coupling] Computing coupling scores (sample interaction)...")

        # sample_gradients[layer_idx] = list of [intermediate_size] tensors
        sample_gradients = {i: [] for i in range(self.num_layers)}

        num_processed = 0
        pbar = tqdm(dataloader, desc="[NTK-Coupling] Collecting", disable=not self.verbose)

        for batch in pbar:
            if num_processed >= num_samples:
                break

            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                input_ids = batch.get('input_ids')
                attention_mask = batch.get('attention_mask')
            else:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device) if len(batch) > 1 else None

            labels = input_ids.clone()
            batch_size = input_ids.size(0)

            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()

                if num_processed == 0 and self.verbose:
                    collected_layers = sum(1 for i in range(self.num_layers) if self.gradients[i])
                    print(f"[NTK-Coupling] Gradient collected for {collected_layers}/{self.num_layers} layers")

                for layer_idx in range(self.num_layers):
                    if self.gradients[layer_idx]:
                        grad = self.gradients[layer_idx][-1]  # [batch, seq, intermediate_size]
                        # flatten seq dimension: [batch, seq*intermediate_size] -> take mean over seq
                        grad_per_sample = grad.mean(dim=1)  # [batch, intermediate_size]
                        for b in range(batch_size):
                            if num_processed + b < num_samples:
                                sample_gradients[layer_idx].append(grad_per_sample[b].cpu())

                num_processed += batch_size
                pbar.set_postfix({"samples": min(num_processed, num_samples)})

            except Exception as e:
                if self.verbose:
                    print(f"[NTK-Coupling] Error: {e}")
                continue
            finally:
                self.model.zero_grad()
                for i in range(self.num_layers):
                    self.activations[i] = []
                    self.gradients[i] = []

        self._remove_hooks()

        from pruning.distributed_utils import is_distributed, all_gather_cat_variable
        if is_distributed():
            for layer_idx in range(self.num_layers):
                if sample_gradients[layer_idx]:
                    local_G = torch.stack(sample_gradients[layer_idx])  # [local_N, D]
                    global_G = all_gather_cat_variable(local_G)  # [total_N, D]
                    sample_gradients[layer_idx] = list(global_G)

        if self.verbose:
            print(f"[NTK-Coupling] Computing kernel off-diagonal norms...")

        coupling_scores = {}

        for layer_idx in tqdm(range(self.num_layers), desc="[NTK-Coupling] Layers", disable=not self.verbose):
            if not sample_gradients[layer_idx]:
                coupling_scores[layer_idx] = torch.zeros(self.intermediate_size)
                continue

            # G matrix: [num_samples, intermediate_size]
            G = torch.stack(sample_gradients[layer_idx])  # [N, D]
            N, D = G.shape

            G_norm = torch.norm(G, dim=0, keepdim=True)  # [1, D]
            G_norm = G_norm.clamp(min=1e-8)
            G_normalized = G / G_norm  # [N, D]

            if self.verbose and layer_idx < 3:
                print(f"[NTK-Coupling] Layer {layer_idx}: G shape={G.shape}, "
                      f"G_raw mean={G.abs().mean():.6e}, G_norm mean={G_normalized.abs().mean():.6e}")

            # coupling = ||K_j - diag(K_j)||_F²
            neuron_coupling = torch.zeros(D)

            for j in range(D):
                g_j = G_normalized[:, j:j+1]
                K_j = g_j @ g_j.T  # [N, N]
                # off-diagonal: K_j - diag(diag(K_j))
                off_diag = K_j - torch.diag(torch.diag(K_j))
                coupling = (off_diag ** 2).sum().item()
                neuron_coupling[j] = coupling

            coupling_scores[layer_idx] = neuron_coupling

            del G, G_normalized

        if self.verbose:
            print(f"[NTK-Coupling] Processed {num_processed} samples")
            for layer_idx in range(min(3, self.num_layers)):
                scores = coupling_scores[layer_idx]
                print(f"  Layer {layer_idx}: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

        return coupling_scores

    def _compute_head_dynamics_scores(
        self,
        dataloader,
        num_samples: int = 128,
        head_mode: str = "o"
    ) -> Dict[int, torch.Tensor]:
        self._register_head_hooks(head_mode)
        self.model.train()

        for param in self.model.parameters():
            param.requires_grad = True

        num_heads = self.config.num_attention_heads
        head_dim = self.hidden_size // num_heads

        if self.verbose:
            print(f"\n[NTK-Head-Dynamics] Computing head dynamics scores...")
            print(f"[NTK-Head-Dynamics] Mode: {head_mode}, Heads: {num_heads}")

        dynamics_scores = {i: torch.zeros(num_heads, device='cpu')
                          for i in range(self.num_layers)}

        num_processed = 0
        pbar = tqdm(dataloader, desc="[NTK-Head-Dynamics]", disable=not self.verbose)

        for batch in pbar:
            if num_processed >= num_samples:
                break

            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                input_ids = batch.get('input_ids')
                attention_mask = batch.get('attention_mask')
            else:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device) if len(batch) > 1 else None

            labels = input_ids.clone()

            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()

                if num_processed == 0 and self.verbose:
                    collected_layers = sum(1 for i in range(self.num_layers) if self.head_gradients[i])
                    print(f"[NTK-Head-Dynamics] Gradient collected for {collected_layers}/{self.num_layers} layers")

                for layer_idx in range(self.num_layers):
                    if self.head_gradients[layer_idx]:
                        grad = self.head_gradients[layer_idx][-1]  # [batch, seq, num_heads * head_dim]
                        batch_size, seq_len, hidden = grad.shape
                        grad_reshaped = grad.view(batch_size, seq_len, num_heads, head_dim)
                        # gradient norm² per head: sum over batch, seq, head_dim
                        grad_norm_sq = (grad_reshaped ** 2).sum(dim=(0, 1, 3))  # [num_heads]
                        dynamics_scores[layer_idx] += grad_norm_sq.cpu()

                num_processed += input_ids.size(0)
                pbar.set_postfix({"samples": num_processed})

            except Exception as e:
                if self.verbose:
                    print(f"[NTK-Head-Dynamics] Error: {e}")
                continue
            finally:
                self.model.zero_grad()
                for i in range(self.num_layers):
                    self.head_activations[i] = []
                    self.head_gradients[i] = []

        self._remove_hooks()

        if self.verbose:
            print(f"[NTK-Head-Dynamics] Processed {num_processed} samples")
            for layer_idx in range(min(3, self.num_layers)):
                scores = dynamics_scores[layer_idx]
                print(f"  Layer {layer_idx}: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

        return dynamics_scores

    def _compute_head_coupling_scores(
        self,
        dataloader,
        num_samples: int = 128,
        head_mode: str = "o"
    ) -> Dict[int, torch.Tensor]:
        self._register_head_hooks(head_mode)
        self.model.train()

        for param in self.model.parameters():
            param.requires_grad = True

        num_heads = self.config.num_attention_heads
        head_dim = self.hidden_size // num_heads

        if self.verbose:
            print(f"\n[NTK-Head-Coupling] Computing head coupling scores...")
            print(f"[NTK-Head-Coupling] Mode: {head_mode}, Heads: {num_heads}")

        sample_gradients = {i: [] for i in range(self.num_layers)}

        num_processed = 0
        pbar = tqdm(dataloader, desc="[NTK-Head-Coupling] Collecting", disable=not self.verbose)

        for batch in pbar:
            if num_processed >= num_samples:
                break

            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                input_ids = batch.get('input_ids')
                attention_mask = batch.get('attention_mask')
            else:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device) if len(batch) > 1 else None

            labels = input_ids.clone()
            batch_size = input_ids.size(0)

            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()

                if num_processed == 0 and self.verbose:
                    collected_layers = sum(1 for i in range(self.num_layers) if self.head_gradients[i])
                    print(f"[NTK-Head-Coupling] Gradient collected for {collected_layers}/{self.num_layers} layers")

                for layer_idx in range(self.num_layers):
                    if self.head_gradients[layer_idx]:
                        grad = self.head_gradients[layer_idx][-1]  # [batch, seq, num_heads * head_dim]
                        batch_size_actual, seq_len, hidden = grad.shape
                        grad_reshaped = grad.view(batch_size_actual, seq_len, num_heads, head_dim)
                        grad_per_head = grad_reshaped.mean(dim=1).sum(dim=-1)  # [batch, num_heads]

                        for b in range(batch_size_actual):
                            if num_processed + b < num_samples:
                                sample_gradients[layer_idx].append(grad_per_head[b].cpu())

                num_processed += batch_size
                pbar.set_postfix({"samples": min(num_processed, num_samples)})

            except Exception as e:
                if self.verbose:
                    print(f"[NTK-Head-Coupling] Error: {e}")
                continue
            finally:
                self.model.zero_grad()
                for i in range(self.num_layers):
                    self.head_activations[i] = []
                    self.head_gradients[i] = []

        self._remove_hooks()

        if self.verbose:
            print(f"[NTK-Head-Coupling] Computing kernel off-diagonal norms...")

        coupling_scores = {}

        for layer_idx in tqdm(range(self.num_layers), desc="[NTK-Head-Coupling] Layers", disable=not self.verbose):
            if not sample_gradients[layer_idx]:
                coupling_scores[layer_idx] = torch.zeros(num_heads)
                continue

            # G matrix: [num_samples, num_heads]
            G = torch.stack(sample_gradients[layer_idx])  # [N, H]
            N, H = G.shape

            G_norm = torch.norm(G, dim=0, keepdim=True)  # [1, H]
            G_norm = G_norm.clamp(min=1e-8)
            G_normalized = G / G_norm  # [N, H]

            if self.verbose and layer_idx < 3:
                print(f"[NTK-Head-Coupling] Layer {layer_idx}: G shape={G.shape}, "
                      f"G_raw mean={G.abs().mean():.6e}, G_norm mean={G_normalized.abs().mean():.6e}")

            head_coupling = torch.zeros(H)

            for h in range(H):
                g_h = G_normalized[:, h:h+1]
                K_h = g_h @ g_h.T  # [N, N]
                off_diag = K_h - torch.diag(torch.diag(K_h))
                coupling = (off_diag ** 2).sum().item()
                head_coupling[h] = coupling

            coupling_scores[layer_idx] = head_coupling

            del G

        if self.verbose:
            print(f"[NTK-Head-Coupling] Processed {num_processed} samples")
            for layer_idx in range(min(3, self.num_layers)):
                scores = coupling_scores[layer_idx]
                print(f"  Layer {layer_idx}: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

        return coupling_scores

    # ==================== Dimension NTK Functions ====================

    def _compute_dimension_dynamics_scores(
        self,
        dataloader,
        num_samples: int = 128,
        head_mode: str = "o"
    ) -> Dict[int, torch.Tensor]:
        self._register_head_hooks(head_mode)
        self.model.train()

        for param in self.model.parameters():
            param.requires_grad = True

        num_heads = self.config.num_attention_heads
        head_dim = self.hidden_size // num_heads

        if self.verbose:
            print(f"\n[NTK-Dim-Dynamics] Computing dimension dynamics scores...")
            print(f"[NTK-Dim-Dynamics] Mode: {head_mode}, Head dim: {head_dim}")

        accumulated = {i: torch.zeros(head_dim, device='cpu')
                       for i in range(self.num_layers)}

        num_processed = 0
        pbar = tqdm(dataloader, desc="[NTK-Dim-Dynamics] Collecting", disable=not self.verbose)

        for batch in pbar:
            if num_processed >= num_samples:
                break

            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                input_ids = batch.get('input_ids')
                attention_mask = batch.get('attention_mask')
            else:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device) if len(batch) > 1 else None

            labels = input_ids.clone()
            batch_size = input_ids.size(0)

            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()

                for layer_idx in range(self.num_layers):
                    if self.head_gradients[layer_idx]:
                        grad = self.head_gradients[layer_idx][-1]  # [batch, seq, hidden]
                        batch_size_actual, seq_len, hidden = grad.shape
                        # Reshape to [batch, seq, num_heads, head_dim]
                        grad_reshaped = grad.view(batch_size_actual, seq_len, num_heads, head_dim)
                        # Mean over batch, seq, heads → gradient norm² per dimension
                        grad_per_dim = grad_reshaped.mean(dim=(0, 1, 2))  # [head_dim]
                        # Accumulate ||grad||² per dimension
                        accumulated[layer_idx] += (grad_per_dim ** 2).cpu()

                num_processed += batch_size
                pbar.set_postfix({"samples": min(num_processed, num_samples)})

            except Exception as e:
                if self.verbose:
                    print(f"[NTK-Dim-Dynamics] Error: {e}")
                continue
            finally:
                self.model.zero_grad()
                for i in range(self.num_layers):
                    self.head_activations[i] = []
                    self.head_gradients[i] = []

        self._remove_hooks()

        from pruning.distributed_utils import all_reduce_dict
        all_reduce_dict(accumulated)

        if self.verbose:
            print(f"[NTK-Dim-Dynamics] Processed {num_processed} samples")
            for layer_idx in range(min(3, self.num_layers)):
                scores = accumulated[layer_idx]
                print(f"  Layer {layer_idx}: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

        return accumulated

    def _compute_dimension_coupling_scores(
        self,
        dataloader,
        num_samples: int = 128,
        head_mode: str = "o"
    ) -> Dict[int, torch.Tensor]:
        self._register_head_hooks(head_mode)
        self.model.train()

        for param in self.model.parameters():
            param.requires_grad = True

        num_heads = self.config.num_attention_heads
        head_dim = self.hidden_size // num_heads

        if self.verbose:
            print(f"\n[NTK-Dim-Coupling] Computing dimension coupling scores...")
            print(f"[NTK-Dim-Coupling] Mode: {head_mode}, Head dim: {head_dim}")

        sample_gradients = {i: [] for i in range(self.num_layers)}

        num_processed = 0
        pbar = tqdm(dataloader, desc="[NTK-Dim-Coupling] Collecting", disable=not self.verbose)

        for batch in pbar:
            if num_processed >= num_samples:
                break

            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                input_ids = batch.get('input_ids')
                attention_mask = batch.get('attention_mask')
            else:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device) if len(batch) > 1 else None

            labels = input_ids.clone()
            batch_size = input_ids.size(0)

            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()

                if num_processed == 0 and self.verbose:
                    collected_layers = sum(1 for i in range(self.num_layers) if self.head_gradients[i])
                    print(f"[NTK-Dim-Coupling] Gradient collected for {collected_layers}/{self.num_layers} layers")

                for layer_idx in range(self.num_layers):
                    if self.head_gradients[layer_idx]:
                        grad = self.head_gradients[layer_idx][-1]  # [batch, seq, hidden]
                        batch_size_actual, seq_len, hidden = grad.shape
                        # Reshape to [batch, seq, num_heads, head_dim]
                        grad_reshaped = grad.view(batch_size_actual, seq_len, num_heads, head_dim)
                        # Mean over seq, heads → [batch, head_dim]
                        grad_per_dim = grad_reshaped.mean(dim=(1, 2))  # [batch, head_dim]

                        for b in range(batch_size_actual):
                            if num_processed + b < num_samples:
                                sample_gradients[layer_idx].append(grad_per_dim[b].cpu())

                num_processed += batch_size
                pbar.set_postfix({"samples": min(num_processed, num_samples)})

            except Exception as e:
                if self.verbose:
                    print(f"[NTK-Dim-Coupling] Error: {e}")
                continue
            finally:
                self.model.zero_grad()
                for i in range(self.num_layers):
                    self.head_activations[i] = []
                    self.head_gradients[i] = []

        self._remove_hooks()

        from pruning.distributed_utils import is_distributed, all_gather_cat_variable
        if is_distributed():
            for layer_idx in range(self.num_layers):
                if sample_gradients[layer_idx]:
                    local_G = torch.stack(sample_gradients[layer_idx])  # [local_N, head_dim]
                    global_G = all_gather_cat_variable(local_G)  # [total_N, head_dim]
                    sample_gradients[layer_idx] = [global_G[i] for i in range(global_G.shape[0])]

        if self.verbose:
            print(f"[NTK-Dim-Coupling] Computing kernel off-diagonal norms...")

        coupling_scores = {}

        for layer_idx in tqdm(range(self.num_layers), desc="[NTK-Dim-Coupling] Layers", disable=not self.verbose):
            if not sample_gradients[layer_idx]:
                coupling_scores[layer_idx] = torch.zeros(head_dim)
                continue

            # G matrix: [num_samples, head_dim]
            G = torch.stack(sample_gradients[layer_idx])  # [N, D]
            N, D = G.shape

            G_norm = torch.norm(G, dim=0, keepdim=True)  # [1, D]
            G_norm = G_norm.clamp(min=1e-8)
            G_normalized = G / G_norm  # [N, D]

            if self.verbose and layer_idx < 3:
                print(f"[NTK-Dim-Coupling] Layer {layer_idx}: G shape={G.shape}, "
                      f"G_raw mean={G.abs().mean():.6e}, G_norm mean={G_normalized.abs().mean():.6e}")

            dim_coupling = torch.zeros(D)

            for d in range(D):
                g_d = G_normalized[:, d:d+1]
                K_d = g_d @ g_d.T  # [N, N]
                off_diag = K_d - torch.diag(torch.diag(K_d))
                coupling = (off_diag ** 2).sum().item()
                dim_coupling[d] = coupling

            coupling_scores[layer_idx] = dim_coupling

            del G

        if self.verbose:
            print(f"[NTK-Dim-Coupling] Processed {num_processed} samples")
            for layer_idx in range(min(3, self.num_layers)):
                scores = coupling_scores[layer_idx]
                print(f"  Layer {layer_idx}: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

        return coupling_scores

    def _apply_ntk_modifiers(
        self,
        base_scores: Dict[int, torch.Tensor],
        dynamics_scores: Dict[int, torch.Tensor] = None,
        coupling_scores: Dict[int, torch.Tensor] = None
    ) -> Dict[int, torch.Tensor]:
        def safe_normalize(scores: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
            cleaned = {}
            for k, v in scores.items():
                v_clean = v.clone()
                v_clean = torch.where(torch.isnan(v_clean), torch.zeros_like(v_clean), v_clean)
                v_clean = torch.where(torch.isinf(v_clean), torch.zeros_like(v_clean), v_clean)
                cleaned[k] = v_clean

            all_scores = torch.cat([s.flatten() for s in cleaned.values()])
            min_val = all_scores.min()
            max_val = all_scores.max()

            if max_val - min_val < 1e-8:
                return {k: torch.ones_like(v) for k, v in cleaned.items()}

            normalized = {}
            for layer_idx, s in cleaned.items():
                norm_s = (s - min_val) / (max_val - min_val)
                norm_s = torch.where(torch.isnan(norm_s), torch.zeros_like(norm_s), norm_s)
                normalized[layer_idx] = norm_s

            return normalized

        base_norm = safe_normalize(base_scores)

        dyn_norm = safe_normalize(dynamics_scores) if dynamics_scores is not None else None
        coup_norm = safe_normalize(coupling_scores) if coupling_scores is not None else None

        modified = {}
        for layer_idx in base_scores.keys():
            score = base_norm[layer_idx].clone()

            if dyn_norm is not None:
                score = score * dyn_norm[layer_idx]

            if coup_norm is not None:
                score = score * coup_norm[layer_idx]

            if torch.isnan(score).any():
                if self.verbose:
                    print(f"[Warning] Layer {layer_idx}: NaN detected, using base scores")
                score = base_scores[layer_idx].clone()
                score_min, score_max = score.min(), score.max()
                if score_max - score_min > 1e-8:
                    score = (score - score_min) / (score_max - score_min)
                else:
                    score = torch.ones_like(score)

            modified[layer_idx] = score

        return modified

    def _get_mlp_module(self, layer_idx: int):
        try:
            if hasattr(self.model, 'model'):
                base = self.model.model
            else:
                base = self.model
            if hasattr(base, 'model'):
                base = base.model
            if hasattr(base, 'layers'):
                return base.layers[layer_idx].mlp
        except (AttributeError, IndexError):
            pass
        return None

    def _register_hooks_for_ntk(self):
        self.activations = {i: [] for i in range(self.num_layers)}
        self.gradients = {i: [] for i in range(self.num_layers)}
        self.hooks = []

        for layer_idx in range(self.num_layers):
            mlp = self._get_mlp_module(layer_idx)
            if mlp is None:
                continue

            down_proj = get_down_proj_module(mlp)
            if down_proj is None:
                continue
            if hasattr(down_proj, 'base_layer'):
                target_module = down_proj.base_layer
            else:
                target_module = down_proj

            def make_forward_hook(idx):
                def hook(module, input, output):
                    if len(input) > 0 and input[0] is not None:
                        self.activations[idx].append(input[0].detach())
                return hook

            def make_backward_hook(idx):
                def hook(module, grad_input, grad_output):
                    if grad_input is not None and len(grad_input) > 0 and grad_input[0] is not None:
                        self.gradients[idx].append(grad_input[0].detach())
                return hook

            h1 = target_module.register_forward_hook(make_forward_hook(layer_idx))
            h2 = target_module.register_full_backward_hook(make_backward_hook(layer_idx))
            self.hooks.append(h1)
            self.hooks.append(h2)

    def _register_hooks(self):
        self.activations = {i: [] for i in range(self.num_layers)}
        self.gradients = {i: [] for i in range(self.num_layers)}
        self.weights = {}

        if self.ffn_mode == "all":
            self.up_activations = {i: [] for i in range(self.num_layers)}
            self.up_gradients = {i: [] for i in range(self.num_layers)}
            self.up_weights = {}
            self.gate_activations = {i: [] for i in range(self.num_layers)}
            self.gate_gradients = {i: [] for i in range(self.num_layers)}
            self.gate_weights = {}

        self.q_activations = {i: [] for i in range(self.num_layers)}
        self.k_activations = {i: [] for i in range(self.num_layers)}

        for layer_idx in range(self.num_layers):
            mlp = self._get_mlp_module(layer_idx)
            if mlp is None:
                continue

            if self.ffn_mode in ["down", "all"]:
                down_proj = get_down_proj_module(mlp)
                if down_proj is None:
                    continue
                if hasattr(down_proj, 'base_layer'):
                    target_module = down_proj.base_layer
                else:
                    target_module = down_proj

                self.weights[layer_idx] = target_module.weight.data.detach()

                def make_forward_hook(idx):
                    def hook(module, input, output):
                        if len(input) > 0 and input[0] is not None:
                            self.activations[idx].append(input[0].detach())
                    return hook

                def make_backward_hook(idx):
                    def hook(module, grad_input, grad_output):
                        if grad_input is not None and len(grad_input) > 0 and grad_input[0] is not None:
                            self.gradients[idx].append(grad_input[0].detach())
                    return hook

                h1 = target_module.register_forward_hook(make_forward_hook(layer_idx))
                h2 = target_module.register_full_backward_hook(make_backward_hook(layer_idx))
                self.hooks.append(h1)
                self.hooks.append(h2)

            if self.ffn_mode in ["up", "all"]:
                up_proj = getattr(mlp, 'up_proj', None) or getattr(mlp, 'fc1', None)
                if up_proj is None:
                    continue
                if hasattr(up_proj, 'base_layer'):
                    up_module = up_proj.base_layer
                else:
                    up_module = up_proj

                if self.ffn_mode == "all":
                    self.up_weights[layer_idx] = up_module.weight.data.detach()
                elif self.ffn_mode == "up":
                    self.weights[layer_idx] = up_module.weight.data.detach()

                if self.ffn_mode == "up":
                    def make_up_forward_hook(idx):
                        def hook(module, input, output):
                            if output is not None:
                                self.activations[idx].append(output.detach())
                        return hook

                    def make_up_backward_hook(idx):
                        def hook(module, grad_input, grad_output):
                            if grad_output is not None and len(grad_output) > 0 and grad_output[0] is not None:
                                self.gradients[idx].append(grad_output[0].detach())
                        return hook
                else:
                    def make_up_forward_hook(idx):
                        def hook(module, input, output):
                            if output is not None:
                                self.up_activations[idx].append(output.detach())
                        return hook

                    def make_up_backward_hook(idx):
                        def hook(module, grad_input, grad_output):
                            if grad_output is not None and len(grad_output) > 0 and grad_output[0] is not None:
                                self.up_gradients[idx].append(grad_output[0].detach())
                        return hook

                h3 = up_module.register_forward_hook(make_up_forward_hook(layer_idx))
                h4 = up_module.register_full_backward_hook(make_up_backward_hook(layer_idx))
                self.hooks.append(h3)
                self.hooks.append(h4)

            if self.ffn_mode == "all" and hasattr(mlp, 'gate_proj'):
                gate_proj = mlp.gate_proj
                if hasattr(gate_proj, 'base_layer'):
                    gate_module = gate_proj.base_layer
                else:
                    gate_module = gate_proj

                self.gate_weights[layer_idx] = gate_module.weight.data.detach()

                def make_gate_forward_hook(idx):
                    def hook(module, input, output):
                        if output is not None:
                            self.gate_activations[idx].append(output.detach())
                    return hook

                def make_gate_backward_hook(idx):
                    def hook(module, grad_input, grad_output):
                        if grad_output is not None and len(grad_output) > 0 and grad_output[0] is not None:
                            self.gate_gradients[idx].append(grad_output[0].detach())
                    return hook

                h5 = gate_module.register_forward_hook(make_gate_forward_hook(layer_idx))
                h6 = gate_module.register_full_backward_hook(make_gate_backward_hook(layer_idx))
                self.hooks.append(h5)
                self.hooks.append(h6)

            attn = self._get_attn_module(layer_idx)
            if attn is not None:
                # q_proj hook
                q_proj = attn.q_proj
                if hasattr(q_proj, 'base_layer'):
                    q_module = q_proj.base_layer
                else:
                    q_module = q_proj

                def make_q_forward_hook(idx):
                    def hook(module, input, output):
                        if output is not None:
                            # output: [batch, seq, num_heads * head_dim]
                            self.q_activations[idx].append(output.detach())
                    return hook

                h_q = q_module.register_forward_hook(make_q_forward_hook(layer_idx))
                self.hooks.append(h_q)

                # k_proj hook
                k_proj = attn.k_proj
                if hasattr(k_proj, 'base_layer'):
                    k_module = k_proj.base_layer
                else:
                    k_module = k_proj

                def make_k_forward_hook(idx):
                    def hook(module, input, output):
                        if output is not None:
                            # output: [batch, seq, num_kv_heads * head_dim]
                            self.k_activations[idx].append(output.detach())
                    return hook

                h_k = k_module.register_forward_hook(make_k_forward_hook(layer_idx))
                self.hooks.append(h_k)

    def _get_attn_module(self, layer_idx: int):
        try:
            if hasattr(self.model, 'base_model'):
                layers = self.model.base_model.model.layers
            elif hasattr(self.model, 'model'):
                layers = self.model.model.layers
            else:
                layers = self.model.layers
            return layers[layer_idx].self_attn
        except:
            return None

    def compute_qk_activation_importance(self) -> Dict[int, torch.Tensor]:
        num_heads = self.config.num_attention_heads
        num_kv_heads = getattr(self.config, 'num_key_value_heads', num_heads)
        head_dim = self.hidden_size // num_heads
        num_queries_per_kv = num_heads // num_kv_heads

        head_importance = {}

        for layer_idx in range(self.num_layers):
            if not self.q_activations.get(layer_idx) or not self.k_activations.get(layer_idx):
                head_importance[layer_idx] = torch.zeros(num_heads)
                continue

            # Q activations: [batch, seq, num_heads * head_dim]
            q_acts = torch.cat(self.q_activations[layer_idx], dim=0)  # [total_tokens, seq, q_dim]
            k_acts = torch.cat(self.k_activations[layer_idx], dim=0)  # [total_tokens, seq, k_dim]

            # Per-head activation magnitude
            # Q: [total_tokens, seq, num_heads, head_dim] -> [num_heads]
            q_reshaped = q_acts.view(-1, q_acts.shape[1], num_heads, head_dim)
            q_per_head = q_reshaped.abs().mean(dim=(0, 1, 3))  # [num_heads]

            # K: [total_tokens, seq, num_kv_heads, head_dim] -> [num_kv_heads]
            k_reshaped = k_acts.view(-1, k_acts.shape[1], num_kv_heads, head_dim)
            k_per_kv_head = k_reshaped.abs().mean(dim=(0, 1, 3))  # [num_kv_heads]

            k_expanded = k_per_kv_head.repeat_interleave(num_queries_per_kv)  # [num_heads]

            # Q×K element-wise (activation only)
            qk_importance = q_per_head * k_expanded

            head_importance[layer_idx] = qk_importance.cpu()

        return head_importance

    def _compute_weight_norm_head_scores(
        self,
        p: float = 2.0
    ) -> Dict[int, torch.Tensor]:
        num_heads = self.config.num_attention_heads
        head_dim = self.hidden_size // num_heads

        head_scores = {}

        for layer_idx in range(self.num_layers):
            attn = self._get_attention_module(layer_idx)
            if attn is None:
                head_scores[layer_idx] = torch.ones(num_heads)
                continue

            # Q: [hidden_size, hidden_size] -> [hidden_size, num_heads, head_dim]
            # K: [hidden_size, hidden_size] -> [hidden_size, num_heads, head_dim]
            # V: [hidden_size, hidden_size] -> [hidden_size, num_heads, head_dim]
            # O: [hidden_size, hidden_size] -> [num_heads, head_dim, hidden_size]
            q_proj = attn.q_proj.weight if not hasattr(attn.q_proj, 'base_layer') else attn.q_proj.base_layer.weight
            k_proj = attn.k_proj.weight if not hasattr(attn.k_proj, 'base_layer') else attn.k_proj.base_layer.weight
            v_proj = attn.v_proj.weight if not hasattr(attn.v_proj, 'base_layer') else attn.v_proj.base_layer.weight
            o_proj_mod = get_attn_output_proj(attn)
            o_proj = o_proj_mod.weight if not hasattr(o_proj_mod, 'base_layer') else o_proj_mod.base_layer.weight

            # Q, K, V: [num_heads * head_dim, hidden_size] -> reshape to [num_heads, head_dim, hidden_size]
            q_reshaped = q_proj.detach().float().view(num_heads, head_dim, -1)
            k_reshaped = k_proj.detach().float().view(num_heads, head_dim, -1)
            v_reshaped = v_proj.detach().float().view(num_heads, head_dim, -1)

            # O: [hidden_size, num_heads * head_dim] -> reshape to [hidden_size, num_heads, head_dim]
            o_reshaped = o_proj.detach().float().view(-1, num_heads, head_dim)

            if p == 2.0:
                q_norms = torch.sqrt((q_reshaped ** 2).sum(dim=(1, 2)))  # [num_heads]
                k_norms = torch.sqrt((k_reshaped ** 2).sum(dim=(1, 2)))
                v_norms = torch.sqrt((v_reshaped ** 2).sum(dim=(1, 2)))
                o_norms = torch.sqrt((o_reshaped ** 2).sum(dim=(0, 2)))
            else:
                q_norms = (q_reshaped.abs() ** p).sum(dim=(1, 2)) ** (1/p)
                k_norms = (k_reshaped.abs() ** p).sum(dim=(1, 2)) ** (1/p)
                v_norms = (v_reshaped.abs() ** p).sum(dim=(1, 2)) ** (1/p)
                o_norms = (o_reshaped.abs() ** p).sum(dim=(0, 2)) ** (1/p)

            layer_importance = (q_norms + k_norms + v_norms + o_norms).cpu()
            head_scores[layer_idx] = layer_importance

        if self.verbose:
            print(f"\n[Weight-Norm-Head] Final head scores:")
            for layer_idx in range(min(3, self.num_layers)):
                scores = head_scores[layer_idx]
                print(f"  Layer {layer_idx}: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

        return head_scores

    def _compute_weight_taylor_head_scores(
        self,
        dataloader,
        num_samples: int = 128
    ) -> Dict[int, torch.Tensor]:
        num_heads = self.config.num_attention_heads
        head_dim = self.hidden_size // num_heads

        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True

        accumulated = {i: torch.zeros(num_heads, device='cpu') for i in range(self.num_layers)}

        num_processed = 0
        pbar = tqdm(dataloader, desc="[Weight-Taylor-Head] Calibration", disable=not self.verbose)

        for batch in pbar:
            if num_processed >= num_samples:
                break

            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                input_ids = batch.get('input_ids')
                attention_mask = batch.get('attention_mask')
            else:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device) if len(batch) > 1 else None

            labels = input_ids.clone()

            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()

                for layer_idx in range(self.num_layers):
                    attn = self._get_attention_module(layer_idx)
                    if attn is None:
                        continue

                    layer_importance = torch.zeros(num_heads, device='cpu')

                    for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                        proj = getattr(attn, proj_name)
                        if hasattr(proj, 'base_layer'):
                            proj = proj.base_layer

                        weight = proj.weight.data.detach().float()
                        if proj.weight.grad is not None:
                            grad = proj.weight.grad.detach().float()

                            taylor_elementwise = weight.abs() * grad.abs()

                            if proj_name == 'o_proj':
                                # O: [hidden_size, num_heads * head_dim]
                                # reshape to [hidden_size, num_heads, head_dim] -> sum over (0, 2)
                                taylor_reshaped = taylor_elementwise.view(-1, num_heads, head_dim)
                                taylor_scores = taylor_reshaped.sum(dim=(0, 2))  # [num_heads]
                            else:
                                # Q, K, V: [num_heads * head_dim, hidden_size]
                                # reshape to [num_heads, head_dim, hidden_size] -> sum over (1, 2)
                                taylor_reshaped = taylor_elementwise.view(num_heads, head_dim, -1)
                                taylor_scores = taylor_reshaped.sum(dim=(1, 2))  # [num_heads]

                            layer_importance += taylor_scores.cpu()

                    accumulated[layer_idx] += layer_importance

                num_processed += input_ids.size(0)
                pbar.set_postfix({"samples": num_processed})

            except Exception as e:
                if self.verbose:
                    print(f"[Weight-Taylor-Head] Error: {e}")
                continue
            finally:
                self.model.zero_grad(set_to_none=True)
                del input_ids, attention_mask, labels
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if num_processed > 0:
            batch_count = num_processed / dataloader.batch_size
            for layer_idx in range(self.num_layers):
                accumulated[layer_idx] /= batch_count

        if self.verbose:
            print(f"[Weight-Taylor-Head] Processed {num_processed} samples")
            print(f"\n[Weight-Taylor-Head] Final head scores:")
            for layer_idx in range(min(3, self.num_layers)):
                scores = accumulated[layer_idx]
                print(f"  Layer {layer_idx}: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

        # DEBUG: Always print to verify function is called
        print(f"[DEBUG] _compute_weight_taylor_head_scores called, samples={num_processed}")
        print(f"[DEBUG] Head Layer 0 scores: min={accumulated[0].min():.6f}, max={accumulated[0].max():.6f}, mean={accumulated[0].mean():.6f}")

        return accumulated

    def _compute_weight_ganda_head_scores(
        self,
        dataloader,
        num_samples: int = 128
    ) -> Dict[int, torch.Tensor]:
        num_heads = self.config.num_attention_heads
        head_dim = self.hidden_size // num_heads

        self.q_activations = {i: [] for i in range(self.num_layers)}
        self.k_activations = {i: [] for i in range(self.num_layers)}
        self.v_activations = {i: [] for i in range(self.num_layers)}
        self.o_activations = {i: [] for i in range(self.num_layers)}
        self.hooks = []

        for layer_idx in range(self.num_layers):
            attn = self._get_attention_module(layer_idx)
            if attn is None:
                continue

            # Q proj output hook
            q_proj = attn.q_proj
            q_target = q_proj.base_layer if hasattr(q_proj, 'base_layer') else q_proj

            def make_q_hook(idx):
                def hook(module, input, output):
                    if output is not None:
                        output.requires_grad_(True)
                        output.retain_grad()
                        self.q_activations[idx].append(output)
                return hook
            self.hooks.append(q_target.register_forward_hook(make_q_hook(layer_idx)))

            # K proj output hook
            k_proj = attn.k_proj
            k_target = k_proj.base_layer if hasattr(k_proj, 'base_layer') else k_proj

            def make_k_hook(idx):
                def hook(module, input, output):
                    if output is not None:
                        output.requires_grad_(True)
                        output.retain_grad()
                        self.k_activations[idx].append(output)
                return hook
            self.hooks.append(k_target.register_forward_hook(make_k_hook(layer_idx)))

            # V proj output hook
            v_proj = attn.v_proj
            v_target = v_proj.base_layer if hasattr(v_proj, 'base_layer') else v_proj

            def make_v_hook(idx):
                def hook(module, input, output):
                    if output is not None:
                        output.requires_grad_(True)
                        output.retain_grad()
                        self.v_activations[idx].append(output)
                return hook
            self.hooks.append(v_target.register_forward_hook(make_v_hook(layer_idx)))

            # O proj input hook (attention output) — o_proj or dense
            o_proj = get_attn_output_proj(attn)
            if o_proj is None:
                continue
            o_target = o_proj.base_layer if hasattr(o_proj, 'base_layer') else o_proj

            def make_o_hook(idx):
                def hook(module, input, output):
                    if len(input) > 0 and input[0] is not None:
                        inp = input[0]
                        inp.requires_grad_(True)
                        inp.retain_grad()
                        self.o_activations[idx].append(inp)
                return hook
            self.hooks.append(o_target.register_forward_hook(make_o_hook(layer_idx)))

        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True

        accumulated = {i: torch.zeros(num_heads, device='cpu') for i in range(self.num_layers)}

        num_processed = 0
        pbar = tqdm(dataloader, desc="[Activation-GandA-Head] Calibration", disable=not self.verbose)

        for batch in pbar:
            if num_processed >= num_samples:
                break

            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                input_ids = batch.get('input_ids')
                attention_mask = batch.get('attention_mask')
            else:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device) if len(batch) > 1 else None

            labels = input_ids.clone()

            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()

                for layer_idx in range(self.num_layers):
                    layer_importance = torch.zeros(num_heads, device='cpu')

                    if self.q_activations[layer_idx]:
                        q_act = self.q_activations[layer_idx][-1]
                        if q_act.grad is not None:
                            q_grad = q_act.grad.detach().float()
                            q_val = q_act.detach().float()
                            # [batch, seq, num_heads * head_dim] -> [batch, seq, num_heads, head_dim]
                            q_grad_reshaped = q_grad.view(q_grad.shape[0], q_grad.shape[1], num_heads, head_dim)
                            q_val_reshaped = q_val.view(q_val.shape[0], q_val.shape[1], num_heads, head_dim)
                            ganda = (q_grad_reshaped.abs() * q_val_reshaped.abs()).sum(dim=(0, 1, 3))  # [num_heads]
                            layer_importance += ganda.cpu()

                    if self.k_activations[layer_idx]:
                        k_act = self.k_activations[layer_idx][-1]
                        if k_act.grad is not None:
                            k_grad = k_act.grad.detach().float()
                            k_val = k_act.detach().float()
                            k_grad_reshaped = k_grad.view(k_grad.shape[0], k_grad.shape[1], num_heads, head_dim)
                            k_val_reshaped = k_val.view(k_val.shape[0], k_val.shape[1], num_heads, head_dim)
                            ganda = (k_grad_reshaped.abs() * k_val_reshaped.abs()).sum(dim=(0, 1, 3))
                            layer_importance += ganda.cpu()

                    if self.v_activations[layer_idx]:
                        v_act = self.v_activations[layer_idx][-1]
                        if v_act.grad is not None:
                            v_grad = v_act.grad.detach().float()
                            v_val = v_act.detach().float()
                            v_grad_reshaped = v_grad.view(v_grad.shape[0], v_grad.shape[1], num_heads, head_dim)
                            v_val_reshaped = v_val.view(v_val.shape[0], v_val.shape[1], num_heads, head_dim)
                            ganda = (v_grad_reshaped.abs() * v_val_reshaped.abs()).sum(dim=(0, 1, 3))
                            layer_importance += ganda.cpu()

                    if self.o_activations[layer_idx]:
                        o_act = self.o_activations[layer_idx][-1]
                        if o_act.grad is not None:
                            o_grad = o_act.grad.detach().float()
                            o_val = o_act.detach().float()
                            o_grad_reshaped = o_grad.view(o_grad.shape[0], o_grad.shape[1], num_heads, head_dim)
                            o_val_reshaped = o_val.view(o_val.shape[0], o_val.shape[1], num_heads, head_dim)
                            ganda = (o_grad_reshaped.abs() * o_val_reshaped.abs()).sum(dim=(0, 1, 3))
                            layer_importance += ganda.cpu()

                    accumulated[layer_idx] += layer_importance

                num_processed += input_ids.size(0)
                pbar.set_postfix({"samples": num_processed})

            except Exception as e:
                if self.verbose:
                    print(f"[Activation-GandA-Head] Error: {e}")
                continue
            finally:
                self.model.zero_grad(set_to_none=True)
                for i in range(self.num_layers):
                    self.q_activations[i] = []
                    self.k_activations[i] = []
                    self.v_activations[i] = []
                    self.o_activations[i] = []
                del input_ids, attention_mask, labels
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        if num_processed > 0:
            batch_count = num_processed / dataloader.batch_size
            for layer_idx in range(self.num_layers):
                accumulated[layer_idx] /= batch_count

        if self.verbose:
            print(f"[Activation-GandA-Head] Processed {num_processed} samples")
            print(f"\n[Activation-GandA-Head] Final head scores:")
            for layer_idx in range(min(3, self.num_layers)):
                scores = accumulated[layer_idx]
                print(f"  Layer {layer_idx}: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

        # DEBUG: Always print to verify function is called
        print(f"[DEBUG] _compute_weight_ganda_head_scores called, samples={num_processed}")
        print(f"[DEBUG] Head Layer 0 scores: min={accumulated[0].min():.6f}, max={accumulated[0].max():.6f}, mean={accumulated[0].mean():.6f}")

        return accumulated

    def _compute_weight_wanda_head_scores(
        self,
        dataloader,
        num_samples: int = 128
    ) -> Dict[int, torch.Tensor]:
        num_heads = self.config.num_attention_heads
        num_kv_heads = getattr(self.config, 'num_key_value_heads', num_heads)
        head_dim = self.hidden_size // num_heads
        num_queries_per_kv = num_heads // num_kv_heads
        is_gqa = (num_kv_heads != num_heads)

        if self.verbose:
            print(f"\n[Weight-WANDA-Head] Computing WANDA Head scores: |W| × |X|")
            if is_gqa:
                print(f"[Weight-WANDA-Head] GQA detected: {num_heads} Q heads, {num_kv_heads} KV heads")

        q_input_activations = {i: [] for i in range(self.num_layers)}
        k_input_activations = {i: [] for i in range(self.num_layers)}
        v_input_activations = {i: [] for i in range(self.num_layers)}
        o_input_activations = {i: [] for i in range(self.num_layers)}
        self.hooks = []

        for layer_idx in range(self.num_layers):
            attn = self._get_attention_module(layer_idx)
            if attn is None:
                continue

            # Q proj input hook
            q_proj = attn.q_proj
            q_target = q_proj.base_layer if hasattr(q_proj, 'base_layer') else q_proj

            def make_q_hook(idx):
                def hook(module, input, output):
                    if len(input) > 0 and input[0] is not None:
                        inp = input[0].detach()
                        if inp.dim() == 3:
                            inp = inp.view(-1, inp.size(-1))  # [batch*seq, hidden_size]
                        rms = torch.sqrt((inp ** 2).mean(dim=0))  # [hidden_size]
                        q_input_activations[idx].append(rms.cpu())
                return hook
            self.hooks.append(q_target.register_forward_hook(make_q_hook(layer_idx)))

            # K proj input hook
            k_proj = attn.k_proj
            k_target = k_proj.base_layer if hasattr(k_proj, 'base_layer') else k_proj

            def make_k_hook(idx):
                def hook(module, input, output):
                    if len(input) > 0 and input[0] is not None:
                        inp = input[0].detach()
                        if inp.dim() == 3:
                            inp = inp.view(-1, inp.size(-1))
                        rms = torch.sqrt((inp ** 2).mean(dim=0))
                        k_input_activations[idx].append(rms.cpu())
                return hook
            self.hooks.append(k_target.register_forward_hook(make_k_hook(layer_idx)))

            # V proj input hook
            v_proj = attn.v_proj
            v_target = v_proj.base_layer if hasattr(v_proj, 'base_layer') else v_proj

            def make_v_hook(idx):
                def hook(module, input, output):
                    if len(input) > 0 and input[0] is not None:
                        inp = input[0].detach()
                        if inp.dim() == 3:
                            inp = inp.view(-1, inp.size(-1))
                        rms = torch.sqrt((inp ** 2).mean(dim=0))
                        v_input_activations[idx].append(rms.cpu())
                return hook
            self.hooks.append(v_target.register_forward_hook(make_v_hook(layer_idx)))

            # O proj input hook (attention output) — o_proj or dense
            o_proj = get_attn_output_proj(attn)
            if o_proj is None:
                continue
            o_target = o_proj.base_layer if hasattr(o_proj, 'base_layer') else o_proj

            def make_o_hook(idx):
                def hook(module, input, output):
                    if len(input) > 0 and input[0] is not None:
                        inp = input[0].detach()
                        if inp.dim() == 3:
                            inp = inp.view(-1, inp.size(-1))  # [batch*seq, num_heads * head_dim]
                        rms = torch.sqrt((inp ** 2).mean(dim=0))  # [num_heads * head_dim]
                        o_input_activations[idx].append(rms.cpu())
                return hook
            self.hooks.append(o_target.register_forward_hook(make_o_hook(layer_idx)))

        self.model.eval()
        num_processed = 0
        pbar = tqdm(dataloader, desc="[Weight-WANDA-Head] Collecting activations", disable=not self.verbose)

        with torch.no_grad():
            for batch in pbar:
                if num_processed >= num_samples:
                    break

                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}
                    input_ids = batch.get('input_ids')
                    attention_mask = batch.get('attention_mask')
                else:
                    input_ids = batch[0].to(self.device)
                    attention_mask = batch[1].to(self.device) if len(batch) > 1 else None

                try:
                    _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    num_processed += input_ids.size(0)
                    pbar.set_postfix({"samples": num_processed})
                except Exception as e:
                    if self.verbose:
                        print(f"[Weight-WANDA-Head] Error: {e}")
                    continue

        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        q_act_mean = {}
        k_act_mean = {}
        v_act_mean = {}
        o_act_mean = {}

        for layer_idx in range(self.num_layers):
            if q_input_activations[layer_idx]:
                q_act_mean[layer_idx] = torch.stack(q_input_activations[layer_idx]).mean(dim=0)
            else:
                q_act_mean[layer_idx] = torch.ones(self.hidden_size)

            if k_input_activations[layer_idx]:
                k_act_mean[layer_idx] = torch.stack(k_input_activations[layer_idx]).mean(dim=0)
            else:
                k_act_mean[layer_idx] = torch.ones(self.hidden_size)

            if v_input_activations[layer_idx]:
                v_act_mean[layer_idx] = torch.stack(v_input_activations[layer_idx]).mean(dim=0)
            else:
                v_act_mean[layer_idx] = torch.ones(self.hidden_size)

            if o_input_activations[layer_idx]:
                o_act_mean[layer_idx] = torch.stack(o_input_activations[layer_idx]).mean(dim=0)
            else:
                o_act_mean[layer_idx] = torch.ones(num_heads * head_dim)

        head_scores = {}

        for layer_idx in range(self.num_layers):
            attn = self._get_attention_module(layer_idx)
            if attn is None:
                head_scores[layer_idx] = torch.ones(num_heads)
                continue

            q_proj = attn.q_proj
            k_proj = attn.k_proj
            v_proj = attn.v_proj
            o_proj = get_attn_output_proj(attn)

            q_weight = q_proj.base_layer.weight.data if hasattr(q_proj, 'base_layer') else q_proj.weight.data
            k_weight = k_proj.base_layer.weight.data if hasattr(k_proj, 'base_layer') else k_proj.weight.data
            v_weight = v_proj.base_layer.weight.data if hasattr(v_proj, 'base_layer') else v_proj.weight.data
            o_weight = o_proj.base_layer.weight.data if hasattr(o_proj, 'base_layer') else o_proj.weight.data

            layer_importance = torch.zeros(num_heads)

            # Q proj: [num_heads * head_dim, hidden_size]
            q_act = q_act_mean[layer_idx].to(q_weight.device)
            q_wanda_full = (torch.abs(q_weight) * q_act.unsqueeze(0)).sum(dim=1)  # [num_heads * head_dim]
            q_wanda_per_head = q_wanda_full.view(num_heads, head_dim).sum(dim=1).cpu()  # [num_heads]
            layer_importance += q_wanda_per_head

            k_act = k_act_mean[layer_idx].to(k_weight.device)
            k_wanda_full = (torch.abs(k_weight) * k_act.unsqueeze(0)).sum(dim=1)  # [num_kv_heads * head_dim]
            k_wanda_per_kv = k_wanda_full.view(num_kv_heads, head_dim).sum(dim=1).cpu()  # [num_kv_heads]
            k_wanda_expanded = k_wanda_per_kv.repeat_interleave(num_queries_per_kv)  # [num_heads]
            layer_importance += k_wanda_expanded

            v_act = v_act_mean[layer_idx].to(v_weight.device)
            v_wanda_full = (torch.abs(v_weight) * v_act.unsqueeze(0)).sum(dim=1)  # [num_kv_heads * head_dim]
            v_wanda_per_kv = v_wanda_full.view(num_kv_heads, head_dim).sum(dim=1).cpu()  # [num_kv_heads]
            v_wanda_expanded = v_wanda_per_kv.repeat_interleave(num_queries_per_kv)  # [num_heads]
            layer_importance += v_wanda_expanded

            # O proj: [hidden_size, num_heads * head_dim]
            o_act = o_act_mean[layer_idx].to(o_weight.device)
            o_wanda_full = (torch.abs(o_weight) * o_act.unsqueeze(0)).sum(dim=0)  # [num_heads * head_dim]
            o_wanda_per_head = o_wanda_full.view(num_heads, head_dim).sum(dim=1).cpu()  # [num_heads]
            layer_importance += o_wanda_per_head

            if is_gqa:
                grouped = layer_importance.view(num_kv_heads, num_queries_per_kv)
                group_avg = grouped.mean(dim=1)
                layer_importance = group_avg.repeat_interleave(num_queries_per_kv)

            head_scores[layer_idx] = layer_importance

        if self.verbose:
            print(f"[Weight-WANDA-Head] Processed {num_processed} samples")
            print(f"\n[Weight-WANDA-Head] Final head scores:")
            for layer_idx in range(min(3, self.num_layers)):
                scores = head_scores[layer_idx]
                print(f"  Layer {layer_idx}: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

        # DEBUG: Always print to verify function is called
        print(f"[DEBUG] _compute_weight_wanda_head_scores called, samples={num_processed}")
        print(f"[DEBUG] Head Layer 0 scores: min={head_scores[0].min():.6f}, max={head_scores[0].max():.6f}, mean={head_scores[0].mean():.6f}")

        return head_scores

    def _compute_weight_wag_head_scores(
        self,
        dataloader,
        num_samples: int = 128,
        wag_mode: str = "qkvo"  # "qkvo", "o", "q", "k", "v"
    ) -> Dict[int, torch.Tensor]:
        num_heads = self.config.num_attention_heads
        num_kv_heads = getattr(self.config, 'num_key_value_heads', num_heads)
        head_dim = self.hidden_size // num_heads
        num_queries_per_kv = num_heads // num_kv_heads
        is_gqa = (num_kv_heads != num_heads)

        if self.verbose:
            print(f"\n[Weight-WAG-Head] Computing WAG Head scores: |W × act × grad| (mode={wag_mode})")
            if is_gqa:
                print(f"[Weight-WAG-Head] GQA detected: {num_heads} Q heads, {num_kv_heads} KV heads")

        q_activations = {i: [] for i in range(self.num_layers)}
        k_activations = {i: [] for i in range(self.num_layers)}
        v_activations = {i: [] for i in range(self.num_layers)}
        o_activations = {i: [] for i in range(self.num_layers)}
        self.hooks = []

        for layer_idx in range(self.num_layers):
            attn = self._get_attention_module(layer_idx)
            if attn is None:
                continue

            # Q proj output hook
            if wag_mode in ["qkvo", "q"]:
                q_proj = attn.q_proj
                q_target = q_proj.base_layer if hasattr(q_proj, 'base_layer') else q_proj

                def make_q_hook(idx):
                    def hook(module, input, output):
                        if output is not None:
                            output.requires_grad_(True)
                            output.retain_grad()
                            q_activations[idx].append(output)
                    return hook
                self.hooks.append(q_target.register_forward_hook(make_q_hook(layer_idx)))

            # K proj output hook
            if wag_mode in ["qkvo", "k"]:
                k_proj = attn.k_proj
                k_target = k_proj.base_layer if hasattr(k_proj, 'base_layer') else k_proj

                def make_k_hook(idx):
                    def hook(module, input, output):
                        if output is not None:
                            output.requires_grad_(True)
                            output.retain_grad()
                            k_activations[idx].append(output)
                    return hook
                self.hooks.append(k_target.register_forward_hook(make_k_hook(layer_idx)))

            # V proj output hook
            if wag_mode in ["qkvo", "v"]:
                v_proj = attn.v_proj
                v_target = v_proj.base_layer if hasattr(v_proj, 'base_layer') else v_proj

                def make_v_hook(idx):
                    def hook(module, input, output):
                        if output is not None:
                            output.requires_grad_(True)
                            output.retain_grad()
                            v_activations[idx].append(output)
                    return hook
                self.hooks.append(v_target.register_forward_hook(make_v_hook(layer_idx)))

            # O proj input hook — o_proj or dense
            if wag_mode in ["qkvo", "o"]:
                o_proj = get_attn_output_proj(attn)
                if o_proj is None:
                    continue
                o_target = o_proj.base_layer if hasattr(o_proj, 'base_layer') else o_proj

                def make_o_hook(idx):
                    def hook(module, input, output):
                        if len(input) > 0 and input[0] is not None:
                            inp = input[0]
                            inp.requires_grad_(True)
                            inp.retain_grad()
                            o_activations[idx].append(inp)
                    return hook
                self.hooks.append(o_target.register_forward_hook(make_o_hook(layer_idx)))

        # Enable gradients
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True

        accumulated = {i: torch.zeros(num_heads, device='cpu') for i in range(self.num_layers)}
        num_processed = 0
        pbar = tqdm(dataloader, desc=f"[Weight-WAG-Head] Calibration ({wag_mode})", disable=not self.verbose)

        for batch in pbar:
            if num_processed >= num_samples:
                break

            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                input_ids = batch.get('input_ids')
                attention_mask = batch.get('attention_mask')
            else:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device) if len(batch) > 1 else None

            labels = input_ids.clone()

            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()

                for layer_idx in range(self.num_layers):
                    attn = self._get_attention_module(layer_idx)
                    if attn is None:
                        continue

                    layer_importance = torch.zeros(num_heads, device='cpu')
                    count = 0

                    # Q WAG
                    if wag_mode in ["qkvo", "q"] and q_activations[layer_idx]:
                        q_act = q_activations[layer_idx][-1]
                        if q_act.grad is not None:
                            q_grad = q_act.grad.detach().float()
                            q_val = q_act.detach().float()
                            q_proj = attn.q_proj
                            q_weight = q_proj.base_layer.weight if hasattr(q_proj, 'base_layer') else q_proj.weight
                            # Q weight: [num_heads * head_dim, hidden] -> per head weight norm
                            w_norm = q_weight.detach().abs().view(num_heads, head_dim, -1).mean(dim=(1, 2)).float()
                            # ganda per head
                            ganda = (q_grad.abs() * q_val.abs()).sum(dim=(0, 1))  # [num_heads * head_dim]
                            ganda_per_head = ganda.view(num_heads, head_dim).sum(dim=1).cpu()
                            wag = ganda_per_head * w_norm.cpu()
                            layer_importance += wag
                            count += 1

                    # K WAG
                    if wag_mode in ["qkvo", "k"] and k_activations[layer_idx]:
                        k_act = k_activations[layer_idx][-1]
                        if k_act.grad is not None:
                            k_grad = k_act.grad.detach().float()
                            k_val = k_act.detach().float()
                            k_proj = attn.k_proj
                            k_weight = k_proj.base_layer.weight if hasattr(k_proj, 'base_layer') else k_proj.weight
                            w_norm = k_weight.detach().abs().view(num_kv_heads, head_dim, -1).mean(dim=(1, 2)).float()
                            ganda = (k_grad.abs() * k_val.abs()).sum(dim=(0, 1))
                            ganda_per_kv = ganda.view(num_kv_heads, head_dim).sum(dim=1).cpu()
                            wag = ganda_per_kv * w_norm.cpu()
                            # Expand to num_heads for GQA
                            wag_expanded = wag.repeat_interleave(num_queries_per_kv)
                            layer_importance += wag_expanded
                            count += 1

                    # V WAG
                    if wag_mode in ["qkvo", "v"] and v_activations[layer_idx]:
                        v_act = v_activations[layer_idx][-1]
                        if v_act.grad is not None:
                            v_grad = v_act.grad.detach().float()
                            v_val = v_act.detach().float()
                            v_proj = attn.v_proj
                            v_weight = v_proj.base_layer.weight if hasattr(v_proj, 'base_layer') else v_proj.weight
                            w_norm = v_weight.detach().abs().view(num_kv_heads, head_dim, -1).mean(dim=(1, 2)).float()

                            ganda = (v_grad.abs() * v_val.abs()).sum(dim=(0, 1))
                            ganda_per_kv = ganda.view(num_kv_heads, head_dim).sum(dim=1).cpu()
                            wag = ganda_per_kv * w_norm.cpu()
                            wag_expanded = wag.repeat_interleave(num_queries_per_kv)
                            layer_importance += wag_expanded
                            count += 1

                    # O WAG
                    if wag_mode in ["qkvo", "o"] and o_activations[layer_idx]:
                        o_act = o_activations[layer_idx][-1]
                        if o_act.grad is not None:
                            o_grad = o_act.grad.detach().float()
                            o_val = o_act.detach().float()
                            o_proj = get_attn_output_proj(attn)
                            o_weight = o_proj.base_layer.weight if hasattr(o_proj, 'base_layer') else o_proj.weight
                            # O weight: [hidden, num_heads * head_dim] -> per head weight norm (column)
                            w_norm = o_weight.detach().abs().view(-1, num_heads, head_dim).mean(dim=(0, 2)).float()

                            ganda = (o_grad.abs() * o_val.abs()).sum(dim=(0, 1))
                            ganda_per_head = ganda.view(num_heads, head_dim).sum(dim=1).cpu()
                            wag = ganda_per_head * w_norm.cpu()
                            layer_importance += wag
                            count += 1

                    # Normalize if multiple sources
                    if count > 1:
                        layer_importance /= count

                    accumulated[layer_idx] += layer_importance

                num_processed += input_ids.size(0)
                pbar.set_postfix({"samples": num_processed})

            except Exception as e:
                if self.verbose:
                    print(f"[Weight-WAG-Head] Error: {e}")
                continue
            finally:
                self.model.zero_grad(set_to_none=True)
                for i in range(self.num_layers):
                    q_activations[i] = []
                    k_activations[i] = []
                    v_activations[i] = []
                    o_activations[i] = []
                del input_ids, attention_mask, labels
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        if num_processed > 0:
            batch_count = num_processed / dataloader.batch_size
            for layer_idx in range(self.num_layers):
                accumulated[layer_idx] /= batch_count

        if self.verbose:
            print(f"[Weight-WAG-Head] Processed {num_processed} samples")
            print(f"\n[Weight-WAG-Head] Final head scores:")
            for layer_idx in range(min(3, self.num_layers)):
                scores = accumulated[layer_idx]
                print(f"  Layer {layer_idx}: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

        return accumulated

    # ==================== FFN Weight-based Importance Methods ====================

    def _compute_weight_norm_ffn_scores(
        self,
        p: float = 2.0
    ) -> Dict[int, torch.Tensor]:
        ffn_scores = {}

        for layer_idx in range(self.num_layers):
            mlp = self._get_mlp_module(layer_idx)
            if mlp is None:
                ffn_scores[layer_idx] = torch.ones(self.intermediate_size)
                continue

            if is_swiglu_ffn(mlp):
                # LLaMA SwiGLU: gate_proj + up_proj + down_proj
                gate_w = mlp.gate_proj.base_layer.weight if hasattr(mlp.gate_proj, 'base_layer') else mlp.gate_proj.weight
                up_w = mlp.up_proj.base_layer.weight if hasattr(mlp.up_proj, 'base_layer') else mlp.up_proj.weight
                down_w = mlp.down_proj.base_layer.weight if hasattr(mlp.down_proj, 'base_layer') else mlp.down_proj.weight

                gate_norms = torch.norm(gate_w.detach().float(), p=p, dim=1)
                up_norms = torch.norm(up_w.detach().float(), p=p, dim=1)
                down_norms = torch.norm(down_w.detach().float(), p=p, dim=0)
                layer_importance = (gate_norms + up_norms + down_norms).cpu()
            elif hasattr(mlp, 'fc1'):
                # Phi-2 Standard MLP: fc1 + fc2
                fc1_w = mlp.fc1.base_layer.weight if hasattr(mlp.fc1, 'base_layer') else mlp.fc1.weight
                fc2_w = mlp.fc2.base_layer.weight if hasattr(mlp.fc2, 'base_layer') else mlp.fc2.weight

                fc1_norms = torch.norm(fc1_w.detach().float(), p=p, dim=1)
                fc2_norms = torch.norm(fc2_w.detach().float(), p=p, dim=0)
                layer_importance = (fc1_norms + fc2_norms).cpu()
            else:
                layer_importance = torch.ones(self.intermediate_size)
            ffn_scores[layer_idx] = layer_importance

        if self.verbose:
            print(f"\n[Weight-Norm-FFN] Final FFN scores:")
            for layer_idx in range(min(3, self.num_layers)):
                scores = ffn_scores[layer_idx]
                print(f"  Layer {layer_idx}: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

        return ffn_scores

    def _compute_weight_taylor_ffn_scores(
        self,
        dataloader,
        num_samples: int = 128
    ) -> Dict[int, torch.Tensor]:
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True

        accumulated = {i: torch.zeros(self.intermediate_size, device='cpu') for i in range(self.num_layers)}

        num_processed = 0
        pbar = tqdm(dataloader, desc="[Weight-Taylor-FFN] Calibration", disable=not self.verbose)

        for batch in pbar:
            if num_processed >= num_samples:
                break

            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                input_ids = batch.get('input_ids')
                attention_mask = batch.get('attention_mask')
            else:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device) if len(batch) > 1 else None

            labels = input_ids.clone()

            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()

                for layer_idx in range(self.num_layers):
                    mlp = self._get_mlp_module(layer_idx)
                    if mlp is None:
                        continue

                    layer_importance = torch.zeros(self.intermediate_size, device='cpu')

                    for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                        proj = getattr(mlp, proj_name)
                        if hasattr(proj, 'base_layer'):
                            proj = proj.base_layer

                        weight = proj.weight.data.detach().float()
                        if proj.weight.grad is not None:
                            grad = proj.weight.grad.detach().float()

                            if proj_name == 'down_proj':
                                # down_proj: [hidden_size, intermediate_size]
                                taylor_scores = (weight.abs() * grad.abs()).sum(dim=0)  # [intermediate_size]
                            else:
                                # gate_proj, up_proj: [intermediate_size, hidden_size]
                                taylor_scores = (weight.abs() * grad.abs()).sum(dim=1)  # [intermediate_size]

                            layer_importance += taylor_scores.cpu()

                    accumulated[layer_idx] += layer_importance

                num_processed += input_ids.size(0)
                pbar.set_postfix({"samples": num_processed})

            except Exception as e:
                if self.verbose:
                    print(f"[Weight-Taylor-FFN] Error: {e}")
                continue
            finally:
                self.model.zero_grad(set_to_none=True)
                del input_ids, attention_mask, labels
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if num_processed > 0:
            batch_count = num_processed / dataloader.batch_size
            for layer_idx in range(self.num_layers):
                accumulated[layer_idx] /= batch_count

        if self.verbose:
            print(f"[Weight-Taylor-FFN] Processed {num_processed} samples")
            print(f"\n[Weight-Taylor-FFN] Final FFN scores:")
            for layer_idx in range(min(3, self.num_layers)):
                scores = accumulated[layer_idx]
                print(f"  Layer {layer_idx}: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

        # DEBUG: Always print to verify function is called
        print(f"[DEBUG] _compute_weight_taylor_ffn_scores called, samples={num_processed}")
        print(f"[DEBUG] Layer 0 scores: min={accumulated[0].min():.6f}, max={accumulated[0].max():.6f}, mean={accumulated[0].mean():.6f}")

        return accumulated

    def _compute_weight_ganda_ffn_scores(
        self,
        dataloader,
        num_samples: int = 128
    ) -> Dict[int, torch.Tensor]:
        self.gate_out_activations = {i: [] for i in range(self.num_layers)}
        self.up_out_activations = {i: [] for i in range(self.num_layers)}
        self.down_in_activations = {i: [] for i in range(self.num_layers)}
        self.hooks = []

        for layer_idx in range(self.num_layers):
            mlp = self._get_mlp_module(layer_idx)
            if mlp is None:
                continue

            if hasattr(mlp, 'gate_proj'):
                gate_proj = mlp.gate_proj
                gate_target = gate_proj.base_layer if hasattr(gate_proj, 'base_layer') else gate_proj

                def make_gate_hook(idx):
                    def hook(module, input, output):
                        if output is not None:
                            output.requires_grad_(True)
                            output.retain_grad()
                            self.gate_out_activations[idx].append(output)
                    return hook
                self.hooks.append(gate_target.register_forward_hook(make_gate_hook(layer_idx)))

            # up_proj/fc1 output hook
            up_proj = getattr(mlp, 'up_proj', None) or getattr(mlp, 'fc1', None)
            if up_proj is not None:
                up_target = up_proj.base_layer if hasattr(up_proj, 'base_layer') else up_proj

                def make_up_hook(idx):
                    def hook(module, input, output):
                        if output is not None:
                            output.requires_grad_(True)
                            output.retain_grad()
                            self.up_out_activations[idx].append(output)
                    return hook
                self.hooks.append(up_target.register_forward_hook(make_up_hook(layer_idx)))

            down_proj = get_down_proj_module(mlp)
            if down_proj is None:
                continue
            down_target = down_proj.base_layer if hasattr(down_proj, 'base_layer') else down_proj

            def make_down_hook(idx):
                def hook(module, input, output):
                    if len(input) > 0 and input[0] is not None:
                        inp = input[0]
                        inp.requires_grad_(True)
                        inp.retain_grad()
                        self.down_in_activations[idx].append(inp)
                return hook
            self.hooks.append(down_target.register_forward_hook(make_down_hook(layer_idx)))

        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True

        accumulated = {i: torch.zeros(self.intermediate_size, device='cpu') for i in range(self.num_layers)}

        num_processed = 0
        pbar = tqdm(dataloader, desc=f"[Activation-GandA-FFN] Calibration", disable=not self.verbose)

        for batch in pbar:
            if num_processed >= num_samples:
                break

            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                input_ids = batch.get('input_ids')
                attention_mask = batch.get('attention_mask')
            else:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device) if len(batch) > 1 else None

            labels = input_ids.clone()

            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()

                for layer_idx in range(self.num_layers):
                    layer_importance = torch.zeros(self.intermediate_size, device='cpu')

                    # gate activation ganda
                    if self.gate_out_activations[layer_idx]:
                        gate_act = self.gate_out_activations[layer_idx][-1]
                        if gate_act.grad is not None:
                            gate_grad = gate_act.grad.detach().float()
                            gate_val = gate_act.detach().float()
                            ganda = (gate_grad.abs() * gate_val.abs()).sum(dim=(0, 1))
                            layer_importance += ganda.cpu()

                    # up activation ganda
                    if self.up_out_activations[layer_idx]:
                        up_act = self.up_out_activations[layer_idx][-1]
                        if up_act.grad is not None:
                            up_grad = up_act.grad.detach().float()
                            up_val = up_act.detach().float()
                            ganda = (up_grad.abs() * up_val.abs()).sum(dim=(0, 1))
                            layer_importance += ganda.cpu()

                    # down input activation ganda
                    if self.down_in_activations[layer_idx]:
                        down_act = self.down_in_activations[layer_idx][-1]
                        if down_act.grad is not None:
                            down_grad = down_act.grad.detach().float()
                            down_val = down_act.detach().float()
                            ganda = (down_grad.abs() * down_val.abs()).sum(dim=(0, 1))
                            layer_importance += ganda.cpu()

                    accumulated[layer_idx] += layer_importance

                num_processed += input_ids.size(0)
                pbar.set_postfix({"samples": num_processed})

            except Exception as e:
                if self.verbose:
                    print(f"[Activation-GandA-FFN] Error: {e}")
                continue
            finally:
                self.model.zero_grad(set_to_none=True)
                for i in range(self.num_layers):
                    self.gate_out_activations[i] = []
                    self.up_out_activations[i] = []
                    self.down_in_activations[i] = []
                del input_ids, attention_mask, labels
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        if num_processed > 0:
            batch_count = num_processed / dataloader.batch_size
            for layer_idx in range(self.num_layers):
                accumulated[layer_idx] /= batch_count

        if self.verbose:
            print(f"[Activation-GandA-FFN] Processed {num_processed} samples")
            print(f"\n[Activation-GandA-FFN] Final FFN scores:")
            for layer_idx in range(min(3, self.num_layers)):
                scores = accumulated[layer_idx]
                print(f"  Layer {layer_idx}: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

        # DEBUG: Always print to verify function is called
        print(f"[DEBUG] _compute_weight_ganda_ffn_scores called, samples={num_processed}")
        print(f"[DEBUG] Layer 0 scores: min={accumulated[0].min():.6f}, max={accumulated[0].max():.6f}, mean={accumulated[0].mean():.6f}")

        return accumulated

    def _compute_weight_wanda_ffn_scores(
        self,
        dataloader,
        num_samples: int = 128
    ) -> Dict[int, torch.Tensor]:
        if self.verbose:
            print(f"\n[Weight-WANDA-FFN] Computing WANDA FFN scores: |W| × |X|")

        gate_input_activations = {i: [] for i in range(self.num_layers)}
        up_input_activations = {i: [] for i in range(self.num_layers)}
        down_input_activations = {i: [] for i in range(self.num_layers)}
        self.hooks = []

        for layer_idx in range(self.num_layers):
            mlp = self._get_mlp_module(layer_idx)
            if mlp is None:
                continue

            if hasattr(mlp, 'gate_proj'):
                gate_proj = mlp.gate_proj
                gate_target = gate_proj.base_layer if hasattr(gate_proj, 'base_layer') else gate_proj

                def make_gate_hook(idx):
                    def hook(module, input, output):
                        if len(input) > 0 and input[0] is not None:
                            inp = input[0].detach()
                            if inp.dim() == 3:
                                inp = inp.view(-1, inp.size(-1))
                            rms = torch.sqrt((inp ** 2).mean(dim=0))
                            gate_input_activations[idx].append(rms.cpu())
                    return hook
                self.hooks.append(gate_target.register_forward_hook(make_gate_hook(layer_idx)))

            # up_proj/fc1 input hook
            up_proj = getattr(mlp, 'up_proj', None) or getattr(mlp, 'fc1', None)
            if up_proj is not None:
                up_target = up_proj.base_layer if hasattr(up_proj, 'base_layer') else up_proj

                def make_up_hook(idx):
                    def hook(module, input, output):
                        if len(input) > 0 and input[0] is not None:
                            inp = input[0].detach()
                            if inp.dim() == 3:
                                inp = inp.view(-1, inp.size(-1))
                            rms = torch.sqrt((inp ** 2).mean(dim=0))
                            up_input_activations[idx].append(rms.cpu())
                    return hook
                self.hooks.append(up_target.register_forward_hook(make_up_hook(layer_idx)))

            # down_proj/fc2 input hook
            down_proj = get_down_proj_module(mlp)
            if down_proj is None:
                continue
            down_target = down_proj.base_layer if hasattr(down_proj, 'base_layer') else down_proj

            def make_down_hook(idx):
                def hook(module, input, output):
                    if len(input) > 0 and input[0] is not None:
                        inp = input[0].detach()
                        if inp.dim() == 3:
                            inp = inp.view(-1, inp.size(-1))  # [batch*seq, intermediate_size]
                        rms = torch.sqrt((inp ** 2).mean(dim=0))  # [intermediate_size]
                        down_input_activations[idx].append(rms.cpu())
                return hook
            self.hooks.append(down_target.register_forward_hook(make_down_hook(layer_idx)))

        self.model.eval()
        num_processed = 0
        pbar = tqdm(dataloader, desc="[Weight-WANDA-FFN] Collecting activations", disable=not self.verbose)

        with torch.no_grad():
            for batch in pbar:
                if num_processed >= num_samples:
                    break

                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}
                    input_ids = batch.get('input_ids')
                    attention_mask = batch.get('attention_mask')
                else:
                    input_ids = batch[0].to(self.device)
                    attention_mask = batch[1].to(self.device) if len(batch) > 1 else None

                try:
                    _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    num_processed += input_ids.size(0)
                    pbar.set_postfix({"samples": num_processed})
                except Exception as e:
                    if self.verbose:
                        print(f"[Weight-WANDA-FFN] Error: {e}")
                    continue

        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        gate_act_mean = {}
        up_act_mean = {}
        down_act_mean = {}

        for layer_idx in range(self.num_layers):
            if gate_input_activations[layer_idx]:
                gate_act_mean[layer_idx] = torch.stack(gate_input_activations[layer_idx]).mean(dim=0)
            else:
                gate_act_mean[layer_idx] = torch.ones(self.config.hidden_size)

            if up_input_activations[layer_idx]:
                up_act_mean[layer_idx] = torch.stack(up_input_activations[layer_idx]).mean(dim=0)
            else:
                up_act_mean[layer_idx] = torch.ones(self.config.hidden_size)

            if down_input_activations[layer_idx]:
                down_act_mean[layer_idx] = torch.stack(down_input_activations[layer_idx]).mean(dim=0)
            else:
                down_act_mean[layer_idx] = torch.ones(self.intermediate_size)

        ffn_scores = {}

        for layer_idx in range(self.num_layers):
            mlp = self._get_mlp_module(layer_idx)
            if mlp is None:
                ffn_scores[layer_idx] = torch.ones(self.intermediate_size)
                continue

            if is_swiglu_ffn(mlp):
                # LLaMA SwiGLU
                gate_proj = mlp.gate_proj
                up_proj = mlp.up_proj
                down_proj = mlp.down_proj

                gate_weight = gate_proj.base_layer.weight.data if hasattr(gate_proj, 'base_layer') else gate_proj.weight.data
                up_weight = up_proj.base_layer.weight.data if hasattr(up_proj, 'base_layer') else up_proj.weight.data
                down_weight = down_proj.base_layer.weight.data if hasattr(down_proj, 'base_layer') else down_proj.weight.data

                gate_act = gate_act_mean[layer_idx].to(gate_weight.device)
                gate_wanda = (torch.abs(gate_weight) * gate_act.unsqueeze(0)).sum(dim=1)

                up_act = up_act_mean[layer_idx].to(up_weight.device)
                up_wanda = (torch.abs(up_weight) * up_act.unsqueeze(0)).sum(dim=1)

                down_act = down_act_mean[layer_idx].to(down_weight.device)
                down_wanda = (torch.abs(down_weight) * down_act.unsqueeze(0)).sum(dim=0)

                layer_importance = gate_wanda.cpu() + up_wanda.cpu() + down_wanda.cpu()
            elif hasattr(mlp, 'fc1'):
                # Phi-2 Standard MLP
                fc1 = mlp.fc1
                fc2 = mlp.fc2

                fc1_weight = fc1.base_layer.weight.data if hasattr(fc1, 'base_layer') else fc1.weight.data
                fc2_weight = fc2.base_layer.weight.data if hasattr(fc2, 'base_layer') else fc2.weight.data

                up_act = up_act_mean[layer_idx].to(fc1_weight.device)
                fc1_wanda = (torch.abs(fc1_weight) * up_act.unsqueeze(0)).sum(dim=1)

                down_act = down_act_mean[layer_idx].to(fc2_weight.device)
                fc2_wanda = (torch.abs(fc2_weight) * down_act.unsqueeze(0)).sum(dim=0)

                layer_importance = fc1_wanda.cpu() + fc2_wanda.cpu()
            else:
                layer_importance = torch.ones(self.intermediate_size)
            ffn_scores[layer_idx] = layer_importance

        if self.verbose:
            print(f"[Weight-WANDA-FFN] Processed {num_processed} samples")
            print(f"\n[Weight-WANDA-FFN] Final FFN scores:")
            for layer_idx in range(min(3, self.num_layers)):
                scores = ffn_scores[layer_idx]
                print(f"  Layer {layer_idx}: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

        # DEBUG: Always print to verify function is called
        print(f"[DEBUG] _compute_weight_wanda_ffn_scores called, samples={num_processed}")
        print(f"[DEBUG] Layer 0 scores: min={ffn_scores[0].min():.6f}, max={ffn_scores[0].max():.6f}, mean={ffn_scores[0].mean():.6f}")

        return ffn_scores

    def _compute_weight_wag_ffn_scores(
        self,
        dataloader,
        num_samples: int = 128,
        wag_mode: str = "all"  # "all", "down", "up", "gate"
    ) -> Dict[int, torch.Tensor]:
        if self.verbose:
            print(f"\n[Weight-WAG-FFN] Computing WAG FFN scores: |W × act × grad| (mode={wag_mode})")

        self.gate_out_activations = {i: [] for i in range(self.num_layers)}
        self.up_out_activations = {i: [] for i in range(self.num_layers)}
        self.down_in_activations = {i: [] for i in range(self.num_layers)}
        self.hooks = []

        for layer_idx in range(self.num_layers):
            mlp = self._get_mlp_module(layer_idx)
            if mlp is None:
                continue

            if wag_mode in ["all", "gate"] and hasattr(mlp, 'gate_proj'):
                gate_proj = mlp.gate_proj
                gate_target = gate_proj.base_layer if hasattr(gate_proj, 'base_layer') else gate_proj

                def make_gate_hook(idx):
                    def hook(module, input, output):
                        if output is not None:
                            output.requires_grad_(True)
                            output.retain_grad()
                            self.gate_out_activations[idx].append(output)
                    return hook
                self.hooks.append(gate_target.register_forward_hook(make_gate_hook(layer_idx)))

            # up_proj/fc1 output hook
            if wag_mode in ["all", "up"]:
                up_proj = getattr(mlp, 'up_proj', None) or getattr(mlp, 'fc1', None)
                if up_proj is not None:
                    up_target = up_proj.base_layer if hasattr(up_proj, 'base_layer') else up_proj

                    def make_up_hook(idx):
                        def hook(module, input, output):
                            if output is not None:
                                output.requires_grad_(True)
                                output.retain_grad()
                                self.up_out_activations[idx].append(output)
                        return hook
                    self.hooks.append(up_target.register_forward_hook(make_up_hook(layer_idx)))

            # down_proj/fc2 input hook
            if wag_mode in ["all", "down"]:
                down_proj = get_down_proj_module(mlp)
                if down_proj is None:
                    continue
                down_target = down_proj.base_layer if hasattr(down_proj, 'base_layer') else down_proj

                def make_down_hook(idx):
                    def hook(module, input, output):
                        if len(input) > 0 and input[0] is not None:
                            inp = input[0]
                            inp.requires_grad_(True)
                            inp.retain_grad()
                            self.down_in_activations[idx].append(inp)
                    return hook
                self.hooks.append(down_target.register_forward_hook(make_down_hook(layer_idx)))

        # Enable gradients
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True

        accumulated = {i: torch.zeros(self.intermediate_size, device='cpu') for i in range(self.num_layers)}
        num_processed = 0
        pbar = tqdm(dataloader, desc=f"[Weight-WAG-FFN] Calibration ({wag_mode})", disable=not self.verbose)

        for batch in pbar:
            if num_processed >= num_samples:
                break

            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                input_ids = batch.get('input_ids')
                attention_mask = batch.get('attention_mask')
            else:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device) if len(batch) > 1 else None

            labels = input_ids.clone()

            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()

                for layer_idx in range(self.num_layers):
                    mlp = self._get_mlp_module(layer_idx)
                    if mlp is None:
                        continue

                    layer_importance = torch.zeros(self.intermediate_size, device='cpu')
                    count = 0

                    if wag_mode in ["all", "gate"] and hasattr(mlp, 'gate_proj') and self.gate_out_activations[layer_idx]:
                        gate_act = self.gate_out_activations[layer_idx][-1]
                        if gate_act.grad is not None:
                            gate_grad = gate_act.grad.detach().float()
                            gate_val = gate_act.detach().float()
                            # Get gate weight
                            gate_proj = mlp.gate_proj
                            gate_weight = gate_proj.base_layer.weight if hasattr(gate_proj, 'base_layer') else gate_proj.weight
                            w_norm = gate_weight.detach().abs().mean(dim=1).float()  # [intermediate]

                            # |grad × act| per neuron
                            ganda = (gate_grad.abs() * gate_val.abs()).sum(dim=(0, 1))  # [intermediate]
                            # WAG = ganda × weight
                            wag = ganda.cpu() * w_norm.cpu()
                            layer_importance += wag
                            count += 1

                    # Up WAG: |W_up/fc1 × act × grad|
                    if wag_mode in ["all", "up"] and self.up_out_activations[layer_idx]:
                        up_act = self.up_out_activations[layer_idx][-1]
                        if up_act.grad is not None:
                            up_grad = up_act.grad.detach().float()
                            up_val = up_act.detach().float()
                            up_proj = getattr(mlp, 'up_proj', None) or getattr(mlp, 'fc1', None)
                            up_weight = up_proj.base_layer.weight if hasattr(up_proj, 'base_layer') else up_proj.weight
                            w_norm = up_weight.detach().abs().mean(dim=1).float()

                            ganda = (up_grad.abs() * up_val.abs()).sum(dim=(0, 1))
                            wag = ganda.cpu() * w_norm.cpu()
                            layer_importance += wag
                            count += 1

                    # Down WAG: |W_down/fc2 × act × grad|
                    if wag_mode in ["all", "down"] and self.down_in_activations[layer_idx]:
                        down_act = self.down_in_activations[layer_idx][-1]
                        if down_act.grad is not None:
                            down_grad = down_act.grad.detach().float()
                            down_val = down_act.detach().float()
                            down_proj = get_down_proj_module(mlp)
                            down_weight = down_proj.base_layer.weight if hasattr(down_proj, 'base_layer') else down_proj.weight
                            w_norm = down_weight.detach().abs().mean(dim=0).float()  # [intermediate]

                            ganda = (down_grad.abs() * down_val.abs()).sum(dim=(0, 1))
                            wag = ganda.cpu() * w_norm.cpu()
                            layer_importance += wag
                            count += 1

                    # Normalize if multiple sources
                    if count > 1:
                        layer_importance /= count

                    accumulated[layer_idx] += layer_importance

                num_processed += input_ids.size(0)
                pbar.set_postfix({"samples": num_processed})

            except Exception as e:
                if self.verbose:
                    print(f"[Weight-WAG-FFN] Error: {e}")
                continue
            finally:
                self.model.zero_grad(set_to_none=True)
                for i in range(self.num_layers):
                    self.gate_out_activations[i] = []
                    self.up_out_activations[i] = []
                    self.down_in_activations[i] = []
                del input_ids, attention_mask, labels
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        if num_processed > 0:
            batch_count = num_processed / dataloader.batch_size
            for layer_idx in range(self.num_layers):
                accumulated[layer_idx] /= batch_count

        if self.verbose:
            print(f"[Weight-WAG-FFN] Processed {num_processed} samples")
            print(f"\n[Weight-WAG-FFN] Final FFN scores:")
            for layer_idx in range(min(3, self.num_layers)):
                scores = accumulated[layer_idx]
                print(f"  Layer {layer_idx}: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

        return accumulated

    def _compute_qk_activation_head_scores(
        self,
        dataloader,
        num_samples: int = 128
    ) -> Dict[int, torch.Tensor]:
        self._register_hooks()
        self.model.eval()

        num_heads = self.config.num_attention_heads

        if self.verbose:
            print(f"[QK-Activation] Num heads: {num_heads}")

        num_processed = 0
        pbar = tqdm(dataloader, desc="[QK-Activation] Forward pass", disable=not self.verbose)

        with torch.no_grad():
            for batch in pbar:
                if num_processed >= num_samples:
                    break

                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}
                    input_ids = batch.get('input_ids')
                    attention_mask = batch.get('attention_mask')
                else:
                    input_ids = batch[0].to(self.device)
                    attention_mask = batch[1].to(self.device) if len(batch) > 1 else None

                try:
                    _ = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )

                    num_processed += input_ids.shape[0]
                    pbar.set_postfix(samples=num_processed)

                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Error during forward pass: {e}")
                    continue

        self._remove_hooks()

        head_scores = self.compute_qk_activation_importance()

        if self.verbose:
            print(f"\n[QK-Activation] Final head scores:")
            for layer_idx in range(min(3, self.num_layers)):
                scores = head_scores[layer_idx]
                print(f"  Layer {layer_idx}: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

        return head_scores

    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def compute_taylor_scores(
        self,
        dataloader,
        num_samples: int = 128
    ) -> Dict[int, torch.Tensor]:
        if self.ffn_mode in ["weight_norm", "wn"]:
            if self.verbose:
                print(f"\n[Weight-Norm-FFN] Computing weight norm FFN scores...")
            return self._compute_weight_norm_ffn_scores()

        if self.ffn_mode in ["weight_taylor", "wt"]:
            if self.verbose:
                print(f"\n[Weight-Taylor-FFN] Computing weight taylor FFN scores...")
            return self._compute_weight_taylor_ffn_scores(dataloader, num_samples)

        if self.ffn_mode in ["weight_ganda", "wg"]:
            if self.verbose:
                print(f"\n[Activation-GandA-FFN] Computing activation ganda FFN scores...")
            return self._compute_weight_ganda_ffn_scores(dataloader, num_samples)

        if self.ffn_mode in ["weight_wanda", "ww"]:
            if self.verbose:
                print(f"\n[Weight-WANDA-FFN] Computing WANDA FFN scores...")
            return self._compute_weight_wanda_ffn_scores(dataloader, num_samples)

        if self.ffn_mode in ["weight_wag", "wag", "wag_down"]:
            if self.verbose:
                print(f"\n[Weight-WAG-FFN] Computing WAG FFN scores (down)...")
            return self._compute_weight_wag_ffn_scores(dataloader, num_samples, wag_mode="down")

        if self.ffn_mode == "wag_up":
            if self.verbose:
                print(f"\n[Weight-WAG-FFN] Computing WAG FFN scores (up)...")
            return self._compute_weight_wag_ffn_scores(dataloader, num_samples, wag_mode="up")

        if self.ffn_mode == "wag_gate":
            if self.verbose:
                print(f"\n[Weight-WAG-FFN] Computing WAG FFN scores (gate)...")
            return self._compute_weight_wag_ffn_scores(dataloader, num_samples, wag_mode="gate")

        if self.ffn_mode == "wag_all":
            if self.verbose:
                print(f"\n[Weight-WAG-FFN] Computing WAG FFN scores (all: gate+up+down)...")
            return self._compute_weight_wag_ffn_scores(dataloader, num_samples, wag_mode="all")

        if self._is_standalone_ntk_method():
            if self.verbose:
                print(f"\n[{self.importance_method.upper()}] Computing standalone NTK scores with {self._get_method_formula()}...")

            if self.importance_method == "dynamics":
                standalone_scores = self._compute_dynamics_scores(dataloader, num_samples)
            else:  # coupling
                standalone_scores = self._compute_coupling_scores(dataloader, num_samples)

            if self.verbose:
                print(f"\n[{self.importance_method.upper()}] Final scores:")
                for layer_idx in range(min(3, self.num_layers)):
                    scores = standalone_scores[layer_idx]
                    print(f"  Layer {layer_idx}: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

            return standalone_scores

        if self._is_sti_method():
            if self.verbose:
                print(f"\n[{self.importance_method.upper()}] Computing STI scores with {self._get_method_formula()}...")

            sti_scores = self._compute_ffn_sti_scores(dataloader, num_samples)

            if self.importance_method == "sti_magnitude":
                if self.verbose:
                    print(f"\n[{self.importance_method.upper()}] Applying magnitude (ganda) correction...")

                original_method = self.importance_method
                self.importance_method = "ganda"
                ganda_scores = self.compute_taylor_scores(dataloader, num_samples)
                self.importance_method = original_method

                # STI × normalized_ganda
                ganda_normalized = self._normalize_scores(ganda_scores)
                for layer_idx in range(self.num_layers):
                    sti_scores[layer_idx] = sti_scores[layer_idx] * ganda_normalized[layer_idx]

            if self.verbose:
                print(f"\n[{self.importance_method.upper()}] Final STI scores:")
                for layer_idx in range(min(3, self.num_layers)):
                    scores = sti_scores[layer_idx]
                    print(f"  Layer {layer_idx}: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

            return sti_scores

        self._register_hooks()
        if self.head_masks is not None or self.ffn_masks is not None:
            self._register_premasking_hooks()
        self.model.train()

        for param in self.model.parameters():
            param.requires_grad = True

        if self.verbose:
            print(f"\n[{self.importance_method.upper()}] Computing importance with {self._get_method_formula()}...")
            print(f"[{self.importance_method.upper()}] FFN mode: {self.ffn_mode}")

        accumulated = {i: torch.zeros(self.intermediate_size, device='cpu')
                       for i in range(self.num_layers)}

        if self.ffn_mode == "all":
            accumulated_up = {i: torch.zeros(self.intermediate_size, device='cpu')
                             for i in range(self.num_layers)}
            accumulated_gate = {i: torch.zeros(self.intermediate_size, device='cpu')
                               for i in range(self.num_layers)}

        num_processed = 0

        pbar = tqdm(dataloader, desc="[Taylor] Calibration", disable=not self.verbose)

        for batch in pbar:
            if num_processed >= num_samples:
                break

            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                input_ids = batch.get('input_ids')
                attention_mask = batch.get('attention_mask')
            else:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device) if len(batch) > 1 else None

            labels = input_ids.clone()

            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()

                for layer_idx in range(self.num_layers):
                    if self.activations[layer_idx] and self.gradients[layer_idx]:
                        act = self.activations[layer_idx][-1]
                        grad = self.gradients[layer_idx][-1]

                        if act.shape == grad.shape:
                            weight = self.weights.get(layer_idx)
                            importance = self._compute_importance(act, grad, weight)
                            importance = importance.mean(dim=(0, 1))
                            accumulated[layer_idx] += importance.cpu()

                    if self.ffn_mode == "all" and self.up_activations[layer_idx] and self.up_gradients[layer_idx]:
                        up_act = self.up_activations[layer_idx][-1]
                        up_grad = self.up_gradients[layer_idx][-1]

                        if up_act.shape == up_grad.shape:
                            up_weight = self.up_weights.get(layer_idx)
                            importance_up = self._compute_importance(up_act, up_grad, up_weight)
                            importance_up = importance_up.mean(dim=(0, 1))
                            accumulated_up[layer_idx] += importance_up.cpu()

                    if self.ffn_mode == "all" and self.gate_activations[layer_idx] and self.gate_gradients[layer_idx]:
                        gate_act = self.gate_activations[layer_idx][-1]
                        gate_grad = self.gate_gradients[layer_idx][-1]

                        if gate_act.shape == gate_grad.shape:
                            gate_weight = self.gate_weights.get(layer_idx)
                            importance_gate = self._compute_importance(gate_act, gate_grad, gate_weight)
                            importance_gate = importance_gate.mean(dim=(0, 1))
                            accumulated_gate[layer_idx] += importance_gate.cpu()

                num_processed += input_ids.size(0)
                pbar.set_postfix({"samples": num_processed})

            except Exception as e:
                if self.verbose:
                    print(f"[{self.importance_method.upper()}] Error: {e}")
                continue
            finally:
                self.model.zero_grad(set_to_none=True)
                for i in range(self.num_layers):
                    self.activations[i] = []
                    self.gradients[i] = []
                    if self.ffn_mode == "all":
                        self.up_activations[i] = []
                        self.up_gradients[i] = []
                        self.gate_activations[i] = []
                        self.gate_gradients[i] = []
                del input_ids, attention_mask, labels
                if 'outputs' in dir():
                    del outputs
                if 'loss' in dir():
                    del loss
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self._remove_hooks()

        if num_processed > 0:
            batch_count = num_processed / dataloader.batch_size
            for layer_idx in range(self.num_layers):
                accumulated[layer_idx] /= batch_count

                if self.ffn_mode == "all":
                    accumulated_up[layer_idx] /= batch_count
                    accumulated_gate[layer_idx] /= batch_count

        if self.ffn_mode == "all":
            if self.verbose:
                print(f"\n[{self.importance_method.upper()}] Averaging importance from up_proj, gate_proj, down_proj...")
            for layer_idx in range(self.num_layers):
                accumulated[layer_idx] = (accumulated[layer_idx] + accumulated_up[layer_idx] + accumulated_gate[layer_idx]) / 3.0

        if self.verbose:
            print(f"[{self.importance_method.upper()}] Processed {num_processed} samples")
            for layer_idx in range(min(3, self.num_layers)):
                scores = accumulated[layer_idx]
                print(f"  Layer {layer_idx}: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

        if self._is_ntk_method():
            if self.verbose:
                print(f"\n[{self.importance_method.upper()}] Applying NTK modifiers...")

            dynamics_scores = None
            coupling_scores = None

            if self.importance_method in ["ganda_ntk_dyn", "ganda_ntk_combined"]:
                dynamics_scores = self._compute_dynamics_scores(dataloader, num_samples)

            if self.importance_method in ["ganda_ntk_coupling", "ganda_ntk_combined"]:
                coupling_scores = self._compute_coupling_scores(dataloader, num_samples)

            accumulated = self._apply_ntk_modifiers(accumulated, dynamics_scores, coupling_scores)

            if self.verbose:
                print(f"\n[{self.importance_method.upper()}] Final scores after NTK modification:")
                for layer_idx in range(min(3, self.num_layers)):
                    scores = accumulated[layer_idx]
                    print(f"  Layer {layer_idx}: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

        from pruning.distributed_utils import all_reduce_dict
        all_reduce_dict(accumulated)

        return accumulated

    # ==================== Head Taylor Importance ====================

    def _get_attention_module(self, layer_idx: int):
        try:
            if hasattr(self.model, 'model'):
                base = self.model.model
            else:
                base = self.model
            if hasattr(base, 'model'):
                base = base.model
            if hasattr(base, 'layers'):
                return base.layers[layer_idx].self_attn
            else:
                if self.verbose and layer_idx == 0:
                    print(f"[DEBUG] _get_attention_module: base has no 'layers' attribute")
                    print(f"[DEBUG] base type: {type(base)}, attrs: {[a for a in dir(base) if not a.startswith('_')][:10]}")
        except (AttributeError, IndexError) as e:
            if self.verbose and layer_idx == 0:
                print(f"[DEBUG] _get_attention_module error: {e}")
            pass
        return None

    def _register_head_hooks(self, head_mode: str = "o"):
        self.head_activations = {i: [] for i in range(self.num_layers)}
        self.head_gradients = {i: [] for i in range(self.num_layers)}
        self.head_weights = {}

        registered_count = 0
        for layer_idx in range(self.num_layers):
            attn = self._get_attention_module(layer_idx)
            if attn is None:
                if self.verbose:
                    print(f"[DEBUG] Layer {layer_idx}: attention module NOT FOUND!")
                continue

            if head_mode == "o":
                target = get_attn_output_proj(attn)
            else:  # "v"
                target = attn.v_proj

            if target is None:
                continue

            if hasattr(target, 'base_layer'):
                target_module = target.base_layer
                if self.verbose and layer_idx == 0:
                    print(f"[DEBUG] Layer {layer_idx}: Using base_layer of {type(target).__name__}")
            else:
                target_module = target
                if self.verbose and layer_idx == 0:
                    print(f"[DEBUG] Layer {layer_idx}: Using direct module {type(target_module).__name__}")

            self.head_weights[layer_idx] = target_module.weight.data.detach()

            if head_mode == "o":
                # o_proj/dense input (attention output)
                def make_forward_hook(idx):
                    def hook(module, input, output):
                        if len(input) > 0 and input[0] is not None:
                            self.head_activations[idx].append(input[0].detach())
                    return hook

                def make_backward_hook(idx):
                    def hook(module, grad_input, grad_output):
                        if grad_input is not None and len(grad_input) > 0 and grad_input[0] is not None:
                            self.head_gradients[idx].append(grad_input[0].detach())
                    return hook
            else:  # "v" - v_proj output
                def make_forward_hook(idx):
                    def hook(module, input, output):
                        if output is not None:
                            self.head_activations[idx].append(output.detach())
                    return hook

                def make_backward_hook(idx):
                    def hook(module, grad_input, grad_output):
                        if grad_output is not None and len(grad_output) > 0 and grad_output[0] is not None:
                            self.head_gradients[idx].append(grad_output[0].detach())
                    return hook

            h1 = target_module.register_forward_hook(make_forward_hook(layer_idx))
            h2 = target_module.register_full_backward_hook(make_backward_hook(layer_idx))
            self.hooks.append(h1)
            self.hooks.append(h2)
            registered_count += 1

        if self.verbose:
            print(f"[DEBUG] Registered hooks for {registered_count}/{self.num_layers} layers")

    def _register_joint_hooks(self, head_mode: str = "o"):
        self.activations = {i: [] for i in range(self.num_layers)}
        self.gradients = {i: [] for i in range(self.num_layers)}
        self.weights = {}

        if self.ffn_mode == "all":
            self.up_activations = {i: [] for i in range(self.num_layers)}
            self.up_gradients = {i: [] for i in range(self.num_layers)}
            self.up_weights = {}
            self.gate_activations = {i: [] for i in range(self.num_layers)}
            self.gate_gradients = {i: [] for i in range(self.num_layers)}
            self.gate_weights = {}

        self.head_activations = {i: [] for i in range(self.num_layers)}
        self.head_gradients = {i: [] for i in range(self.num_layers)}
        self.head_weights = {}

        self.hooks = []

        for layer_idx in range(self.num_layers):
            # ===== FFN Hooks =====
            mlp = self._get_mlp_module(layer_idx)
            if mlp is not None:
                # down_proj/fc2 hooks
                if self.ffn_mode in ["down", "all"]:
                    down_proj = get_down_proj_module(mlp)
                    if down_proj is not None:
                        if hasattr(down_proj, 'base_layer'):
                            target_module = down_proj.base_layer
                        else:
                            target_module = down_proj

                        self.weights[layer_idx] = target_module.weight.data.detach()

                        def make_ffn_forward_hook(idx):
                            def hook(module, input, output):
                                if len(input) > 0 and input[0] is not None:
                                    self.activations[idx].append(input[0].detach())
                            return hook

                        def make_ffn_backward_hook(idx):
                            def hook(module, grad_input, grad_output):
                                if grad_input is not None and len(grad_input) > 0 and grad_input[0] is not None:
                                    self.gradients[idx].append(grad_input[0].detach())
                            return hook

                        h1 = target_module.register_forward_hook(make_ffn_forward_hook(layer_idx))
                        h2 = target_module.register_full_backward_hook(make_ffn_backward_hook(layer_idx))
                        self.hooks.extend([h1, h2])

                if self.ffn_mode == "all":
                    up_proj = getattr(mlp, 'up_proj', None) or getattr(mlp, 'fc1', None)
                    if up_proj is not None:
                        if hasattr(up_proj, 'base_layer'):
                            up_module = up_proj.base_layer
                        else:
                            up_module = up_proj

                        self.up_weights[layer_idx] = up_module.weight.data.detach()

                        def make_up_forward_hook(idx):
                            def hook(module, input, output):
                                if output is not None:
                                    self.up_activations[idx].append(output.detach())
                            return hook

                        def make_up_backward_hook(idx):
                            def hook(module, grad_input, grad_output):
                                if grad_output is not None and len(grad_output) > 0 and grad_output[0] is not None:
                                    self.up_gradients[idx].append(grad_output[0].detach())
                            return hook

                        h1 = up_module.register_forward_hook(make_up_forward_hook(layer_idx))
                        h2 = up_module.register_full_backward_hook(make_up_backward_hook(layer_idx))
                        self.hooks.extend([h1, h2])

                    if hasattr(mlp, 'gate_proj'):
                        gate_proj = mlp.gate_proj
                        if hasattr(gate_proj, 'base_layer'):
                            gate_module = gate_proj.base_layer
                        else:
                            gate_module = gate_proj

                        self.gate_weights[layer_idx] = gate_module.weight.data.detach()

                    def make_gate_forward_hook(idx):
                        def hook(module, input, output):
                            if output is not None:
                                self.gate_activations[idx].append(output.detach())
                        return hook

                    def make_gate_backward_hook(idx):
                        def hook(module, grad_input, grad_output):
                            if grad_output is not None and len(grad_output) > 0 and grad_output[0] is not None:
                                self.gate_gradients[idx].append(grad_output[0].detach())
                        return hook

                    h1 = gate_module.register_forward_hook(make_gate_forward_hook(layer_idx))
                    h2 = gate_module.register_full_backward_hook(make_gate_backward_hook(layer_idx))
                    self.hooks.extend([h1, h2])

                elif self.ffn_mode == "up":
                    up_proj = getattr(mlp, 'up_proj', None) or getattr(mlp, 'fc1', None)
                    if up_proj is None:
                        continue
                    if hasattr(up_proj, 'base_layer'):
                        up_module = up_proj.base_layer
                    else:
                        up_module = up_proj

                    self.weights[layer_idx] = up_module.weight.data.detach()

                    def make_up_forward_hook_single(idx):
                        def hook(module, input, output):
                            if output is not None:
                                self.activations[idx].append(output.detach())
                        return hook

                    def make_up_backward_hook_single(idx):
                        def hook(module, grad_input, grad_output):
                            if grad_output is not None and len(grad_output) > 0 and grad_output[0] is not None:
                                self.gradients[idx].append(grad_output[0].detach())
                        return hook

                    h1 = up_module.register_forward_hook(make_up_forward_hook_single(layer_idx))
                    h2 = up_module.register_full_backward_hook(make_up_backward_hook_single(layer_idx))
                    self.hooks.extend([h1, h2])

            # ===== Head Hooks =====
            attn = self._get_attention_module(layer_idx)
            if attn is not None:
                if head_mode == "o":
                    target = get_attn_output_proj(attn)
                else:  # "v"
                    target = attn.v_proj

                if target is None:
                    continue

                if hasattr(target, 'base_layer'):
                    target_module = target.base_layer
                else:
                    target_module = target

                self.head_weights[layer_idx] = target_module.weight.data.detach()

                if head_mode == "o":
                    def make_head_forward_hook(idx):
                        def hook(module, input, output):
                            if len(input) > 0 and input[0] is not None:
                                self.head_activations[idx].append(input[0].detach())
                        return hook

                    def make_head_backward_hook(idx):
                        def hook(module, grad_input, grad_output):
                            if grad_input is not None and len(grad_input) > 0 and grad_input[0] is not None:
                                self.head_gradients[idx].append(grad_input[0].detach())
                        return hook
                else:  # "v"
                    def make_head_forward_hook(idx):
                        def hook(module, input, output):
                            if output is not None:
                                self.head_activations[idx].append(output.detach())
                        return hook

                    def make_head_backward_hook(idx):
                        def hook(module, grad_input, grad_output):
                            if grad_output is not None and len(grad_output) > 0 and grad_output[0] is not None:
                                self.head_gradients[idx].append(grad_output[0].detach())
                        return hook

                h1 = target_module.register_forward_hook(make_head_forward_hook(layer_idx))
                h2 = target_module.register_full_backward_hook(make_head_backward_hook(layer_idx))
                self.hooks.extend([h1, h2])

        if self.verbose:
            print(f"[Joint] Registered FFN + Head hooks for {self.num_layers} layers")

    def compute_joint_scores(
        self,
        dataloader,
        num_samples: int = 128,
        head_mode: str = "o"
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        ffn_is_weight_mode = self.ffn_mode in ["weight_norm", "wn", "weight_taylor", "wt", "weight_ganda", "wg", "weight_wanda", "ww", "weight_wag", "wag", "wag_down", "wag_up", "wag_gate", "wag_all"]
        head_is_weight_mode = head_mode in ["weight_norm", "wn", "weight_taylor", "wt", "weight_ganda", "wg", "weight_wanda", "ww", "weight_wag", "wag", "wag_o", "wag_q", "wag_k", "wag_v", "wag_qkvo"]

        if ffn_is_weight_mode:
            if self.ffn_mode in ["weight_norm", "wn"]:
                ffn_scores = self._compute_weight_norm_ffn_scores()
            elif self.ffn_mode in ["weight_taylor", "wt"]:
                ffn_scores = self._compute_weight_taylor_ffn_scores(dataloader, num_samples)
            elif self.ffn_mode in ["weight_ganda", "wg"]:
                ffn_scores = self._compute_weight_ganda_ffn_scores(dataloader, num_samples)
            elif self.ffn_mode in ["weight_wanda", "ww"]:
                ffn_scores = self._compute_weight_wanda_ffn_scores(dataloader, num_samples)
            elif self.ffn_mode in ["weight_wag", "wag", "wag_down"]:
                ffn_scores = self._compute_weight_wag_ffn_scores(dataloader, num_samples, wag_mode="down")
            elif self.ffn_mode == "wag_up":
                ffn_scores = self._compute_weight_wag_ffn_scores(dataloader, num_samples, wag_mode="up")
            elif self.ffn_mode == "wag_gate":
                ffn_scores = self._compute_weight_wag_ffn_scores(dataloader, num_samples, wag_mode="gate")
            elif self.ffn_mode == "wag_all":
                ffn_scores = self._compute_weight_wag_ffn_scores(dataloader, num_samples, wag_mode="all")
            else:
                ffn_scores = self._compute_weight_norm_ffn_scores()  # fallback

            if head_is_weight_mode:
                head_scores = self.compute_taylor_head_scores(dataloader, num_samples, head_mode)
                return ffn_scores, head_scores

            head_scores = self.compute_taylor_head_scores(dataloader, num_samples, head_mode)
            return ffn_scores, head_scores

        if head_is_weight_mode:
            ffn_scores = self.compute_taylor_scores(dataloader, num_samples)
            head_scores = self.compute_taylor_head_scores(dataloader, num_samples, head_mode)
            return ffn_scores, head_scores

        self._register_joint_hooks(head_mode)

        if self.head_masks is not None or self.ffn_masks is not None:
            self._register_premasking_hooks()

        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True

        num_heads = self.config.num_attention_heads
        head_dim = self.hidden_size // num_heads

        if self.verbose:
            print(f"\n[Joint] Computing FFN + Head importance with {self._get_method_formula()}...")
            print(f"[Joint] FFN mode: {self.ffn_mode}, Head mode: {head_mode}")
            print(f"[Joint] Num samples: {num_samples}")

        ffn_accumulated = {i: torch.zeros(self.intermediate_size, device='cpu')
                          for i in range(self.num_layers)}
        if self.ffn_mode == "all":
            ffn_accumulated_up = {i: torch.zeros(self.intermediate_size, device='cpu')
                                 for i in range(self.num_layers)}
            ffn_accumulated_gate = {i: torch.zeros(self.intermediate_size, device='cpu')
                                   for i in range(self.num_layers)}

        head_accumulated = {i: torch.zeros(num_heads, device='cpu')
                           for i in range(self.num_layers)}

        num_processed = 0
        pbar = tqdm(dataloader, desc="[Joint] Calibration", disable=not self.verbose)

        for batch in pbar:
            if num_processed >= num_samples:
                break

            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                input_ids = batch.get('input_ids')
                attention_mask = batch.get('attention_mask')
            else:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device) if len(batch) > 1 else None

            labels = input_ids.clone()

            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()

                for layer_idx in range(self.num_layers):
                    if self.activations[layer_idx] and self.gradients[layer_idx]:
                        act = self.activations[layer_idx][-1]
                        grad = self.gradients[layer_idx][-1]

                        if act.shape == grad.shape:
                            weight = self.weights.get(layer_idx)
                            importance = self._compute_importance(act, grad, weight)
                            importance = importance.mean(dim=(0, 1))
                            ffn_accumulated[layer_idx] += importance.cpu()

                    if self.ffn_mode == "all" and self.up_activations[layer_idx] and self.up_gradients[layer_idx]:
                        up_act = self.up_activations[layer_idx][-1]
                        up_grad = self.up_gradients[layer_idx][-1]

                        if up_act.shape == up_grad.shape:
                            up_weight = self.up_weights.get(layer_idx)
                            importance_up = self._compute_importance(up_act, up_grad, up_weight)
                            importance_up = importance_up.mean(dim=(0, 1))
                            ffn_accumulated_up[layer_idx] += importance_up.cpu()

                    if self.ffn_mode == "all" and self.gate_activations[layer_idx] and self.gate_gradients[layer_idx]:
                        gate_act = self.gate_activations[layer_idx][-1]
                        gate_grad = self.gate_gradients[layer_idx][-1]

                        if gate_act.shape == gate_grad.shape:
                            gate_weight = self.gate_weights.get(layer_idx)
                            importance_gate = self._compute_importance(gate_act, gate_grad, gate_weight)
                            importance_gate = importance_gate.mean(dim=(0, 1))
                            ffn_accumulated_gate[layer_idx] += importance_gate.cpu()

                for layer_idx in range(self.num_layers):
                    if self.head_activations[layer_idx] and self.head_gradients[layer_idx]:
                        act = self.head_activations[layer_idx][-1]
                        grad = self.head_gradients[layer_idx][-1]

                        if act.shape == grad.shape:
                            weight = self.head_weights.get(layer_idx)
                            importance = self._compute_importance(act, grad, weight)
                            importance = importance.mean(dim=(0, 1))  # [num_heads * head_dim]

                            importance = importance.view(num_heads, head_dim).mean(dim=1)
                            head_accumulated[layer_idx] += importance.cpu()

                num_processed += input_ids.size(0)
                pbar.set_postfix({"samples": num_processed})

            except Exception as e:
                if self.verbose:
                    print(f"[Joint] Error: {e}")
                continue
            finally:
                self.model.zero_grad()
                for i in range(self.num_layers):
                    self.activations[i] = []
                    self.gradients[i] = []
                    if self.ffn_mode == "all":
                        self.up_activations[i] = []
                        self.up_gradients[i] = []
                        self.gate_activations[i] = []
                        self.gate_gradients[i] = []
                    self.head_activations[i] = []
                    self.head_gradients[i] = []

        self._remove_hooks()

        if num_processed > 0:
            batch_count = num_processed / dataloader.batch_size
            for layer_idx in range(self.num_layers):
                ffn_accumulated[layer_idx] /= batch_count
                head_accumulated[layer_idx] /= batch_count

                if self.ffn_mode == "all":
                    ffn_accumulated_up[layer_idx] /= batch_count
                    ffn_accumulated_gate[layer_idx] /= batch_count

        if self.ffn_mode == "all":
            for layer_idx in range(self.num_layers):
                ffn_accumulated[layer_idx] = (
                    ffn_accumulated[layer_idx] +
                    ffn_accumulated_up[layer_idx] +
                    ffn_accumulated_gate[layer_idx]
                ) / 3

        if self.verbose:
            print(f"\n[Joint] Computed scores for {num_processed} samples")
            print(f"[Joint] FFN scores: {self.num_layers} layers × {self.intermediate_size} neurons")
            print(f"[Joint] Head scores: {self.num_layers} layers × {num_heads} heads")

        return ffn_accumulated, head_accumulated

    def compute_taylor_head_scores(
        self,
        dataloader,
        num_samples: int = 128,
        head_mode: str = "o"
    ) -> Dict[int, torch.Tensor]:
        if head_mode in ["weight_norm", "wn"]:
            if self.verbose:
                print(f"\n[{self.importance_method.upper()}-Head] Computing Weight Norm importance...")
                print(f"[{self.importance_method.upper()}-Head] Mode: weight_norm (||W_o|| + ||W_q|| + ||W_k|| + ||W_v||)")

            return self._compute_weight_norm_head_scores()

        if head_mode in ["weight_taylor", "wt"]:
            if self.verbose:
                print(f"\n[{self.importance_method.upper()}-Head] Computing Weight Taylor importance...")
                print(f"[{self.importance_method.upper()}-Head] Mode: weight_taylor (|W ⊙ ∇W|)")

            return self._compute_weight_taylor_head_scores(dataloader, num_samples)

        if head_mode in ["weight_ganda", "wg"]:
            if self.verbose:
                print(f"\n[{self.importance_method.upper()}-Head] Computing Weight GandA importance...")
                print(f"[{self.importance_method.upper()}-Head] Mode: weight_ganda (|∇W × activation|)")

            return self._compute_weight_ganda_head_scores(dataloader, num_samples)

        if head_mode in ["weight_wanda", "ww"]:
            if self.verbose:
                print(f"\n[{self.importance_method.upper()}-Head] Computing Weight WANDA importance...")
                print(f"[{self.importance_method.upper()}-Head] Mode: weight_wanda (|W × activation|)")

            return self._compute_weight_wanda_head_scores(dataloader, num_samples)

        if head_mode in ["weight_wag", "wag", "wag_o"]:
            if self.verbose:
                print(f"\n[{self.importance_method.upper()}-Head] Computing Weight WAG importance (O)...")
                print(f"[{self.importance_method.upper()}-Head] Mode: weight_wag (|W × act × grad|)")

            return self._compute_weight_wag_head_scores(dataloader, num_samples, wag_mode="o")

        if head_mode == "wag_q":
            if self.verbose:
                print(f"\n[{self.importance_method.upper()}-Head] Computing Weight WAG importance (Q)...")
            return self._compute_weight_wag_head_scores(dataloader, num_samples, wag_mode="q")

        if head_mode == "wag_k":
            if self.verbose:
                print(f"\n[{self.importance_method.upper()}-Head] Computing Weight WAG importance (K)...")
            return self._compute_weight_wag_head_scores(dataloader, num_samples, wag_mode="k")

        if head_mode == "wag_v":
            if self.verbose:
                print(f"\n[{self.importance_method.upper()}-Head] Computing Weight WAG importance (V)...")
            return self._compute_weight_wag_head_scores(dataloader, num_samples, wag_mode="v")

        if head_mode == "wag_qkvo":
            if self.verbose:
                print(f"\n[{self.importance_method.upper()}-Head] Computing Weight WAG importance (Q+K+V+O)...")
            return self._compute_weight_wag_head_scores(dataloader, num_samples, wag_mode="qkvo")

        if head_mode == "qk":
            if self.verbose:
                print(f"\n[{self.importance_method.upper()}-Head] Computing QK activation-only importance...")
                print(f"[{self.importance_method.upper()}-Head] Mode: qk (|Q_act| × |K_act|, no gradient)")

            return self._compute_qk_activation_head_scores(dataloader, num_samples)

        if self._is_sti_method():
            if self.verbose:
                print(f"\n[{self.importance_method.upper()}-Head] Computing STI scores with {self._get_method_formula()}...")

            sti_scores = self._compute_head_sti_scores(dataloader, num_samples, head_mode)

            # sti_magnitude: STI × ganda
            if self.importance_method == "sti_magnitude":
                if self.verbose:
                    print(f"\n[{self.importance_method.upper()}-Head] Applying magnitude (ganda) correction...")

                original_method = self.importance_method
                self.importance_method = "ganda"
                ganda_scores = self.compute_taylor_head_scores(dataloader, num_samples, head_mode)
                self.importance_method = original_method

                ganda_normalized = self._normalize_scores(ganda_scores)
                for layer_idx in range(self.num_layers):
                    sti_scores[layer_idx] = sti_scores[layer_idx] * ganda_normalized[layer_idx]

            if self.verbose:
                print(f"\n[{self.importance_method.upper()}-Head] Final STI scores:")
                for layer_idx in range(min(3, self.num_layers)):
                    scores = sti_scores[layer_idx]
                    print(f"  Layer {layer_idx}: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

            return sti_scores

        self._register_head_hooks(head_mode)
        if self.head_masks is not None or self.ffn_masks is not None:
            self._register_premasking_hooks()
        self.model.train()

        for param in self.model.parameters():
            param.requires_grad = True

        num_heads = self.config.num_attention_heads
        head_dim = self.hidden_size // num_heads

        if self.verbose:
            print(f"\n[{self.importance_method.upper()}-Head] Computing head importance with {self._get_method_formula()}...")
            print(f"[{self.importance_method.upper()}-Head] Mode: {head_mode}, Heads: {num_heads}, Head dim: {head_dim}")

        accumulated = {i: torch.zeros(num_heads, device='cpu')
                       for i in range(self.num_layers)}

        num_processed = 0
        pbar = tqdm(dataloader, desc="[Taylor-Head] Calibration", disable=not self.verbose)

        for batch in pbar:
            if num_processed >= num_samples:
                break

            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                input_ids = batch.get('input_ids')
                attention_mask = batch.get('attention_mask')
            else:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device) if len(batch) > 1 else None

            labels = input_ids.clone()

            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()

                for layer_idx in range(self.num_layers):
                    if self.head_activations[layer_idx] and self.head_gradients[layer_idx]:
                        act = self.head_activations[layer_idx][-1]  # [batch, seq, num_heads * head_dim]
                        grad = self.head_gradients[layer_idx][-1]

                        if act.shape == grad.shape:
                            # Importance per head using selected method
                            weight = self.head_weights.get(layer_idx)
                            importance = self._compute_importance(act, grad, weight)  # [batch, seq, num_heads * head_dim]
                            # Reshape to [batch, seq, num_heads, head_dim]
                            importance = importance.view(importance.shape[0], importance.shape[1], num_heads, head_dim)
                            # Mean over batch, seq, head_dim -> [num_heads]
                            head_importance = importance.mean(dim=(0, 1, 3))
                            accumulated[layer_idx] += head_importance.cpu()

                num_processed += input_ids.size(0)
                pbar.set_postfix({"samples": num_processed})

            except Exception as e:
                if self.verbose:
                    print(f"[{self.importance_method.upper()}-Head] Error: {e}")
                continue
            finally:
                self.model.zero_grad(set_to_none=True)
                for i in range(self.num_layers):
                    self.head_activations[i] = []
                    self.head_gradients[i] = []
                del input_ids, attention_mask, labels
                if 'outputs' in dir():
                    del outputs
                if 'loss' in dir():
                    del loss
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self._remove_hooks()

        if num_processed > 0:
            batch_count = num_processed / dataloader.batch_size
            for layer_idx in range(self.num_layers):
                accumulated[layer_idx] /= batch_count

        if self.verbose:
            print(f"[{self.importance_method.upper()}-Head] Processed {num_processed} samples")
            for layer_idx in range(min(3, self.num_layers)):
                scores = accumulated[layer_idx]
                print(f"  Layer {layer_idx}: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

        if self._is_standalone_ntk_method():
            if self.verbose:
                print(f"\n[{self.importance_method.upper()}-Head] Computing standalone NTK scores...")

            if self.importance_method == "dynamics":
                standalone_scores = self._compute_head_dynamics_scores(dataloader, num_samples, head_mode)
            else:  # coupling
                standalone_scores = self._compute_head_coupling_scores(dataloader, num_samples, head_mode)

            if self.verbose:
                print(f"\n[{self.importance_method.upper()}-Head] Final scores:")
                for layer_idx in range(min(3, self.num_layers)):
                    scores = standalone_scores[layer_idx]
                    print(f"  Layer {layer_idx}: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

            return standalone_scores

        if self._is_ntk_method():
            if self.verbose:
                print(f"\n[{self.importance_method.upper()}-Head] Applying NTK modifiers...")

            dynamics_scores = None
            coupling_scores = None

            if self.importance_method in ["ganda_ntk_dyn", "ganda_ntk_combined"]:
                dynamics_scores = self._compute_head_dynamics_scores(dataloader, num_samples, head_mode)

            if self.importance_method in ["ganda_ntk_coupling", "ganda_ntk_combined"]:
                coupling_scores = self._compute_head_coupling_scores(dataloader, num_samples, head_mode)

            accumulated = self._apply_ntk_modifiers(accumulated, dynamics_scores, coupling_scores)

            if self.verbose:
                print(f"\n[{self.importance_method.upper()}-Head] Final scores after NTK modification:")
                for layer_idx in range(min(3, self.num_layers)):
                    scores = accumulated[layer_idx]
                    print(f"  Layer {layer_idx}: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

        return accumulated

    # ==================== Dimension Taylor Importance ====================

    def compute_taylor_dimension_scores(
        self,
        dataloader,
        num_samples: int = 128,
        head_mode: str = "o"
    ) -> Dict[int, torch.Tensor]:
        if self._is_sti_method():
            if self.verbose:
                print(f"\n[{self.importance_method.upper()}-Dim] Computing STI scores with {self._get_method_formula()}...")

            sti_scores = self._compute_dimension_sti_scores(dataloader, num_samples, head_mode)

            # sti_magnitude: STI × ganda
            if self.importance_method == "sti_magnitude":
                if self.verbose:
                    print(f"\n[{self.importance_method.upper()}-Dim] Applying magnitude (ganda) correction...")

                original_method = self.importance_method
                self.importance_method = "ganda"
                ganda_scores = self.compute_taylor_dimension_scores(dataloader, num_samples, head_mode)
                self.importance_method = original_method

                ganda_normalized = self._normalize_scores(ganda_scores)
                for layer_idx in range(self.num_layers):
                    sti_scores[layer_idx] = sti_scores[layer_idx] * ganda_normalized[layer_idx]

            if self.verbose:
                print(f"\n[{self.importance_method.upper()}-Dim] Final STI scores:")
                for layer_idx in range(min(3, self.num_layers)):
                    scores = sti_scores[layer_idx]
                    print(f"  Layer {layer_idx}: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

            return sti_scores

        self._register_head_hooks(head_mode)
        self.model.train()

        for param in self.model.parameters():
            param.requires_grad = True

        num_heads = self.config.num_attention_heads
        head_dim = self.hidden_size // num_heads

        if self.verbose:
            print(f"\n[{self.importance_method.upper()}-Dim] Computing dimension importance with {self._get_method_formula()}...")
            print(f"[{self.importance_method.upper()}-Dim] Head dim: {head_dim}")

        accumulated = {i: torch.zeros(head_dim, device='cpu')
                       for i in range(self.num_layers)}

        num_processed = 0
        pbar = tqdm(dataloader, desc="[Taylor-Dim] Calibration", disable=not self.verbose)

        for batch in pbar:
            if num_processed >= num_samples:
                break

            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                input_ids = batch.get('input_ids')
                attention_mask = batch.get('attention_mask')
            else:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device) if len(batch) > 1 else None

            labels = input_ids.clone()

            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()

                if num_processed == 0 and self.verbose:
                    act_counts = [len(self.head_activations[i]) for i in range(min(3, self.num_layers))]
                    grad_counts = [len(self.head_gradients[i]) for i in range(min(3, self.num_layers))]
                    print(f"[DEBUG] After 1st batch - Activations captured (layers 0-2): {act_counts}")
                    print(f"[DEBUG] After 1st batch - Gradients captured (layers 0-2): {grad_counts}")

                    for layer_idx in range(min(3, self.num_layers)):
                        has_act = len(self.head_activations[layer_idx]) > 0
                        has_grad = len(self.head_gradients[layer_idx]) > 0
                        if has_act:
                            act = self.head_activations[layer_idx][-1]
                            print(f"[DEBUG] Layer {layer_idx} activation shape: {act.shape}, sum: {act.sum().item():.6f}")
                        if has_grad:
                            grad = self.head_gradients[layer_idx][-1]
                            print(f"[DEBUG] Layer {layer_idx} gradient shape: {grad.shape}, sum: {grad.sum().item():.6f}")

                for layer_idx in range(self.num_layers):
                    if self.head_activations[layer_idx] and self.head_gradients[layer_idx]:
                        act = self.head_activations[layer_idx][-1]
                        grad = self.head_gradients[layer_idx][-1]

                        if act.shape == grad.shape:
                            # Importance using selected method
                            weight = self.head_weights.get(layer_idx)
                            importance = self._compute_importance(act, grad, weight)
                            # Reshape to [batch, seq, num_heads, head_dim]
                            importance = importance.view(importance.shape[0], importance.shape[1], num_heads, head_dim)
                            # Mean over batch, seq, heads -> [head_dim]
                            dim_importance = importance.mean(dim=(0, 1, 2))
                            accumulated[layer_idx] += dim_importance.cpu()

                num_processed += input_ids.size(0)
                pbar.set_postfix({"samples": num_processed})

            except Exception as e:
                if self.verbose:
                    print(f"[{self.importance_method.upper()}-Dim] Error: {e}")
                continue
            finally:
                self.model.zero_grad()
                for i in range(self.num_layers):
                    self.head_activations[i] = []
                    self.head_gradients[i] = []

        self._remove_hooks()

        if num_processed > 0:
            batch_count = num_processed / dataloader.batch_size
            for layer_idx in range(self.num_layers):
                accumulated[layer_idx] /= batch_count

        if self.verbose:
            print(f"[{self.importance_method.upper()}-Dim] Processed {num_processed} samples")
            for layer_idx in range(min(3, self.num_layers)):
                scores = accumulated[layer_idx]
                print(f"  Layer {layer_idx}: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

        if self._is_standalone_ntk_method():
            if self.verbose:
                print(f"\n[{self.importance_method.upper()}-Dim] Computing standalone NTK scores...")

            if self.importance_method == "dynamics":
                standalone_scores = self._compute_dimension_dynamics_scores(dataloader, num_samples, head_mode)
            else:  # coupling
                standalone_scores = self._compute_dimension_coupling_scores(dataloader, num_samples, head_mode)

            if self.verbose:
                print(f"\n[{self.importance_method.upper()}-Dim] Final scores:")
                for layer_idx in range(min(3, self.num_layers)):
                    scores = standalone_scores[layer_idx]
                    print(f"  Layer {layer_idx}: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

            return standalone_scores

        if self._is_ntk_method():
            if self.verbose:
                print(f"\n[{self.importance_method.upper()}-Dim] Applying NTK modifiers...")

            dynamics_scores = None
            coupling_scores = None

            if self.importance_method in ["ganda_ntk_dyn", "ganda_ntk_combined"]:
                dynamics_scores = self._compute_dimension_dynamics_scores(dataloader, num_samples, head_mode)

            if self.importance_method in ["ganda_ntk_coupling", "ganda_ntk_combined"]:
                coupling_scores = self._compute_dimension_coupling_scores(dataloader, num_samples, head_mode)

            accumulated = self._apply_ntk_modifiers(accumulated, dynamics_scores, coupling_scores)

            if self.verbose:
                print(f"\n[{self.importance_method.upper()}-Dim] Final scores after NTK modification:")
                for layer_idx in range(min(3, self.num_layers)):
                    scores = accumulated[layer_idx]
                    print(f"  Layer {layer_idx}: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

        return accumulated

    def compute_taylor_dimension_scores_qkv(
        self,
        dataloader,
        num_samples: int = 128
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        num_heads = self.config.num_attention_heads
        head_dim = self.hidden_size // num_heads

        if self.verbose:
            print(f"\n[{self.importance_method.upper()}-Dim-QKV] Computing separate Q, K, V dimension importance...")
            print(f"[{self.importance_method.upper()}-Dim-QKV] Head dim: {head_dim}, Num heads: {num_heads}")

        q_scores = self._compute_single_proj_dimension_scores(dataloader, num_samples, "q")
        k_scores = self._compute_single_proj_dimension_scores(dataloader, num_samples, "k")
        v_scores = self._compute_single_proj_dimension_scores(dataloader, num_samples, "v")

        # QK = (Q + K) / 2
        qk_dim_scores = {}
        v_dim_scores = {}

        for layer_idx in range(self.num_layers):
            qk_dim_scores[layer_idx] = (q_scores[layer_idx] + k_scores[layer_idx]) / 2
            v_dim_scores[layer_idx] = v_scores[layer_idx]

        if self.verbose:
            print(f"\n[{self.importance_method.upper()}-Dim-QKV] QK scores (Q+K)/2:")
            for layer_idx in range(min(3, self.num_layers)):
                scores = qk_dim_scores[layer_idx]
                print(f"  Layer {layer_idx}: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")
            print(f"\n[{self.importance_method.upper()}-Dim-QKV] V scores:")
            for layer_idx in range(min(3, self.num_layers)):
                scores = v_dim_scores[layer_idx]
                print(f"  Layer {layer_idx}: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

        return qk_dim_scores, v_dim_scores

    def _compute_single_proj_dimension_scores(
        self,
        dataloader,
        num_samples: int,
        proj_type: str  # "q", "k", or "v"
    ) -> Dict[int, torch.Tensor]:
        self._register_qkv_hooks(proj_type)
        self.model.train()

        for param in self.model.parameters():
            param.requires_grad = True

        num_heads = self.config.num_attention_heads
        head_dim = self.hidden_size // num_heads

        accumulated = {i: torch.zeros(head_dim, device='cpu') for i in range(self.num_layers)}

        num_processed = 0
        pbar = tqdm(dataloader, desc=f"[{self.importance_method.upper()}-Dim-{proj_type.upper()}]", disable=not self.verbose)

        for batch in pbar:
            if num_processed >= num_samples:
                break

            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                input_ids = batch.get('input_ids')
                attention_mask = batch.get('attention_mask')
            else:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device) if len(batch) > 1 else None

            labels = input_ids.clone()

            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()

                for layer_idx in range(self.num_layers):
                    if self.head_activations[layer_idx] and self.head_gradients[layer_idx]:
                        act = self.head_activations[layer_idx][-1]
                        grad = self.head_gradients[layer_idx][-1]

                        if act.shape == grad.shape:
                            weight = self.head_weights.get(layer_idx)
                            importance = self._compute_importance(act, grad, weight)
                            # Reshape to [batch, seq, num_heads, head_dim]
                            importance = importance.view(importance.shape[0], importance.shape[1], num_heads, head_dim)
                            # Mean over batch, seq, heads -> [head_dim]
                            dim_importance = importance.mean(dim=(0, 1, 2))
                            accumulated[layer_idx] += dim_importance.cpu()

                num_processed += input_ids.size(0)
                pbar.set_postfix({"samples": num_processed})

            except Exception as e:
                if self.verbose:
                    print(f"[{self.importance_method.upper()}-Dim-{proj_type.upper()}] Error: {e}")
                continue
            finally:
                self.model.zero_grad()
                for i in range(self.num_layers):
                    self.head_activations[i] = []
                    self.head_gradients[i] = []

        self._remove_hooks()

        if num_processed > 0:
            batch_count = num_processed / dataloader.batch_size
            for layer_idx in range(self.num_layers):
                accumulated[layer_idx] /= batch_count

        return accumulated

    def _register_qkv_hooks(self, proj_type: str):
        self.head_activations = {i: [] for i in range(self.num_layers)}
        self.head_gradients = {i: [] for i in range(self.num_layers)}
        self.head_weights = {}

        for layer_idx in range(self.num_layers):
            attn = self._get_attention_module(layer_idx)
            if attn is None:
                continue

            if proj_type == "q":
                target = attn.q_proj
            elif proj_type == "k":
                target = attn.k_proj
            else:  # "v"
                target = attn.v_proj

            if hasattr(target, 'base_layer'):
                target_module = target.base_layer
            else:
                target_module = target

            self.head_weights[layer_idx] = target_module.weight.data.detach()

            def make_forward_hook(idx):
                def hook(module, input, output):
                    if output is not None:
                        self.head_activations[idx].append(output.detach())
                return hook

            def make_backward_hook(idx):
                def hook(module, grad_input, grad_output):
                    if grad_output is not None and len(grad_output) > 0 and grad_output[0] is not None:
                        self.head_gradients[idx].append(grad_output[0].detach())
                return hook

            h1 = target_module.register_forward_hook(make_forward_hook(layer_idx))
            h2 = target_module.register_full_backward_hook(make_backward_hook(layer_idx))
            self.hooks.append(h1)
            self.hooks.append(h2)

    # ==================== Joint All Importance (FFN + Head + Dim + Embedding) ====================

    def compute_all_importance(
        self,
        dataloader,
        num_samples: int = 128,
        head_mode: str = "o",
        compute_ffn: bool = True,
        compute_head: bool = True,
        compute_dim: bool = True,
        compute_embedding: bool = True
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, torch.Tensor], torch.Tensor]:
        if self.verbose:
            print(f"\n[Joint All] Computing all importance scores in single pass...")
            print(f"[Joint All] Method: {self.importance_method}")
            print(f"[Joint All] FFN mode: {self.ffn_mode}")
            print(f"[Joint All] FFN: {compute_ffn}, Head: {compute_head}, Dim: {compute_dim}, Embedding: {compute_embedding}")

        ffn_is_weight_mode = self.ffn_mode in ["weight_norm", "wn", "weight_taylor", "wt", "weight_ganda", "wg", "weight_wanda", "ww", "weight_wag", "wag", "wag_down", "wag_up", "wag_gate", "wag_all"]

        if compute_ffn and ffn_is_weight_mode:
            if self.verbose:
                print(f"[Joint All] Using weight-based FFN mode: {self.ffn_mode}")

            if self.ffn_mode in ["weight_norm", "wn"]:
                ffn_scores = self._compute_weight_norm_ffn_scores()
            elif self.ffn_mode in ["weight_taylor", "wt"]:
                ffn_scores = self._compute_weight_taylor_ffn_scores(dataloader, num_samples)
            elif self.ffn_mode in ["weight_ganda", "wg"]:
                ffn_scores = self._compute_weight_ganda_ffn_scores(dataloader, num_samples)
            elif self.ffn_mode in ["weight_wanda", "ww"]:
                ffn_scores = self._compute_weight_wanda_ffn_scores(dataloader, num_samples)
            elif self.ffn_mode in ["weight_wag", "wag", "wag_down"]:
                ffn_scores = self._compute_weight_wag_ffn_scores(dataloader, num_samples, wag_mode="down")
            elif self.ffn_mode == "wag_up":
                ffn_scores = self._compute_weight_wag_ffn_scores(dataloader, num_samples, wag_mode="up")
            elif self.ffn_mode == "wag_gate":
                ffn_scores = self._compute_weight_wag_ffn_scores(dataloader, num_samples, wag_mode="gate")
            elif self.ffn_mode == "wag_all":
                ffn_scores = self._compute_weight_wag_ffn_scores(dataloader, num_samples, wag_mode="all")
            else:
                ffn_scores = self._compute_weight_norm_ffn_scores()  # fallback

            _, head_scores, dim_scores, embed_scores = self.compute_all_importance(
                dataloader, num_samples, head_mode,
                compute_ffn=False, compute_head=compute_head,
                compute_dim=compute_dim, compute_embedding=compute_embedding
            )
            return ffn_scores, head_scores, dim_scores, embed_scores

        head_is_weight_mode = head_mode in ["weight_norm", "wn", "weight_taylor", "wt", "weight_ganda", "wg", "weight_wanda", "ww", "weight_wag", "wag", "wag_o", "wag_q", "wag_k", "wag_v", "wag_qkvo"]

        if compute_head and head_is_weight_mode:
            if self.verbose:
                print(f"[Joint All] Using weight-based Head mode: {head_mode}")

            head_scores = self.compute_taylor_head_scores(dataloader, num_samples, head_mode)

            ffn_scores, _, dim_scores, embed_scores = self.compute_all_importance(
                dataloader, num_samples, head_mode,
                compute_ffn=compute_ffn, compute_head=False,
                compute_dim=compute_dim, compute_embedding=compute_embedding
            )
            return ffn_scores, head_scores, dim_scores, embed_scores

        self._remove_hooks()
        self.hooks = []

        if self.head_masks is not None or self.ffn_masks is not None or self.dim_masks is not None or self.embedding_mask is not None:
            self._register_premasking_hooks()
            if self.verbose:
                print(f"[Joint All] Pre-masking hooks registered for iterative pruning")

        if self.verbose:
            print(f"[Joint All] Head mode: {head_mode} ({'v_proj output' if head_mode == 'v' else 'o_proj input'})")

        num_heads = self.config.num_attention_heads
        head_dim = self.hidden_size // num_heads

        # Storage for activations and gradients
        ffn_activations = {i: [] for i in range(self.num_layers)}
        ffn_gradients = {i: [] for i in range(self.num_layers)}
        ffn_weights = {}
        head_activations = {i: [] for i in range(self.num_layers)}
        head_gradients = {i: [] for i in range(self.num_layers)}
        head_weights = {}
        attn_ln_activations = {i: [] for i in range(self.num_layers)}
        ffn_ln_activations = {i: [] for i in range(self.num_layers)}

        # Register hooks for all types
        for layer_idx in range(self.num_layers):
            layer = self.model.model.layers[layer_idx]
            mlp = self._get_mlp_module(layer_idx)
            attn = self._get_attention_module(layer_idx)

            # FFN hooks: down_proj/fc2 input
            if compute_ffn and mlp is not None:
                down_proj = get_down_proj_module(mlp)
                if down_proj is None:
                    continue
                target = down_proj.base_layer if hasattr(down_proj, 'base_layer') else down_proj

                ffn_weights[layer_idx] = target.weight.data.detach()

                def make_ffn_fwd(idx):
                    def hook(module, input, output):
                        if len(input) > 0 and input[0] is not None:
                            ffn_activations[idx].append(input[0].detach())
                    return hook

                def make_ffn_bwd(idx):
                    def hook(module, grad_input, grad_output):
                        if grad_input is not None and len(grad_input) > 0 and grad_input[0] is not None:
                            ffn_gradients[idx].append(grad_input[0].detach())
                    return hook

                self.hooks.append(target.register_forward_hook(make_ffn_fwd(layer_idx)))
                self.hooks.append(target.register_full_backward_hook(make_ffn_bwd(layer_idx)))

            # - "v": v_proj output
            if (compute_head or compute_dim) and attn is not None:
                if head_mode == "v":
                    v_proj = attn.v_proj
                    target = v_proj.base_layer if hasattr(v_proj, 'base_layer') else v_proj

                    head_weights[layer_idx] = target.weight.data.detach()

                    def make_head_fwd(idx):
                        def hook(module, input, output):
                            if output is not None:
                                head_activations[idx].append(output.detach())
                        return hook

                    def make_head_bwd(idx):
                        def hook(module, grad_input, grad_output):
                            if grad_output is not None and len(grad_output) > 0 and grad_output[0] is not None:
                                head_gradients[idx].append(grad_output[0].detach())
                        return hook
                else:
                    o_proj = get_attn_output_proj(attn)
                    if o_proj is None:
                        continue
                    target = o_proj.base_layer if hasattr(o_proj, 'base_layer') else o_proj

                    head_weights[layer_idx] = target.weight.data.detach()

                    def make_head_fwd(idx):
                        def hook(module, input, output):
                            if len(input) > 0 and input[0] is not None:
                                head_activations[idx].append(input[0].detach())
                        return hook

                    def make_head_bwd(idx):
                        def hook(module, grad_input, grad_output):
                            if grad_input is not None and len(grad_input) > 0 and grad_input[0] is not None:
                                head_gradients[idx].append(grad_input[0].detach())
                        return hook

                self.hooks.append(target.register_forward_hook(make_head_fwd(layer_idx)))
                self.hooks.append(target.register_full_backward_hook(make_head_bwd(layer_idx)))

            # Embedding hooks: layernorm outputs
            if compute_embedding:
                # input_layernorm output (-> Q, K, V)
                def make_attn_ln_hook(idx):
                    def hook(module, input, output):
                        if output is not None:
                            output.requires_grad_(True)
                            output.retain_grad()
                            attn_ln_activations[idx].append(output)
                    return hook
                self.hooks.append(layer.input_layernorm.register_forward_hook(make_attn_ln_hook(layer_idx)))

                # post_attention_layernorm output (-> FFN)
                def make_ffn_ln_hook(idx):
                    def hook(module, input, output):
                        if output is not None:
                            output.requires_grad_(True)
                            output.retain_grad()
                            ffn_ln_activations[idx].append(output)
                    return hook
                self.hooks.append(layer.post_attention_layernorm.register_forward_hook(make_ffn_ln_hook(layer_idx)))

        # Initialize accumulators
        ffn_accumulated = {i: torch.zeros(self.intermediate_size, device='cpu') for i in range(self.num_layers)}
        head_accumulated = {i: torch.zeros(num_heads, device='cpu') for i in range(self.num_layers)}
        dim_accumulated = {i: torch.zeros(head_dim, device='cpu') for i in range(self.num_layers)}
        embed_accumulated = torch.zeros(self.hidden_size, device='cpu')

        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True

        num_processed = 0
        pbar = tqdm(dataloader, desc="[Joint All] Calibration", disable=not self.verbose)

        for batch in pbar:
            if num_processed >= num_samples:
                break

            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                input_ids = batch.get('input_ids')
                attention_mask = batch.get('attention_mask')
            else:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device) if len(batch) > 1 else None

            labels = input_ids.clone()

            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()

                # Process FFN importance
                if compute_ffn:
                    for layer_idx in range(self.num_layers):
                        if ffn_activations[layer_idx] and ffn_gradients[layer_idx]:
                            act = ffn_activations[layer_idx][-1]
                            grad = ffn_gradients[layer_idx][-1]
                            weight = ffn_weights.get(layer_idx)
                            importance = self._compute_importance(act, grad, weight)
                            if self.use_correct_normalization:
                                importance = importance.sum(dim=(0, 1))  # [intermediate_size]
                            else:
                                importance = importance.mean(dim=(0, 1))  # [intermediate_size] (legacy)
                            ffn_accumulated[layer_idx] += importance.cpu()

                # Process Head and Dimension importance
                if compute_head or compute_dim:
                    for layer_idx in range(self.num_layers):
                        if head_activations[layer_idx] and head_gradients[layer_idx]:
                            act = head_activations[layer_idx][-1]  # [batch, seq, hidden_size]
                            grad = head_gradients[layer_idx][-1]
                            batch_size, seq_len, hidden = act.shape

                            weight = head_weights.get(layer_idx)
                            importance = self._compute_importance(act, grad, weight)
                            # Reshape to [batch, seq, num_heads, head_dim]
                            importance = importance.view(batch_size, seq_len, num_heads, head_dim)

                            if compute_head:
                                # Head importance: sum/mean over batch, seq, head_dim -> [num_heads]
                                if self.use_correct_normalization:
                                    head_imp = importance.sum(dim=(0, 1, 3))
                                else:
                                    head_imp = importance.mean(dim=(0, 1, 3))  # legacy
                                head_accumulated[layer_idx] += head_imp.cpu()

                            if compute_dim:
                                # Dimension importance: sum/mean over batch, seq, num_heads -> [head_dim]
                                if self.use_correct_normalization:
                                    dim_imp = importance.sum(dim=(0, 1, 2))
                                else:
                                    dim_imp = importance.mean(dim=(0, 1, 2))  # legacy
                                dim_accumulated[layer_idx] += dim_imp.cpu()

                # Process Embedding importance
                if compute_embedding:
                    for layer_idx in range(self.num_layers):
                        # Attention layernorm
                        if attn_ln_activations[layer_idx]:
                            act = attn_ln_activations[layer_idx][-1]
                            if act.grad is not None:
                                if self.use_correct_normalization:
                                    importance = (act * act.grad).abs().sum(dim=(0, 1))  # [hidden_size]
                                else:
                                    importance = (act * act.grad).abs().mean(dim=(0, 1))  # legacy
                                embed_accumulated += importance.cpu()

                        # FFN layernorm
                        if ffn_ln_activations[layer_idx]:
                            act = ffn_ln_activations[layer_idx][-1]
                            if act.grad is not None:
                                if self.use_correct_normalization:
                                    importance = (act * act.grad).abs().sum(dim=(0, 1))
                                else:
                                    importance = (act * act.grad).abs().mean(dim=(0, 1))  # legacy
                                embed_accumulated += importance.cpu()

                # Clear for next batch
                for layer_idx in range(self.num_layers):
                    ffn_activations[layer_idx].clear()
                    ffn_gradients[layer_idx].clear()
                    head_activations[layer_idx].clear()
                    head_gradients[layer_idx].clear()
                    attn_ln_activations[layer_idx].clear()
                    ffn_ln_activations[layer_idx].clear()

                self.model.zero_grad(set_to_none=True)
                num_processed += input_ids.shape[0]
                pbar.set_postfix({'processed': num_processed})

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                if self.verbose:
                    print(f"[Joint All] Error in batch: {e}")
                continue

        # Cleanup
        self._remove_hooks()
        self._remove_premasking_hooks()

        # Normalize by number of samples
        if num_processed > 0:
            for layer_idx in range(self.num_layers):
                ffn_accumulated[layer_idx] /= num_processed
                head_accumulated[layer_idx] /= num_processed
                dim_accumulated[layer_idx] /= num_processed
            embed_accumulated /= (num_processed * self.num_layers * 2)  # 2 layernorms per layer

        if self.verbose:
            print(f"\n[Joint All] Computed importance scores for {num_processed} samples")
            for layer_idx in range(min(3, self.num_layers)):
                print(f"  Layer {layer_idx}:")
                if compute_ffn:
                    print(f"    FFN: min={ffn_accumulated[layer_idx].min():.6f}, max={ffn_accumulated[layer_idx].max():.6f}")
                if compute_head:
                    print(f"    Head: min={head_accumulated[layer_idx].min():.6f}, max={head_accumulated[layer_idx].max():.6f}")
                if compute_dim:
                    print(f"    Dim: min={dim_accumulated[layer_idx].min():.6f}, max={dim_accumulated[layer_idx].max():.6f}")
            if compute_embedding:
                print(f"  Embedding: min={embed_accumulated.min():.6f}, max={embed_accumulated.max():.6f}")

        from pruning.distributed_utils import all_reduce_dict, all_reduce_sum
        all_reduce_dict(ffn_accumulated)
        all_reduce_dict(head_accumulated)
        all_reduce_dict(dim_accumulated)
        embed_accumulated = all_reduce_sum(embed_accumulated)

        return ffn_accumulated, head_accumulated, dim_accumulated, embed_accumulated

    # ==================== Embedding Importance Methods ====================

    def compute_embedding_importance(
        self,
        dataloader,
        num_samples: int = 128,
        embedding_mode: str = "all"
    ) -> torch.Tensor:
        if self.verbose:
            print(f"\n[Embedding Importance] Computing embedding importance...")
            print(f"[Embedding Importance] Method: {self.importance_method}")
            print(f"[Embedding Importance] Mode: {embedding_mode}")

        if self.importance_method in ["wanda"]:
            return self._compute_wanda_embedding_importance(dataloader, num_samples, embedding_mode)
        elif self.importance_method in ["taylor"]:
            return self._compute_taylor_embedding_importance(dataloader, num_samples, embedding_mode)
        else:  # ganda (default)
            return self._compute_ganda_embedding_importance(dataloader, num_samples, embedding_mode)

    def _compute_ganda_embedding_importance(
        self,
        dataloader,
        num_samples: int = 128,
        embedding_mode: str = "all"
    ) -> torch.Tensor:
        if self.verbose:
            print(f"[Embedding-GandA] Computing |grad × activation| importance...")
            print(f"[Embedding-GandA] Measuring at: input_layernorm (→QKV) + post_attention_layernorm (→FFN)")

        attn_activations = {i: [] for i in range(self.num_layers)}
        ffn_activations = {i: [] for i in range(self.num_layers)}
        self.hooks = []

        for layer_idx in range(self.num_layers):
            layer = self.model.model.layers[layer_idx]

            def make_attn_hook(idx):
                def hook(module, input, output):
                    if output is not None:
                        output.requires_grad_(True)
                        output.retain_grad()
                        attn_activations[idx].append(output)
                return hook
            self.hooks.append(layer.input_layernorm.register_forward_hook(make_attn_hook(layer_idx)))

            def make_ffn_hook(idx):
                def hook(module, input, output):
                    if output is not None:
                        output.requires_grad_(True)
                        output.retain_grad()
                        ffn_activations[idx].append(output)
                return hook
            self.hooks.append(layer.post_attention_layernorm.register_forward_hook(make_ffn_hook(layer_idx)))

        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True

        accumulated = torch.zeros(self.hidden_size, device='cpu')
        num_processed = 0
        pbar = tqdm(dataloader, desc="[Embedding-GandA] Calibration", disable=not self.verbose)

        for batch in pbar:
            if num_processed >= num_samples:
                break

            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                input_ids = batch.get('input_ids')
                attention_mask = batch.get('attention_mask')
            else:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device) if len(batch) > 1 else None

            labels = input_ids.clone()

            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()

                for layer_idx in range(self.num_layers):
                    if attn_activations[layer_idx]:
                        act = attn_activations[layer_idx][-1]
                        if act.grad is not None:
                            grad = act.grad.detach().float()
                            val = act.detach().float()
                            ganda = (grad.abs() * val.abs()).sum(dim=(0, 1))
                            accumulated += ganda.cpu()

                    if ffn_activations[layer_idx]:
                        act = ffn_activations[layer_idx][-1]
                        if act.grad is not None:
                            grad = act.grad.detach().float()
                            val = act.detach().float()
                            ganda = (grad.abs() * val.abs()).sum(dim=(0, 1))
                            accumulated += ganda.cpu()

                num_processed += input_ids.size(0)
                pbar.set_postfix({"samples": num_processed})

            except Exception as e:
                if self.verbose:
                    print(f"[Embedding-GandA] Error: {e}")
                continue
            finally:
                self.model.zero_grad(set_to_none=True)
                for i in range(self.num_layers):
                    attn_activations[i] = []
                    ffn_activations[i] = []
                del input_ids, attention_mask, labels
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        if num_processed > 0:
            accumulated /= num_processed

        from pruning.distributed_utils import all_reduce_sum
        accumulated = all_reduce_sum(accumulated)

        if self.verbose:
            print(f"[Embedding-GandA] Processed {num_processed} samples")
            print(f"[Embedding-GandA] Measured {self.num_layers} layers × 2 positions (attn input + ffn input)")
            print(f"[Embedding-GandA] Importance stats: min={accumulated.min():.6f}, max={accumulated.max():.6f}, mean={accumulated.mean():.6f}")

        return accumulated

    def _compute_taylor_embedding_importance(
        self,
        dataloader,
        num_samples: int = 128,
        embedding_mode: str = "all"
    ) -> torch.Tensor:
        if self.verbose:
            print(f"[Embedding-Taylor] Computing |grad × weight| importance...")

        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True

        accumulated = torch.zeros(self.hidden_size, device='cpu')
        num_processed = 0
        pbar = tqdm(dataloader, desc="[Embedding-Taylor] Calibration", disable=not self.verbose)

        for batch in pbar:
            if num_processed >= num_samples:
                break

            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                input_ids = batch.get('input_ids')
                attention_mask = batch.get('attention_mask')
            else:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device) if len(batch) > 1 else None

            labels = input_ids.clone()

            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()

                batch_importance = torch.zeros(self.hidden_size, device='cpu')

                for layer_idx in range(self.num_layers):
                    layer = self.model.model.layers[layer_idx]
                    attn = layer.self_attn
                    mlp = layer.mlp

                    for proj in [attn.q_proj, attn.k_proj, attn.v_proj]:
                        weight = proj.base_layer.weight if hasattr(proj, 'base_layer') else proj.weight
                        if weight.grad is not None:
                            taylor = (weight.grad.abs() * weight.data.abs()).sum(dim=0)
                            batch_importance += taylor.cpu()

                    o_proj = get_attn_output_proj(attn)
                    if o_proj is not None:
                        weight = o_proj.base_layer.weight if hasattr(o_proj, 'base_layer') else o_proj.weight
                        if weight.grad is not None:
                            taylor = (weight.grad.abs() * weight.data.abs()).sum(dim=1)
                            batch_importance += taylor.cpu()

                    for proj in get_up_proj_modules(mlp):
                        weight = proj.base_layer.weight if hasattr(proj, 'base_layer') else proj.weight
                        if weight.grad is not None:
                            taylor = (weight.grad.abs() * weight.data.abs()).sum(dim=0)
                            batch_importance += taylor.cpu()

                    down_proj = get_down_proj_module(mlp)
                    if down_proj is not None:
                        weight = down_proj.base_layer.weight if hasattr(down_proj, 'base_layer') else down_proj.weight
                        if weight.grad is not None:
                            taylor = (weight.grad.abs() * weight.data.abs()).sum(dim=1)
                            batch_importance += taylor.cpu()

                accumulated += batch_importance
                num_processed += input_ids.size(0)
                pbar.set_postfix({"samples": num_processed})

            except Exception as e:
                if self.verbose:
                    print(f"[Embedding-Taylor] Error: {e}")
                continue
            finally:
                self.model.zero_grad(set_to_none=True)
                del input_ids, attention_mask, labels
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if num_processed > 0:
            accumulated /= num_processed

        from pruning.distributed_utils import all_reduce_sum
        accumulated = all_reduce_sum(accumulated)

        if self.verbose:
            print(f"[Embedding-Taylor] Processed {num_processed} samples")
            print(f"[Embedding-Taylor] Importance stats: min={accumulated.min():.6f}, max={accumulated.max():.6f}, mean={accumulated.mean():.6f}")

        return accumulated

    def _compute_wanda_embedding_importance(
        self,
        dataloader,
        num_samples: int = 128,
        embedding_mode: str = "all"
    ) -> torch.Tensor:
        if self.verbose:
            print(f"[Embedding-WANDA] Computing |weight × activation| importance...")

        attn_activations = {i: [] for i in range(self.num_layers)}  # input_layernorm output
        ffn_activations = {i: [] for i in range(self.num_layers)}   # post_attention_layernorm output
        self.hooks = []

        for layer_idx in range(self.num_layers):
            layer = self.model.model.layers[layer_idx]

            # 1. Attention input: input_layernorm output (→ Q, K, V)
            def make_attn_hook(idx):
                def hook(module, input, output):
                    if output is not None:
                        out = output.detach()
                        if out.dim() == 3:
                            out = out.view(-1, out.size(-1))
                        rms = torch.sqrt((out ** 2).mean(dim=0))
                        attn_activations[idx].append(rms.cpu())
                return hook

            self.hooks.append(layer.input_layernorm.register_forward_hook(make_attn_hook(layer_idx)))

            # 2. FFN input: post_attention_layernorm output (→ gate_proj, up_proj)
            def make_ffn_hook(idx):
                def hook(module, input, output):
                    if output is not None:
                        out = output.detach()
                        if out.dim() == 3:
                            out = out.view(-1, out.size(-1))
                        rms = torch.sqrt((out ** 2).mean(dim=0))
                        ffn_activations[idx].append(rms.cpu())
                return hook

            self.hooks.append(layer.post_attention_layernorm.register_forward_hook(make_ffn_hook(layer_idx)))

        self.model.eval()
        num_processed = 0
        pbar = tqdm(dataloader, desc="[Embedding-WANDA] Collecting activations", disable=not self.verbose)

        with torch.no_grad():
            for batch in pbar:
                if num_processed >= num_samples:
                    break

                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}
                    input_ids = batch.get('input_ids')
                    attention_mask = batch.get('attention_mask')
                else:
                    input_ids = batch[0].to(self.device)
                    attention_mask = batch[1].to(self.device) if len(batch) > 1 else None

                try:
                    _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    num_processed += input_ids.size(0)
                    pbar.set_postfix({"samples": num_processed})
                except Exception as e:
                    if self.verbose:
                        print(f"[Embedding-WANDA] Error: {e}")
                    continue

        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        attn_act_mean = {}
        ffn_act_mean = {}
        for layer_idx in range(self.num_layers):
            if attn_activations[layer_idx]:
                attn_act_mean[layer_idx] = torch.stack(attn_activations[layer_idx]).mean(dim=0)
            else:
                attn_act_mean[layer_idx] = torch.ones(self.hidden_size)

            if ffn_activations[layer_idx]:
                ffn_act_mean[layer_idx] = torch.stack(ffn_activations[layer_idx]).mean(dim=0)
            else:
                ffn_act_mean[layer_idx] = torch.ones(self.hidden_size)

        accumulated = torch.zeros(self.hidden_size, device='cpu')

        for layer_idx in range(self.num_layers):
            layer = self.model.model.layers[layer_idx]
            attn = layer.self_attn
            mlp = layer.mlp
            attn_act = attn_act_mean[layer_idx]
            ffn_act = ffn_act_mean[layer_idx]

            for proj in [attn.q_proj, attn.k_proj, attn.v_proj]:
                weight = proj.base_layer.weight.data if hasattr(proj, 'base_layer') else proj.weight.data
                wanda = (torch.abs(weight.cpu()) * attn_act.unsqueeze(0)).sum(dim=0)
                accumulated += wanda

            # O projection — o_proj or dense
            o_proj = get_attn_output_proj(attn)
            if o_proj is not None:
                weight = o_proj.base_layer.weight.data if hasattr(o_proj, 'base_layer') else o_proj.weight.data
                wanda = torch.abs(weight.cpu()).sum(dim=1)
                accumulated += wanda

            for proj in get_up_proj_modules(mlp):
                weight = proj.base_layer.weight.data if hasattr(proj, 'base_layer') else proj.weight.data
                wanda = (torch.abs(weight.cpu()) * ffn_act.unsqueeze(0)).sum(dim=0)
                accumulated += wanda

            # down_proj/fc2
            down_proj = get_down_proj_module(mlp)
            if down_proj is not None:
                weight = down_proj.base_layer.weight.data if hasattr(down_proj, 'base_layer') else down_proj.weight.data
                wanda = torch.abs(weight.cpu()).sum(dim=1)
                accumulated += wanda

        from pruning.distributed_utils import all_reduce_sum
        accumulated = all_reduce_sum(accumulated)

        if self.verbose:
            print(f"[Embedding-WANDA] Processed {num_processed} samples")
            print(f"[Embedding-WANDA] Importance stats: min={accumulated.min():.6f}, max={accumulated.max():.6f}, mean={accumulated.mean():.6f}")

        return accumulated


def compute_taylor_head_masks(
    model: nn.Module,
    dataloader,
    head_pruning_ratio: float,
    num_samples: int = 128,
    device: str = "cuda",
    verbose: bool = True,
    head_mode: str = "o",
    global_pruning: bool = True,
    importance_method: str = "ganda",
    enable_iterative_pruning: bool = False,
    pruning_step_size: float = 0.05,
    iterative_start_ratio_ffn: float = 0.0,
    iterative_start_ratio_head: float = 0.0,
    iterative_start_ratio_emb: float = 0.0,
    use_ntk_adjustment: bool = False,
    ntk_adjustment_alpha: float = 0.3,
    ntk_adjustment_mode: str = "previous",
    ntk_sensitivity_direction: str = "normal",
    ntk_target_mode: str = "default",
    ntk_granularity: str = "layer",
    use_iterative_premasking: bool = False,
    use_gradient_checkpointing: bool = False,
    skip_first_layer: int = 0,
    skip_last_layer: int = 0,
    ntk_norm_method: str = "robust",
    allow_full_block_pruning: bool = False
) -> Dict[int, torch.Tensor]:
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    num_skip = skip_first_layer + skip_last_layer
    if num_skip > 0 and num_skip < num_layers:
        adjusted_head_ratio = head_pruning_ratio * num_layers / (num_layers - num_skip)

        if adjusted_head_ratio > 1.0:
            print(f"[WARNING] Adjusted Head ratio {adjusted_head_ratio:.2%} > 100%, clamping to 100%")
            adjusted_head_ratio = 1.0

        if verbose:
            print(f"[Taylor Head Mask] Skip layers: {num_skip} (first={skip_first_layer}, last={skip_last_layer}), adjusting ratio {head_pruning_ratio:.2%} -> {adjusted_head_ratio:.2%}")
    else:
        adjusted_head_ratio = head_pruning_ratio

    skip_layers = set()
    if skip_first_layer > 0:
        skip_layers.update(range(min(skip_first_layer, num_layers)))
    if skip_last_layer > 0:
        skip_layers.update(range(max(0, num_layers - skip_last_layer), num_layers))

    if enable_iterative_pruning and head_pruning_ratio > 0:
        return _compute_taylor_head_masks_iterative(
            model=model,
            dataloader=dataloader,
            target_ratio=adjusted_head_ratio,
            step_size=pruning_step_size,
            start_ratio=iterative_start_ratio_head,
            num_samples=num_samples,
            device=device,
            verbose=verbose,
            head_mode=head_mode,
            global_pruning=global_pruning,
            importance_method=importance_method,
            use_ntk_adjustment=use_ntk_adjustment,
            ntk_adjustment_alpha=ntk_adjustment_alpha,
            ntk_adjustment_mode=ntk_adjustment_mode,
            ntk_sensitivity_direction=ntk_sensitivity_direction,
            ntk_target_mode=ntk_target_mode,
            ntk_granularity=ntk_granularity,
            use_iterative_premasking=use_iterative_premasking,
            use_gradient_checkpointing=use_gradient_checkpointing,
            skip_first_layer=skip_first_layer,
            skip_last_layer=skip_last_layer,
            ntk_norm_method=ntk_norm_method
        )

    calculator = TaylorImportanceCalculator(
        model, device, verbose, importance_method=importance_method,
        use_gradient_checkpointing=use_gradient_checkpointing
    )
    head_scores = calculator.compute_taylor_head_scores(dataloader, num_samples, head_mode)

    if global_pruning:
        non_skip_layers = [i for i in range(num_layers) if i not in skip_layers]
        score_tensors = [head_scores[layer_idx] for layer_idx in non_skip_layers]
        all_scores = torch.cat(score_tensors) if score_tensors else torch.tensor([])

        all_indices = [(layer_idx, head_idx)
                       for layer_idx in non_skip_layers
                       for head_idx in range(num_heads)]

        total_heads = len(all_scores)
        total_heads_all = num_layers * num_heads
        total_target_prune = min(int(total_heads_all * head_pruning_ratio), total_heads)
        num_to_prune = total_target_prune

        if verbose:
            print(f"\n[Taylor Head Mask] Total heads (excluding skip): {total_heads}")
            print(f"[Taylor Head Mask] Total heads (all layers): {total_heads_all}")
            print(f"[Taylor Head Mask] Original ratio: {head_pruning_ratio:.2%}")
            print(f"[Taylor Head Mask] Target prune count: {total_target_prune}, actual: {num_to_prune}")

        sorted_indices = sorted(range(len(all_scores)), key=lambda i: all_scores[i])
        prune_indices = set(sorted_indices[:num_to_prune])

        head_masks = {}
        for layer_idx in range(num_layers):
            head_masks[layer_idx] = torch.ones(num_heads)

        for flat_idx in prune_indices:
            layer_idx, head_idx = all_indices[flat_idx]
            head_masks[layer_idx][head_idx] = 0.0

        if not allow_full_block_pruning:
            for layer_idx in range(num_layers):
                if layer_idx in skip_layers:
                    continue
                if head_masks[layer_idx].sum() == 0:
                    scores = head_scores[layer_idx]
                    best_head = scores.argmax().item()
                    head_masks[layer_idx][best_head] = 1.0
    else:
        # Layer-wise pruning
        head_masks = {}
        min_heads = 0 if allow_full_block_pruning else 1
        for layer_idx in range(num_layers):
            if layer_idx in skip_layers:
                head_masks[layer_idx] = torch.ones(num_heads)
                continue

            scores = head_scores[layer_idx]
            num_to_keep = max(min_heads, int(num_heads * (1 - adjusted_head_ratio)))
            if num_to_keep > 0:
                _, top_indices = torch.topk(scores, num_to_keep)
                mask = torch.zeros(num_heads)
                mask[top_indices] = 1.0
            else:
                mask = torch.zeros(num_heads)
            head_masks[layer_idx] = mask

    if verbose:
        print(f"\n[Taylor Head Mask] Head masks created:")
        for layer_idx in range(num_layers):
            kept = int(head_masks[layer_idx].sum().item())
            skip_marker = " (skip)" if layer_idx in skip_layers else ""
            print(f"  Layer {layer_idx}: {kept}/{num_heads} heads kept{skip_marker}")

    del calculator, head_scores
    model.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return head_masks


def _register_ntk_mask_hooks(model, masks, mask_type, num_layers):
    hooks = []
    if masks is None:
        return hooks

    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    for layer_idx in range(num_layers):
        if layer_idx not in masks:
            continue
        mask = masks[layer_idx]
        if mask.sum().item() >= mask.numel():
            continue

        layer = model.model.layers[layer_idx]

        if mask_type == "ffn":
            def make_ffn_hook(m):
                def hook(module, input):
                    x = input[0]
                    m_exp = m.to(device=x.device, dtype=x.dtype)
                    if x.dim() == 3:
                        m_exp = m_exp.view(1, 1, -1)
                    elif x.dim() == 2:
                        m_exp = m_exp.view(1, -1)
                    return (x * m_exp,)
                return hook
            down_proj = get_down_proj_module(layer.mlp)
            if down_proj is not None:
                hooks.append(down_proj.register_forward_pre_hook(make_ffn_hook(mask)))

        elif mask_type == "head":
            def make_head_hook(m, nh=num_heads, hd=head_dim):
                def hook(module, input):
                    x = input[0]
                    if x.dim() == 3:
                        bs, sl, _ = x.shape
                        x_r = x.view(bs, sl, nh, hd)
                        m_exp = m.to(device=x.device, dtype=x.dtype).view(1, 1, nh, 1)
                        return ((x_r * m_exp).view(bs, sl, -1),)
                    return input
                return hook
            o_proj = get_attn_output_proj(layer.self_attn)
            if o_proj is not None:
                hooks.append(o_proj.register_forward_pre_hook(make_head_hook(mask)))

        elif mask_type == "dimension":
            def make_dim_hook(m, nh=num_heads, hd=head_dim):
                def hook(module, input):
                    x = input[0]
                    if x.dim() == 3:
                        bs, sl, _ = x.shape
                        x_r = x.view(bs, sl, nh, hd)
                        m_exp = m.to(device=x.device, dtype=x.dtype).view(1, 1, 1, hd)
                        return ((x_r * m_exp).view(bs, sl, -1),)
                    return input
                return hook
            o_proj = get_attn_output_proj(layer.self_attn)
            if o_proj is not None:
                hooks.append(o_proj.register_forward_pre_hook(make_dim_hook(mask)))

    return hooks


def _compute_ntk_sensitivity(
    model: nn.Module,
    dataloader,
    current_masks: Dict[int, torch.Tensor],
    prev_masks: Optional[Dict[int, torch.Tensor]],
    mask_type: str = "ffn",  # "ffn", "head", "dimension"
    num_samples: int = 10,
    device: str = "cuda",
    verbose: bool = False,
    ntk_target_mode: str = "default",  # "default" or "all"
    ntk_method: str = "frobenius",  # "frobenius", "eigenvalue", "delta", "quadform"
    ntk_eigenvalue_k: int = 5,  # top-k eigenvalues for eigenvalue method
    skip_layers: Optional[set] = None
) -> Dict[int, float]:
    num_layers = model.config.num_hidden_layers
    if skip_layers is None:
        skip_layers = set()

    if verbose:
        method_str = f" (method={ntk_method}"
        if ntk_method == "eigenvalue":
            method_str += f", k={ntk_eigenvalue_k})"
        else:
            method_str += ")"
        skip_str = f", skip={len(skip_layers)} layers" if skip_layers else ""
        print(f"  [NTK] Computing layer sensitivities (samples={num_samples}){method_str}{skip_str}")

    pruned_counts = {}
    for layer_idx in range(num_layers):
        if layer_idx in skip_layers:
            continue
        if prev_masks is not None and layer_idx in prev_masks:
            prev_kept = prev_masks[layer_idx].sum().item()
            curr_kept = current_masks[layer_idx].sum().item()
            pruned_counts[layer_idx] = prev_kept - curr_kept
        else:
            total = current_masks[layer_idx].numel()
            curr_kept = current_masks[layer_idx].sum().item()
            pruned_counts[layer_idx] = total - curr_kept

    model.eval()

    target_params = {}  # {layer_idx: [param1, param2, ...]}
    for layer_idx in range(num_layers):
        if layer_idx in skip_layers:
            continue
        mlp = model.model.layers[layer_idx].mlp
        attn = model.model.layers[layer_idx].self_attn

        if mask_type == "ffn":
            if ntk_target_mode == "all":
                params = []
                if hasattr(mlp, 'up_proj'):
                    params.append(mlp.up_proj.weight)
                if hasattr(mlp, 'gate_proj'):
                    params.append(mlp.gate_proj.weight)
                if hasattr(mlp, 'down_proj'):
                    params.append(mlp.down_proj.weight)
                if not params and hasattr(mlp, 'fc1'):
                    params.append(mlp.fc1.weight)
                    if hasattr(mlp, 'fc2'):
                        params.append(mlp.fc2.weight)
                if params:
                    target_params[layer_idx] = params
            else:
                if hasattr(mlp, 'down_proj'):
                    target_params[layer_idx] = [mlp.down_proj.weight]
                elif hasattr(mlp, 'fc2'):
                    target_params[layer_idx] = [mlp.fc2.weight]
        elif mask_type == "head":
            if ntk_target_mode == "all":
                params = []
                for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    if hasattr(attn, proj_name):
                        params.append(getattr(attn, proj_name).weight)
                if params:
                    target_params[layer_idx] = params
            else:
                o_proj = get_attn_output_proj(attn)
                if o_proj is not None:
                    weight = o_proj.base_layer.weight if hasattr(o_proj, 'base_layer') else o_proj.weight
                    target_params[layer_idx] = [weight]
        elif mask_type == "dimension":
            if ntk_target_mode == "all":
                params = []
                for proj_name in ['q_proj', 'k_proj', 'v_proj']:
                    if hasattr(attn, proj_name):
                        params.append(getattr(attn, proj_name).weight)
                if params:
                    target_params[layer_idx] = params
            else:
                if hasattr(attn, 'v_proj'):
                    target_params[layer_idx] = [attn.v_proj.weight]

    if verbose:
        for layer_idx in range(min(1, num_layers)):
            if layer_idx in target_params:
                param_names = [f"{p.shape}" for p in target_params[layer_idx]]
                print(f"  [NTK] Layer {layer_idx} target params ({ntk_target_mode}): {param_names}")

    # ==================== Temporarily enable gradients for frozen models ====================
    original_requires_grad = {}  # {param_id: original_value}
    for layer_idx, params in target_params.items():
        for param in params:
            param_id = id(param)
            if param_id not in original_requires_grad:
                original_requires_grad[param_id] = param.requires_grad
                param.requires_grad_(True)

    effective_num_samples = num_samples

    cached_batches = []
    data_iter = iter(dataloader)
    for _ in range(effective_num_samples):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        cached_batches.append(batch)

    mask_hooks = _register_ntk_mask_hooks(model, current_masks, mask_type, num_layers)

    layer_sensitivities = {}

    for layer_idx in range(num_layers):
        if layer_idx in skip_layers:
            layer_sensitivities[layer_idx] = 1.0
            continue
        if layer_idx not in target_params:
            layer_sensitivities[layer_idx] = 1.0
            continue

        layer_grads = []

        for batch in cached_batches:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(device)

            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            # Forward + Backward
            model.zero_grad(set_to_none=True)
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            loss.backward()

            grads = []
            for param in target_params[layer_idx]:
                if param.grad is not None:
                    grads.append(param.grad.detach().flatten())
            if grads:
                grad = torch.cat(grads, dim=0)
                layer_grads.append(grad)

            del outputs, loss
            model.zero_grad(set_to_none=True)

        from pruning.distributed_utils import is_distributed, all_gather_cat_variable
        if is_distributed() and len(layer_grads) > 0:
            local_G = torch.stack(layer_grads)  # [local_samples, param_size]
            G_full = all_gather_cat_variable(local_G)  # [total_samples, param_size]
            layer_grads = [G_full[i] for i in range(G_full.shape[0])]

        if len(layer_grads) == 0:
            layer_sensitivities[layer_idx] = 1.0
            if verbose and layer_idx == 0:
                print(f"    [NTK Debug] Layer {layer_idx}: NO gradients collected!")
        else:
            G = torch.stack(layer_grads)  # [num_samples, param_size]
            K = torch.mm(G, G.t())  # [num_samples, num_samples]

            if ntk_method == "eigenvalue":
                try:
                    eigenvalues = torch.linalg.eigvalsh(K)
                    k = min(ntk_eigenvalue_k, len(eigenvalues))
                    sensitivity_value = eigenvalues[-k:].sum().item()
                except Exception as e:
                    if verbose:
                        print(f"    [NTK Warning] Layer {layer_idx} eigenvalue failed: {e}, falling back to Frobenius")
                    sensitivity_value = torch.norm(K, p='fro').item()
            elif ntk_method == "delta":
                # trace(K) = sum of eigenvalues = total variance
                sensitivity_value = torch.trace(K).item()
            elif ntk_method == "quadform":
                # Quadratic form: 1ᵀK1 = Σ_{a,b} K_{a,b} = ||G^T 1||_2^2
                sensitivity_value = K.sum().item()
            else:
                sensitivity_value = torch.norm(K, p='fro').item()

            num_pruned = max(1, pruned_counts.get(layer_idx, 1))
            layer_sensitivities[layer_idx] = sensitivity_value / num_pruned

            if verbose and layer_idx < 3:
                print(f"    [NTK Debug] Layer {layer_idx}: {len(layer_grads)} grads, sens_value={sensitivity_value:.4f}, pruned={num_pruned}, raw_sens={layer_sensitivities[layer_idx]:.4f} ({ntk_method})")

        del layer_grads
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    active_sensitivities = {k: v for k, v in layer_sensitivities.items() if k not in skip_layers}
    if verbose and active_sensitivities:
        raw_values = list(active_sensitivities.values())
        print(f"  [NTK Debug] RAW sensitivities (active {len(raw_values)} layers): min={min(raw_values):.4f}, max={max(raw_values):.4f}, mean={sum(raw_values)/len(raw_values):.4f}")

    if active_sensitivities:
        mean_sensitivity = sum(active_sensitivities.values()) / len(active_sensitivities)
        if mean_sensitivity > 0:
            for layer_idx in layer_sensitivities:
                if layer_idx not in skip_layers:
                    layer_sensitivities[layer_idx] = layer_sensitivities[layer_idx] / mean_sensitivity

    for hook in mask_hooks:
        hook.remove()

    # ==================== Restore original requires_grad states ====================
    for layer_idx, params in target_params.items():
        for param in params:
            param_id = id(param)
            if param_id in original_requires_grad:
                param.requires_grad_(original_requires_grad[param_id])

    if verbose:
        print(f"  [NTK] Sensitivity computed for {num_layers} layers (GPU, layer-sequential, masked={len(mask_hooks) > 0})")
        sorted_sens = sorted(layer_sensitivities.items(), key=lambda x: x[1], reverse=True)
        print(f"  [NTK] Most sensitive layers: {[(l, f'{s:.3f}') for l, s in sorted_sens[:3]]}")

    return layer_sensitivities


def _compute_ntk_sensitivity_per_unit(
    model: nn.Module,
    dataloader,
    current_masks: Dict[int, torch.Tensor],
    prev_masks: Optional[Dict[int, torch.Tensor]],
    mask_type: str = "ffn",  # "ffn", "head", "dimension"
    num_samples: int = 10,
    device: str = "cuda",
    verbose: bool = False,
    dimension_multiple: int = 16,
    num_heads: int = 32,
    ntk_target_mode: str = "default",  # "default" or "all"
    ntk_unit_mode: str = "group",  # "group" or "neuron"
    group_assignments: Optional[Dict[int, List[torch.Tensor]]] = None,
    ntk_method: str = "frobenius",  # "frobenius", "eigenvalue", "delta", "quadform"
    ntk_eigenvalue_k: int = 5,  # top-k eigenvalues for eigenvalue method
    normalize_per_layer: bool = True,
    skip_layers: Optional[set] = None
) -> Dict[int, torch.Tensor]:
    num_layers = model.config.num_hidden_layers
    if skip_layers is None:
        skip_layers = set()

    ffn_unit_size = 1 if (mask_type == "ffn" and ntk_unit_mode == "neuron") else dimension_multiple

    use_importance_groups = (group_assignments is not None and mask_type == "ffn")

    use_layer_sequential = True

    if verbose:
        group_str = "importance-based" if use_importance_groups else "index-based"
        skip_str = f", skip={len(skip_layers)} layers" if skip_layers else ""
        print(f"  [NTK-PerUnit] Computing per-unit sensitivities (type={mask_type}, samples={num_samples}, unit_mode={ntk_unit_mode}, groups={group_str}, compute=layer-sequential GPU{skip_str})")

    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    model.eval()

    target_params = {}  # {layer_idx: list of params}
    unit_sizes = {}

    for layer_idx in range(num_layers):
        if layer_idx in skip_layers:
            continue
        mlp = model.model.layers[layer_idx].mlp
        attn = model.model.layers[layer_idx].self_attn

        if mask_type == "ffn":
            if hasattr(mlp, 'down_proj'):
                intermediate_size = mlp.down_proj.weight.shape[1]
                unit_sizes[layer_idx] = intermediate_size // ffn_unit_size
                if ntk_target_mode == "all":
                    params = []
                    if hasattr(mlp, 'up_proj'):
                        params.append(('row', mlp.up_proj.weight))  # [inter, hidden]
                    if hasattr(mlp, 'gate_proj'):
                        params.append(('row', mlp.gate_proj.weight))  # [inter, hidden]
                    params.append(('col', mlp.down_proj.weight))  # [hidden, inter]
                    target_params[layer_idx] = params
                else:
                    target_params[layer_idx] = [('col', mlp.down_proj.weight)]
            elif hasattr(mlp, 'fc2'):
                intermediate_size = mlp.fc2.weight.shape[1]
                unit_sizes[layer_idx] = intermediate_size // ffn_unit_size
                if ntk_target_mode == "all" and hasattr(mlp, 'fc1'):
                    target_params[layer_idx] = [('row', mlp.fc1.weight), ('col', mlp.fc2.weight)]
                else:
                    target_params[layer_idx] = [('col', mlp.fc2.weight)]

        elif mask_type == "head":
            o_proj = get_attn_output_proj(attn)
            if o_proj is not None:
                unit_sizes[layer_idx] = num_heads
                o_weight = o_proj.base_layer.weight if hasattr(o_proj, 'base_layer') else o_proj.weight
                if ntk_target_mode == "all":
                    params = []
                    for proj_name in ['q_proj', 'k_proj', 'v_proj']:
                        if hasattr(attn, proj_name):
                            proj = getattr(attn, proj_name)
                            w = proj.base_layer.weight if hasattr(proj, 'base_layer') else proj.weight
                            params.append(('row', w))  # [n_h*hd, hidden]
                    params.append(('col', o_weight))  # [hidden, n_h*hd]
                    target_params[layer_idx] = params
                else:
                    target_params[layer_idx] = [('col', o_weight)]

        elif mask_type == "dimension":
            if hasattr(attn, 'v_proj'):
                head_dim = model.config.hidden_size // num_heads
                unit_sizes[layer_idx] = head_dim
                if ntk_target_mode == "all":
                    params = []
                    for proj_name in ['q_proj', 'k_proj', 'v_proj']:
                        if hasattr(attn, proj_name):
                            params.append(('row', getattr(attn, proj_name).weight))  # [n_h*hd, hidden]
                    target_params[layer_idx] = params
                else:
                    target_params[layer_idx] = [('row', attn.v_proj.weight)]

    if verbose:
        for layer_idx in range(min(1, num_layers)):
            if layer_idx in target_params:
                info = [(mode, p.shape) for mode, p in target_params[layer_idx]]
                print(f"  [NTK-PerUnit] Layer {layer_idx} targets ({ntk_target_mode}): {info}")

    effective_num_samples = num_samples

    unit_sensitivities = {}
    head_dim = model.config.hidden_size // num_heads

    # ==================== Temporarily enable gradients for frozen models ====================
    original_requires_grad = {}  # {(layer_idx, param_id): original_value}
    for layer_idx, params in target_params.items():
        for split_mode, param in params:
            param_id = id(param)
            if param_id not in original_requires_grad:
                original_requires_grad[param_id] = param.requires_grad
                param.requires_grad_(True)

    mask_hooks = _register_ntk_mask_hooks(model, current_masks, mask_type, num_layers)

    if use_layer_sequential:
        if verbose:
            print(f"  [NTK-PerUnit] Using layer-sequential mode (GPU compute, {num_layers} layers × {effective_num_samples} samples)")

        cached_samples = []
        data_iter = iter(dataloader)
        while len(cached_samples) < effective_num_samples:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            bsz = batch['input_ids'].shape[0]
            for i in range(bsz):
                single = {k: v[i:i+1] for k, v in batch.items() if isinstance(v, torch.Tensor)}
                cached_samples.append(single)
                if len(cached_samples) >= effective_num_samples:
                    break

        for layer_idx in range(num_layers):
            if layer_idx in skip_layers:
                if mask_type == "ffn":
                    dp = get_down_proj_module(model.model.layers[layer_idx].mlp)
                    inter_size = dp.weight.shape[1] if dp is not None else 0
                    unit_sensitivities[layer_idx] = torch.ones(inter_size // ffn_unit_size)
                elif mask_type == "head":
                    unit_sensitivities[layer_idx] = torch.ones(num_heads)
                continue

            if layer_idx not in target_params:
                if layer_idx in unit_sizes:
                    unit_sensitivities[layer_idx] = torch.ones(unit_sizes[layer_idx])
                continue

            if use_importance_groups and layer_idx in group_assignments:
                num_units = len(group_assignments[layer_idx])
            else:
                num_units = unit_sizes[layer_idx]

            layer_grads = {u: [] for u in range(num_units)}

            for sample_idx, batch in enumerate(cached_samples):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(device)

                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                # Forward + Backward
                model.zero_grad(set_to_none=True)
                outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                loss.backward()

                for unit_idx in range(num_units):
                    unit_grads = []

                    for split_mode, param in target_params[layer_idx]:
                        if param.grad is None:
                            continue
                        grad = param.grad.detach()

                        if mask_type == "ffn":
                            if use_importance_groups and layer_idx in group_assignments:
                                group_indices = group_assignments[layer_idx][unit_idx]
                                if split_mode == "col":
                                    unit_grads.append(grad[:, group_indices].flatten())
                                else:
                                    unit_grads.append(grad[group_indices, :].flatten())
                            else:
                                start_idx = unit_idx * ffn_unit_size
                                end_idx = start_idx + ffn_unit_size
                                if split_mode == "col":
                                    unit_grads.append(grad[:, start_idx:end_idx].flatten())
                                else:
                                    unit_grads.append(grad[start_idx:end_idx, :].flatten())

                        elif mask_type == "head":
                            start_idx = unit_idx * head_dim
                            end_idx = start_idx + head_dim
                            if split_mode == "col":
                                unit_grads.append(grad[:, start_idx:end_idx].flatten())
                            else:
                                unit_grads.append(grad[start_idx:end_idx, :].flatten())

                        elif mask_type == "dimension":
                            indices = [h * head_dim + unit_idx for h in range(num_heads)]
                            if split_mode == "row":
                                unit_grads.append(grad[indices, :].flatten())
                            else:
                                unit_grads.append(grad[:, indices].flatten())

                    if unit_grads:
                        combined_grad = torch.cat(unit_grads, dim=0)
                        layer_grads[unit_idx].append(combined_grad)

                del outputs, loss
                model.zero_grad(set_to_none=True)

            from pruning.distributed_utils import is_distributed, all_gather_cat_variable
            if is_distributed():
                for unit_idx in range(num_units):
                    if layer_grads[unit_idx]:
                        local_G = torch.stack(layer_grads[unit_idx])
                        global_G = all_gather_cat_variable(local_G)
                        layer_grads[unit_idx] = [global_G[i] for i in range(global_G.shape[0])]

            sensitivities = torch.zeros(num_units, device=device)

            total_grads_collected = sum(len(layer_grads[u]) for u in range(num_units))
            if verbose and layer_idx == 0:
                print(f"    [NTK Debug] Layer {layer_idx}: collected {total_grads_collected} gradients for {num_units} units")

            for unit_idx in range(num_units):
                grads = layer_grads[unit_idx]
                if len(grads) == 0:
                    sensitivities[unit_idx] = 1.0
                    continue

                G = torch.stack(grads)  # [num_samples, param_size]
                K = torch.mm(G, G.t())  # [num_samples, num_samples]

                if ntk_method == "eigenvalue":
                    try:
                        eigenvalues = torch.linalg.eigvalsh(K)
                        k = min(ntk_eigenvalue_k, len(eigenvalues))
                        sensitivities[unit_idx] = eigenvalues[-k:].sum()
                    except:
                        sensitivities[unit_idx] = torch.norm(K, p='fro')
                elif ntk_method == "delta":
                    sensitivities[unit_idx] = torch.trace(K)
                elif ntk_method == "quadform":
                    sensitivities[unit_idx] = K.sum()
                else:
                    sensitivities[unit_idx] = torch.norm(K, p='fro')

            sensitivities = torch.nan_to_num(sensitivities, nan=1.0, posinf=1.0, neginf=1.0)

            raw_min, raw_max, raw_mean, raw_std = sensitivities.min().item(), sensitivities.max().item(), sensitivities.mean().item(), sensitivities.std().item()
            if verbose and layer_idx < 3:
                print(f"    [NTK Debug] Layer {layer_idx} RAW: min={raw_min:.4f}, max={raw_max:.4f}, mean={raw_mean:.4f}, std={raw_std:.4f}")

            if normalize_per_layer:
                mean_sens = sensitivities.mean()
                if mean_sens > 0 and not torch.isnan(mean_sens):
                    sensitivities = sensitivities / mean_sens
                else:
                    sensitivities = torch.ones_like(sensitivities)

            unit_sensitivities[layer_idx] = sensitivities.cpu()

            del layer_grads
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            if verbose and layer_idx < 3:
                sens = unit_sensitivities[layer_idx]
                print(f"    Layer {layer_idx}: min={sens.min():.3f}, max={sens.max():.3f}, mean={sens.mean():.3f}")

    else:
        layer_unit_gradients = {}
        for layer_idx in range(num_layers):
            if layer_idx in unit_sizes:
                layer_unit_gradients[layer_idx] = {u: [] for u in range(unit_sizes[layer_idx])}

        sample_count = 0
        data_iter = iter(dataloader)

        while sample_count < effective_num_samples:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(device)

            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            # Forward + Backward
            model.zero_grad(set_to_none=True)
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            loss.backward()

            for layer_idx in range(num_layers):
                if layer_idx not in target_params:
                    continue

                num_units = unit_sizes[layer_idx]

                for unit_idx in range(num_units):
                    unit_grads = []

                    for split_mode, param in target_params[layer_idx]:
                        if param.grad is None:
                            continue
                        grad = param.grad.detach()

                        if mask_type == "ffn":
                            if use_importance_groups:
                                group_indices = group_assignments[layer_idx][unit_idx]
                                if split_mode == "col":
                                    unit_grads.append(grad[:, group_indices].flatten())
                                else:
                                    unit_grads.append(grad[group_indices, :].flatten())
                            else:
                                start_idx = unit_idx * ffn_unit_size
                                end_idx = start_idx + ffn_unit_size
                                if split_mode == "col":
                                    unit_grads.append(grad[:, start_idx:end_idx].flatten())
                                else:
                                    unit_grads.append(grad[start_idx:end_idx, :].flatten())

                        elif mask_type == "head":
                            start_idx = unit_idx * head_dim
                            end_idx = start_idx + head_dim
                            if split_mode == "col":
                                unit_grads.append(grad[:, start_idx:end_idx].flatten())
                            else:
                                unit_grads.append(grad[start_idx:end_idx, :].flatten())

                        elif mask_type == "dimension":
                            indices = [h * head_dim + unit_idx for h in range(num_heads)]
                            if split_mode == "row":
                                unit_grads.append(grad[indices, :].flatten())
                            else:
                                unit_grads.append(grad[:, indices].flatten())

                    if unit_grads:
                        combined_grad = torch.cat(unit_grads, dim=0)
                        combined_grad = combined_grad.cpu()
                        layer_unit_gradients[layer_idx][unit_idx].append(combined_grad)

            sample_count += 1

            del outputs, loss
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        from pruning.distributed_utils import is_distributed, all_gather_cat_variable
        if is_distributed():
            for layer_idx in layer_unit_gradients:
                for unit_idx in layer_unit_gradients[layer_idx]:
                    grads = layer_unit_gradients[layer_idx][unit_idx]
                    if grads:
                        local_G = torch.stack(grads)  # [local_samples, param_size]
                        global_G = all_gather_cat_variable(local_G)  # [total_samples, param_size]
                        layer_unit_gradients[layer_idx][unit_idx] = [global_G[i] for i in range(global_G.shape[0])]

        for layer_idx in range(num_layers):
            if layer_idx in skip_layers:
                continue
            if layer_idx not in layer_unit_gradients:
                if layer_idx in unit_sizes:
                    unit_sensitivities[layer_idx] = torch.ones(unit_sizes[layer_idx])
                continue

            num_units = unit_sizes[layer_idx]
            sensitivities = torch.zeros(num_units)

            for unit_idx in range(num_units):
                grads = layer_unit_gradients[layer_idx][unit_idx]
                if len(grads) == 0:
                    sensitivities[unit_idx] = 1.0
                    continue

                G = torch.stack(grads)
                K = torch.mm(G, G.t())

                if ntk_method == "eigenvalue":
                    try:
                        eigenvalues = torch.linalg.eigvalsh(K)
                        k = min(ntk_eigenvalue_k, len(eigenvalues))
                        sensitivities[unit_idx] = eigenvalues[-k:].sum().item()
                    except:
                        sensitivities[unit_idx] = torch.norm(K, p='fro').item()
                elif ntk_method == "delta":
                    sensitivities[unit_idx] = torch.trace(K).item()
                elif ntk_method == "quadform":
                    sensitivities[unit_idx] = K.sum().item()
                else:
                    sensitivities[unit_idx] = torch.norm(K, p='fro').item()

            sensitivities = torch.nan_to_num(sensitivities, nan=1.0, posinf=1.0, neginf=1.0)

            if normalize_per_layer:
                mean_sensitivity = sensitivities.mean()
                if mean_sensitivity > 0 and not torch.isnan(mean_sensitivity):
                    sensitivities = sensitivities / mean_sensitivity
                else:
                    sensitivities = torch.ones_like(sensitivities)

            unit_sensitivities[layer_idx] = sensitivities

            del layer_unit_gradients[layer_idx]

    for hook in mask_hooks:
        hook.remove()

    # ==================== Restore original requires_grad states ====================
    for layer_idx, params in target_params.items():
        for split_mode, param in params:
            param_id = id(param)
            if param_id in original_requires_grad:
                param.requires_grad_(original_requires_grad[param_id])

    if verbose:
        print(f"  [NTK-PerUnit] Sensitivity computed for {len(unit_sensitivities)} layers (masked={len(mask_hooks) > 0})")
        if not use_layer_sequential:
            for layer_idx in list(unit_sensitivities.keys())[:3]:
                sens = unit_sensitivities[layer_idx]
                print(f"    Layer {layer_idx}: min={sens.min():.3f}, max={sens.max():.3f}, "
                      f"mean={sens.mean():.3f}, std={sens.std():.3f}")

    return unit_sensitivities


def _compute_taylor_head_masks_iterative(
    model: nn.Module,
    dataloader,
    target_ratio: float,
    step_size: float,
    start_ratio: float = 0.0,
    num_samples: int = 128,
    device: str = "cuda",
    verbose: bool = True,
    head_mode: str = "o",
    global_pruning: bool = True,
    importance_method: str = "ganda",
    use_ntk_adjustment: bool = False,
    ntk_adjustment_alpha: float = 0.3,
    ntk_adjustment_mode: str = "previous",
    ntk_sensitivity_direction: str = "normal",
    ntk_target_mode: str = "default",
    ntk_granularity: str = "layer",
    use_iterative_premasking: bool = False,
    use_gradient_checkpointing: bool = False,
    skip_first_layer: int = 0,
    skip_last_layer: int = 0,
    ntk_norm_method: str = "robust"
) -> Dict[int, torch.Tensor]:
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    skip_layers = set()
    if skip_first_layer > 0:
        skip_layers.update(range(min(skip_first_layer, num_layers)))
    if skip_last_layer > 0:
        skip_layers.update(range(max(0, num_layers - skip_last_layer), num_layers))

    effective_start = min(start_ratio, target_ratio) if start_ratio > 0 else 0.0
    steps = []
    if effective_start > 0:
        steps.append(effective_start)
        current_ratio = effective_start + step_size
    else:
        current_ratio = step_size
    while current_ratio < target_ratio:
        steps.append(current_ratio)
        current_ratio += step_size
    steps.append(target_ratio)

    if verbose:
        print(f"\n{'='*60}")
        print(f"[Iterative Head Pruning] Starting iterative head pruning")
        print(f"[Iterative Head Pruning] Target ratio: {target_ratio:.1%}")
        print(f"[Iterative Head Pruning] Step size: {step_size:.1%}")
        if effective_start > 0:
            print(f"[Iterative Head Pruning] Hybrid: one-shot to {effective_start:.1%}, then iterative")
        print(f"[Iterative Head Pruning] Steps: {len(steps)}")
        print(f"[Iterative Head Pruning] Process: 0% → {' → '.join([f'{s:.0%}' for s in steps])}")
        if skip_layers:
            print(f"[Iterative Head Pruning] Skip layers: {sorted(skip_layers)}")
        if use_iterative_premasking:
            print(f"[Iterative Head Pruning] Pre-masking: enabled (pruned heads output=0 during forward)")
        if use_ntk_adjustment:
            print(f"[Iterative Head Pruning] NTK Adjustment: enabled (alpha={ntk_adjustment_alpha}, mode={ntk_adjustment_mode}, direction={ntk_sensitivity_direction})")
        print(f"{'='*60}")

    sens_sign = -1.0 if ntk_sensitivity_direction == "inverse" else 1.0

    current_masks = {}
    for layer_idx in range(num_layers):
        current_masks[layer_idx] = torch.ones(num_heads)

    layer_sensitivities = None  # {layer_idx: sensitivity_score}
    prev_masks = None

    for step_idx, current_target in enumerate(steps):
        if verbose:
            print(f"\n[Iterative Head Step {step_idx + 1}/{len(steps)}] Pruning to {current_target:.1%}")

        if use_iterative_premasking and step_idx > 0:
            calculator = TaylorImportanceCalculator(
                model, device, verbose=False,
                importance_method=importance_method,
                head_masks=current_masks,
                use_gradient_checkpointing=use_gradient_checkpointing
            )
        else:
            calculator = TaylorImportanceCalculator(
                model, device, verbose=False,
                importance_method=importance_method,
                use_gradient_checkpointing=use_gradient_checkpointing
            )
        head_scores = calculator.compute_taylor_head_scores(dataloader, num_samples, head_mode)

        del calculator
        model.zero_grad(set_to_none=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if not use_iterative_premasking:
            for layer_idx in range(num_layers):
                head_scores[layer_idx] = head_scores[layer_idx] * current_masks[layer_idx].to(head_scores[layer_idx].device)

        # ===== Helper function for head pruning =====
        def apply_head_pruning(scores_dict, target, skip_layers_set, is_global):
            masks = {}
            if is_global:
                # Global pruning
                non_skip_layers = [i for i in range(num_layers) if i not in skip_layers_set]
                score_tensors = [scores_dict[layer_idx] for layer_idx in non_skip_layers]
                all_scores_tensor = torch.cat(score_tensors) if score_tensors else torch.tensor([])
                all_indices = [(layer_idx, head_idx)
                               for layer_idx in non_skip_layers
                               for head_idx in range(num_heads)]
                total_heads = len(all_scores_tensor)
                num_to_prune = int(total_heads * target)
                _, sorted_indices = torch.sort(all_scores_tensor)
                sorted_indices = sorted_indices.tolist()
                prune_indices = set(sorted_indices[:num_to_prune])
                for layer_idx in range(num_layers):
                    masks[layer_idx] = torch.ones(num_heads)
                for flat_idx in prune_indices:
                    layer_idx, head_idx = all_indices[flat_idx]
                    masks[layer_idx][head_idx] = 0.0
                if not allow_full_block_pruning:
                    for layer_idx in range(num_layers):
                        if layer_idx in skip_layers_set:
                            continue
                        if masks[layer_idx].sum() == 0:
                            scores = scores_dict[layer_idx]
                            best_head = scores.argmax().item()
                            masks[layer_idx][best_head] = 1.0
            else:
                # Layer-wise pruning
                min_heads = 0 if allow_full_block_pruning else 1
                for layer_idx in range(num_layers):
                    if layer_idx in skip_layers_set:
                        masks[layer_idx] = torch.ones(num_heads)
                        continue
                    scores = scores_dict[layer_idx]
                    num_to_keep = max(min_heads, int(num_heads * (1 - target)))
                    if num_to_keep > 0:
                        _, top_indices = torch.topk(scores, num_to_keep)
                        mask = torch.zeros(num_heads)
                        mask[top_indices] = 1.0
                    else:
                        mask = torch.zeros(num_heads)
                    masks[layer_idx] = mask
            return masks

        if use_ntk_adjustment and ntk_adjustment_mode == "current":
            if verbose:
                print(f"  [NTK-current] 1st pruning (temporary)")
            temp_masks = apply_head_pruning(head_scores, current_target, skip_layers, global_pruning)

            if step_idx == 0:
                temp_masks = {i: torch.ones(num_heads) for i in range(num_layers)}

            if global_pruning and ntk_granularity == "layer":
                if verbose:
                    print(f"  [NTK-current] Computing layer-level NTK (Global mode)")
                current_sensitivities = _compute_ntk_sensitivity(
                    model=model,
                    dataloader=dataloader,
                    current_masks=temp_masks,
                    prev_masks=current_masks,
                    mask_type="head",
                    num_samples=num_samples,
                    device=device,
                    verbose=verbose,
                    ntk_target_mode=ntk_target_mode,
                    skip_layers=skip_layers
                )
                if verbose:
                    print(f"  [NTK-current] Adjusting scores (alpha={ntk_adjustment_alpha})")
                adjusted_scores = _apply_ntk_adjustment_to_scores(
                    scores=head_scores,
                    sensitivities=current_sensitivities,
                    alpha=ntk_adjustment_alpha,
                    sens_sign=sens_sign,
                    granularity="layer",
                    unit_mode="head",
                    dimension_multiple=1,
                    intermediate_size=num_heads,
                    num_layers=num_layers,
                    verbose=verbose,
                    ntk_norm_method=ntk_norm_method,
                    skip_layers=skip_layers
                )
            elif global_pruning and ntk_granularity == "unit":
                if verbose:
                    print(f"  [NTK-current] Computing per-head NTK (Global+Unit mode)")
                current_sensitivities = _compute_ntk_sensitivity_per_unit(
                    model=model,
                    dataloader=dataloader,
                    current_masks=temp_masks,
                    prev_masks=current_masks,
                    mask_type="head",
                    num_samples=num_samples,
                    device=device,
                    verbose=verbose,
                    num_heads=num_heads,
                    ntk_target_mode=ntk_target_mode,
                    normalize_per_layer=False,
                    skip_layers=skip_layers
                )
                if ntk_norm_method == "legacy":
                    all_sens = torch.cat([current_sensitivities[l] for l in range(num_layers) if l in current_sensitivities])
                    nonzero_mask = all_sens > 0
                    global_mean = all_sens[nonzero_mask].mean() if nonzero_mask.any() else all_sens.mean()
                    if global_mean > 0:
                        for layer_idx in current_sensitivities:
                            current_sensitivities[layer_idx] = current_sensitivities[layer_idx] / global_mean
                    if verbose:
                        print(f"  [NTK-current] Global normalization (legacy): original_mean={global_mean:.4f}")

                if verbose:
                    print(f"  [NTK-current] Adjusting scores per-head (alpha={ntk_adjustment_alpha})")
                adjusted_scores = _apply_ntk_adjustment_to_scores(
                    scores=head_scores,
                    sensitivities=current_sensitivities,
                    alpha=ntk_adjustment_alpha,
                    sens_sign=sens_sign,
                    granularity="unit",
                    unit_mode="head",
                    dimension_multiple=1,
                    intermediate_size=num_heads,
                    num_layers=num_layers,
                    verbose=verbose,
                    ntk_norm_method=ntk_norm_method,
                    skip_layers=skip_layers
                )
            else:
                if verbose:
                    print(f"  [NTK-current] Computing per-head NTK (Layer-wise mode)")
                current_sensitivities = _compute_ntk_sensitivity_per_unit(
                    model=model,
                    dataloader=dataloader,
                    current_masks=temp_masks,
                    prev_masks=current_masks,
                    mask_type="head",
                    num_samples=num_samples,
                    device=device,
                    verbose=verbose,
                    num_heads=num_heads,
                    ntk_target_mode=ntk_target_mode,
                    skip_layers=skip_layers
                )
                if verbose:
                    print(f"  [NTK-current] Adjusting scores per-head (alpha={ntk_adjustment_alpha})")
                adjusted_scores = _apply_ntk_adjustment_to_scores(
                    scores=head_scores,
                    sensitivities=current_sensitivities,
                    alpha=ntk_adjustment_alpha,
                    sens_sign=sens_sign,
                    granularity="unit",
                    unit_mode="head",
                    dimension_multiple=1,
                    intermediate_size=num_heads,
                    num_layers=num_layers,
                    verbose=verbose,
                    ntk_norm_method=ntk_norm_method,
                    skip_layers=skip_layers,
                    layer_wise=True
                )

            if verbose:
                print(f"  [NTK-current] 2nd pruning (final, no forward)")
            current_masks = apply_head_pruning(adjusted_scores, current_target, skip_layers, global_pruning)

        else:
            if use_ntk_adjustment and layer_sensitivities is not None:
                if verbose:
                    print(f"  [NTK-previous] Applying sensitivity adjustment (alpha={ntk_adjustment_alpha})")
                prev_head_granularity = "layer" if (global_pruning and ntk_granularity == "layer") else "unit"
                head_scores = _apply_ntk_adjustment_to_scores(
                    scores=head_scores,
                    sensitivities=layer_sensitivities,
                    alpha=ntk_adjustment_alpha,
                    sens_sign=sens_sign,
                    granularity=prev_head_granularity,
                    unit_mode="head",
                    dimension_multiple=1,
                    intermediate_size=num_heads,
                    num_layers=num_layers,
                    verbose=verbose,
                    ntk_norm_method=ntk_norm_method,
                    skip_layers=skip_layers,
                    layer_wise=not global_pruning
                )

            current_masks = apply_head_pruning(head_scores, current_target, skip_layers, global_pruning)

            if use_ntk_adjustment and step_idx < len(steps) - 1:
                if global_pruning and ntk_granularity == "layer":
                    layer_sensitivities = _compute_ntk_sensitivity(
                        model=model,
                        dataloader=dataloader,
                        current_masks=current_masks,
                        prev_masks=None if step_idx == 0 else prev_masks,
                        mask_type="head",
                        num_samples=num_samples,
                        device=device,
                        verbose=verbose,
                        ntk_target_mode=ntk_target_mode,
                        skip_layers=skip_layers
                    )
                elif global_pruning and ntk_granularity == "unit":
                    layer_sensitivities = _compute_ntk_sensitivity_per_unit(
                        model=model,
                        dataloader=dataloader,
                        current_masks=current_masks,
                        prev_masks=None if step_idx == 0 else prev_masks,
                        mask_type="head",
                        num_samples=num_samples,
                        device=device,
                        verbose=verbose,
                        num_heads=num_heads,
                        ntk_target_mode=ntk_target_mode,
                        normalize_per_layer=False,
                        skip_layers=skip_layers
                    )
                    if ntk_norm_method == "legacy":
                        all_sens = torch.cat([layer_sensitivities[l] for l in range(num_layers) if l in layer_sensitivities])
                        nonzero_mask = all_sens > 0
                        global_mean = all_sens[nonzero_mask].mean() if nonzero_mask.any() else all_sens.mean()
                        if global_mean > 0:
                            for layer_idx in layer_sensitivities:
                                layer_sensitivities[layer_idx] = layer_sensitivities[layer_idx] / global_mean
                        if verbose:
                            print(f"  [NTK-previous] Global normalization (legacy): original_mean={global_mean:.4f}")
                else:
                    layer_sensitivities = _compute_ntk_sensitivity_per_unit(
                        model=model,
                        dataloader=dataloader,
                        current_masks=current_masks,
                        prev_masks=None if step_idx == 0 else prev_masks,
                        mask_type="head",
                        num_samples=num_samples,
                        device=device,
                        verbose=verbose,
                        num_heads=num_heads,
                        ntk_target_mode=ntk_target_mode,
                        skip_layers=skip_layers
                    )

            if use_ntk_adjustment:
                prev_masks = {k: v.clone() for k, v in current_masks.items()}

        if verbose:
            total_kept = sum(mask.sum().item() for mask in current_masks.values())
            total_heads_all = num_layers * num_heads
            actual_ratio = 1 - (total_kept / total_heads_all)
            print(f"  → Actual pruning: {actual_ratio:.1%} ({int(total_heads_all - total_kept)}/{total_heads_all} pruned)")

    if verbose:
        print(f"\n[Iterative Head Pruning] Final head mask distribution:")
        for layer_idx in range(num_layers):
            kept = int(current_masks[layer_idx].sum().item())
            skip_marker = " (skip)" if layer_idx in skip_layers else ""
            print(f"  Layer {layer_idx}: {kept}/{num_heads} heads kept{skip_marker}")

    model.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return current_masks


def _create_dimension_masks_qkv(
    qk_dim_scores: Dict[int, torch.Tensor],
    v_dim_scores: Dict[int, torch.Tensor],
    dimension_pruning_ratio: float,
    dimension_group_size: int = 16,
    global_pruning: bool = True,
    verbose: bool = True
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    num_layers = len(qk_dim_scores)
    head_dim = qk_dim_scores[0].size(0)
    num_groups_per_layer = head_dim // dimension_group_size
    device = qk_dim_scores[0].device if qk_dim_scores[0].is_cuda else "cpu"

    if verbose:
        print(f"\n[Dim Mask QKV] Creating separate QK/V masks")
        print(f"[Dim Mask QKV] Head dim: {head_dim}, Group size: {dimension_group_size}")
        print(f"[Dim Mask QKV] Pruning ratio: {dimension_pruning_ratio:.2%}")
        print(f"[Dim Mask QKV] Global pruning: {global_pruning}")

    if not global_pruning:
        qk_dim_masks = {}
        v_dim_masks = {}

        for layer_idx in range(num_layers):
            qk_score = qk_dim_scores[layer_idx].to(device)
            v_score = v_dim_scores[layer_idx].to(device)

            qk_sorted_scores, qk_sorted_indices = torch.sort(qk_score, descending=True)
            v_sorted_scores, v_sorted_indices = torch.sort(v_score, descending=True)

            combined_scores = (qk_sorted_scores + v_sorted_scores) / 2

            combined_sorted_scores, combined_order = torch.sort(combined_scores, descending=True)

            num_groups_to_keep = int(num_groups_per_layer * (1 - dimension_pruning_ratio))
            num_groups_to_keep = max(1, num_groups_to_keep)

            qk_mask = torch.zeros(head_dim, device=device)
            v_mask = torch.zeros(head_dim, device=device)

            for group_idx in range(num_groups_to_keep):
                start_idx = group_idx * dimension_group_size
                end_idx = start_idx + dimension_group_size

                group_positions = combined_order[start_idx:end_idx]

                qk_mask[qk_sorted_indices[group_positions]] = 1.0
                v_mask[v_sorted_indices[group_positions]] = 1.0

            qk_dim_masks[layer_idx] = qk_mask.cpu()
            v_dim_masks[layer_idx] = v_mask.cpu()

        return qk_dim_masks, v_dim_masks

    all_groups = []

    for layer_idx in sorted(qk_dim_scores.keys()):
        qk_score = qk_dim_scores[layer_idx].to(device)
        v_score = v_dim_scores[layer_idx].to(device)

        qk_sorted_scores, qk_sorted_indices = torch.sort(qk_score, descending=True)
        v_sorted_scores, v_sorted_indices = torch.sort(v_score, descending=True)

        combined_scores = (qk_sorted_scores + v_sorted_scores) / 2

        combined_sorted_scores, combined_order = torch.sort(combined_scores, descending=True)

        for group_idx in range(num_groups_per_layer):
            start_idx = group_idx * dimension_group_size
            end_idx = start_idx + dimension_group_size

            group_positions = combined_order[start_idx:end_idx]

            qk_group_indices = qk_sorted_indices[group_positions]
            v_group_indices = v_sorted_indices[group_positions]

            group_avg = combined_sorted_scores[start_idx:end_idx].mean().item()

            all_groups.append({
                'layer_idx': layer_idx,
                'group_idx': group_idx,
                'combined_avg': group_avg,
                'qk_original_indices': qk_group_indices.cpu(),
                'v_original_indices': v_group_indices.cpu()
            })

    all_groups.sort(key=lambda x: x['combined_avg'], reverse=True)

    total_groups = len(all_groups)
    num_groups_to_keep = int(total_groups * (1 - dimension_pruning_ratio))
    num_groups_to_keep = max(num_layers, num_groups_to_keep)

    selected_groups = all_groups[:num_groups_to_keep]

    if verbose:
        print(f"[Dim Mask QKV] Global: {total_groups} groups → {num_groups_to_keep} kept")

    qk_dim_masks = {}
    v_dim_masks = {}

    layer_groups = {i: [] for i in range(num_layers)}
    for group in selected_groups:
        layer_groups[group['layer_idx']].append(group)

    for layer_idx in range(num_layers):
        groups = layer_groups[layer_idx]

        qk_mask = torch.zeros(head_dim)
        v_mask = torch.zeros(head_dim)

        for group in groups:
            qk_mask[group['qk_original_indices']] = 1.0
            v_mask[group['v_original_indices']] = 1.0

        qk_dim_masks[layer_idx] = qk_mask
        v_dim_masks[layer_idx] = v_mask

        if verbose and layer_idx < 3:
            qk_kept = int(qk_mask.sum().item())
            v_kept = int(v_mask.sum().item())
            print(f"  Layer {layer_idx}: QK kept={qk_kept}/{head_dim}, V kept={v_kept}/{head_dim}")

    return qk_dim_masks, v_dim_masks


def compute_taylor_dimension_masks(
    model: nn.Module,
    dataloader,
    dimension_pruning_ratio: float,
    num_samples: int = 128,
    dimension_group_size: int = 16,
    device: str = "cuda",
    verbose: bool = True,
    head_mode: str = "o",
    global_pruning: bool = True,
    importance_method: str = "ganda",
    enable_iterative_pruning: bool = False,
    pruning_step_size: float = 0.05,
    iterative_start_ratio_ffn: float = 0.0,
    iterative_start_ratio_head: float = 0.0,
    iterative_start_ratio_emb: float = 0.0,
    use_ntk_adjustment: bool = False,
    ntk_adjustment_alpha: float = 0.3,
    ntk_adjustment_mode: str = "previous",
    ntk_sensitivity_direction: str = "normal",
    ntk_target_mode: str = "default",
    separate_qkv_dimension: bool = False,
    use_iterative_premasking: bool = False,
    ntk_norm_method: str = "robust",
    ntk_granularity: str = "unit"
):
    num_layers = model.config.num_hidden_layers
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    num_groups = head_dim // dimension_group_size

    if enable_iterative_pruning and dimension_pruning_ratio > 0:
        return _compute_taylor_dimension_masks_iterative(
            model=model,
            dataloader=dataloader,
            target_ratio=dimension_pruning_ratio,
            step_size=pruning_step_size,
            num_samples=num_samples,
            dimension_group_size=dimension_group_size,
            device=device,
            verbose=verbose,
            head_mode=head_mode,
            global_pruning=global_pruning,
            importance_method=importance_method,
            use_ntk_adjustment=use_ntk_adjustment,
            ntk_adjustment_alpha=ntk_adjustment_alpha,
            ntk_adjustment_mode=ntk_adjustment_mode,
            ntk_sensitivity_direction=ntk_sensitivity_direction,
            ntk_target_mode=ntk_target_mode,
            separate_qkv_dimension=separate_qkv_dimension,
            use_iterative_premasking=use_iterative_premasking,
            ntk_norm_method=ntk_norm_method,
            ntk_granularity=ntk_granularity
        )

    calculator = TaylorImportanceCalculator(model, device, verbose, importance_method=importance_method)

    if separate_qkv_dimension:
        qk_dim_scores, v_dim_scores = calculator.compute_taylor_dimension_scores_qkv(dataloader, num_samples)
        return _create_dimension_masks_qkv(
            qk_dim_scores=qk_dim_scores,
            v_dim_scores=v_dim_scores,
            dimension_pruning_ratio=dimension_pruning_ratio,
            dimension_group_size=dimension_group_size,
            global_pruning=global_pruning,
            verbose=verbose
        )

    dim_scores = calculator.compute_taylor_dimension_scores(dataloader, num_samples, head_mode)

    if verbose:
        print(f"\n[Taylor Dim Mask] Head dim: {head_dim}, Group size: {dimension_group_size}")
        print(f"[Taylor Dim Mask] Pruning ratio: {dimension_pruning_ratio:.2%}")
        print(f"[Taylor Dim Mask] Mode: {'Global (sorted grouping)' if global_pruning else 'Layer-wise (sorted grouping)'}")

    if global_pruning:
        dim_masks = _create_dimension_mask_sorted_grouping_global(
            all_scores=dim_scores,
            pruning_ratio=dimension_pruning_ratio,
            group_size=dimension_group_size
        )
    else:
        dim_masks = {}
        for layer_idx in range(num_layers):
            dim_masks[layer_idx] = _create_dimension_mask_sorted_grouping(
                scores=dim_scores[layer_idx],
                pruning_ratio=dimension_pruning_ratio,
                group_size=dimension_group_size
            )

    if verbose:
        print(f"\n[Taylor Dim Mask] Dimension masks created (scattered indices):")
        for layer_idx in range(min(5, num_layers)):
            kept = int(dim_masks[layer_idx].sum().item())
            kept_indices = torch.where(dim_masks[layer_idx] == 1.0)[0]
            sample_indices = kept_indices[:8].tolist() if len(kept_indices) > 0 else []
            print(f"  Layer {layer_idx}: {kept}/{head_dim} dims kept, sample indices: {sample_indices}...")

    return dim_masks


def _create_dimension_mask_sorted_grouping(
    scores: torch.Tensor,
    pruning_ratio: float,
    group_size: int = 16
) -> torch.Tensor:
    head_dim = scores.shape[0]

    num_to_keep = int(head_dim * (1 - pruning_ratio))
    num_to_keep = max(group_size, num_to_keep)
    num_to_keep = (num_to_keep // group_size) * group_size

    sorted_indices = torch.argsort(scores, descending=True)

    keep_indices = sorted_indices[:num_to_keep]

    mask = torch.zeros(head_dim)
    mask[keep_indices] = 1.0

    return mask


def _create_dimension_mask_sorted_grouping_global(
    all_scores: Dict[int, torch.Tensor],
    pruning_ratio: float,
    group_size: int = 16
) -> Dict[int, torch.Tensor]:
    num_layers = len(all_scores)
    head_dim = all_scores[0].shape[0]
    num_groups_per_layer = head_dim // group_size

    all_groups = []

    for layer_idx in range(num_layers):
        scores = all_scores[layer_idx]

        sorted_indices = torch.argsort(scores, descending=True)
        sorted_scores = scores[sorted_indices]

        for group_idx in range(num_groups_per_layer):
            start_idx = group_idx * group_size
            end_idx = start_idx + group_size

            group_original_indices = sorted_indices[start_idx:end_idx]

            group_score = sorted_scores[start_idx:end_idx].mean().item()

            all_groups.append({
                'score': group_score,
                'layer_idx': layer_idx,
                'group_idx': group_idx,
                'original_indices': group_original_indices
            })

    all_groups.sort(key=lambda x: x['score'], reverse=True)

    total_groups = len(all_groups)
    total_target_groups = num_layers * num_groups_per_layer
    total_target_prune = int(total_target_groups * pruning_ratio)
    num_groups_to_keep = max(1, total_groups - min(total_target_prune, total_groups))

    selected_groups = all_groups[:num_groups_to_keep]

    masks = {}
    for layer_idx in range(num_layers):
        masks[layer_idx] = torch.zeros(head_dim)

    for group_info in selected_groups:
        layer_idx = group_info['layer_idx']
        original_indices = group_info['original_indices']
        masks[layer_idx][original_indices] = 1.0

    for layer_idx in range(num_layers):
        if masks[layer_idx].sum() == 0:
            scores = all_scores[layer_idx]
            sorted_indices = torch.argsort(scores, descending=True)
            masks[layer_idx][sorted_indices[:group_size]] = 1.0

    return masks


def _compute_taylor_dimension_masks_iterative(
    model: nn.Module,
    dataloader,
    target_ratio: float,
    step_size: float,
    num_samples: int = 128,
    dimension_group_size: int = 16,
    device: str = "cuda",
    verbose: bool = True,
    head_mode: str = "o",
    global_pruning: bool = True,
    importance_method: str = "ganda",
    use_ntk_adjustment: bool = False,
    ntk_adjustment_alpha: float = 0.3,
    ntk_adjustment_mode: str = "previous",
    ntk_sensitivity_direction: str = "normal",
    ntk_target_mode: str = "default",
    separate_qkv_dimension: bool = False,
    use_iterative_premasking: bool = False,
    ntk_norm_method: str = "robust",
    ntk_granularity: str = "unit"
):
    if separate_qkv_dimension:
        if verbose:
            print(f"\n[Iterative Dim] separate_qkv_dimension=True → Using one-shot QKV mode")
        calculator = TaylorImportanceCalculator(model, device, verbose, importance_method=importance_method)
        qk_dim_scores, v_dim_scores = calculator.compute_taylor_dimension_scores_qkv(dataloader, num_samples)
        return _create_dimension_masks_qkv(
            qk_dim_scores=qk_dim_scores,
            v_dim_scores=v_dim_scores,
            dimension_pruning_ratio=target_ratio,
            dimension_group_size=dimension_group_size,
            global_pruning=global_pruning,
            verbose=verbose
        )
    num_layers = model.config.num_hidden_layers
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    num_groups = head_dim // dimension_group_size

    steps = []
    current_ratio = step_size
    while current_ratio < target_ratio:
        steps.append(current_ratio)
        current_ratio += step_size
    steps.append(target_ratio)

    if verbose:
        print(f"\n{'='*60}")
        print(f"[Iterative Dim Pruning] Starting iterative dimension pruning")
        print(f"[Iterative Dim Pruning] Target ratio: {target_ratio:.1%}")
        print(f"[Iterative Dim Pruning] Step size: {step_size:.1%}")
        print(f"[Iterative Dim Pruning] Steps: {len(steps)}")
        print(f"[Iterative Dim Pruning] Process: 0% → {' → '.join([f'{s:.0%}' for s in steps])}")
        if use_iterative_premasking:
            print(f"[Iterative Dim Pruning] Pre-masking: enabled (pruned dims output=0 during forward)")
        if use_ntk_adjustment:
            print(f"[Iterative Dim Pruning] NTK Adjustment: enabled (alpha={ntk_adjustment_alpha}, mode={ntk_adjustment_mode}, direction={ntk_sensitivity_direction})")
        print(f"{'='*60}")

    sens_sign = -1.0 if ntk_sensitivity_direction == "inverse" else 1.0

    current_masks = {}
    for layer_idx in range(num_layers):
        current_masks[layer_idx] = torch.ones(head_dim)

    layer_sensitivities = None  # {layer_idx: sensitivity_score}
    prev_masks = None

    for step_idx, current_target in enumerate(steps):
        if verbose:
            print(f"\n[Iterative Dim Step {step_idx + 1}/{len(steps)}] Pruning to {current_target:.1%}")

        if use_iterative_premasking and step_idx > 0:
            calculator = TaylorImportanceCalculator(
                model, device, verbose=False,
                importance_method=importance_method,
                dim_masks=current_masks
            )
        else:
            calculator = TaylorImportanceCalculator(
                model, device, verbose=False,
                importance_method=importance_method
            )
        dim_scores = calculator.compute_taylor_dimension_scores(dataloader, num_samples, head_mode)

        if not use_iterative_premasking:
            for layer_idx in range(num_layers):
                dim_scores[layer_idx] = dim_scores[layer_idx] * current_masks[layer_idx].to(dim_scores[layer_idx].device)

        # ===== Helper function for dimension pruning =====
        def apply_dim_pruning(scores_dict, target, is_global):
            masks = {}
            if is_global:
                masks = _create_dimension_mask_sorted_grouping_global(
                    all_scores=scores_dict,
                    pruning_ratio=target,
                    group_size=dimension_group_size
                )
            else:
                for layer_idx in range(num_layers):
                    masks[layer_idx] = _create_dimension_mask_sorted_grouping(
                        scores=scores_dict[layer_idx],
                        pruning_ratio=target,
                        group_size=dimension_group_size
                    )
            return masks

        num_heads = model.config.num_attention_heads

        if use_ntk_adjustment and ntk_adjustment_mode == "current":
            if verbose:
                print(f"  [NTK-current] 1st pruning (temporary)")
            temp_masks = apply_dim_pruning(dim_scores, current_target, global_pruning)

            if step_idx == 0:
                temp_masks = {i: torch.ones(head_dim) for i in range(num_layers)}

            if global_pruning:
                if verbose:
                    print(f"  [NTK-current] Computing layer-level NTK (Global mode)")
                current_sensitivities = _compute_ntk_sensitivity(
                    model=model,
                    dataloader=dataloader,
                    current_masks=temp_masks,
                    prev_masks=current_masks,
                    mask_type="dimension",
                    num_samples=num_samples,
                    device=device,
                    verbose=verbose,
                    ntk_target_mode=ntk_target_mode,
                    skip_layers=skip_layers
                )
                if verbose:
                    print(f"  [NTK-current] Adjusting scores (alpha={ntk_adjustment_alpha})")
                adjusted_scores = _apply_ntk_adjustment_to_scores(
                    scores=dim_scores,
                    sensitivities=current_sensitivities,
                    alpha=ntk_adjustment_alpha,
                    sens_sign=sens_sign,
                    granularity="layer",
                    unit_mode="group",
                    dimension_multiple=dimension_group_size,
                    intermediate_size=head_dim,
                    num_layers=num_layers,
                    verbose=verbose,
                    ntk_norm_method=ntk_norm_method,
                    skip_layers=skip_layers
                )
            else:
                if verbose:
                    print(f"  [NTK-current] Computing per-dimension NTK (Layer-wise mode)")
                current_sensitivities = _compute_ntk_sensitivity_per_unit(
                    model=model,
                    dataloader=dataloader,
                    current_masks=temp_masks,
                    prev_masks=current_masks,
                    mask_type="dimension",
                    num_samples=num_samples,
                    device=device,
                    verbose=verbose,
                    dimension_multiple=dimension_group_size,
                    num_heads=num_heads,
                    ntk_target_mode=ntk_target_mode,
                    skip_layers=skip_layers
                )
                if verbose:
                    print(f"  [NTK-current] Adjusting scores per-dimension (alpha={ntk_adjustment_alpha})")
                adjusted_scores = _apply_ntk_adjustment_to_scores(
                    scores=dim_scores,
                    sensitivities=current_sensitivities,
                    alpha=ntk_adjustment_alpha,
                    sens_sign=sens_sign,
                    granularity="unit",
                    unit_mode="group",
                    dimension_multiple=dimension_group_size,
                    intermediate_size=head_dim,
                    num_layers=num_layers,
                    verbose=verbose,
                    ntk_norm_method=ntk_norm_method,
                    skip_layers=skip_layers,
                    layer_wise=True
                )

            if verbose:
                print(f"  [NTK-current] 2nd pruning (final, no forward)")
            current_masks = apply_dim_pruning(adjusted_scores, current_target, global_pruning)

        else:
            if use_ntk_adjustment and layer_sensitivities is not None:
                if verbose:
                    print(f"  [NTK-previous] Applying sensitivity adjustment (alpha={ntk_adjustment_alpha})")

                prev_granularity = ntk_granularity if ntk_granularity else ("layer" if global_pruning else "unit")
                dim_scores = _apply_ntk_adjustment_to_scores(
                    scores=dim_scores,
                    sensitivities=layer_sensitivities,
                    alpha=ntk_adjustment_alpha,
                    sens_sign=sens_sign,
                    granularity=prev_granularity,
                    unit_mode="group",
                    dimension_multiple=dimension_group_size,
                    intermediate_size=head_dim,
                    num_layers=num_layers,
                    verbose=verbose,
                    ntk_norm_method=ntk_norm_method,
                    skip_layers=skip_layers,
                    layer_wise=not global_pruning
                )

            current_masks = apply_dim_pruning(dim_scores, current_target, global_pruning)

            if use_ntk_adjustment and step_idx < len(steps) - 1:
                if global_pruning:
                    layer_sensitivities = _compute_ntk_sensitivity(
                        model=model,
                        dataloader=dataloader,
                        current_masks=current_masks,
                        prev_masks=None if step_idx == 0 else prev_masks,
                        mask_type="dimension",
                        num_samples=num_samples,
                        device=device,
                        verbose=verbose,
                        ntk_target_mode=ntk_target_mode,
                        skip_layers=skip_layers
                    )
                else:
                    layer_sensitivities = _compute_ntk_sensitivity_per_unit(
                        model=model,
                        dataloader=dataloader,
                        current_masks=current_masks,
                        prev_masks=None if step_idx == 0 else prev_masks,
                        mask_type="dimension",
                        num_samples=num_samples,
                        device=device,
                        verbose=verbose,
                        dimension_multiple=dimension_group_size,
                        num_heads=num_heads,
                        ntk_target_mode=ntk_target_mode,
                        skip_layers=skip_layers
                    )

            if use_ntk_adjustment:
                prev_masks = {k: v.clone() for k, v in current_masks.items()}

        if verbose:
            total_kept = sum(mask.sum().item() for mask in current_masks.values())
            total_dims = num_layers * head_dim
            actual_ratio = 1 - (total_kept / total_dims)
            print(f"  → Actual pruning: {actual_ratio:.1%} ({int(total_dims - total_kept)}/{total_dims} pruned)")

    if verbose:
        print(f"\n[Iterative Dim Pruning] Final dimension mask distribution:")
        for layer_idx in range(min(5, num_layers)):
            kept = int(current_masks[layer_idx].sum().item())
            print(f"  Layer {layer_idx}: {kept}/{head_dim} dims kept")

    return current_masks


def _create_head_mask(
    scores: torch.Tensor,
    pruning_ratio: float,
    allow_full_block_pruning: bool = False
) -> torch.Tensor:
    num_heads = scores.shape[0]

    min_heads = 0 if allow_full_block_pruning else 1
    num_to_keep = max(min_heads, int(num_heads * (1 - pruning_ratio)))

    sorted_indices = torch.argsort(scores, descending=True)

    keep_indices = sorted_indices[:num_to_keep]

    mask = torch.zeros(num_heads)
    mask[keep_indices] = 1.0

    return mask


def _create_head_mask_global(
    all_scores: Dict[int, torch.Tensor],
    pruning_ratio: float,
    skip_first_layer: int = 0,
    skip_last_layer: int = 0,
    allow_full_block_pruning: bool = False
) -> Dict[int, torch.Tensor]:
    num_layers = len(all_scores)
    num_heads = all_scores[0].shape[0]

    skip_layers = set()
    if skip_first_layer > 0:
        skip_layers.update(range(min(skip_first_layer, num_layers)))
    if skip_last_layer > 0:
        skip_layers.update(range(max(0, num_layers - skip_last_layer), num_layers))

    non_skip_layers = [i for i in range(num_layers) if i not in skip_layers]
    score_tensors = [all_scores[layer_idx] for layer_idx in non_skip_layers]
    all_head_scores_tensor = torch.cat(score_tensors) if score_tensors else torch.tensor([])

    all_indices = [(layer_idx, head_idx)
                   for layer_idx in non_skip_layers
                   for head_idx in range(num_heads)]

    total_heads_all = num_layers * num_heads
    total_heads_available = len(all_head_scores_tensor)
    num_to_prune = min(int(total_heads_all * pruning_ratio), total_heads_available)

    _, sorted_indices = torch.sort(all_head_scores_tensor)
    prune_indices = set(sorted_indices[:num_to_prune].tolist())

    masks = {i: torch.ones(num_heads) for i in range(num_layers)}

    for flat_idx in prune_indices:
        layer_idx, head_idx = all_indices[flat_idx]
        masks[layer_idx][head_idx] = 0.0

    if not allow_full_block_pruning:
        for layer_idx in non_skip_layers:
            if masks[layer_idx].sum() == 0:
                scores = all_scores[layer_idx]
                best_head = scores.argmax().item()
                masks[layer_idx][best_head] = 1.0

    return masks


def _create_ffn_mask_sorted_grouping(
    scores: torch.Tensor,
    pruning_ratio: float,
    group_size: int = 16,
    allow_full_block_pruning: bool = False
) -> torch.Tensor:
    intermediate_size = scores.shape[0]

    num_to_keep = int(intermediate_size * (1 - pruning_ratio))
    min_keep = 0 if allow_full_block_pruning else group_size
    num_to_keep = max(min_keep, num_to_keep)
    num_to_keep = (num_to_keep // group_size) * group_size

    sorted_indices = torch.argsort(scores, descending=True)

    keep_indices = sorted_indices[:num_to_keep]

    mask = torch.zeros(intermediate_size)
    mask[keep_indices] = 1.0

    return mask


def _create_ffn_mask_sorted_grouping_global(
    all_scores: Dict[int, torch.Tensor],
    pruning_ratio: float,
    group_size: int = 16,
    skip_layers: set = None,
    allow_full_block_pruning: bool = False
) -> Dict[int, torch.Tensor]:
    if skip_layers is None:
        skip_layers = set()

    num_layers = len(all_scores)
    intermediate_size = all_scores[0].shape[0]
    num_groups_per_layer = intermediate_size // group_size

    all_groups = []

    for layer_idx in range(num_layers):
        if layer_idx in skip_layers:
            continue

        scores = all_scores[layer_idx]

        sorted_indices = torch.argsort(scores, descending=True)
        sorted_scores = scores[sorted_indices]

        for group_idx in range(num_groups_per_layer):
            start_idx = group_idx * group_size
            end_idx = start_idx + group_size

            group_original_indices = sorted_indices[start_idx:end_idx]

            group_score = sorted_scores[start_idx:end_idx].mean().item()

            all_groups.append({
                'score': group_score,
                'layer_idx': layer_idx,
                'group_idx': group_idx,
                'original_indices': group_original_indices
            })

    all_groups.sort(key=lambda x: x['score'], reverse=True)

    total_groups_all = num_layers * num_groups_per_layer
    total_groups_available = len(all_groups)
    total_target_prune = min(int(total_groups_all * pruning_ratio), total_groups_available)
    min_groups = 0 if allow_full_block_pruning else 1
    num_groups_to_keep = max(min_groups, total_groups_available - total_target_prune)

    selected_groups = all_groups[:num_groups_to_keep]

    masks = {}
    for layer_idx in range(num_layers):
        if layer_idx in skip_layers:
            masks[layer_idx] = torch.ones(intermediate_size)
        else:
            masks[layer_idx] = torch.zeros(intermediate_size)

    for group in selected_groups:
        layer_idx = group['layer_idx']
        original_indices = group['original_indices']
        masks[layer_idx][original_indices] = 1.0

    return masks


def _create_ffn_mask_sorted_grouping_global_with_groups(
    all_scores: Dict[int, torch.Tensor],
    pruning_ratio: float,
    group_size: int = 16,
    skip_layers: set = None
) -> Tuple[Dict[int, torch.Tensor], Dict[int, List[torch.Tensor]]]:
    if skip_layers is None:
        skip_layers = set()

    num_layers = len(all_scores)
    intermediate_size = all_scores[0].shape[0]
    num_groups_per_layer = intermediate_size // group_size

    group_assignments = {l: [] for l in range(num_layers)}

    all_groups = []

    for layer_idx in range(num_layers):
        if layer_idx in skip_layers:
            for group_idx in range(num_groups_per_layer):
                start_idx = group_idx * group_size
                end_idx = start_idx + group_size
                group_assignments[layer_idx].append(
                    torch.arange(start_idx, end_idx)
                )
            continue

        scores = all_scores[layer_idx]

        sorted_indices = torch.argsort(scores, descending=True)
        sorted_scores = scores[sorted_indices]

        for group_idx in range(num_groups_per_layer):
            start_idx = group_idx * group_size
            end_idx = start_idx + group_size

            group_original_indices = sorted_indices[start_idx:end_idx]

            group_assignments[layer_idx].append(group_original_indices.clone())

            group_score = sorted_scores[start_idx:end_idx].mean().item()

            all_groups.append({
                'score': group_score,
                'layer_idx': layer_idx,
                'group_idx': group_idx,
                'original_indices': group_original_indices
            })

    all_groups.sort(key=lambda x: x['score'], reverse=True)

    total_groups_all = num_layers * num_groups_per_layer
    total_groups_available = len(all_groups)
    total_target_prune = min(int(total_groups_all * pruning_ratio), total_groups_available)
    num_groups_to_keep = max(1, total_groups_available - total_target_prune)

    selected_groups = all_groups[:num_groups_to_keep]

    masks = {}
    for layer_idx in range(num_layers):
        if layer_idx in skip_layers:
            masks[layer_idx] = torch.ones(intermediate_size)
        else:
            masks[layer_idx] = torch.zeros(intermediate_size)

    for group in selected_groups:
        layer_idx = group['layer_idx']
        original_indices = group['original_indices']
        masks[layer_idx][original_indices] = 1.0

    return masks, group_assignments


def _is_weight_based_ffn_mode(ffn_mode: str) -> bool:
    return ffn_mode in ["weight_norm", "wn", "weight_taylor", "wt", "weight_ganda", "wg", "weight_wanda", "ww", "weight_wag", "wag", "wag_down", "wag_up", "wag_gate", "wag_all"]


def _compute_joint_masks_iterative(
    model: nn.Module,
    dataloader,
    ffn_target_ratio: float,
    head_target_ratio: float,
    step_size: float,
    num_samples: int = 128,
    dimension_multiple: int = 1,
    device: str = "cuda",
    verbose: bool = True,
    ffn_mode: str = "down",
    head_mode: str = "o",
    importance_method: str = "ganda",
    global_ffn_pruning: bool = True,
    global_head_pruning: bool = True,
    use_ntk_adjustment: bool = False,
    ntk_adjustment_alpha: float = 0.3,
    use_iterative_premasking: bool = False,
    skip_first_layer_head: int = 0,
    skip_last_layer_head: int = 0,
    skip_first_layer_ffn: int = 0,
    skip_last_layer_ffn: int = 0,
    allow_full_block_pruning: bool = False
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    num_layers = model.config.num_hidden_layers
    intermediate_size = getattr(model.config, 'intermediate_size', model.config.hidden_size * 4)
    num_heads = model.config.num_attention_heads

    skip_layers_ffn = set()
    if skip_first_layer_ffn > 0:
        skip_layers_ffn.update(range(min(skip_first_layer_ffn, num_layers)))
    if skip_last_layer_ffn > 0:
        skip_layers_ffn.update(range(max(0, num_layers - skip_last_layer_ffn), num_layers))

    skip_layers_head = set()
    if skip_first_layer_head > 0:
        skip_layers_head.update(range(min(skip_first_layer_head, num_layers)))
    if skip_last_layer_head > 0:
        skip_layers_head.update(range(max(0, num_layers - skip_last_layer_head), num_layers))

    max_ratio = max(ffn_target_ratio, head_target_ratio)
    steps = []
    current_ratio = step_size
    while current_ratio < max_ratio:
        steps.append(current_ratio)
        current_ratio += step_size
    steps.append(max_ratio)

    if verbose:
        print(f"\n{'='*70}")
        print(f"[Joint Iterative Pruning] Starting iterative FFN + Head pruning")
        print(f"[Joint Iterative Pruning] FFN target: {ffn_target_ratio:.1%}, Head target: {head_target_ratio:.1%}")
        print(f"[Joint Iterative Pruning] Step size: {step_size:.1%}")
        print(f"[Joint Iterative Pruning] Steps: {len(steps)}")
        print(f"[Joint Iterative Pruning] Process: 0% → {' → '.join([f'{s:.0%}' for s in steps])}")
        if skip_layers_ffn:
            print(f"[Joint Iterative Pruning] Skip layers (FFN): {sorted(skip_layers_ffn)} ({len(skip_layers_ffn)} layers)")
        if skip_layers_head:
            print(f"[Joint Iterative Pruning] Skip layers (HEAD): {sorted(skip_layers_head)} ({len(skip_layers_head)} layers)")
        if use_iterative_premasking:
            print(f"[Joint Iterative Pruning] Pre-masking: enabled")
        if use_ntk_adjustment:
            print(f"[Joint Iterative Pruning] NTK Adjustment: enabled (alpha={ntk_adjustment_alpha})")
        print(f"{'='*70}")

    ffn_masks = {i: torch.ones(intermediate_size) for i in range(num_layers)}
    head_masks = {i: torch.ones(num_heads) for i in range(num_layers)}

    group_size = dimension_multiple
    num_groups_per_layer = intermediate_size // group_size

    for step_idx, current_step_ratio in enumerate(steps):
        ffn_current_ratio = min(current_step_ratio, ffn_target_ratio)
        head_current_ratio = min(current_step_ratio, head_target_ratio)

        if verbose:
            print(f"\n[Joint Step {step_idx + 1}/{len(steps)}] FFN: {ffn_current_ratio:.1%}, Head: {head_current_ratio:.1%}")

        if use_iterative_premasking and step_idx > 0:
            calculator = TaylorImportanceCalculator(
                model, device, verbose=False,
                ffn_mode=ffn_mode,
                importance_method=importance_method,
                ffn_masks=ffn_masks,
                head_masks=head_masks
            )
        else:
            calculator = TaylorImportanceCalculator(
                model, device, verbose=False,
                ffn_mode=ffn_mode,
                importance_method=importance_method
            )

        ffn_scores, head_scores = calculator.compute_joint_scores(dataloader, num_samples, head_mode)

        del calculator
        gc.collect()
        torch.cuda.empty_cache()

        if not use_iterative_premasking:
            for layer_idx in range(num_layers):
                ffn_scores[layer_idx] = ffn_scores[layer_idx] * ffn_masks[layer_idx].to(ffn_scores[layer_idx].device)
                head_scores[layer_idx] = head_scores[layer_idx] * head_masks[layer_idx].to(head_scores[layer_idx].device)

        if ffn_current_ratio > 0:
            if global_ffn_pruning:
                ffn_masks = _create_ffn_mask_sorted_grouping_global(
                    all_scores=ffn_scores,
                    pruning_ratio=ffn_current_ratio,
                    group_size=group_size,
                    skip_layers=skip_layers_ffn,
                    allow_full_block_pruning=allow_full_block_pruning
                )
            else:
                for layer_idx in range(num_layers):
                    if layer_idx in skip_layers_ffn:
                        ffn_masks[layer_idx] = torch.ones(intermediate_size)
                    else:
                        ffn_masks[layer_idx] = _create_ffn_mask_sorted_grouping(
                            scores=ffn_scores[layer_idx],
                            pruning_ratio=ffn_current_ratio,
                            group_size=group_size,
                            allow_full_block_pruning=allow_full_block_pruning
                        )

        if head_current_ratio > 0:
            if global_head_pruning:
                non_skip_layers_head = [i for i in range(num_layers) if i not in skip_layers_head]
                score_tensors = [head_scores[layer_idx] for layer_idx in non_skip_layers_head]
                all_head_scores_tensor = torch.cat(score_tensors) if score_tensors else torch.tensor([])

                all_indices = [(layer_idx, head_idx)
                               for layer_idx in non_skip_layers_head
                               for head_idx in range(num_heads)]

                total_heads = len(all_head_scores_tensor)
                num_to_prune = int(total_heads * head_current_ratio)
                _, sorted_indices = torch.sort(all_head_scores_tensor)
                prune_indices = set(sorted_indices[:num_to_prune].tolist())

                for layer_idx in range(num_layers):
                    head_masks[layer_idx] = torch.ones(num_heads)

                for flat_idx in prune_indices:
                    layer_idx, head_idx = all_indices[flat_idx]
                    head_masks[layer_idx][head_idx] = 0.0

                if not allow_full_block_pruning:
                    for layer_idx in range(num_layers):
                        if layer_idx in skip_layers_head:
                            continue
                        if head_masks[layer_idx].sum() == 0:
                            scores = head_scores[layer_idx]
                            best_head = scores.argmax().item()
                            head_masks[layer_idx][best_head] = 1.0
            else:
                min_heads = 0 if allow_full_block_pruning else 1
                for layer_idx in range(num_layers):
                    if layer_idx in skip_layers_head:
                        head_masks[layer_idx] = torch.ones(num_heads)
                        continue

                    scores = head_scores[layer_idx]
                    num_to_keep = max(min_heads, int(num_heads * (1 - head_current_ratio)))
                    head_masks[layer_idx] = torch.zeros(num_heads)
                    if num_to_keep > 0:
                        _, top_indices = torch.topk(scores, num_to_keep)
                        head_masks[layer_idx][top_indices] = 1.0

        if verbose:
            ffn_kept = sum(m.sum().item() for m in ffn_masks.values())
            ffn_total = num_layers * intermediate_size
            head_kept = sum(m.sum().item() for m in head_masks.values())
            head_total = num_layers * num_heads
            print(f"  FFN: {int(ffn_kept)}/{ffn_total} neurons kept ({ffn_kept/ffn_total:.1%})")
            print(f"  Head: {int(head_kept)}/{head_total} heads kept ({head_kept/head_total:.1%})")

    if verbose:
        print(f"\n{'='*70}")
        print(f"[Joint Iterative Pruning] Completed!")
        print(f"[Joint Iterative Pruning] Final FFN masks: {num_layers} layers")
        print(f"[Joint Iterative Pruning] Final Head masks: {num_layers} layers")
        if skip_layers_ffn:
            print(f"[Joint Iterative Pruning] Skip layers (FFN): {sorted(skip_layers_ffn)}")
        if skip_layers_head:
            print(f"[Joint Iterative Pruning] Skip layers (HEAD): {sorted(skip_layers_head)}")
        print(f"{'='*70}")

    return ffn_masks, head_masks


def compute_joint_masks(
    model: nn.Module,
    dataloader,
    ffn_pruning_ratio: float,
    head_pruning_ratio: float,
    num_samples: int = 128,
    dimension_multiple: int = 1,
    device: str = "cuda",
    verbose: bool = True,
    ffn_mode: str = "down",
    head_mode: str = "o",
    importance_method: str = "ganda",
    global_ffn_pruning: bool = True,
    global_head_pruning: bool = True,
    enable_iterative_pruning: bool = False,
    pruning_step_size: float = 0.05,
    use_ntk_adjustment: bool = False,
    ntk_adjustment_alpha: float = 0.3,
    use_iterative_premasking: bool = False,
    skip_first_layer_head: int = 0,
    skip_last_layer_head: int = 0,
    skip_first_layer_ffn: int = 0,
    skip_last_layer_ffn: int = 0,
    allow_full_block_pruning: bool = False
) -> Tuple[Optional[Dict[int, torch.Tensor]], Optional[Dict[int, torch.Tensor]]]:
    num_layers = model.config.num_hidden_layers
    intermediate_size = getattr(model.config, 'intermediate_size', model.config.hidden_size * 4)
    num_heads = model.config.num_attention_heads

    num_skip_ffn = skip_first_layer_ffn + skip_last_layer_ffn
    num_skip_head = skip_first_layer_head + skip_last_layer_head

    if num_skip_ffn > 0 and num_skip_ffn < num_layers:
        adjusted_ffn_ratio = ffn_pruning_ratio * num_layers / (num_layers - num_skip_ffn)
        if adjusted_ffn_ratio > 1.0:
            print(f"[WARNING] Adjusted FFN ratio {adjusted_ffn_ratio:.2%} > 100%, clamping to 100%")
            adjusted_ffn_ratio = 1.0
    else:
        adjusted_ffn_ratio = ffn_pruning_ratio

    if num_skip_head > 0 and num_skip_head < num_layers:
        adjusted_head_ratio = head_pruning_ratio * num_layers / (num_layers - num_skip_head)
        if adjusted_head_ratio > 1.0:
            print(f"[WARNING] Adjusted Head ratio {adjusted_head_ratio:.2%} > 100%, clamping to 100%")
            adjusted_head_ratio = 1.0
    else:
        adjusted_head_ratio = head_pruning_ratio

    if enable_iterative_pruning and (ffn_pruning_ratio > 0 or head_pruning_ratio > 0):
        return _compute_joint_masks_iterative(
            model=model,
            dataloader=dataloader,
            ffn_target_ratio=ffn_pruning_ratio if global_ffn_pruning else adjusted_ffn_ratio,
            head_target_ratio=adjusted_head_ratio,
            step_size=pruning_step_size,
            num_samples=num_samples,
            dimension_multiple=dimension_multiple,
            device=device,
            verbose=verbose,
            ffn_mode=ffn_mode,
            head_mode=head_mode,
            importance_method=importance_method,
            global_ffn_pruning=global_ffn_pruning,
            global_head_pruning=global_head_pruning,
            use_ntk_adjustment=use_ntk_adjustment,
            ntk_adjustment_alpha=ntk_adjustment_alpha,
            use_iterative_premasking=use_iterative_premasking,
            skip_first_layer_head=skip_first_layer_head,
            skip_last_layer_head=skip_last_layer_head,
            skip_first_layer_ffn=skip_first_layer_ffn,
            skip_last_layer_ffn=skip_last_layer_ffn,
            allow_full_block_pruning=allow_full_block_pruning
        )

    if verbose:
        print(f"\n[Joint Masks] Computing FFN + Head masks in single calibration pass")
        print(f"[Joint Masks] Target FFN ratio: {ffn_pruning_ratio:.2%}, Head ratio: {head_pruning_ratio:.2%}")
        if num_skip_ffn > 0 or num_skip_head > 0:
            print(f"[Joint Masks] Skip layers (FFN): first={skip_first_layer_ffn}, last={skip_last_layer_ffn} ({num_skip_ffn} layers)")
            print(f"[Joint Masks] Skip layers (HEAD): first={skip_first_layer_head}, last={skip_last_layer_head} ({num_skip_head} layers)")
            print(f"[Joint Masks] Adjusted FFN ratio: {adjusted_ffn_ratio:.2%}, Head ratio: {adjusted_head_ratio:.2%}")
        print(f"[Joint Masks] Num samples: {num_samples}")

    calculator = TaylorImportanceCalculator(
        model, device, verbose,
        ffn_mode=ffn_mode,
        importance_method=importance_method
    )
    ffn_scores, head_scores = calculator.compute_joint_scores(dataloader, num_samples, head_mode)

    del calculator
    gc.collect()
    torch.cuda.empty_cache()

    ffn_masks = None
    head_masks = None

    if adjusted_ffn_ratio > 0:
        group_size = dimension_multiple
        num_groups_per_layer = intermediate_size // group_size

        skip_layers_ffn = set()
        if skip_first_layer_ffn > 0:
            skip_layers_ffn.update(range(min(skip_first_layer_ffn, num_layers)))
        if skip_last_layer_ffn > 0:
            skip_layers_ffn.update(range(max(0, num_layers - skip_last_layer_ffn), num_layers))

        if global_ffn_pruning:
            ffn_masks = _create_ffn_mask_sorted_grouping_global(
                all_scores=ffn_scores,
                pruning_ratio=ffn_pruning_ratio,
                group_size=group_size,
                skip_layers=skip_layers_ffn,
                allow_full_block_pruning=allow_full_block_pruning
            )
        else:
            ffn_masks = {}
            for layer_idx in range(num_layers):
                if layer_idx in skip_layers_ffn:
                    ffn_masks[layer_idx] = torch.ones(intermediate_size)
                else:
                    ffn_masks[layer_idx] = _create_ffn_mask_sorted_grouping(
                        scores=ffn_scores[layer_idx],
                        pruning_ratio=adjusted_ffn_ratio,
                        group_size=group_size,
                        allow_full_block_pruning=allow_full_block_pruning
                    )

        if verbose:
            print(f"\n[Joint Masks] FFN masks created:")
            for layer_idx in range(min(5, num_layers)):
                kept = int(ffn_masks[layer_idx].sum().item())
                print(f"  Layer {layer_idx}: {kept}/{intermediate_size} neurons kept")

    if adjusted_head_ratio > 0:
        skip_layers_head = set()
        if skip_first_layer_head > 0:
            skip_layers_head.update(range(min(skip_first_layer_head, num_layers)))
        if skip_last_layer_head > 0:
            skip_layers_head.update(range(max(0, num_layers - skip_last_layer_head), num_layers))

        if global_head_pruning:
            non_skip_layers_head = [i for i in range(num_layers) if i not in skip_layers_head]
            score_tensors = [head_scores[layer_idx] for layer_idx in non_skip_layers_head]
            all_scores_tensor = torch.cat(score_tensors) if score_tensors else torch.tensor([])

            all_indices = [(layer_idx, head_idx)
                           for layer_idx in non_skip_layers_head
                           for head_idx in range(num_heads)]

            total_heads_available = len(all_scores_tensor)
            total_heads_all = num_layers * num_heads
            total_target_prune = min(int(total_heads_all * head_pruning_ratio), total_heads_available)
            num_to_prune = total_target_prune

            _, sorted_indices = torch.sort(all_scores_tensor)
            prune_indices = set(sorted_indices[:num_to_prune].tolist())

            head_masks = {}
            for layer_idx in range(num_layers):
                if layer_idx in skip_layers_head:
                    head_masks[layer_idx] = torch.ones(num_heads)
                else:
                    head_masks[layer_idx] = torch.ones(num_heads)

            for flat_idx in prune_indices:
                layer_idx, head_idx = all_indices[flat_idx]
                head_masks[layer_idx][head_idx] = 0.0

            if not allow_full_block_pruning:
                for layer_idx in range(num_layers):
                    if layer_idx not in skip_layers_head and head_masks[layer_idx].sum() == 0:
                        scores = head_scores[layer_idx]
                        best_head = scores.argmax().item()
                        head_masks[layer_idx][best_head] = 1.0
        else:
            # Layer-wise Head pruning
            head_masks = {}
            min_heads = 0 if allow_full_block_pruning else 1
            for layer_idx in range(num_layers):
                if layer_idx in skip_layers_head:
                    head_masks[layer_idx] = torch.ones(num_heads)
                    continue

                scores = head_scores[layer_idx]
                num_to_keep = max(min_heads, int(num_heads * (1 - adjusted_head_ratio)))
                if num_to_keep > 0:
                    _, top_indices = torch.topk(scores, num_to_keep)
                    mask = torch.zeros(num_heads)
                    mask[top_indices] = 1.0
                else:
                    mask = torch.zeros(num_heads)
                head_masks[layer_idx] = mask

        if verbose:
            print(f"\n[Joint Masks] Head masks created:")
            for layer_idx in range(min(5, num_layers)):
                kept = int(head_masks[layer_idx].sum().item())
                skip_mark = " (skipped)" if layer_idx in skip_layers_head else ""
                print(f"  Layer {layer_idx}: {kept}/{num_heads} heads kept{skip_mark}")

    if verbose:
        print(f"\n[Joint Masks] Done! FFN: {'created' if ffn_masks else 'skipped'}, Head: {'created' if head_masks else 'skipped'}")

    return ffn_masks, head_masks


def compute_embedding_masks(
    model: nn.Module,
    dataloader,
    embedding_pruning_ratio: float,
    num_samples: int = 128,
    dimension_multiple: int = 64,
    device: str = "cuda",
    verbose: bool = True,
    importance_method: str = "ganda",
    enable_iterative_pruning: bool = False,
    pruning_step_size: float = 0.05,
    iterative_start_ratio_ffn: float = 0.0,
    iterative_start_ratio_head: float = 0.0,
    iterative_start_ratio_emb: float = 0.0,
    use_ntk_adjustment: bool = False,
    ntk_adjustment_alpha: float = 0.3,
    use_iterative_premasking: bool = False
) -> torch.Tensor:
    hidden_size = model.config.hidden_size

    if embedding_pruning_ratio <= 0:
        if verbose:
            print(f"[Embedding Mask] Pruning ratio is 0, returning all-ones mask")
        return torch.ones(hidden_size)

    if verbose:
        print(f"\n{'='*60}")
        print(f"[Embedding Mask] Computing embedding masks")
        print(f"[Embedding Mask] Hidden size: {hidden_size}")
        print(f"[Embedding Mask] Pruning ratio: {embedding_pruning_ratio:.2%}")
        print(f"[Embedding Mask] Dimension multiple: {dimension_multiple}")
        print(f"[Embedding Mask] Importance method: {importance_method}")
        print(f"[Embedding Mask] Iterative pruning: {enable_iterative_pruning}")
        if enable_iterative_pruning and use_iterative_premasking:
            print(f"[Embedding Mask] Iterative pre-masking: enabled")
        print(f"[Embedding Mask] NTK adjustment: {use_ntk_adjustment}")
        print(f"{'='*60}")

    if enable_iterative_pruning:
        return _compute_embedding_masks_iterative(
            model=model,
            dataloader=dataloader,
            target_ratio=embedding_pruning_ratio,
            step_size=pruning_step_size,
            start_ratio=iterative_start_ratio_emb,
            num_samples=num_samples,
            dimension_multiple=dimension_multiple,
            device=device,
            verbose=verbose,
            importance_method=importance_method,
            use_ntk_adjustment=use_ntk_adjustment,
            ntk_adjustment_alpha=ntk_adjustment_alpha,
            use_iterative_premasking=use_iterative_premasking
        )

    calculator = TaylorImportanceCalculator(
        model=model,
        device=device,
        verbose=verbose,
        ffn_mode="down",
        importance_method=importance_method,
        use_correct_normalization=use_correct_normalization
    )

    embed_scores = calculator.compute_embedding_importance(dataloader, num_samples)

    if use_ntk_adjustment:
        if verbose:
            print(f"[Embedding Mask] Applying NTK adjustment (alpha={ntk_adjustment_alpha})...")

        ntk_sensitivity = _compute_embedding_ntk_sensitivity(
            model, dataloader, num_samples, device, verbose
        )

        if ntk_sensitivity is not None:
            ntk_norm = ntk_sensitivity / (ntk_sensitivity.max() + 1e-8)
            embed_norm = embed_scores / (embed_scores.max() + 1e-8)

            adjusted_scores = embed_norm * (1 + ntk_adjustment_alpha * ntk_norm)
            embed_scores = adjusted_scores

            if verbose:
                print(f"[Embedding Mask] NTK adjustment applied")

    mask = _create_embedding_mask(
        embed_scores,
        embedding_pruning_ratio,
        dimension_multiple,
        verbose
    )

    return mask


def _compute_embedding_masks_iterative(
    model: nn.Module,
    dataloader,
    target_ratio: float,
    step_size: float,
    start_ratio: float = 0.0,
    num_samples: int = 128,
    dimension_multiple: int = 64,
    device: str = "cuda",
    verbose: bool = True,
    importance_method: str = "ganda",
    use_ntk_adjustment: bool = False,
    ntk_adjustment_alpha: float = 0.3,
    use_iterative_premasking: bool = False
) -> torch.Tensor:
    hidden_size = model.config.hidden_size
    current_ratio = 0.0
    current_mask = torch.ones(hidden_size)

    effective_start = min(start_ratio, target_ratio) if start_ratio > 0 else 0.0
    remaining = target_ratio - effective_start
    num_steps = max(1, int(remaining / step_size)) if effective_start > 0 else int(target_ratio / step_size)
    if num_steps == 0:
        num_steps = 1
    if effective_start > 0:
        num_steps += 1

    if verbose:
        print(f"\n[Embedding Iterative] Starting iterative pruning")
        print(f"[Embedding Iterative] Target ratio: {target_ratio:.2%}")
        print(f"[Embedding Iterative] Step size: {step_size:.2%}")
        if effective_start > 0:
            print(f"[Embedding Iterative] Hybrid: one-shot to {effective_start:.2%}, then iterative")
        print(f"[Embedding Iterative] Number of steps: {num_steps}")
        if use_iterative_premasking:
            print(f"[Embedding Iterative] Pre-masking: enabled (pruned hidden dims output=0 during forward)")

    for step in range(num_steps):
        if effective_start > 0:
            if step == 0:
                current_ratio = effective_start
            else:
                current_ratio = min(effective_start + step * step_size, target_ratio)
        else:
            current_ratio = min((step + 1) * step_size, target_ratio)

        if verbose:
            print(f"\n[Embedding Iterative] Step {step+1}/{num_steps}: ratio={current_ratio:.2%}")

        if use_iterative_premasking and step > 0:
            calculator = TaylorImportanceCalculator(
                model=model,
                device=device,
                verbose=False,
                ffn_mode="down",
                importance_method=importance_method,
                embedding_mask=current_mask,
                use_correct_normalization=use_correct_normalization
            )
        else:
            calculator = TaylorImportanceCalculator(
                model=model,
                device=device,
                verbose=False,
                ffn_mode="down",
                importance_method=importance_method,
                use_correct_normalization=use_correct_normalization
            )

        embed_scores = calculator.compute_embedding_importance(dataloader, num_samples)

        if not use_iterative_premasking:
            pruned_dims = (current_mask == 0)
            embed_scores[pruned_dims] = embed_scores.max() + 1

        if use_ntk_adjustment:
            ntk_sensitivity = _compute_embedding_ntk_sensitivity(
                model, dataloader, num_samples, device, False
            )
            if ntk_sensitivity is not None:
                ntk_norm = ntk_sensitivity / (ntk_sensitivity.max() + 1e-8)
                embed_norm = embed_scores / (embed_scores.max() + 1e-8)
                embed_scores = embed_norm * (1 + ntk_adjustment_alpha * ntk_norm)

        current_mask = _create_embedding_mask(
            embed_scores,
            current_ratio,
            dimension_multiple,
            verbose=False
        )

        kept = current_mask.sum().item()
        if verbose:
            print(f"[Embedding Iterative] Step {step+1} result: {int(kept)}/{hidden_size} dims kept ({kept/hidden_size:.1%})")

    if verbose:
        final_kept = current_mask.sum().item()
        print(f"\n[Embedding Iterative] Final: {int(final_kept)}/{hidden_size} dims kept ({final_kept/hidden_size:.1%})")

    return current_mask


def _compute_embedding_ntk_sensitivity(
    model: nn.Module,
    dataloader,
    num_samples: int,
    device: str,
    verbose: bool
) -> Optional[torch.Tensor]:
    hidden_size = model.config.hidden_size
    sensitivity = torch.zeros(hidden_size, device='cpu')

    # ==================== Temporarily enable gradients for frozen models ====================
    original_requires_grad = {}  # {param_id: original_value}
    for param in model.parameters():
        param_id = id(param)
        original_requires_grad[param_id] = param.requires_grad
        param.requires_grad_(True)

    model.train()

    cached_samples = []
    data_iter = iter(dataloader)
    while len(cached_samples) < num_samples:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        if isinstance(batch, dict):
            bsz = batch['input_ids'].shape[0]
            for i in range(bsz):
                single = {k: v[i:i+1] for k, v in batch.items() if isinstance(v, torch.Tensor)}
                cached_samples.append(single)
                if len(cached_samples) >= num_samples:
                    break
        else:
            bsz = batch[0].shape[0]
            for i in range(bsz):
                single = tuple(v[i:i+1] for v in batch)
                cached_samples.append(single)
                if len(cached_samples) >= num_samples:
                    break

    num_processed = 0
    pbar = tqdm(cached_samples, desc="[Embedding NTK] Computing sensitivity", disable=not verbose)

    for sample in pbar:
        if isinstance(sample, dict):
            input_ids = sample['input_ids'].to(device)
            attention_mask = sample.get('attention_mask', torch.ones_like(sample['input_ids'])).to(device)
        else:
            input_ids = sample[0].to(device)
            attention_mask = sample[1].to(device) if len(sample) > 1 else None

        labels = input_ids.clone()

        try:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()

            for layer in model.model.layers:
                col_projs = [layer.self_attn.q_proj, layer.self_attn.k_proj,
                            layer.self_attn.v_proj] + get_up_proj_modules(layer.mlp)
                for proj in col_projs:
                    weight = proj.base_layer.weight if hasattr(proj, 'base_layer') else proj.weight
                    if weight.grad is not None:
                        grad_norm = (weight.grad ** 2).sum(dim=0)
                        sensitivity += grad_norm.cpu()

                row_projs = []
                o_proj = get_attn_output_proj(layer.self_attn)
                if o_proj is not None:
                    row_projs.append(o_proj)
                dp = get_down_proj_module(layer.mlp)
                if dp is not None:
                    row_projs.append(dp)
                for proj in row_projs:
                    weight = proj.base_layer.weight if hasattr(proj, 'base_layer') else proj.weight
                    if weight.grad is not None:
                        grad_norm = (weight.grad ** 2).sum(dim=1)
                        sensitivity += grad_norm.cpu()

            num_processed += 1
            pbar.set_postfix({"samples": num_processed})

        except Exception as e:
            if verbose:
                print(f"[Embedding NTK] Error: {e}")
            continue
        finally:
            model.zero_grad(set_to_none=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if num_processed > 0:
        sensitivity /= num_processed

    # ==================== Restore original requires_grad states ====================
    for param in model.parameters():
        param_id = id(param)
        if param_id in original_requires_grad:
            param.requires_grad_(original_requires_grad[param_id])

    model.eval()  # Restore eval mode

    return sensitivity


def _compute_embedding_ntk_sensitivity_matrix(
    model: nn.Module,
    dataloader,
    num_samples: int,
    device: str,
    verbose: bool,
    ntk_method: str = "frobenius",  # "frobenius", "eigenvalue", "delta", "quadform"
    ntk_eigenvalue_k: int = 5
) -> Optional[torch.Tensor]:
    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers

    # ==================== Temporarily enable gradients for frozen models ====================
    original_requires_grad = {}

    for layer in model.model.layers:
        # QKV projections
        projs = [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]
        # Attention output
        o_proj = get_attn_output_proj(layer.self_attn)
        if o_proj is not None:
            projs.append(o_proj)
        # FFN projections
        projs.extend(get_up_proj_modules(layer.mlp))
        dp = get_down_proj_module(layer.mlp)
        if dp is not None:
            projs.append(dp)

        for proj in projs:
            weight = proj.base_layer.weight if hasattr(proj, 'base_layer') else proj.weight
            param_id = id(weight)
            if param_id not in original_requires_grad:
                original_requires_grad[param_id] = weight.requires_grad
                weight.requires_grad_(True)

    model.train()

    cached_samples = []
    data_iter = iter(dataloader)
    while len(cached_samples) < num_samples:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        bsz = batch['input_ids'].shape[0] if isinstance(batch, dict) else batch[0].shape[0]
        for i in range(bsz):
            if isinstance(batch, dict):
                single = {k: v[i:i+1] for k, v in batch.items() if isinstance(v, torch.Tensor)}
            else:
                single = tuple(v[i:i+1] for v in batch)
            cached_samples.append(single)
            if len(cached_samples) >= num_samples:
                break

    S = len(cached_samples)
    if S == 0:
        model.eval()
        return None

    # K_accum[d][i][j] = Σ_layer Σ_proj <grad_proj_d_i, grad_proj_d_j>
    K_accum = torch.zeros(hidden_size, S, S, device='cpu')

    if verbose:
        print(f"[Embedding NTK-Matrix] Layer-by-layer K accumulation: "
              f"{num_layers} layers × {S} samples = {num_layers * S} forward/backward passes")

    chunk_size = 256

    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]

        proj_info = [
            ('col', layer.self_attn.q_proj),
            ('col', layer.self_attn.k_proj),
            ('col', layer.self_attn.v_proj),
        ]
        # FFN up projections (col: hidden → intermediate)
        for up_module in get_up_proj_modules(layer.mlp):
            proj_info.append(('col', up_module))
        # Attention output (row: intermediate → hidden)
        o_proj = get_attn_output_proj(layer.self_attn)
        if o_proj is not None:
            proj_info.append(('row', o_proj))
        # FFN down projection (row: intermediate → hidden)
        dp = get_down_proj_module(layer.mlp)
        if dp is not None:
            proj_info.append(('row', dp))

        # sample_proj_grads[proj_idx] = [sample_0_grad, sample_1_grad, ...]
        sample_proj_grads = [[] for _ in range(len(proj_info))]

        for s, batch in enumerate(cached_samples):
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch.get('attention_mask',
                                          torch.ones_like(batch['input_ids'])).to(device)
            else:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device) if len(batch) > 1 else None

            try:
                model.zero_grad(set_to_none=True)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                loss = outputs.loss
                loss.backward()

                for p_idx, (split_mode, proj) in enumerate(proj_info):
                    weight = proj.base_layer.weight if hasattr(proj, 'base_layer') else proj.weight
                    if weight.grad is not None:
                        sample_proj_grads[p_idx].append(weight.grad.detach().cpu().clone())
                    else:
                        sample_proj_grads[p_idx].append(torch.zeros_like(weight.data).cpu())

            except Exception as e:
                if verbose:
                    print(f"[Embedding NTK-Matrix] Layer {layer_idx}, Sample {s} Error: {e}")
                for p_idx, (split_mode, proj) in enumerate(proj_info):
                    weight = proj.base_layer.weight if hasattr(proj, 'base_layer') else proj.weight
                    sample_proj_grads[p_idx].append(torch.zeros_like(weight.data).cpu())
            finally:
                model.zero_grad(set_to_none=True)
                del input_ids
                if attention_mask is not None:
                    del attention_mask

        # K_d[i,j] += Σ_proj <grad_proj_d_i, grad_proj_d_j>

        for p_idx, (split_mode, _) in enumerate(proj_info):
            grads = sample_proj_grads[p_idx]
            if len(grads) < S:
                continue

            # Stack: [S, out_dim, hidden] (col) or [S, hidden, in_dim] (row)
            G = torch.stack(grads)  # CPU tensor

            for d_start in range(0, hidden_size, chunk_size):
                d_end = min(d_start + chunk_size, hidden_size)

                if split_mode == 'col':
                    # G: [S, out_dim, hidden_size]
                    G_chunk = G[:, :, d_start:d_end].permute(2, 0, 1).contiguous()
                    # G_chunk: [chunk, S, out_dim]
                else:
                    # G: [S, hidden_size, in_dim]
                    G_chunk = G[:, d_start:d_end, :].permute(1, 0, 2).contiguous()
                    # G_chunk: [chunk, S, in_dim]

                # K_chunk: [chunk, S, S]
                K_chunk = torch.bmm(G_chunk, G_chunk.transpose(1, 2))
                K_accum[d_start:d_end] += K_chunk

                del G_chunk, K_chunk

            del G

        del sample_proj_grads
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if verbose:
            print(f"  [Embedding NTK-Matrix] Layer {layer_idx + 1}/{num_layers} done")

    if verbose:
        print(f"[Embedding NTK-Matrix] Computing sensitivity from K matrices...")

    sensitivities = torch.zeros(hidden_size)

    for d in range(hidden_size):
        K = K_accum[d]  # [S, S]

        if ntk_method == "eigenvalue":
            try:
                eigenvalues = torch.linalg.eigvalsh(K)
                k = min(ntk_eigenvalue_k, len(eigenvalues))
                sensitivities[d] = eigenvalues[-k:].sum()
            except:
                sensitivities[d] = torch.norm(K, p='fro')
        elif ntk_method == "delta":
            sensitivities[d] = torch.trace(K)
        elif ntk_method == "quadform":
            sensitivities[d] = K.sum()
        else:  # frobenius
            sensitivities[d] = torch.norm(K, p='fro')

    sensitivities = torch.nan_to_num(sensitivities, nan=1.0, posinf=1.0, neginf=1.0)

    if verbose:
        print(f"[Embedding NTK-Matrix] Raw sensitivity: min={sensitivities.min():.4f}, max={sensitivities.max():.4f}, "
              f"mean={sensitivities.mean():.4f}, std={sensitivities.std():.4f}")

    del K_accum
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # ==================== Restore original requires_grad states ====================
    for param in model.parameters():
        param_id = id(param)
        if param_id in original_requires_grad:
            param.requires_grad_(original_requires_grad[param_id])

    model.eval()

    return sensitivities


def _create_embedding_mask(
    scores: torch.Tensor,
    pruning_ratio: float,
    dimension_multiple: int,
    verbose: bool
) -> torch.Tensor:
    hidden_size = scores.size(0)

    if dimension_multiple > 1:
        num_groups = hidden_size // dimension_multiple

        sorted_scores, sorted_indices = torch.sort(scores)

        group_scores = []
        group_indices = []

        for g in range(num_groups):
            start = g * dimension_multiple
            end = start + dimension_multiple
            group_score = sorted_scores[start:end].mean().item()
            group_scores.append(group_score)
            group_indices.append(sorted_indices[start:end].tolist())

        group_scores = torch.tensor(group_scores)

        num_to_prune = int(num_groups * pruning_ratio)

        _, group_rank = torch.sort(group_scores)
        prune_group_indices = group_rank[:num_to_prune].tolist()

        mask = torch.ones(hidden_size)
        pruned_dims = set()
        for g in prune_group_indices:
            for dim_idx in group_indices[g]:
                mask[dim_idx] = 0.0
                pruned_dims.add(dim_idx)

        if verbose:
            kept_groups = num_groups - num_to_prune
            kept_dims = int(mask.sum().item())
            print(f"[Embedding Mask] Sort-then-group strategy")
            print(f"[Embedding Mask] Groups: {kept_groups}/{num_groups} kept ({kept_groups/num_groups:.1%})")
            print(f"[Embedding Mask] Dims: {kept_dims}/{hidden_size} kept ({kept_dims/hidden_size:.1%})")
            if pruned_dims:
                pruned_list = sorted(pruned_dims)
                contiguous_count = sum(1 for i in range(len(pruned_list)-1) if pruned_list[i+1] - pruned_list[i] == 1)
                print(f"[Embedding Mask] Pruned dims are {'scattered' if contiguous_count < len(pruned_list)//2 else 'mostly contiguous'}")

    else:
        num_to_prune = int(hidden_size * pruning_ratio)
        _, sorted_indices = torch.sort(scores)
        prune_dims = set(sorted_indices[:num_to_prune].tolist())

        mask = torch.ones(hidden_size)
        for d in prune_dims:
            mask[d] = 0.0

        if verbose:
            print(f"[Embedding Mask] Dims: {int(mask.sum())}/{hidden_size} kept ({mask.sum()/hidden_size:.1%})")

    return mask


# ==================== NTK Factor Computation (Robust Normalization) ====================

def _compute_ntk_factors(
    sensitivities: Dict[int, torch.Tensor],
    method: str = "robust",      # "legacy" or "robust"
    alpha: float = 0.5,
    direction: str = "normal",   # "normal" or "inverse"
    factor_range: tuple = (0.25, 4.0),
    verbose: bool = False,
    skip_layers: Optional[set] = None,
    layer_wise: bool = False
) -> Dict[int, torch.Tensor]:
    if not sensitivities:
        return {}
    if skip_layers is None:
        skip_layers = set()

    sens_sign = 1.0 if direction == "normal" else -1.0

    active_keys = sorted([l for l in sensitivities.keys() if l not in skip_layers])
    if not active_keys:
        return {l: torch.ones_like(sensitivities[l]) for l in sensitivities}

    if layer_wise:
        result = {}
        for l in active_keys:
            layer_sens = sensitivities[l]
            if method == "robust":
                s = torch.log1p(layer_sens)
                median_s = s.median()
                mad = (s - median_s).abs().median()
                z = (s - median_s) / (mad + 1e-8)
                z = z.clamp(-3.0, 3.0)
                layer_factors = torch.exp(alpha * sens_sign * z)
                layer_factors = layer_factors.clamp(factor_range[0], factor_range[1])
            else:  # legacy
                nonzero_mask = layer_sens > 0
                layer_mean = layer_sens[nonzero_mask].mean() if nonzero_mask.any() else layer_sens.mean()
                if layer_mean > 0:
                    norm_sens = layer_sens / layer_mean
                else:
                    norm_sens = torch.ones_like(layer_sens)
                layer_factors = 1.0 + alpha * sens_sign * (norm_sens - 1.0)
            result[l] = layer_factors

        if verbose:
            all_factors = torch.cat([result[l] for l in active_keys])
            print(f"    [NTK factors] method={method} (layer-wise), range=[{all_factors.min():.4f}, {all_factors.max():.4f}], "
                  f"mean={all_factors.mean():.4f}, median={all_factors.median():.4f}")
    else:
        all_sens = torch.cat([sensitivities[l] for l in active_keys])

        if method == "robust":
            s = torch.log1p(all_sens)
            median_s = s.median()
            mad = (s - median_s).abs().median()
            z = (s - median_s) / (mad + 1e-8)
            z = z.clamp(-3.0, 3.0)
            all_factors = torch.exp(alpha * sens_sign * z)
            all_factors = all_factors.clamp(factor_range[0], factor_range[1])
        else:  # legacy
            nonzero_mask = all_sens > 0
            global_mean = all_sens[nonzero_mask].mean() if nonzero_mask.any() else all_sens.mean()
            if global_mean > 0:
                norm_sens = all_sens / global_mean
            else:
                norm_sens = torch.ones_like(all_sens)
            all_factors = 1.0 + alpha * sens_sign * (norm_sens - 1.0)

        if verbose:
            print(f"    [NTK factors] method={method} (global), range=[{all_factors.min():.4f}, {all_factors.max():.4f}], "
                  f"mean={all_factors.mean():.4f}, median={all_factors.median():.4f}")

        result = {}
        offset = 0
        for l in active_keys:
            size = sensitivities[l].shape[0]
            result[l] = all_factors[offset:offset+size]
            offset += size

    for l in sensitivities:
        if l in skip_layers:
            result[l] = torch.ones_like(sensitivities[l])
    return result


def _compute_ntk_factors_flat(
    sensitivity: torch.Tensor,
    method: str = "robust",
    alpha: float = 0.5,
    direction: str = "normal",
    factor_range: tuple = (0.25, 4.0),
    verbose: bool = False
) -> torch.Tensor:
    if sensitivity.numel() == 0:
        return torch.ones_like(sensitivity)

    sens_sign = 1.0 if direction == "normal" else -1.0
    if method == "robust":
        s = torch.log1p(sensitivity)
        median_s = s.median()
        mad = (s - median_s).abs().median()
        z = (s - median_s) / (mad + 1e-8)
        z = z.clamp(-3.0, 3.0)
        factors = torch.exp(alpha * sens_sign * z)
        factors = factors.clamp(factor_range[0], factor_range[1])
    else:
        nonzero_mask = sensitivity > 0
        mean_s = sensitivity[nonzero_mask].mean() if nonzero_mask.any() else sensitivity.mean()
        if mean_s > 0:
            norm_s = sensitivity / mean_s
        else:
            norm_s = torch.ones_like(sensitivity)
        factors = 1.0 + alpha * sens_sign * (norm_s - 1.0)
    if verbose:
        print(f"    [NTK factors flat] method={method}, range=[{factors.min():.4f}, {factors.max():.4f}], "
              f"mean={factors.mean():.4f}, median={factors.median():.4f}")
    return factors


# ==================== NTK Score Adjustment Helper ====================

def _apply_ntk_adjustment_to_scores(
    scores: Dict[int, torch.Tensor],
    sensitivities: Dict[int, torch.Tensor],
    alpha: float,
    sens_sign: float,
    granularity: str,
    unit_mode: str,
    dimension_multiple: int,
    intermediate_size: int,
    num_layers: int,
    verbose: bool = False,
    group_assignments: Optional[Dict[int, List[torch.Tensor]]] = None,
    ntk_norm_method: str = "robust",
    skip_layers: Optional[set] = None,
    layer_wise: bool = False
) -> Dict[int, torch.Tensor]:
    if ntk_norm_method == "robust":
        direction = "normal" if sens_sign > 0 else "inverse"
        factors = _compute_ntk_factors(
            sensitivities=sensitivities,
            method="robust",
            alpha=alpha,
            direction=direction,
            verbose=verbose,
            skip_layers=skip_layers,
            layer_wise=layer_wise
        )
        adjusted_scores = {}
        use_importance_groups = (group_assignments is not None)
        for layer_idx in range(num_layers):
            if layer_idx not in scores:
                continue
            if layer_idx not in factors:
                adjusted_scores[layer_idx] = scores[layer_idx]
                continue

            layer_factors = factors[layer_idx]
            if granularity == "layer":
                avg_factor = layer_factors.mean().item()
                adjusted_scores[layer_idx] = scores[layer_idx] * avg_factor
                if verbose and layer_idx < 3:
                    print(f"    Layer {layer_idx}: robust factor={avg_factor:.4f}")
            else:
                if unit_mode == "neuron":
                    unit_size = 1
                elif unit_mode == "head":
                    unit_size = 1
                else:
                    unit_size = dimension_multiple

                neuron_adjustment = torch.ones(intermediate_size)

                if use_importance_groups and layer_idx in group_assignments:
                    for unit_idx in range(len(layer_factors)):
                        if unit_idx < len(group_assignments[layer_idx]):
                            group_indices = group_assignments[layer_idx][unit_idx]
                            neuron_adjustment[group_indices] = layer_factors[unit_idx].item()
                else:
                    for unit_idx in range(len(layer_factors)):
                        start_idx = unit_idx * unit_size
                        end_idx = min(start_idx + unit_size, intermediate_size)
                        neuron_adjustment[start_idx:end_idx] = layer_factors[unit_idx].item()

                adjusted_scores[layer_idx] = scores[layer_idx] * neuron_adjustment.to(scores[layer_idx].device)
                if verbose and layer_idx < 3:
                    print(f"    Layer {layer_idx}: robust factor range=[{layer_factors.min():.3f}, {layer_factors.max():.3f}]")
        return adjusted_scores

    adjusted_scores = {}
    use_importance_groups = (group_assignments is not None)

    for layer_idx in range(num_layers):
        if layer_idx not in scores:
            continue

        if layer_idx not in sensitivities:
            adjusted_scores[layer_idx] = scores[layer_idx]
            continue

        if granularity == "layer":
            sensitivity = sensitivities[layer_idx]
            if isinstance(sensitivity, torch.Tensor):
                sensitivity = sensitivity.item() if sensitivity.numel() == 1 else sensitivity.mean().item()
            adjustment_factor = 1.0 + alpha * sens_sign * (sensitivity - 1.0)
            adjusted_scores[layer_idx] = scores[layer_idx] * adjustment_factor
            if verbose and layer_idx < 3:
                print(f"    Layer {layer_idx}: sensitivity={sensitivity:.4f}, factor={adjustment_factor:.4f}")
        else:
            unit_sens = sensitivities[layer_idx]
            if unit_mode == "neuron":
                unit_size = 1
            elif unit_mode == "head":
                unit_size = 1
            else:  # "group"
                unit_size = dimension_multiple

            neuron_adjustment = torch.ones(intermediate_size)

            if use_importance_groups and layer_idx in group_assignments:
                for unit_idx, sens in enumerate(unit_sens):
                    if unit_idx < len(group_assignments[layer_idx]):
                        group_indices = group_assignments[layer_idx][unit_idx]
                        sens_val = sens.item() if isinstance(sens, torch.Tensor) else sens
                        adjustment_factor = 1.0 + alpha * sens_sign * (sens_val - 1.0)
                        neuron_adjustment[group_indices] = adjustment_factor
            else:
                for unit_idx, sens in enumerate(unit_sens):
                    start_idx = unit_idx * unit_size
                    end_idx = min(start_idx + unit_size, intermediate_size)
                    sens_val = sens.item() if isinstance(sens, torch.Tensor) else sens
                    adjustment_factor = 1.0 + alpha * sens_sign * (sens_val - 1.0)
                    neuron_adjustment[start_idx:end_idx] = adjustment_factor

            adjusted_scores[layer_idx] = scores[layer_idx] * neuron_adjustment.to(scores[layer_idx].device)
            if verbose and layer_idx < 3:
                print(f"    Layer {layer_idx}: unit_sens range=[{unit_sens.min():.3f}, {unit_sens.max():.3f}]")

    return adjusted_scores


# ==================== NTK Revival Counting Helper ====================

def _compute_ntk_revival_stats(
    pre_ntk_ffn_scores: Optional[Dict[int, torch.Tensor]],
    pre_ntk_head_scores: Optional[Dict[int, torch.Tensor]],
    pre_ntk_embed_scores: Optional[torch.Tensor],
    actual_ffn_masks: Dict[int, torch.Tensor],
    actual_head_masks: Dict[int, torch.Tensor],
    actual_embedding_mask: torch.Tensor,
    ffn_pruning_ratio: float,
    head_pruning_ratio: float,
    embedding_pruning_ratio: float,
    num_layers: int,
    intermediate_size: int,
    num_heads: int,
    hidden_size: int,
    ffn_dimension_multiple: int,
    embedding_dimension_multiple: int,
    global_ffn_pruning: bool,
    global_head_pruning: bool,
    skip_first_layer_head: int,
    skip_last_layer_head: int,
    skip_first_layer_ffn: int,
    skip_last_layer_ffn: int,
    allow_full_block_pruning: bool,
    compute_ffn: bool,
    compute_head: bool,
    compute_embedding: bool,
    verbose: bool = False
) -> Dict:
    stats = {
        'ffn_revived': 0, 'head_revived': 0, 'embedding_revived': 0,
        'ffn_revived_per_layer': {}, 'head_revived_per_layer': {},
        'ffn_newly_pruned': 0, 'head_newly_pruned': 0, 'embedding_newly_pruned': 0,
    }

    # FFN revival
    if compute_ffn and pre_ntk_ffn_scores is not None and ffn_pruning_ratio > 0:
        ffn_skip_set = set()
        if skip_first_layer_ffn > 0:
            ffn_skip_set.update(range(min(skip_first_layer_ffn, num_layers)))
        if skip_last_layer_ffn > 0:
            ffn_skip_set.update(range(max(0, num_layers - skip_last_layer_ffn), num_layers))

        if global_ffn_pruning:
            no_ntk_ffn = _create_ffn_mask_sorted_grouping_global(
                pre_ntk_ffn_scores, ffn_pruning_ratio, ffn_dimension_multiple,
                skip_layers=ffn_skip_set, allow_full_block_pruning=allow_full_block_pruning)
        else:
            no_ntk_ffn = {}
            for l in range(num_layers):
                if l in ffn_skip_set:
                    no_ntk_ffn[l] = torch.ones(intermediate_size)
                else:
                    no_ntk_ffn[l] = _create_ffn_mask_sorted_grouping(
                        pre_ntk_ffn_scores[l], ffn_pruning_ratio, ffn_dimension_multiple,
                        allow_full_block_pruning=allow_full_block_pruning)

        for l in actual_ffn_masks:
            if l in no_ntk_ffn:
                revived = int(((no_ntk_ffn[l] == 0) & (actual_ffn_masks[l] == 1)).sum().item())
                newly_pruned = int(((no_ntk_ffn[l] == 1) & (actual_ffn_masks[l] == 0)).sum().item())
                stats['ffn_revived'] += revived
                stats['ffn_newly_pruned'] += newly_pruned
                if revived > 0:
                    stats['ffn_revived_per_layer'][l] = revived

    # Head revival
    if compute_head and pre_ntk_head_scores is not None and head_pruning_ratio > 0:
        if global_head_pruning:
            no_ntk_head = _create_head_mask_global(
                pre_ntk_head_scores, head_pruning_ratio,
                skip_first_layer=skip_first_layer_head, skip_last_layer=skip_last_layer_head,
                allow_full_block_pruning=allow_full_block_pruning)
        else:
            head_skip_set = set()
            if skip_first_layer_head > 0:
                head_skip_set.update(range(min(skip_first_layer_head, num_layers)))
            if skip_last_layer_head > 0:
                head_skip_set.update(range(max(0, num_layers - skip_last_layer_head), num_layers))
            no_ntk_head = {}
            for l in range(num_layers):
                if l in head_skip_set:
                    no_ntk_head[l] = torch.ones(num_heads)
                else:
                    no_ntk_head[l] = _create_head_mask(
                        pre_ntk_head_scores[l], head_pruning_ratio,
                        allow_full_block_pruning=allow_full_block_pruning)

        for l in actual_head_masks:
            if l in no_ntk_head:
                revived = int(((no_ntk_head[l] == 0) & (actual_head_masks[l] == 1)).sum().item())
                newly_pruned = int(((no_ntk_head[l] == 1) & (actual_head_masks[l] == 0)).sum().item())
                stats['head_revived'] += revived
                stats['head_newly_pruned'] += newly_pruned
                if revived > 0:
                    stats['head_revived_per_layer'][l] = revived

    # Embedding revival
    if compute_embedding and pre_ntk_embed_scores is not None and embedding_pruning_ratio > 0:
        no_ntk_embed = _create_embedding_mask(
            pre_ntk_embed_scores, embedding_pruning_ratio, embedding_dimension_multiple, verbose=False)
        revived = int(((no_ntk_embed == 0) & (actual_embedding_mask == 1)).sum().item())
        newly_pruned = int(((no_ntk_embed == 1) & (actual_embedding_mask == 0)).sum().item())
        stats['embedding_revived'] = revived
        stats['embedding_newly_pruned'] = newly_pruned

    if verbose:
        total_revived = stats['ffn_revived'] + stats['head_revived'] + stats['embedding_revived']
        total_newly_pruned = stats['ffn_newly_pruned'] + stats['head_newly_pruned'] + stats['embedding_newly_pruned']
        print(f"\n[NTK Revival Stats]")
        print(f"  FFN:       {stats['ffn_revived']} revived, {stats['ffn_newly_pruned']} newly pruned")
        print(f"  Head:      {stats['head_revived']} revived, {stats['head_newly_pruned']} newly pruned")
        print(f"  Embedding: {stats['embedding_revived']} revived, {stats['embedding_newly_pruned']} newly pruned")
        print(f"  Total:     {total_revived} revived, {total_newly_pruned} newly pruned")
        if stats['ffn_revived_per_layer']:
            layers_str = ", ".join(f"L{l}:{c}" for l, c in sorted(stats['ffn_revived_per_layer'].items()))
            print(f"  FFN per-layer: {layers_str}")
        if stats['head_revived_per_layer']:
            layers_str = ", ".join(f"L{l}:{c}" for l, c in sorted(stats['head_revived_per_layer'].items()))
            print(f"  Head per-layer: {layers_str}")

    return stats


# ==================== Joint All Masks (FFN + Head + Dim + Embedding) ====================

def compute_joint_all_masks(
    model: nn.Module,
    dataloader,
    ffn_pruning_ratio: float = 0.0,
    head_pruning_ratio: float = 0.0,
    dimension_pruning_ratio: float = 0.0,
    embedding_pruning_ratio: float = 0.0,
    num_samples: int = 128,
    ffn_dimension_multiple: int = 128,
    dimension_group_size: int = 16,
    embedding_dimension_multiple: int = 128,
    device: str = "cuda",
    verbose: bool = True,
    ffn_mode: str = "down",
    head_mode: str = "o",
    importance_method: str = "ganda",
    global_ffn_pruning: bool = False,
    global_head_pruning: bool = False,
    global_dimension_pruning: bool = False,
    enable_iterative_pruning: bool = False,
    pruning_step_size: float = 0.05,
    iterative_start_ratio_ffn: float = 0.0,
    iterative_start_ratio_head: float = 0.0,
    iterative_start_ratio_emb: float = 0.0,
    use_ntk_adjustment: bool = False,
    ntk_adjustment_alpha: float = 0.3,
    ntk_adjustment_mode: str = "previous",
    ntk_sensitivity_direction: str = "normal",
    ntk_target_mode: str = "default",
    ntk_granularity: str = "layer",
    ntk_unit_mode: str = "group",
    skip_first_layer_head: int = 1,
    skip_last_layer_head: int = 1,
    skip_first_layer_ffn: int = 0,
    skip_last_layer_ffn: int = 0,
    use_ntk_embedding: bool = False,
    ntk_embedding_method: str = "gradient",  # "gradient" (grad norm²) or "ntk_matrix" (actual NTK K=J×Jᵀ)
    ntk_method: str = "frobenius",  # "frobenius", "eigenvalue", "delta", "quadform"
    ntk_eigenvalue_k: int = 5,  # top-k eigenvalues for eigenvalue method
    use_mode_connectivity: bool = False,
    barrier_threshold: float = 0.5,
    barrier_n_points: int = 5,
    barrier_num_samples: int = 5,
    adaptive_step_size: bool = False,
    min_step_size: float = 0.0125,
    barrier_action: str = "subdivide",
    use_connectivity_importance: bool = False,
    connectivity_alpha: float = 0.3,
    connectivity_n_points: int = 5,
    connectivity_direction: str = "inverse",  # "normal" or "inverse"
    use_correct_normalization: bool = False,  # True: sum-based, False: mean-based (legacy)
    ntk_norm_method: str = "robust",
    # Iterative Scale Calibration (Iter-FLAP)
    enable_iterative_calibration: bool = False,
    allow_full_block_pruning: bool = False
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, torch.Tensor], torch.Tensor, Optional[Dict]]:
    if verbose:
        print(f"\n[Joint All Masks] Computing all masks in joint mode...")
        print(f"[Joint All Masks] FFN: {ffn_pruning_ratio:.1%}, Head: {head_pruning_ratio:.1%}, "
              f"Dim: {dimension_pruning_ratio:.1%}, Embedding: {embedding_pruning_ratio:.1%}")
        print(f"[Joint All Masks] Iterative: {enable_iterative_pruning}, Step size: {pruning_step_size:.1%}")
        if use_ntk_adjustment:
            print(f"[Joint All Masks] NTK Adjustment: alpha={ntk_adjustment_alpha}, mode={ntk_adjustment_mode}, "
                  f"direction={ntk_sensitivity_direction}, target={ntk_target_mode}, "
                  f"granularity={ntk_granularity}, unit_mode={ntk_unit_mode}, embedding={use_ntk_embedding}")
        if use_mode_connectivity:
            print(f"[Joint All Masks] Mode Connectivity: threshold={barrier_threshold}, n_points={barrier_n_points}, "
                  f"action={barrier_action}, adaptive_step={adaptive_step_size}")
        if enable_iterative_calibration:
            print(f"[Joint All Masks] Iterative Scale Calibration (Iter-FLAP): enabled")

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    head_dim = hidden_size // num_heads
    intermediate_size = model.config.intermediate_size

    compute_ffn = ffn_pruning_ratio > 0
    compute_head = head_pruning_ratio > 0
    compute_dim = dimension_pruning_ratio > 0
    compute_embedding = embedding_pruning_ratio > 0

    if not enable_iterative_pruning:

        if importance_method == "ntk":
            if verbose:
                print(f"[NTK Importance] Using NTK sensitivity as importance score (standalone mode)")
                print(f"[NTK Importance] target_mode={ntk_target_mode}, granularity={ntk_granularity}, unit_mode={ntk_unit_mode}")

            # Initialize scores
            ffn_scores = {i: torch.zeros(intermediate_size) for i in range(num_layers)}
            head_scores = {i: torch.zeros(num_heads) for i in range(num_layers)}
            dim_scores = {i: torch.zeros(head_dim) for i in range(num_layers)}
            embed_scores = torch.zeros(hidden_size)

            ffn_skip_layers = set()
            if skip_first_layer_ffn > 0:
                ffn_skip_layers.update(range(min(skip_first_layer_ffn, num_layers)))
            if skip_last_layer_ffn > 0:
                ffn_skip_layers.update(range(max(0, num_layers - skip_last_layer_ffn), num_layers))

            head_skip_layers = set()
            if skip_first_layer_head > 0:
                head_skip_layers.update(range(min(skip_first_layer_head, num_layers)))
            if skip_last_layer_head > 0:
                head_skip_layers.update(range(max(0, num_layers - skip_last_layer_head), num_layers))

            # FFN NTK sensitivity as importance score
            if compute_ffn:
                if verbose:
                    print(f"[NTK Importance] Computing FFN importance via NTK sensitivity...")

                # Create dummy masks (all ones - no pruning yet)
                dummy_ffn_masks = {i: torch.ones(intermediate_size) for i in range(num_layers)}

                # Compute NTK sensitivity
                ffn_ntk_sens = _compute_ntk_sensitivity_per_unit(
                    model=model,
                    dataloader=dataloader,
                    current_masks=dummy_ffn_masks,
                    prev_masks=None,
                    mask_type="ffn",
                    num_samples=num_samples,
                    device=device,
                    verbose=verbose,
                    dimension_multiple=ffn_dimension_multiple,
                    ntk_target_mode=ntk_target_mode,
                    ntk_unit_mode=ntk_unit_mode,
                    group_assignments=None,
                    ntk_method=ntk_method,
                    ntk_eigenvalue_k=ntk_eigenvalue_k,
                    normalize_per_layer=not global_ffn_pruning,
                    skip_layers=ffn_skip_layers
                ) if ntk_granularity == "unit" else _compute_ntk_sensitivity(
                    model=model,
                    dataloader=dataloader,
                    current_masks=dummy_ffn_masks,
                    prev_masks=None,
                    mask_type="ffn",
                    num_samples=num_samples,
                    device=device,
                    verbose=verbose,
                    ntk_target_mode=ntk_target_mode,
                    ntk_method=ntk_method,
                    ntk_eigenvalue_k=ntk_eigenvalue_k,
                    skip_layers=ffn_skip_layers
                )

                if ntk_granularity == "unit" and global_ffn_pruning and ffn_ntk_sens is not None:
                    if ntk_norm_method == "legacy":
                        all_sens = torch.cat([ffn_ntk_sens[l] for l in range(num_layers) if l in ffn_ntk_sens])
                        nonzero_mask = all_sens > 0
                        global_mean = all_sens[nonzero_mask].mean() if nonzero_mask.any() else all_sens.mean()
                        if global_mean > 0:
                            for layer_idx in ffn_ntk_sens:
                                ffn_ntk_sens[layer_idx] = ffn_ntk_sens[layer_idx] / global_mean
                        if verbose:
                            print(f"[NTK Importance] FFN Global normalization (legacy): mean={global_mean:.4f}")

                # Convert NTK sensitivity to importance scores
                # Higher sensitivity = more important = should NOT be pruned
                # So we use sensitivity directly as importance score
                if ffn_ntk_sens is not None:
                    for layer_idx in range(num_layers):
                        if layer_idx in ffn_ntk_sens:
                            layer_sens = ffn_ntk_sens[layer_idx]
                            if isinstance(layer_sens, torch.Tensor):
                                # Per-unit sensitivity (unit granularity)
                                if layer_sens.numel() == intermediate_size:
                                    ffn_scores[layer_idx] = layer_sens.cpu()
                                else:
                                    # Group-level sensitivity - expand to neuron level
                                    num_groups = layer_sens.numel()
                                    expanded = layer_sens.repeat_interleave(ffn_dimension_multiple)
                                    if expanded.numel() >= intermediate_size:
                                        ffn_scores[layer_idx] = expanded[:intermediate_size].cpu()
                                    else:
                                        ffn_scores[layer_idx][:expanded.numel()] = expanded.cpu()
                            else:
                                # Single value (layer granularity)
                                ffn_scores[layer_idx] = torch.full((intermediate_size,), float(layer_sens))

                # Memory cleanup
                model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                if verbose:
                    print(f"[NTK Importance] FFN scores computed for {num_layers} layers")

            # Head NTK sensitivity as importance score
            if compute_head:
                if verbose:
                    print(f"[NTK Importance] Computing Head importance via NTK sensitivity...")

                # Create dummy masks (all ones - no pruning yet)
                dummy_head_masks = {i: torch.ones(num_heads) for i in range(num_layers)}

                # Compute NTK sensitivity (head always uses per-unit)
                head_ntk_sens = _compute_ntk_sensitivity_per_unit(
                    model=model,
                    dataloader=dataloader,
                    current_masks=dummy_head_masks,
                    prev_masks=None,
                    mask_type="head",
                    num_samples=num_samples,
                    device=device,
                    verbose=verbose,
                    dimension_multiple=1,
                    num_heads=num_heads,
                    ntk_target_mode=ntk_target_mode,
                    ntk_unit_mode="head",
                    ntk_method=ntk_method,
                    ntk_eigenvalue_k=ntk_eigenvalue_k,
                    normalize_per_layer=not global_head_pruning,
                    skip_layers=head_skip_layers
                )

                if global_head_pruning and head_ntk_sens is not None:
                    if ntk_norm_method == "legacy":
                        all_sens = torch.cat([head_ntk_sens[l] for l in range(num_layers) if l in head_ntk_sens])
                        nonzero_mask = all_sens > 0
                        global_mean = all_sens[nonzero_mask].mean() if nonzero_mask.any() else all_sens.mean()
                        if global_mean > 0:
                            for layer_idx in head_ntk_sens:
                                head_ntk_sens[layer_idx] = head_ntk_sens[layer_idx] / global_mean
                        if verbose:
                            print(f"[NTK Importance] Head Global normalization (legacy): mean={global_mean:.4f}")

                # Convert NTK sensitivity to importance scores
                if head_ntk_sens is not None:
                    for layer_idx in range(num_layers):
                        if layer_idx in head_ntk_sens:
                            layer_sens = head_ntk_sens[layer_idx]
                            if isinstance(layer_sens, torch.Tensor):
                                head_scores[layer_idx] = layer_sens.cpu()
                            else:
                                head_scores[layer_idx] = torch.full((num_heads,), float(layer_sens))

                # Memory cleanup
                model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                if verbose:
                    print(f"[NTK Importance] Head scores computed for {num_layers} layers")

            # Embedding NTK sensitivity as importance score
            if compute_embedding:
                if verbose:
                    print(f"[NTK Importance] Computing Embedding importance via NTK sensitivity...")

                embed_ntk_sens = _compute_embedding_ntk_sensitivity(
                    model=model,
                    dataloader=dataloader,
                    num_samples=num_samples,
                    device=device,
                    verbose=verbose
                )

                if embed_ntk_sens is not None:
                    embed_scores = embed_ntk_sens.cpu()

                # Memory cleanup
                model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                if verbose:
                    print(f"[NTK Importance] Embedding scores computed")

        else:

            # Initialize scores
            ffn_scores = {i: torch.zeros(intermediate_size) for i in range(num_layers)}
            head_scores = {i: torch.zeros(num_heads) for i in range(num_layers)}
            dim_scores = {i: torch.zeros(head_dim) for i in range(num_layers)}
            embed_scores = torch.zeros(hidden_size)

            ffn_skip_layers_set = set()
            if skip_first_layer_ffn > 0:
                ffn_skip_layers_set.update(range(min(skip_first_layer_ffn, num_layers)))
            if skip_last_layer_ffn > 0:
                ffn_skip_layers_set.update(range(max(0, num_layers - skip_last_layer_ffn), num_layers))

            head_skip_layers_set = set()
            if skip_first_layer_head > 0:
                head_skip_layers_set.update(range(min(skip_first_layer_head, num_layers)))
            if skip_last_layer_head > 0:
                head_skip_layers_set.update(range(max(0, num_layers - skip_last_layer_head), num_layers))

            use_calculator_for_ffn = compute_ffn and ffn_mode != "ntk"
            use_calculator_for_head = compute_head and head_mode != "ntk"

            # FFN NTK mode
            if compute_ffn and ffn_mode == "ntk":
                if verbose:
                    print(f"[FFN NTK Mode] Computing FFN importance via NTK sensitivity...")

                dummy_ffn_masks = {i: torch.ones(intermediate_size) for i in range(num_layers)}

                ffn_ntk_sens = _compute_ntk_sensitivity_per_unit(
                    model=model,
                    dataloader=dataloader,
                    current_masks=dummy_ffn_masks,
                    prev_masks=None,
                    mask_type="ffn",
                    num_samples=num_samples,
                    device=device,
                    verbose=verbose,
                    dimension_multiple=ffn_dimension_multiple,
                    ntk_target_mode=ntk_target_mode,
                    ntk_unit_mode=ntk_unit_mode,
                    group_assignments=None,
                    ntk_method=ntk_method,
                    ntk_eigenvalue_k=ntk_eigenvalue_k,
                    normalize_per_layer=not global_ffn_pruning,
                    skip_layers=ffn_skip_layers_set
                ) if ntk_granularity == "unit" else _compute_ntk_sensitivity(
                    model=model,
                    dataloader=dataloader,
                    current_masks=dummy_ffn_masks,
                    prev_masks=None,
                    mask_type="ffn",
                    num_samples=num_samples,
                    device=device,
                    verbose=verbose,
                    ntk_target_mode=ntk_target_mode,
                    ntk_method=ntk_method,
                    ntk_eigenvalue_k=ntk_eigenvalue_k,
                    skip_layers=ffn_skip_layers_set
                )

                if ntk_granularity == "unit" and global_ffn_pruning and ffn_ntk_sens is not None:
                    if ntk_norm_method == "legacy":
                        all_sens = torch.cat([ffn_ntk_sens[l] for l in range(num_layers) if l in ffn_ntk_sens])
                        nonzero_mask = all_sens > 0
                        global_mean = all_sens[nonzero_mask].mean() if nonzero_mask.any() else all_sens.mean()
                        if global_mean > 0:
                            for layer_idx in ffn_ntk_sens:
                                ffn_ntk_sens[layer_idx] = ffn_ntk_sens[layer_idx] / global_mean
                        if verbose:
                            print(f"[FFN NTK Mode] FFN Global normalization (legacy): mean={global_mean:.4f}")

                if ffn_ntk_sens is not None:
                    for layer_idx in range(num_layers):
                        if layer_idx in ffn_ntk_sens:
                            layer_sens = ffn_ntk_sens[layer_idx]
                            if isinstance(layer_sens, torch.Tensor):
                                if layer_sens.numel() == intermediate_size:
                                    ffn_scores[layer_idx] = layer_sens.cpu()
                                else:
                                    num_groups = layer_sens.numel()
                                    expanded = layer_sens.repeat_interleave(ffn_dimension_multiple)
                                    if expanded.numel() >= intermediate_size:
                                        ffn_scores[layer_idx] = expanded[:intermediate_size].cpu()
                                    else:
                                        ffn_scores[layer_idx][:expanded.numel()] = expanded.cpu()
                            else:
                                ffn_scores[layer_idx] = torch.full((intermediate_size,), float(layer_sens))

                model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                if verbose:
                    print(f"[FFN NTK Mode] FFN scores computed for {num_layers} layers")

            # Head NTK mode
            if compute_head and head_mode == "ntk":
                if verbose:
                    print(f"[Head NTK Mode] Computing Head importance via NTK sensitivity...")

                dummy_head_masks = {i: torch.ones(num_heads) for i in range(num_layers)}

                head_ntk_sens = _compute_ntk_sensitivity_per_unit(
                    model=model,
                    dataloader=dataloader,
                    current_masks=dummy_head_masks,
                    prev_masks=None,
                    mask_type="head",
                    num_samples=num_samples,
                    device=device,
                    verbose=verbose,
                    dimension_multiple=1,
                    num_heads=num_heads,
                    ntk_target_mode=ntk_target_mode,
                    ntk_unit_mode="head",
                    ntk_method=ntk_method,
                    ntk_eigenvalue_k=ntk_eigenvalue_k,
                    normalize_per_layer=not global_head_pruning,
                    skip_layers=head_skip_layers_set
                )

                if global_head_pruning and head_ntk_sens is not None:
                    if ntk_norm_method == "legacy":
                        all_sens = torch.cat([head_ntk_sens[l] for l in range(num_layers) if l in head_ntk_sens])
                        nonzero_mask = all_sens > 0
                        global_mean = all_sens[nonzero_mask].mean() if nonzero_mask.any() else all_sens.mean()
                        if global_mean > 0:
                            for layer_idx in head_ntk_sens:
                                head_ntk_sens[layer_idx] = head_ntk_sens[layer_idx] / global_mean
                        if verbose:
                            print(f"[Head NTK Mode] Head Global normalization (legacy): mean={global_mean:.4f}")

                if head_ntk_sens is not None:
                    for layer_idx in range(num_layers):
                        if layer_idx in head_ntk_sens:
                            layer_sens = head_ntk_sens[layer_idx]
                            if isinstance(layer_sens, torch.Tensor):
                                head_scores[layer_idx] = layer_sens.cpu()
                            else:
                                head_scores[layer_idx] = torch.full((num_heads,), float(layer_sens))

                model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                if verbose:
                    print(f"[Head NTK Mode] Head scores computed for {num_layers} layers")

            if use_calculator_for_ffn or use_calculator_for_head or compute_dim or compute_embedding:
                calc_ffn_mode = "down" if ffn_mode == "ntk" else ffn_mode
                calc_head_mode = "o" if head_mode == "ntk" else head_mode

                calculator = TaylorImportanceCalculator(
                    model, device, verbose=verbose,
                    ffn_mode=calc_ffn_mode,
                    importance_method=importance_method,
                    use_correct_normalization=use_correct_normalization
                )

                calc_ffn_scores, calc_head_scores, calc_dim_scores, calc_embed_scores = calculator.compute_all_importance(
                    dataloader, num_samples, calc_head_mode,
                    compute_ffn=use_calculator_for_ffn,
                    compute_head=use_calculator_for_head,
                    compute_dim=compute_dim,
                    compute_embedding=compute_embedding
                )

                if use_calculator_for_ffn:
                    ffn_scores = calc_ffn_scores
                if use_calculator_for_head:
                    head_scores = calc_head_scores
                if compute_dim:
                    dim_scores = calc_dim_scores
                if compute_embedding:
                    embed_scores = calc_embed_scores

        pre_ntk_ffn_scores = None
        pre_ntk_head_scores = None
        pre_ntk_embed_scores = None
        if use_ntk_adjustment and importance_method != "ntk":
            if compute_ffn:
                pre_ntk_ffn_scores = {k: v.clone() for k, v in ffn_scores.items()}
            if compute_head:
                pre_ntk_head_scores = {k: v.clone() for k, v in head_scores.items()}
            if compute_embedding and use_ntk_embedding:
                pre_ntk_embed_scores = embed_scores.clone()

        # NTK adjustment for "current" mode in one-shot pruning
        # Skip if importance_method == "ntk" (already NTK-based scores)
        if use_ntk_adjustment and ntk_adjustment_mode == "current" and importance_method != "ntk":
            if verbose:
                print(f"[One-shot NTK] Applying NTK adjustment (current mode, alpha={ntk_adjustment_alpha})")

            # NTK direction sign
            sens_sign = 1.0 if ntk_sensitivity_direction == "normal" else -1.0

            ffn_skip_layers = set()
            if skip_first_layer_ffn > 0:
                ffn_skip_layers.update(range(min(skip_first_layer_ffn, num_layers)))
            if skip_last_layer_ffn > 0:
                ffn_skip_layers.update(range(max(0, num_layers - skip_last_layer_ffn), num_layers))

            head_skip_layers = set()
            if skip_first_layer_head > 0:
                head_skip_layers.update(range(min(skip_first_layer_head, num_layers)))
            if skip_last_layer_head > 0:
                head_skip_layers.update(range(max(0, num_layers - skip_last_layer_head), num_layers))

            # FFN NTK adjustment
            if compute_ffn and ffn_pruning_ratio > 0:
                if verbose:
                    print(f"  [NTK-current] FFN: 1st pruning → NTK → score adjustment")

                ffn_group_assignments = None
                if global_ffn_pruning:
                    temp_ffn_masks, ffn_group_assignments = _create_ffn_mask_sorted_grouping_global_with_groups(
                        ffn_scores, ffn_pruning_ratio, ffn_dimension_multiple, skip_layers=ffn_skip_layers
                    )
                else:
                    temp_ffn_masks = {
                        l: _create_ffn_mask_sorted_grouping(ffn_scores[l], ffn_pruning_ratio, ffn_dimension_multiple, allow_full_block_pruning=allow_full_block_pruning)
                        if l not in ffn_skip_layers else torch.ones(intermediate_size)
                        for l in range(num_layers)
                    }

                dummy_ffn_masks = {i: torch.ones(intermediate_size) for i in range(num_layers)}
                ffn_ga_for_ntk = None if ntk_unit_mode == "neuron" else ffn_group_assignments
                current_ffn_sens = _compute_ntk_sensitivity_per_unit(
                    model=model,
                    dataloader=dataloader,
                    current_masks=dummy_ffn_masks,
                    prev_masks=None,
                    mask_type="ffn",
                    num_samples=num_samples,
                    device=device,
                    verbose=verbose,
                    dimension_multiple=ffn_dimension_multiple,
                    ntk_target_mode=ntk_target_mode,
                    ntk_unit_mode=ntk_unit_mode,
                    group_assignments=ffn_ga_for_ntk,
                    ntk_method=ntk_method,
                    ntk_eigenvalue_k=ntk_eigenvalue_k,
                    normalize_per_layer=not global_ffn_pruning,
                    skip_layers=ffn_skip_layers
                ) if ntk_granularity == "unit" else _compute_ntk_sensitivity(
                    model=model,
                    dataloader=dataloader,
                    current_masks=dummy_ffn_masks,
                    prev_masks=None,
                    mask_type="ffn",
                    num_samples=num_samples,
                    device=device,
                    verbose=verbose,
                    ntk_target_mode=ntk_target_mode,
                    ntk_method=ntk_method,
                    ntk_eigenvalue_k=ntk_eigenvalue_k,
                    skip_layers=ffn_skip_layers
                )

                if ntk_granularity == "unit" and global_ffn_pruning:
                    if ntk_norm_method == "legacy":
                        all_sens = torch.cat([current_ffn_sens[l] for l in range(num_layers) if l in current_ffn_sens])
                        nonzero_mask = all_sens > 0
                        global_mean = all_sens[nonzero_mask].mean() if nonzero_mask.any() else all_sens.mean()
                        if global_mean > 0:
                            for layer_idx in current_ffn_sens:
                                current_ffn_sens[layer_idx] = current_ffn_sens[layer_idx] / global_mean
                        if verbose:
                            print(f"  [NTK-current] FFN Global normalization (legacy): mean={global_mean:.4f}")

                ffn_scores = _apply_ntk_adjustment_to_scores(
                    scores=ffn_scores,
                    sensitivities=current_ffn_sens,
                    alpha=ntk_adjustment_alpha,
                    sens_sign=sens_sign,
                    granularity=ntk_granularity,
                    unit_mode=ntk_unit_mode,
                    dimension_multiple=ffn_dimension_multiple,
                    intermediate_size=intermediate_size,
                    num_layers=num_layers,
                    verbose=verbose,
                    group_assignments=ffn_ga_for_ntk,
                    ntk_norm_method=ntk_norm_method,
                    skip_layers=ffn_skip_layers,
                    layer_wise=not global_ffn_pruning
                )

                model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            # Head NTK adjustment
            if compute_head and head_pruning_ratio > 0:
                if verbose:
                    print(f"  [NTK-current] Head: 1st pruning → NTK → score adjustment")

                temp_head_masks = _create_head_mask_global(
                    head_scores, head_pruning_ratio,
                    skip_first_layer=skip_first_layer_head, skip_last_layer=skip_last_layer_head,
                    allow_full_block_pruning=allow_full_block_pruning
                ) if global_head_pruning else {
                    l: _create_head_mask(head_scores[l], head_pruning_ratio, allow_full_block_pruning=allow_full_block_pruning)
                    if l not in head_skip_layers else torch.ones(num_heads)
                    for l in range(num_layers)
                }

                dummy_head_masks = {i: torch.ones(num_heads) for i in range(num_layers)}
                current_head_sens = _compute_ntk_sensitivity_per_unit(
                    model=model,
                    dataloader=dataloader,
                    current_masks=dummy_head_masks,
                    prev_masks=None,
                    mask_type="head",
                    num_samples=num_samples,
                    device=device,
                    verbose=verbose,
                    dimension_multiple=1,
                    num_heads=num_heads,
                    ntk_target_mode=ntk_target_mode,
                    ntk_unit_mode="head",
                    ntk_method=ntk_method,
                    ntk_eigenvalue_k=ntk_eigenvalue_k,
                    normalize_per_layer=not global_head_pruning,
                    skip_layers=head_skip_layers
                )

                if global_head_pruning:
                    if ntk_norm_method == "legacy":
                        all_sens = torch.cat([current_head_sens[l] for l in range(num_layers) if l in current_head_sens])
                        nonzero_mask = all_sens > 0
                        global_mean = all_sens[nonzero_mask].mean() if nonzero_mask.any() else all_sens.mean()
                        if global_mean > 0:
                            for layer_idx in current_head_sens:
                                current_head_sens[layer_idx] = current_head_sens[layer_idx] / global_mean
                        if verbose:
                            print(f"  [NTK-current] Head Global normalization (legacy): mean={global_mean:.4f}")

                head_scores = _apply_ntk_adjustment_to_scores(
                    scores=head_scores,
                    sensitivities=current_head_sens,
                    alpha=ntk_adjustment_alpha,
                    sens_sign=sens_sign,
                    granularity="unit",
                    unit_mode="head",
                    dimension_multiple=1,
                    intermediate_size=num_heads,
                    num_layers=num_layers,
                    verbose=verbose,
                    ntk_norm_method=ntk_norm_method,
                    skip_layers=head_skip_layers,
                    layer_wise=not global_head_pruning
                )

                model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            # Embedding NTK adjustment
            if use_ntk_embedding and compute_embedding and embedding_pruning_ratio > 0:
                if verbose:
                    print(f"  [NTK-current] Embedding: Computing NTK sensitivity (method={ntk_embedding_method})...")

                if ntk_embedding_method == "ntk_matrix":
                    embed_sensitivity = _compute_embedding_ntk_sensitivity_matrix(
                        model=model,
                        dataloader=dataloader,
                        num_samples=num_samples,
                        device=device,
                        verbose=verbose,
                        ntk_method=ntk_method,
                        ntk_eigenvalue_k=ntk_eigenvalue_k
                    )
                else:
                    embed_sensitivity = _compute_embedding_ntk_sensitivity(
                        model=model,
                        dataloader=dataloader,
                        num_samples=num_samples,
                        device=device,
                        verbose=verbose
                    )

                if embed_sensitivity is not None:
                    direction = "normal" if sens_sign > 0 else "inverse"
                    adjustment_factor = _compute_ntk_factors_flat(
                        sensitivity=embed_sensitivity,
                        method=ntk_norm_method,
                        alpha=ntk_adjustment_alpha,
                        direction=direction,
                        verbose=verbose
                    )
                    embed_scores = embed_scores * adjustment_factor.to(embed_scores.device)

                    if verbose:
                        print(f"  [NTK-current] Embedding: adjustment factor range=[{adjustment_factor.min():.4f}, {adjustment_factor.max():.4f}]")

                model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        ffn_masks = {}
        head_masks = {}
        dim_masks = {}
        embedding_mask = torch.ones(hidden_size)

        # FFN masks
        if compute_ffn:
            # FFN skip layers set (for global pruning)
            # FFN skip layers set (for mask creation)
            ffn_skip_set = set()
            if skip_first_layer_ffn > 0:
                ffn_skip_set.update(range(min(skip_first_layer_ffn, num_layers)))
            if skip_last_layer_ffn > 0:
                ffn_skip_set.update(range(max(0, num_layers - skip_last_layer_ffn), num_layers))

            if global_ffn_pruning:
                ffn_masks = _create_ffn_mask_sorted_grouping_global(
                    ffn_scores, ffn_pruning_ratio, ffn_dimension_multiple,
                    skip_layers=ffn_skip_set,
                    allow_full_block_pruning=allow_full_block_pruning
                )
            else:
                for layer_idx in range(num_layers):
                    if layer_idx in ffn_skip_set:
                        ffn_masks[layer_idx] = torch.ones(intermediate_size)
                    else:
                        ffn_masks[layer_idx] = _create_ffn_mask_sorted_grouping(
                            ffn_scores[layer_idx], ffn_pruning_ratio, ffn_dimension_multiple,
                            allow_full_block_pruning=allow_full_block_pruning
                        )

        # Head masks
        if compute_head:
            if global_head_pruning:
                ffn_masks_for_head = _create_head_mask_global(
                    head_scores, head_pruning_ratio,
                    skip_first_layer=skip_first_layer_head, skip_last_layer=skip_last_layer_head,
                    allow_full_block_pruning=allow_full_block_pruning
                )
                head_masks = ffn_masks_for_head
            else:
                # Head skip layers set (for mask creation)
                head_skip_set = set()
                if skip_first_layer_head > 0:
                    head_skip_set.update(range(min(skip_first_layer_head, num_layers)))
                if skip_last_layer_head > 0:
                    head_skip_set.update(range(max(0, num_layers - skip_last_layer_head), num_layers))
                for layer_idx in range(num_layers):
                    if layer_idx in head_skip_set:
                        head_masks[layer_idx] = torch.ones(num_heads)
                    else:
                        head_masks[layer_idx] = _create_head_mask(
                            head_scores[layer_idx], head_pruning_ratio,
                            allow_full_block_pruning=allow_full_block_pruning
                        )

        # Dimension masks
        if compute_dim:
            if global_dimension_pruning:
                dim_masks = _create_dimension_mask_sorted_grouping_global(
                    dim_scores, dimension_pruning_ratio, dimension_group_size
                )
            else:
                for layer_idx in range(num_layers):
                    dim_masks[layer_idx] = _create_dimension_mask_sorted_grouping(
                        dim_scores[layer_idx], dimension_pruning_ratio, dimension_group_size
                    )

        # Embedding mask
        if compute_embedding:
            embedding_mask = _create_embedding_mask(
                embed_scores, embedding_pruning_ratio, embedding_dimension_multiple, verbose
            )

        # NTK Revival Counting (one-shot)
        ntk_revival_stats = None
        if use_ntk_adjustment and (pre_ntk_ffn_scores is not None or pre_ntk_head_scores is not None or pre_ntk_embed_scores is not None):
            ntk_revival_stats = _compute_ntk_revival_stats(
                pre_ntk_ffn_scores=pre_ntk_ffn_scores,
                pre_ntk_head_scores=pre_ntk_head_scores,
                pre_ntk_embed_scores=pre_ntk_embed_scores,
                actual_ffn_masks=ffn_masks,
                actual_head_masks=head_masks,
                actual_embedding_mask=embedding_mask,
                ffn_pruning_ratio=ffn_pruning_ratio,
                head_pruning_ratio=head_pruning_ratio,
                embedding_pruning_ratio=embedding_pruning_ratio,
                num_layers=num_layers,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                hidden_size=hidden_size,
                ffn_dimension_multiple=ffn_dimension_multiple,
                embedding_dimension_multiple=embedding_dimension_multiple,
                global_ffn_pruning=global_ffn_pruning,
                global_head_pruning=global_head_pruning,
                skip_first_layer_head=skip_first_layer_head,
                skip_last_layer_head=skip_last_layer_head,
                skip_first_layer_ffn=skip_first_layer_ffn,
                skip_last_layer_ffn=skip_last_layer_ffn,
                allow_full_block_pruning=allow_full_block_pruning,
                compute_ffn=compute_ffn,
                compute_head=compute_head,
                compute_embedding=compute_embedding,
                verbose=verbose
            )

        return ffn_masks, head_masks, dim_masks, embedding_mask, ntk_revival_stats

    max_ratio = max(ffn_pruning_ratio, head_pruning_ratio, dimension_pruning_ratio, embedding_pruning_ratio)

    ffn_start = min(iterative_start_ratio_ffn, ffn_pruning_ratio) if compute_ffn and iterative_start_ratio_ffn > 0 else 0
    head_start = min(iterative_start_ratio_head, head_pruning_ratio) if compute_head and iterative_start_ratio_head > 0 else 0
    emb_start = min(iterative_start_ratio_emb, embedding_pruning_ratio) if compute_embedding and iterative_start_ratio_emb > 0 else 0
    has_hybrid_start = (ffn_start > 0 or head_start > 0 or emb_start > 0)

    steps = []
    if has_hybrid_start:
        effective_start = max(ffn_start, head_start, emb_start)
        steps.append(effective_start)
        current = effective_start + pruning_step_size
    else:
        current = pruning_step_size
    while current < max_ratio:
        steps.append(current)
        current += pruning_step_size
    steps.append(max_ratio)

    if verbose:
        if has_hybrid_start:
            print(f"[Joint All Iterative] Hybrid mode (per-type one-shot start):")
            print(f"  → FFN: one-shot to {ffn_start:.1%}, then iterative to {ffn_pruning_ratio:.1%}")
            print(f"  → Head: one-shot to {head_start:.1%}, then iterative to {head_pruning_ratio:.1%}")
            print(f"  → Embedding: one-shot to {emb_start:.1%}, then iterative to {embedding_pruning_ratio:.1%}")
        print(f"[Joint All Iterative] Steps: {[f'{s:.1%}' for s in steps]}")
        if use_mode_connectivity:
            print(f"[Joint All Iterative] Mode Connectivity: threshold={barrier_threshold}, n_points={barrier_n_points}, action={barrier_action}")

    current_ffn_masks = {i: torch.ones(intermediate_size) for i in range(num_layers)}
    current_head_masks = {i: torch.ones(num_heads) for i in range(num_layers)}
    current_dim_masks = {i: torch.ones(head_dim) for i in range(num_layers)}
    current_embedding_mask = torch.ones(hidden_size)

    connectivity_results = {
        'enabled': use_mode_connectivity,
        'ffn_barrier_history': [],
        'head_barrier_history': [],
        'embedding_barrier_history': [],
        'step_history': [],
        'threshold_exceeded': {'ffn': 0, 'head': 0, 'embedding': 0}
    } if use_mode_connectivity else None

    ffn_sensitivities = None  # FFN NTK sensitivities (for "previous" mode)
    head_sensitivities = None  # Head NTK sensitivities (for "previous" mode)
    embed_sensitivity = None  # Embedding NTK sensitivity (for "previous" mode)
    prev_ffn_masks = None
    prev_head_masks = None
    prev_embedding_mask = None

    cumulative_revival_stats = {
        'ffn_revived': 0, 'head_revived': 0, 'embedding_revived': 0,
        'ffn_revived_per_layer': {}, 'head_revived_per_layer': {},
        'ffn_newly_pruned': 0, 'head_newly_pruned': 0, 'embedding_newly_pruned': 0,
    }

    # NTK direction sign
    sens_sign = 1.0 if ntk_sensitivity_direction == "normal" else -1.0

    ffn_skip_layers = set()
    if skip_first_layer_ffn > 0:
        ffn_skip_layers.update(range(min(skip_first_layer_ffn, num_layers)))
    if skip_last_layer_ffn > 0:
        ffn_skip_layers.update(range(max(0, num_layers - skip_last_layer_ffn), num_layers))

    head_skip_layers = set()
    if skip_first_layer_head > 0:
        head_skip_layers.update(range(min(skip_first_layer_head, num_layers)))
    if skip_last_layer_head > 0:
        head_skip_layers.update(range(max(0, num_layers - skip_last_layer_head), num_layers))

    for step_idx, step_ratio in enumerate(steps):
        if verbose:
            print(f"\n[Joint All Iterative] Step {step_idx + 1}/{len(steps)}: ratio={step_ratio:.1%}")

        if step_idx == 0 and has_hybrid_start:
            step_ffn_ratio = ffn_start
            step_head_ratio = head_start
            step_dim_ratio = min(step_ratio, dimension_pruning_ratio) if compute_dim else 0
            step_embed_ratio = emb_start
            if verbose:
                print(f"  → Hybrid one-shot: FFN={step_ffn_ratio:.1%}, Head={step_head_ratio:.1%}, Emb={step_embed_ratio:.1%}")
        else:
            step_ffn_ratio = min(step_ratio, ffn_pruning_ratio) if compute_ffn else 0
            step_head_ratio = min(step_ratio, head_pruning_ratio) if compute_head else 0
            step_dim_ratio = min(step_ratio, dimension_pruning_ratio) if compute_dim else 0
            step_embed_ratio = min(step_ratio, embedding_pruning_ratio) if compute_embedding else 0

        if step_idx > 0:
            calculator = TaylorImportanceCalculator(
                model, device, verbose=False,
                ffn_mode=ffn_mode,
                importance_method=importance_method,
                ffn_masks=current_ffn_masks,
                head_masks=current_head_masks,
                dim_masks=current_dim_masks,
                embedding_mask=current_embedding_mask,
                use_correct_normalization=use_correct_normalization
            )
        else:
            calculator = TaylorImportanceCalculator(
                model, device, verbose=False,
                ffn_mode=ffn_mode,
                importance_method=importance_method,
                use_correct_normalization=use_correct_normalization
            )

        ffn_scores, head_scores, dim_scores, embed_scores = calculator.compute_all_importance(
            dataloader, num_samples, head_mode,
            compute_ffn=compute_ffn,
            compute_head=compute_head,
            compute_dim=compute_dim,
            compute_embedding=compute_embedding
        )

        step_pre_ntk_ffn = None
        step_pre_ntk_head = None
        step_pre_ntk_embed = None
        if use_ntk_adjustment:
            step_pre_ntk_ffn = {k: v.clone() for k, v in ffn_scores.items()} if compute_ffn else None
            step_pre_ntk_head = {k: v.clone() for k, v in head_scores.items()} if compute_head else None
            step_pre_ntk_embed = embed_scores.clone() if (compute_embedding and use_ntk_embedding) else None

        # ===== NTK Adjustment =====
        if use_ntk_adjustment and ntk_adjustment_mode == "previous":
            if ffn_sensitivities is not None and compute_ffn:
                if verbose:
                    print(f"  [NTK-previous] Applying FFN sensitivity adjustment (alpha={ntk_adjustment_alpha})")
                ffn_scores = _apply_ntk_adjustment_to_scores(
                    scores=ffn_scores,
                    sensitivities=ffn_sensitivities,
                    alpha=ntk_adjustment_alpha,
                    sens_sign=sens_sign,
                    granularity=ntk_granularity,
                    unit_mode=ntk_unit_mode,
                    dimension_multiple=ffn_dimension_multiple,
                    intermediate_size=intermediate_size,
                    num_layers=num_layers,
                    verbose=verbose,
                    ntk_norm_method=ntk_norm_method,
                    skip_layers=ffn_skip_layers,
                    layer_wise=not global_ffn_pruning
                )
            if head_sensitivities is not None and compute_head:
                if verbose:
                    print(f"  [NTK-previous] Applying Head sensitivity adjustment (alpha={ntk_adjustment_alpha})")
                head_scores = _apply_ntk_adjustment_to_scores(
                    scores=head_scores,
                    sensitivities=head_sensitivities,
                    alpha=ntk_adjustment_alpha,
                    sens_sign=sens_sign,
                    granularity="unit",
                    unit_mode="head",
                    dimension_multiple=1,
                    intermediate_size=num_heads,
                    num_layers=num_layers,
                    verbose=verbose,
                    ntk_norm_method=ntk_norm_method,
                    skip_layers=head_skip_layers,
                    layer_wise=not global_head_pruning
                )
            # Embedding NTK adjustment (previous mode)
            if use_ntk_embedding and embed_sensitivity is not None and compute_embedding:
                if verbose:
                    print(f"  [NTK-previous] Applying Embedding sensitivity adjustment (alpha={ntk_adjustment_alpha})")
                direction = "normal" if sens_sign > 0 else "inverse"
                adjustment_factor = _compute_ntk_factors_flat(
                    sensitivity=embed_sensitivity,
                    method=ntk_norm_method,
                    alpha=ntk_adjustment_alpha,
                    direction=direction,
                    verbose=verbose
                )
                embed_scores = embed_scores * adjustment_factor.to(embed_scores.device)
                if verbose:
                    print(f"  [NTK-previous] Embedding: adjustment factor range=[{adjustment_factor.min():.4f}, {adjustment_factor.max():.4f}]")

        elif use_ntk_adjustment and ntk_adjustment_mode == "current":
            if compute_ffn and step_ffn_ratio > 0:
                if verbose:
                    print(f"  [NTK-current] FFN: 1st pruning → NTK (importance-groups) → 2nd pruning")

                ffn_group_assignments = None
                if global_ffn_pruning:
                    temp_ffn_masks, ffn_group_assignments = _create_ffn_mask_sorted_grouping_global_with_groups(
                        ffn_scores, step_ffn_ratio, ffn_dimension_multiple, skip_layers=ffn_skip_layers
                    )
                else:
                    temp_ffn_masks = {
                        l: _create_ffn_mask_sorted_grouping(ffn_scores[l], step_ffn_ratio, ffn_dimension_multiple, allow_full_block_pruning=allow_full_block_pruning)
                        if l not in ffn_skip_layers else torch.ones(intermediate_size)
                        for l in range(num_layers)
                    }

                if step_idx == 0:
                    temp_ffn_masks = {i: torch.ones(intermediate_size) for i in range(num_layers)}

                ffn_ga_for_ntk = None if ntk_unit_mode == "neuron" else ffn_group_assignments
                current_ffn_sens = _compute_ntk_sensitivity_per_unit(
                    model=model,
                    dataloader=dataloader,
                    current_masks=temp_ffn_masks,
                    prev_masks=prev_ffn_masks,
                    mask_type="ffn",
                    num_samples=num_samples,
                    device=device,
                    verbose=verbose,
                    dimension_multiple=ffn_dimension_multiple,
                    ntk_target_mode=ntk_target_mode,
                    ntk_unit_mode=ntk_unit_mode,
                    group_assignments=ffn_ga_for_ntk,
                    ntk_method=ntk_method,
                    ntk_eigenvalue_k=ntk_eigenvalue_k,
                    normalize_per_layer=not global_ffn_pruning,
                    skip_layers=ffn_skip_layers
                ) if ntk_granularity == "unit" else _compute_ntk_sensitivity(
                    model=model,
                    dataloader=dataloader,
                    current_masks=temp_ffn_masks,
                    prev_masks=prev_ffn_masks,
                    mask_type="ffn",
                    num_samples=num_samples,
                    device=device,
                    verbose=verbose,
                    ntk_target_mode=ntk_target_mode,
                    ntk_method=ntk_method,
                    ntk_eigenvalue_k=ntk_eigenvalue_k,
                    skip_layers=ffn_skip_layers
                )
                if ntk_granularity == "unit" and global_ffn_pruning:
                    if ntk_norm_method == "legacy":
                        all_sens = torch.cat([current_ffn_sens[l] for l in range(num_layers) if l in current_ffn_sens])
                        nonzero_mask = all_sens > 0
                        global_mean = all_sens[nonzero_mask].mean() if nonzero_mask.any() else all_sens.mean()
                        if global_mean > 0:
                            for layer_idx in current_ffn_sens:
                                current_ffn_sens[layer_idx] = current_ffn_sens[layer_idx] / global_mean
                        if verbose:
                            print(f"  [NTK-current] FFN Global normalization (legacy): mean={global_mean:.4f}")
                ffn_scores = _apply_ntk_adjustment_to_scores(
                    scores=ffn_scores,
                    sensitivities=current_ffn_sens,
                    alpha=ntk_adjustment_alpha,
                    sens_sign=sens_sign,
                    granularity=ntk_granularity,
                    unit_mode=ntk_unit_mode,
                    dimension_multiple=ffn_dimension_multiple,
                    intermediate_size=intermediate_size,
                    num_layers=num_layers,
                    verbose=verbose,
                    group_assignments=ffn_ga_for_ntk,
                    ntk_norm_method=ntk_norm_method,
                    skip_layers=ffn_skip_layers,
                    layer_wise=not global_ffn_pruning
                )

            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            if compute_head and step_head_ratio > 0:
                if verbose:
                    print(f"  [NTK-current] Head: 1st pruning → NTK → 2nd pruning")
                temp_head_masks = _create_head_mask_global(
                    head_scores, step_head_ratio, skip_first_layer=skip_first_layer_head, skip_last_layer=skip_last_layer_head,
                    allow_full_block_pruning=allow_full_block_pruning
                ) if global_head_pruning else {
                    l: _create_head_mask(head_scores[l], step_head_ratio, allow_full_block_pruning=allow_full_block_pruning)
                    if l not in head_skip_layers else torch.ones(num_heads)
                    for l in range(num_layers)
                }

                if step_idx == 0:
                    temp_head_masks = {i: torch.ones(num_heads) for i in range(num_layers)}

                current_head_sens = _compute_ntk_sensitivity_per_unit(
                    model=model,
                    dataloader=dataloader,
                    current_masks=temp_head_masks,
                    prev_masks=prev_head_masks,
                    mask_type="head",
                    num_samples=num_samples,
                    device=device,
                    verbose=verbose,
                    dimension_multiple=1,
                    num_heads=num_heads,
                    ntk_target_mode=ntk_target_mode,
                    ntk_unit_mode="head",
                    ntk_method=ntk_method,
                    ntk_eigenvalue_k=ntk_eigenvalue_k,
                    normalize_per_layer=not global_head_pruning,
                    skip_layers=head_skip_layers
                )
                if global_head_pruning:
                    if ntk_norm_method == "legacy":
                        all_sens = torch.cat([current_head_sens[l] for l in range(num_layers) if l in current_head_sens])
                        nonzero_mask = all_sens > 0
                        global_mean = all_sens[nonzero_mask].mean() if nonzero_mask.any() else all_sens.mean()
                        if global_mean > 0:
                            for layer_idx in current_head_sens:
                                current_head_sens[layer_idx] = current_head_sens[layer_idx] / global_mean
                        if verbose:
                            print(f"  [NTK-current] Head Global normalization (legacy): mean={global_mean:.4f}")
                head_scores = _apply_ntk_adjustment_to_scores(
                    scores=head_scores,
                    sensitivities=current_head_sens,
                    alpha=ntk_adjustment_alpha,
                    sens_sign=sens_sign,
                    granularity="unit",
                    unit_mode="head",
                    dimension_multiple=1,
                    intermediate_size=num_heads,
                    num_layers=num_layers,
                    verbose=verbose,
                    ntk_norm_method=ntk_norm_method,
                    skip_layers=head_skip_layers,
                    layer_wise=not global_head_pruning
                )

            # Embedding NTK adjustment (current mode)
            if use_ntk_embedding and compute_embedding and step_embed_ratio > 0:
                if verbose:
                    print(f"  [NTK-current] Embedding: Computing NTK sensitivity (method={ntk_embedding_method})...")

                if ntk_embedding_method == "ntk_matrix":
                    embed_sensitivity = _compute_embedding_ntk_sensitivity_matrix(
                        model=model,
                        dataloader=dataloader,
                        num_samples=num_samples,
                        device=device,
                        verbose=verbose,
                        ntk_method=ntk_method,
                        ntk_eigenvalue_k=ntk_eigenvalue_k
                    )
                else:
                    embed_sensitivity = _compute_embedding_ntk_sensitivity(
                        model=model,
                        dataloader=dataloader,
                        num_samples=num_samples,
                        device=device,
                        verbose=verbose
                    )

                if embed_sensitivity is not None:
                    direction = "normal" if sens_sign > 0 else "inverse"
                    adjustment_factor = _compute_ntk_factors_flat(
                        sensitivity=embed_sensitivity,
                        method=ntk_norm_method,
                        alpha=ntk_adjustment_alpha,
                        direction=direction,
                        verbose=verbose
                    )
                    embed_scores = embed_scores * adjustment_factor.to(embed_scores.device)

                    if verbose:
                        print(f"  [NTK-current] Embedding: adjustment factor range=[{adjustment_factor.min():.4f}, {adjustment_factor.max():.4f}]")

                model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        # ===== Connectivity Importance Adjustment (2-pass) =====
        if use_connectivity_importance and step_idx > 0:
            conn_sign = 1.0 if connectivity_direction == "normal" else -1.0
            if verbose:
                print(f"  [Connectivity-2pass] 1st pruning → connectivity → 2nd pruning (direction={connectivity_direction})")

            prev_ffn_for_conn = prev_ffn_masks if prev_ffn_masks is not None else {
                l: torch.ones(intermediate_size) for l in range(num_layers)
            }
            prev_head_for_conn = prev_head_masks if prev_head_masks is not None else {
                l: torch.ones(num_heads) for l in range(num_layers)
            }
            prev_embed_for_conn = prev_embedding_mask if prev_embedding_mask is not None else torch.ones(hidden_size)

            # FFN Connectivity Importance (2-pass)
            if compute_ffn and step_ffn_ratio > 0:
                try:
                    if global_ffn_pruning:
                        temp_ffn_masks = _create_ffn_mask_sorted_grouping_global(
                            ffn_scores, step_ffn_ratio, ffn_dimension_multiple,
                            skip_layers=ffn_skip_layers,
                            allow_full_block_pruning=allow_full_block_pruning
                        )
                    else:
                        temp_ffn_masks = {
                            l: _create_ffn_mask_sorted_grouping(ffn_scores[l], step_ffn_ratio, ffn_dimension_multiple, allow_full_block_pruning=allow_full_block_pruning)
                            if l not in ffn_skip_layers else torch.ones(intermediate_size)
                            for l in range(num_layers)
                        }

                    if verbose:
                        temp_kept = sum(m.sum().item() for m in temp_ffn_masks.values())
                        print(f"    FFN 1st pass: {temp_kept:.0f}/{num_layers * intermediate_size} kept")

                    ffn_conn_importance = _compute_connectivity_importance(
                        model=model,
                        dataloader=dataloader,
                        masks_prev=prev_ffn_for_conn,
                        masks_curr=temp_ffn_masks,
                        mask_type="ffn",
                        n_points=connectivity_n_points,
                        num_samples=barrier_num_samples,
                        device=device,
                        verbose=False
                    )
                    for layer_idx in ffn_scores:
                        if layer_idx in ffn_conn_importance:
                            conn_factor = ffn_conn_importance[layer_idx].to(ffn_scores[layer_idx].device)
                            adjustment = 1.0 + conn_sign * connectivity_alpha * (conn_factor - 1.0)
                            ffn_scores[layer_idx] = ffn_scores[layer_idx] * adjustment
                    if verbose:
                        print(f"    FFN: connectivity adjustment applied (alpha={connectivity_alpha}, sign={conn_sign})")
                except Exception as e:
                    if verbose:
                        print(f"    FFN connectivity failed: {e}")

            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()

            # Head Connectivity Importance (2-pass)
            if compute_head and step_head_ratio > 0:
                try:
                    if global_head_pruning:
                        temp_head_masks = _create_head_mask_global(
                            head_scores, step_head_ratio,
                            skip_first_layer=skip_first_layer_head, skip_last_layer=skip_last_layer_head,
                            allow_full_block_pruning=allow_full_block_pruning
                        )
                    else:
                        temp_head_masks = {
                            l: _create_head_mask(head_scores[l], step_head_ratio, allow_full_block_pruning=allow_full_block_pruning)
                            if l not in head_skip_layers else torch.ones(num_heads)
                            for l in range(num_layers)
                        }

                    if verbose:
                        temp_kept = sum(m.sum().item() for m in temp_head_masks.values())
                        print(f"    Head 1st pass: {temp_kept:.0f}/{num_layers * num_heads} kept")

                    head_conn_importance = _compute_connectivity_importance(
                        model=model,
                        dataloader=dataloader,
                        masks_prev=prev_head_for_conn,
                        masks_curr=temp_head_masks,
                        mask_type="head",
                        n_points=connectivity_n_points,
                        num_samples=barrier_num_samples,
                        device=device,
                        verbose=False
                    )
                    for layer_idx in head_scores:
                        if layer_idx in head_conn_importance:
                            conn_factor = head_conn_importance[layer_idx].to(head_scores[layer_idx].device)
                            adjustment = 1.0 + conn_sign * connectivity_alpha * (conn_factor - 1.0)
                            head_scores[layer_idx] = head_scores[layer_idx] * adjustment
                    if verbose:
                        print(f"    Head: connectivity adjustment applied (alpha={connectivity_alpha}, sign={conn_sign})")
                except Exception as e:
                    if verbose:
                        print(f"    Head connectivity failed: {e}")

            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()

            # Embedding Connectivity Importance (2-pass)
            if compute_embedding and step_embed_ratio > 0:
                try:
                    temp_embedding_mask = _create_embedding_mask(
                        embed_scores, step_embed_ratio, embedding_dimension_multiple, verbose=False
                    )

                    if verbose:
                        temp_kept = temp_embedding_mask.sum().item()
                        print(f"    Embedding 1st pass: {temp_kept:.0f}/{hidden_size} kept")

                    embed_prev_dict = {0: prev_embed_for_conn}
                    embed_curr_dict = {0: temp_embedding_mask}
                    embed_conn_importance = _compute_connectivity_importance(
                        model=model,
                        dataloader=dataloader,
                        masks_prev=embed_prev_dict,
                        masks_curr=embed_curr_dict,
                        mask_type="embedding",
                        n_points=connectivity_n_points,
                        num_samples=barrier_num_samples,
                        device=device,
                        verbose=False
                    )
                    if 0 in embed_conn_importance:
                        conn_factor = embed_conn_importance[0].to(embed_scores.device)
                        adjustment = 1.0 + conn_sign * connectivity_alpha * (conn_factor - 1.0)
                        embed_scores = embed_scores * adjustment
                    if verbose:
                        print(f"    Embedding: connectivity adjustment applied (alpha={connectivity_alpha}, sign={conn_sign})")
                except Exception as e:
                    if verbose:
                        print(f"    Embedding connectivity failed: {e}")

            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()

        if enable_iterative_calibration and step_idx > 0:
            iter_cal_prev_ffn = {k: v.clone() for k, v in current_ffn_masks.items()}
            iter_cal_prev_head = {k: v.clone() for k, v in current_head_masks.items()}

        if compute_ffn and step_ffn_ratio > 0:
            if global_ffn_pruning:
                current_ffn_masks = _create_ffn_mask_sorted_grouping_global(
                    ffn_scores, step_ffn_ratio, ffn_dimension_multiple,
                    skip_layers=ffn_skip_layers,
                    allow_full_block_pruning=allow_full_block_pruning
                )
            else:
                for layer_idx in range(num_layers):
                    if layer_idx in ffn_skip_layers:
                        continue
                    current_ffn_masks[layer_idx] = _create_ffn_mask_sorted_grouping(
                        ffn_scores[layer_idx], step_ffn_ratio, ffn_dimension_multiple,
                        allow_full_block_pruning=allow_full_block_pruning
                    )

        if compute_head and step_head_ratio > 0:
            if global_head_pruning:
                current_head_masks = _create_head_mask_global(
                    head_scores, step_head_ratio,
                    skip_first_layer=skip_first_layer_head, skip_last_layer=skip_last_layer_head,
                    allow_full_block_pruning=allow_full_block_pruning
                )
            else:
                for layer_idx in range(num_layers):
                    if layer_idx in head_skip_layers:
                        continue
                    current_head_masks[layer_idx] = _create_head_mask(
                        head_scores[layer_idx], step_head_ratio,
                        allow_full_block_pruning=allow_full_block_pruning
                    )

        if compute_dim and step_dim_ratio > 0:
            if global_dimension_pruning:
                current_dim_masks = _create_dimension_mask_sorted_grouping_global(
                    dim_scores, step_dim_ratio, dimension_group_size
                )
            else:
                for layer_idx in range(num_layers):
                    current_dim_masks[layer_idx] = _create_dimension_mask_sorted_grouping(
                        dim_scores[layer_idx], step_dim_ratio, dimension_group_size
                    )

        if compute_embedding and step_embed_ratio > 0:
            current_embedding_mask = _create_embedding_mask(
                embed_scores, step_embed_ratio, embedding_dimension_multiple, verbose=False
            )

        if use_ntk_adjustment and (step_pre_ntk_ffn is not None or step_pre_ntk_head is not None or step_pre_ntk_embed is not None):
            step_stats = _compute_ntk_revival_stats(
                pre_ntk_ffn_scores=step_pre_ntk_ffn,
                pre_ntk_head_scores=step_pre_ntk_head,
                pre_ntk_embed_scores=step_pre_ntk_embed,
                actual_ffn_masks=current_ffn_masks,
                actual_head_masks=current_head_masks,
                actual_embedding_mask=current_embedding_mask,
                ffn_pruning_ratio=step_ffn_ratio,
                head_pruning_ratio=step_head_ratio,
                embedding_pruning_ratio=step_embed_ratio,
                num_layers=num_layers,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                hidden_size=hidden_size,
                ffn_dimension_multiple=ffn_dimension_multiple,
                embedding_dimension_multiple=embedding_dimension_multiple,
                global_ffn_pruning=global_ffn_pruning,
                global_head_pruning=global_head_pruning,
                skip_first_layer_head=skip_first_layer_head,
                skip_last_layer_head=skip_last_layer_head,
                skip_first_layer_ffn=skip_first_layer_ffn,
                skip_last_layer_ffn=skip_last_layer_ffn,
                allow_full_block_pruning=allow_full_block_pruning,
                compute_ffn=compute_ffn,
                compute_head=compute_head,
                compute_embedding=compute_embedding,
                verbose=False
            )
            cumulative_revival_stats['ffn_revived'] += step_stats['ffn_revived']
            cumulative_revival_stats['ffn_newly_pruned'] += step_stats['ffn_newly_pruned']
            cumulative_revival_stats['head_revived'] += step_stats['head_revived']
            cumulative_revival_stats['head_newly_pruned'] += step_stats['head_newly_pruned']
            cumulative_revival_stats['embedding_revived'] += step_stats['embedding_revived']
            cumulative_revival_stats['embedding_newly_pruned'] += step_stats['embedding_newly_pruned']
            for l, c in step_stats.get('ffn_revived_per_layer', {}).items():
                cumulative_revival_stats['ffn_revived_per_layer'][l] = cumulative_revival_stats['ffn_revived_per_layer'].get(l, 0) + c
            for l, c in step_stats.get('head_revived_per_layer', {}).items():
                cumulative_revival_stats['head_revived_per_layer'][l] = cumulative_revival_stats['head_revived_per_layer'].get(l, 0) + c
            if verbose:
                print(f"  [NTK Revival Step {step_idx+1}] FFN={step_stats['ffn_revived']}, Head={step_stats['head_revived']}, Emb={step_stats['embedding_revived']}")

        if enable_iterative_calibration and step_idx > 0:
            from pruning.output_scale_calibration import iterative_calibrate_step
            iterative_calibrate_step(
                model=model,
                dataloader=dataloader,
                prev_ffn_masks=iter_cal_prev_ffn,
                prev_head_masks=iter_cal_prev_head,
                curr_ffn_masks=current_ffn_masks,
                curr_head_masks=current_head_masks,
                step_idx=step_idx,
                device=device,
                verbose=verbose
            )
            del iter_cal_prev_ffn, iter_cal_prev_head

        if verbose:
            ffn_total = num_layers * intermediate_size
            head_total = num_layers * num_heads
            dim_total = num_layers * head_dim

            ffn_kept = sum(m.sum().item() for m in current_ffn_masks.values()) if compute_ffn else ffn_total
            head_kept = sum(m.sum().item() for m in current_head_masks.values()) if compute_head else head_total
            dim_kept = sum(m.sum().item() for m in current_dim_masks.values()) if compute_dim else dim_total
            embed_kept = current_embedding_mask.sum().item() if compute_embedding else hidden_size
            print(f"  → FFN kept: {ffn_kept:.0f}/{ffn_total}")
            print(f"  → Head kept: {head_kept:.0f}/{head_total}")
            print(f"  → Dim kept: {dim_kept:.0f}/{dim_total}")
            print(f"  → Embedding kept: {embed_kept:.0f}/{hidden_size}")

        if use_ntk_adjustment and ntk_adjustment_mode == "current" and step_idx < len(steps) - 1:
            prev_ffn_masks = {k: v.clone() for k, v in current_ffn_masks.items()}
            prev_head_masks = {k: v.clone() for k, v in current_head_masks.items()}
            prev_embedding_mask = current_embedding_mask.clone()
            if verbose:
                print(f"  [NTK-current] Updated prev_masks for next step")

        if use_ntk_adjustment and ntk_adjustment_mode == "previous" and step_idx < len(steps) - 1:
            if compute_ffn and step_ffn_ratio > 0:
                if verbose:
                    print(f"  [NTK-previous] Computing FFN sensitivity for next step...")
                ffn_sensitivities = _compute_ntk_sensitivity_per_unit(
                    model=model,
                    dataloader=dataloader,
                    current_masks=current_ffn_masks,
                    prev_masks=prev_ffn_masks,
                    mask_type="ffn",
                    num_samples=num_samples,
                    device=device,
                    verbose=verbose,
                    dimension_multiple=ffn_dimension_multiple,
                    ntk_target_mode=ntk_target_mode,
                    ntk_unit_mode=ntk_unit_mode,
                    ntk_method=ntk_method,
                    ntk_eigenvalue_k=ntk_eigenvalue_k,
                    normalize_per_layer=not global_ffn_pruning,
                    skip_layers=ffn_skip_layers
                ) if ntk_granularity == "unit" else _compute_ntk_sensitivity(
                    model=model,
                    dataloader=dataloader,
                    current_masks=current_ffn_masks,
                    prev_masks=prev_ffn_masks,
                    mask_type="ffn",
                    num_samples=num_samples,
                    device=device,
                    verbose=verbose,
                    ntk_target_mode=ntk_target_mode,
                    ntk_method=ntk_method,
                    ntk_eigenvalue_k=ntk_eigenvalue_k,
                    skip_layers=ffn_skip_layers
                )
                # Global normalization for unit granularity
                if ntk_granularity == "unit" and global_ffn_pruning:
                    if ntk_norm_method == "legacy":
                        all_sens = torch.cat([ffn_sensitivities[l] for l in range(num_layers) if l in ffn_sensitivities])
                        nonzero_mask = all_sens > 0
                        global_mean = all_sens[nonzero_mask].mean() if nonzero_mask.any() else all_sens.mean()
                        if global_mean > 0:
                            for layer_idx in ffn_sensitivities:
                                ffn_sensitivities[layer_idx] = ffn_sensitivities[layer_idx] / global_mean
                        if verbose:
                            print(f"  [NTK-previous] FFN Global normalization (legacy): mean={global_mean:.4f}")

            if compute_head and step_head_ratio > 0:
                if verbose:
                    print(f"  [NTK-previous] Computing Head sensitivity for next step...")
                head_sensitivities = _compute_ntk_sensitivity_per_unit(
                    model=model,
                    dataloader=dataloader,
                    current_masks=current_head_masks,
                    prev_masks=prev_head_masks,
                    mask_type="head",
                    num_samples=num_samples,
                    device=device,
                    verbose=verbose,
                    dimension_multiple=1,
                    num_heads=num_heads,
                    ntk_target_mode=ntk_target_mode,
                    ntk_unit_mode="head",
                    ntk_method=ntk_method,
                    ntk_eigenvalue_k=ntk_eigenvalue_k,
                    normalize_per_layer=not global_head_pruning,
                    skip_layers=head_skip_layers
                )
                # Global normalization for head
                if global_head_pruning:
                    if ntk_norm_method == "legacy":
                        all_sens = torch.cat([head_sensitivities[l] for l in range(num_layers) if l in head_sensitivities])
                        nonzero_mask = all_sens > 0
                        global_mean = all_sens[nonzero_mask].mean() if nonzero_mask.any() else all_sens.mean()
                        if global_mean > 0:
                            for layer_idx in head_sensitivities:
                                head_sensitivities[layer_idx] = head_sensitivities[layer_idx] / global_mean
                        if verbose:
                            print(f"  [NTK-previous] Head Global normalization (legacy): mean={global_mean:.4f}")

            if use_ntk_embedding and compute_embedding and step_embed_ratio > 0:
                if verbose:
                    print(f"  [NTK-previous] Computing Embedding sensitivity for next step (method={ntk_embedding_method})...")
                if ntk_embedding_method == "ntk_matrix":
                    embed_sensitivity = _compute_embedding_ntk_sensitivity_matrix(
                        model=model,
                        dataloader=dataloader,
                        num_samples=num_samples,
                        device=device,
                        verbose=verbose,
                        ntk_method=ntk_method,
                        ntk_eigenvalue_k=ntk_eigenvalue_k
                    )
                else:
                    embed_sensitivity = _compute_embedding_ntk_sensitivity(
                        model=model,
                        dataloader=dataloader,
                        num_samples=num_samples,
                        device=device,
                        verbose=verbose
                    )
                if embed_sensitivity is not None:
                    if ntk_norm_method == "legacy":
                        mean_sens = embed_sensitivity.mean()
                        if mean_sens > 0:
                            embed_sensitivity = embed_sensitivity / mean_sens
                    if verbose:
                        print(f"  [NTK-previous] Embedding sensitivity range: [{embed_sensitivity.min():.4f}, {embed_sensitivity.max():.4f}]")

            prev_ffn_masks = {k: v.clone() for k, v in current_ffn_masks.items()}
            prev_head_masks = {k: v.clone() for k, v in current_head_masks.items()}
            prev_embedding_mask = current_embedding_mask.clone()

        if use_mode_connectivity and step_idx > 0:
            if verbose:
                print(f"  [Mode Connectivity] Measuring barriers...")

            prev_ffn_for_barrier = prev_ffn_masks if prev_ffn_masks is not None else {
                l: torch.ones(intermediate_size) for l in range(num_layers)
            }
            prev_head_for_barrier = prev_head_masks if prev_head_masks is not None else {
                l: torch.ones(num_heads) for l in range(num_layers)
            }

            if compute_ffn and step_ffn_ratio > 0:
                ffn_barrier, ffn_losses = _measure_mode_connectivity_barrier(
                    model=model,
                    dataloader=dataloader,
                    masks_prev=prev_ffn_for_barrier,
                    masks_curr=current_ffn_masks,
                    mask_type="ffn",
                    n_points=barrier_n_points,
                    num_samples=barrier_num_samples,
                    device=device,
                    verbose=False
                )
                if connectivity_results is not None:
                    connectivity_results['ffn_barrier_history'].append(ffn_barrier)
                if verbose:
                    print(f"    FFN barrier: {ffn_barrier:.4f}")
                if ffn_barrier > barrier_threshold:
                    if connectivity_results is not None:
                        connectivity_results['threshold_exceeded']['ffn'] += 1
                    if verbose:
                        print(f"    ⚠️ FFN barrier exceeded threshold!")

            if compute_head and step_head_ratio > 0:
                head_barrier, head_losses = _measure_mode_connectivity_barrier(
                    model=model,
                    dataloader=dataloader,
                    masks_prev=prev_head_for_barrier,
                    masks_curr=current_head_masks,
                    mask_type="head",
                    n_points=barrier_n_points,
                    num_samples=barrier_num_samples,
                    device=device,
                    verbose=False
                )
                if connectivity_results is not None:
                    connectivity_results['head_barrier_history'].append(head_barrier)
                if verbose:
                    print(f"    Head barrier: {head_barrier:.4f}")
                if head_barrier > barrier_threshold:
                    if connectivity_results is not None:
                        connectivity_results['threshold_exceeded']['head'] += 1
                    if verbose:
                        print(f"    ⚠️ Head barrier exceeded threshold!")

            if compute_embedding and step_embed_ratio > 0:
                prev_embed_for_barrier = prev_embedding_mask if prev_embedding_mask is not None else torch.ones(hidden_size)
                embed_prev_dict = {0: prev_embed_for_barrier}
                embed_curr_dict = {0: current_embedding_mask}
                embed_barrier, embed_losses = _measure_mode_connectivity_barrier(
                    model=model,
                    dataloader=dataloader,
                    masks_prev=embed_prev_dict,
                    masks_curr=embed_curr_dict,
                    mask_type="embedding",
                    n_points=barrier_n_points,
                    num_samples=barrier_num_samples,
                    device=device,
                    verbose=False
                )
                if connectivity_results is not None:
                    connectivity_results['embedding_barrier_history'].append(embed_barrier)
                if verbose:
                    print(f"    Embedding barrier: {embed_barrier:.4f}")
                if embed_barrier > barrier_threshold:
                    if connectivity_results is not None:
                        connectivity_results['threshold_exceeded']['embedding'] += 1
                    if verbose:
                        print(f"    ⚠️ Embedding barrier exceeded threshold!")

            if connectivity_results is not None:
                connectivity_results['step_history'].append(step_ratio)

        if use_mode_connectivity:
            prev_ffn_masks = {k: v.clone() for k, v in current_ffn_masks.items()}
            prev_head_masks = {k: v.clone() for k, v in current_head_masks.items()}
            prev_embedding_mask = current_embedding_mask.clone()

        from pruning.distributed_utils import barrier
        barrier()

    if verbose and use_mode_connectivity and connectivity_results is not None:
        print(f"\n[Mode Connectivity] Summary:")
        if connectivity_results['ffn_barrier_history']:
            avg_ffn = sum(connectivity_results['ffn_barrier_history']) / len(connectivity_results['ffn_barrier_history'])
            max_ffn = max(connectivity_results['ffn_barrier_history'])
            print(f"  FFN: avg={avg_ffn:.4f}, max={max_ffn:.4f}, exceeded={connectivity_results['threshold_exceeded']['ffn']}")
        if connectivity_results['head_barrier_history']:
            avg_head = sum(connectivity_results['head_barrier_history']) / len(connectivity_results['head_barrier_history'])
            max_head = max(connectivity_results['head_barrier_history'])
            print(f"  Head: avg={avg_head:.4f}, max={max_head:.4f}, exceeded={connectivity_results['threshold_exceeded']['head']}")
        if connectivity_results['embedding_barrier_history']:
            avg_embed = sum(connectivity_results['embedding_barrier_history']) / len(connectivity_results['embedding_barrier_history'])
            max_embed = max(connectivity_results['embedding_barrier_history'])
            print(f"  Embedding: avg={avg_embed:.4f}, max={max_embed:.4f}, exceeded={connectivity_results['threshold_exceeded']['embedding']}")

    if verbose:
        print(f"\n[Joint All Iterative] Completed!")

    ntk_revival_stats = None
    if use_ntk_adjustment:
        ntk_revival_stats = cumulative_revival_stats
        if verbose:
            print(f"\n[NTK Revival Stats (Cumulative over all iterative steps)]")
            print(f"  FFN: {cumulative_revival_stats['ffn_revived']} revived, {cumulative_revival_stats['ffn_newly_pruned']} newly pruned")
            print(f"  Head: {cumulative_revival_stats['head_revived']} revived, {cumulative_revival_stats['head_newly_pruned']} newly pruned")
            print(f"  Embedding: {cumulative_revival_stats['embedding_revived']} revived, {cumulative_revival_stats['embedding_newly_pruned']} newly pruned")
            if cumulative_revival_stats.get('ffn_revived_per_layer'):
                print(f"  FFN revived per layer: {dict(sorted(cumulative_revival_stats['ffn_revived_per_layer'].items()))}")
            if cumulative_revival_stats.get('head_revived_per_layer'):
                print(f"  Head revived per layer: {dict(sorted(cumulative_revival_stats['head_revived_per_layer'].items()))}")

    return current_ffn_masks, current_head_masks, current_dim_masks, current_embedding_mask, ntk_revival_stats


def compute_taylor_masks(
    model: nn.Module,
    dataloader,
    ffn_pruning_ratio: float,
    num_samples: int = 128,
    dimension_multiple: int = 1,
    device: str = "cuda",
    verbose: bool = True,
    ffn_mode: str = "down",
    importance_method: str = "ganda",
    enable_svd_diversity: bool = False,
    svd_diversity_method: str = "dominant",
    svd_diversity_alpha: float = 1.0,
    enable_iterative_pruning: bool = False,
    pruning_step_size: float = 0.05,
    iterative_start_ratio_ffn: float = 0.0,
    iterative_start_ratio_head: float = 0.0,
    iterative_start_ratio_emb: float = 0.0,
    use_ntk_adjustment: bool = False,
    ntk_adjustment_alpha: float = 0.3,
    ntk_adjustment_mode: str = "previous",
    ntk_sensitivity_direction: str = "normal",
    ntk_target_mode: str = "default",
    ntk_granularity: str = "layer",
    ntk_unit_mode: str = "group",
    use_iterative_premasking: bool = False,
    use_gradient_checkpointing: bool = False,
    skip_first_layer: int = 0,
    skip_last_layer: int = 0,
    global_ffn_pruning: bool = True,
    use_mode_connectivity: bool = False,
    barrier_threshold: float = 0.5,
    barrier_n_points: int = 5,
    barrier_num_samples: int = 5,
    adaptive_step_size: bool = False,
    min_step_size: float = 0.0125,
    barrier_action: str = "subdivide",
    ntk_norm_method: str = "robust",
    allow_full_block_pruning: bool = False
) -> Dict[int, torch.Tensor]:
    num_layers = model.config.num_hidden_layers
    intermediate_size = getattr(model.config, 'intermediate_size', model.config.hidden_size * 4)

    num_skip = skip_first_layer + skip_last_layer
    if num_skip > 0 and num_skip < num_layers:
        adjusted_ffn_ratio = ffn_pruning_ratio * num_layers / (num_layers - num_skip)
        if adjusted_ffn_ratio > 1.0:
            print(f"[WARNING] Adjusted FFN ratio {adjusted_ffn_ratio:.2%} > 100%, clamping to 100%")
            adjusted_ffn_ratio = 1.0
        if verbose:
            print(f"[Taylor FFN] Skip layers: first={skip_first_layer}, last={skip_last_layer} ({num_skip} layers)")
            print(f"[Taylor FFN] Target ratio: {ffn_pruning_ratio:.2%} → Adjusted: {adjusted_ffn_ratio:.2%}")
    else:
        adjusted_ffn_ratio = ffn_pruning_ratio

    if enable_iterative_pruning and ffn_pruning_ratio > 0:
        masks, connectivity_results = _compute_taylor_masks_iterative(
            model=model,
            dataloader=dataloader,
            target_ratio=adjusted_ffn_ratio,
            step_size=pruning_step_size,
            start_ratio=iterative_start_ratio_ffn,
            num_samples=num_samples,
            dimension_multiple=dimension_multiple,
            device=device,
            verbose=verbose,
            ffn_mode=ffn_mode,
            importance_method=importance_method,
            enable_svd_diversity=enable_svd_diversity,
            svd_diversity_method=svd_diversity_method,
            svd_diversity_alpha=svd_diversity_alpha,
            use_ntk_adjustment=use_ntk_adjustment,
            ntk_adjustment_alpha=ntk_adjustment_alpha,
            ntk_adjustment_mode=ntk_adjustment_mode,
            ntk_sensitivity_direction=ntk_sensitivity_direction,
            ntk_target_mode=ntk_target_mode,
            ntk_granularity=ntk_granularity,
            ntk_unit_mode=ntk_unit_mode,
            use_iterative_premasking=use_iterative_premasking,
            use_gradient_checkpointing=use_gradient_checkpointing,
            skip_first_layer=skip_first_layer,
            skip_last_layer=skip_last_layer,
            global_ffn_pruning=global_ffn_pruning,
            use_mode_connectivity=use_mode_connectivity,
            barrier_threshold=barrier_threshold,
            barrier_n_points=barrier_n_points,
            barrier_num_samples=barrier_num_samples,
            adaptive_step_size=adaptive_step_size,
            min_step_size=min_step_size,
            barrier_action=barrier_action,
            ntk_norm_method=ntk_norm_method,
            allow_full_block_pruning=allow_full_block_pruning
        )
        return masks

    calculator = TaylorImportanceCalculator(
        model, device, verbose, ffn_mode=ffn_mode, importance_method=importance_method,
        use_gradient_checkpointing=use_gradient_checkpointing
    )
    taylor_scores = calculator.compute_taylor_scores(dataloader, num_samples)

    diversity_scores = None
    if enable_svd_diversity and svd_diversity_alpha > 0:
        if verbose:
            print(f"\n[SVD-Diversity] Computing diversity scores...")
            print(f"[SVD-Diversity] Method: {svd_diversity_method}, Alpha: {svd_diversity_alpha}")

        diversity_scores = calculator.compute_ffn_diversity_scores(
            dataloader=dataloader,
            num_samples=num_samples,
            block_size=dimension_multiple,
            method=svd_diversity_method
        )

    group_size = dimension_multiple
    num_groups_per_layer = intermediate_size // group_size

    skip_layers = set()
    if skip_first_layer > 0:
        skip_layers.update(range(min(skip_first_layer, num_layers)))
    if skip_last_layer > 0:
        skip_layers.update(range(max(0, num_layers - skip_last_layer), num_layers))

    if global_ffn_pruning:
        if verbose:
            print(f"\n[Taylor Mask] Mode: Global FFN pruning (sorted grouping)")
        ffn_masks = _create_ffn_mask_sorted_grouping_global(
            all_scores=taylor_scores,
            pruning_ratio=ffn_pruning_ratio,
            group_size=group_size,
            skip_layers=skip_layers,
            allow_full_block_pruning=allow_full_block_pruning
        )
    else:
        if verbose:
            print(f"\n[Taylor Mask] Mode: Layer-wise FFN pruning (sorted grouping)")
            print(f"[Taylor Mask] Target pruning ratio per layer: {adjusted_ffn_ratio:.2%}")
            if num_skip > 0:
                print(f"[Taylor Mask] Skip layers: {num_skip}")
        ffn_masks = {}
        for layer_idx in range(num_layers):
            if layer_idx in skip_layers:
                ffn_masks[layer_idx] = torch.ones(intermediate_size)
            else:
                ffn_masks[layer_idx] = _create_ffn_mask_sorted_grouping(
                    scores=taylor_scores[layer_idx],
                    pruning_ratio=adjusted_ffn_ratio,
                    group_size=group_size,
                    allow_full_block_pruning=allow_full_block_pruning
                )

    if verbose:
        print(f"\n[Taylor Mask] FFN masks created:")
        for layer_idx in range(num_layers):
            kept = ffn_masks[layer_idx].sum().item()
            total = ffn_masks[layer_idx].numel()
            skip_mark = " (skip)" if layer_idx in skip_layers else ""
            print(f"  Layer {layer_idx}: {int(kept)}/{total} neurons kept ({kept/total:.1%}){skip_mark}")

    del calculator
    model.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return ffn_masks


def _apply_pruning_from_scores(
    taylor_scores: Dict[int, torch.Tensor],
    current_target: float,
    num_layers: int,
    intermediate_size: int,
    dimension_multiple: int,
    global_ffn_pruning: bool,
    skip_layers: set,
    diversity_scores: Optional[Dict] = None,
    svd_diversity_alpha: float = 1.0,
    allow_full_block_pruning: bool = False
) -> Dict[int, torch.Tensor]:
    group_size = dimension_multiple
    num_groups_per_layer = intermediate_size // group_size
    masks = {}

    if global_ffn_pruning:
        # ===== Global Pruning =====
        all_groups = []
        for layer_idx in range(num_layers):
            if layer_idx in skip_layers:
                continue
            scores = taylor_scores[layer_idx]
            sorted_scores, sorted_indices = torch.sort(scores, descending=True)

            for group_idx in range(num_groups_per_layer):
                start_idx = group_idx * group_size
                end_idx = start_idx + group_size

                group_original_indices = sorted_indices[start_idx:end_idx]
                group_avg_importance = sorted_scores[start_idx:end_idx].mean().item()

                if diversity_scores is not None and layer_idx in diversity_scores:
                    layer_diversity = diversity_scores[layer_idx]
                    if group_idx < len(layer_diversity):
                        diversity_factor = layer_diversity[group_idx].item()
                        group_avg_importance = group_avg_importance * (diversity_factor ** svd_diversity_alpha)

                all_groups.append({
                    'layer_idx': layer_idx,
                    'group_idx': group_idx,
                    'avg_importance': group_avg_importance,
                    'original_indices': group_original_indices
                })

        all_groups.sort(key=lambda x: x['avg_importance'], reverse=True)
        total_groups = len(all_groups)
        num_groups_to_keep = int(total_groups * (1 - current_target))
        min_groups = 0 if allow_full_block_pruning else 1
        num_groups_to_keep = max(min_groups, num_groups_to_keep)
        selected_groups = all_groups[:num_groups_to_keep]

        for layer_idx in range(num_layers):
            if layer_idx in skip_layers:
                masks[layer_idx] = torch.ones(intermediate_size)
            else:
                masks[layer_idx] = torch.zeros(intermediate_size)

        for group in selected_groups:
            layer_idx = group['layer_idx']
            original_indices = group['original_indices']
            masks[layer_idx][original_indices] = 1.0
    else:
        # ===== Layer-wise Pruning =====
        for layer_idx in range(num_layers):
            if layer_idx in skip_layers:
                masks[layer_idx] = torch.ones(intermediate_size)
                continue

            scores = taylor_scores[layer_idx]
            min_keep = 0 if allow_full_block_pruning else group_size
            num_to_keep = max(min_keep, int(intermediate_size * (1 - current_target)))
            num_to_keep = (num_to_keep // group_size) * group_size

            group_scores = []
            for group_idx in range(num_groups_per_layer):
                start_idx = group_idx * group_size
                end_idx = start_idx + group_size
                group_score = scores[start_idx:end_idx].mean().item()

                if diversity_scores is not None and layer_idx in diversity_scores:
                    layer_diversity = diversity_scores[layer_idx]
                    if group_idx < len(layer_diversity):
                        diversity_factor = layer_diversity[group_idx].item()
                        group_score = group_score * (diversity_factor ** svd_diversity_alpha)

                group_scores.append((group_score, group_idx))

            group_scores.sort(key=lambda x: x[0], reverse=True)
            num_groups_to_keep = num_to_keep // group_size
            keep_groups = set(g[1] for g in group_scores[:num_groups_to_keep])

            mask = torch.zeros(intermediate_size)
            for group_idx in keep_groups:
                start_idx = group_idx * group_size
                end_idx = start_idx + group_size
                mask[start_idx:end_idx] = 1.0
            masks[layer_idx] = mask

    return masks


# ==================== Mode Connectivity Helper Functions ====================

def _interpolate_masks(
    masks_a: Dict[int, torch.Tensor],
    masks_b: Dict[int, torch.Tensor],
    t: float
) -> Dict[int, torch.Tensor]:
    interpolated = {}
    for layer_idx in masks_a:
        mask_a = masks_a[layer_idx]
        mask_b = masks_b[layer_idx]
        interpolated[layer_idx] = (1 - t) * mask_a + t * mask_b
    return interpolated


def _evaluate_with_masks(
    model: nn.Module,
    dataloader,
    masks: Dict[int, torch.Tensor],
    mask_type: str = "ffn",
    num_samples: int = 5,
    device: str = "cuda"
) -> float:
    model.eval()
    total_loss = 0.0
    num_evaluated = 0

    hooks = []
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    for layer_idx in range(num_layers):
        if layer_idx not in masks:
            continue

        mask = masks[layer_idx].to(device)

        if mask_type == "ffn":
            mlp = model.model.layers[layer_idx].mlp
            down_proj = get_down_proj_module(mlp)
            if down_proj is not None:
                target = down_proj.base_layer if hasattr(down_proj, 'base_layer') else down_proj

                def make_ffn_hook(m):
                    def hook(module, input):
                        if input[0] is not None:
                            mask_casted = m.unsqueeze(0).unsqueeze(0).to(input[0].dtype)
                            return (input[0] * mask_casted,)
                        return None
                    return hook

                h = target.register_forward_pre_hook(make_ffn_hook(mask))
                hooks.append(h)

        elif mask_type == "head":
            attn = model.model.layers[layer_idx].self_attn
            o_proj = get_attn_output_proj(attn)
            if o_proj is not None:
                target = o_proj.base_layer if hasattr(o_proj, 'base_layer') else o_proj

                def make_head_hook(m, hd, nh):
                    def hook(module, input):
                        # o_proj input masking: [batch, seq, num_heads * head_dim]
                        # forward_pre_hook: (module, input) -> modified_input or None
                        if input[0] is not None:
                            inp = input[0]
                            # Expand head mask to full dimension
                            # mask shape: [num_heads] -> [num_heads * head_dim]
                            expanded_mask = m.repeat_interleave(hd)
                            # Cast mask to input dtype for mixed precision (fp16) support
                            expanded_mask = expanded_mask.unsqueeze(0).unsqueeze(0).to(inp.dtype)
                            return (inp * expanded_mask,)
                        return None
                    return hook

                h = target.register_forward_pre_hook(make_head_hook(mask, head_dim, num_heads))
                hooks.append(h)

        elif mask_type == "embedding":
            if layer_idx == 0:
                embed = model.model.embed_tokens
                target = embed.base_layer if hasattr(embed, 'base_layer') else embed

                def make_embed_hook(m):
                    def hook(module, input, output):
                        # Embedding output masking: [batch, seq, hidden_size]
                        if output is not None:
                            # Cast mask to output dtype for mixed precision (fp16) support
                            mask_casted = m.unsqueeze(0).unsqueeze(0).to(output.dtype)
                            return output * mask_casted
                    return hook

                h = target.register_forward_hook(make_embed_hook(mask))
                hooks.append(h)

    # Evaluation
    data_iter = iter(dataloader)
    with torch.no_grad():
        for _ in range(num_samples):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            total_loss += outputs.loss.item()
            num_evaluated += 1

    for h in hooks:
        h.remove()

    return total_loss / max(num_evaluated, 1)


def _measure_mode_connectivity_barrier(
    model: nn.Module,
    dataloader,
    masks_prev: Dict[int, torch.Tensor],
    masks_curr: Dict[int, torch.Tensor],
    mask_type: str = "ffn",
    n_points: int = 5,
    num_samples: int = 5,
    device: str = "cuda",
    verbose: bool = False
) -> Tuple[float, List[float]]:
    import numpy as np

    losses = []
    t_values = np.linspace(0, 1, n_points)

    for t in t_values:
        interpolated_masks = _interpolate_masks(masks_prev, masks_curr, t)
        loss = _evaluate_with_masks(
            model, dataloader, interpolated_masks,
            mask_type=mask_type, num_samples=num_samples, device=device
        )
        losses.append(loss)

        if verbose:
            print(f"    t={t:.2f}: loss={loss:.4f}")

    baseline = (losses[0] + losses[-1]) / 2
    barrier = max(losses) - baseline

    return barrier, losses


def _get_adaptive_step_size(
    base_step: float,
    barrier_history: List[float],
    min_step: float = 0.0125,  # 1.25%
    high_barrier_threshold: float = 1.0,
    medium_barrier_threshold: float = 0.5
) -> float:
    if len(barrier_history) == 0:
        return base_step

    recent_barrier = barrier_history[-1]

    if recent_barrier > high_barrier_threshold:
        return max(min_step, base_step / 4)
    elif recent_barrier > medium_barrier_threshold:
        return max(min_step, base_step / 2)
    else:
        return base_step


def _compute_connectivity_importance(
    model: nn.Module,
    dataloader,
    masks_prev: Dict[int, torch.Tensor],
    masks_curr: Dict[int, torch.Tensor],
    mask_type: str = "ffn",
    n_points: int = 5,
    num_samples: int = 5,
    device: str = "cuda",
    verbose: bool = False
) -> Dict[int, torch.Tensor]:
    import numpy as np

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    t_values = np.linspace(0, 1, n_points)
    losses = []
    gradients_per_point = []

    for t_idx, t in enumerate(t_values):
        interpolated_masks = _interpolate_masks(masks_prev, masks_curr, t)

        grad_masks = {}
        for layer_idx, mask in interpolated_masks.items():
            grad_masks[layer_idx] = mask.clone().detach().to(device).requires_grad_(True)

        hooks = []
        model.eval()

        for layer_idx in range(num_layers):
            if layer_idx not in grad_masks:
                continue

            mask = grad_masks[layer_idx]

            if mask_type == "ffn":
                mlp = model.model.layers[layer_idx].mlp
                down_proj = get_down_proj_module(mlp)
                if down_proj is not None:
                    target = down_proj.base_layer if hasattr(down_proj, 'base_layer') else down_proj

                    def make_ffn_hook(m, layer_i):
                        def hook(module, input):
                            if input[0] is not None:
                                mask_expanded = m.unsqueeze(0).unsqueeze(0).to(input[0].dtype)
                                return (input[0] * mask_expanded,)
                            return None
                        return hook

                    h = target.register_forward_pre_hook(make_ffn_hook(mask, layer_idx))
                    hooks.append(h)

            elif mask_type == "head":
                attn = model.model.layers[layer_idx].self_attn
                o_proj = get_attn_output_proj(attn)
                if o_proj is not None:
                    target = o_proj.base_layer if hasattr(o_proj, 'base_layer') else o_proj

                    def make_head_hook(m, hd):
                        def hook(module, input):
                            if input[0] is not None:
                                inp = input[0]
                                expanded_mask = m.repeat_interleave(hd)
                                expanded_mask = expanded_mask.unsqueeze(0).unsqueeze(0).to(inp.dtype)
                                return (inp * expanded_mask,)
                            return None
                        return hook

                    h = target.register_forward_pre_hook(make_head_hook(mask, head_dim))
                    hooks.append(h)

            elif mask_type == "embedding":
                if layer_idx == 0:
                    embed = model.model.embed_tokens
                    target = embed.base_layer if hasattr(embed, 'base_layer') else embed

                    def make_embed_hook(m):
                        def hook(module, input, output):
                            if output is not None:
                                mask_expanded = m.unsqueeze(0).unsqueeze(0).to(output.dtype)
                                return output * mask_expanded
                        return hook

                    h = target.register_forward_hook(make_embed_hook(mask))
                    hooks.append(h)

        # Forward pass with gradient
        total_loss = 0.0
        num_evaluated = 0
        data_iter = iter(dataloader)

        for _ in range(num_samples):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            total_loss += loss
            num_evaluated += 1

        avg_loss = total_loss / max(num_evaluated, 1)
        losses.append(avg_loss.item())

        # Backward to get gradients
        avg_loss.backward()

        layer_grads = {}
        for layer_idx, mask in grad_masks.items():
            if mask.grad is not None:
                layer_grads[layer_idx] = mask.grad.abs().detach().clone()
            else:
                layer_grads[layer_idx] = torch.zeros_like(mask)

        gradients_per_point.append(layer_grads)

        for h in hooks:
            h.remove()

        model.zero_grad()
        for mask in grad_masks.values():
            if mask.grad is not None:
                mask.grad.zero_()

        if verbose:
            print(f"    t={t:.2f}: loss={avg_loss.item():.4f}")

    peak_idx = np.argmax(losses)

    if verbose:
        print(f"  Peak loss at t={t_values[peak_idx]:.2f} (idx={peak_idx})")

    connectivity_importance = gradients_per_point[peak_idx]

    for layer_idx in connectivity_importance:
        grad = connectivity_importance[layer_idx]
        if grad.sum() > 0:
            connectivity_importance[layer_idx] = grad / (grad.mean() + 1e-8)

    return connectivity_importance


def _compute_taylor_masks_iterative(
    model: nn.Module,
    dataloader,
    target_ratio: float,
    step_size: float,
    start_ratio: float = 0.0,
    num_samples: int = 128,
    dimension_multiple: int = 1,
    device: str = "cuda",
    verbose: bool = True,
    ffn_mode: str = "down",
    importance_method: str = "ganda",
    enable_svd_diversity: bool = False,
    svd_diversity_method: str = "dominant",
    svd_diversity_alpha: float = 1.0,
    use_ntk_adjustment: bool = False,
    ntk_adjustment_alpha: float = 0.3,
    ntk_adjustment_mode: str = "previous",
    ntk_sensitivity_direction: str = "normal",
    ntk_target_mode: str = "default",
    ntk_granularity: str = "layer",
    ntk_unit_mode: str = "group",
    use_iterative_premasking: bool = False,
    use_gradient_checkpointing: bool = False,
    skip_first_layer: int = 0,
    skip_last_layer: int = 0,
    global_ffn_pruning: bool = True,
    use_mode_connectivity: bool = False,
    barrier_threshold: float = 0.5,
    barrier_n_points: int = 5,
    barrier_num_samples: int = 5,
    adaptive_step_size: bool = False,
    min_step_size: float = 0.0125,
    barrier_action: str = "subdivide",  # "subdivide", "warn", "skip"
    ntk_norm_method: str = "robust",
    allow_full_block_pruning: bool = False
) -> Tuple[Dict[int, torch.Tensor], Optional[Dict]]:
    num_layers = model.config.num_hidden_layers
    intermediate_size = getattr(model.config, 'intermediate_size', model.config.hidden_size * 4)

    skip_layers = set()
    if skip_first_layer > 0:
        skip_layers.update(range(min(skip_first_layer, num_layers)))
    if skip_last_layer > 0:
        skip_layers.update(range(max(0, num_layers - skip_last_layer), num_layers))

    base_step_size = step_size
    effective_start = min(start_ratio, target_ratio) if start_ratio > 0 else 0.0
    steps = []
    if effective_start > 0:
        steps.append(effective_start)
        current_ratio = effective_start + step_size
    else:
        current_ratio = step_size
    while current_ratio < target_ratio:
        steps.append(current_ratio)
        current_ratio += step_size
    steps.append(target_ratio)

    if verbose:
        print(f"\n{'='*60}")
        print(f"[Iterative Pruning] Starting iterative FFN pruning")
        mode_str = "Global" if global_ffn_pruning else "Layer-wise"
        print(f"[Iterative Pruning] Mode: {mode_str}")
        print(f"[Iterative Pruning] Target ratio: {target_ratio:.1%}")
        print(f"[Iterative Pruning] Step size: {step_size:.1%}")
        if effective_start > 0:
            print(f"[Iterative Pruning] Hybrid: one-shot to {effective_start:.1%}, then iterative")
        print(f"[Iterative Pruning] Steps: {len(steps)}")
        print(f"[Iterative Pruning] Process: 0% → {' → '.join([f'{s:.0%}' for s in steps])}")
        if skip_layers:
            print(f"[Iterative Pruning] Skip layers: {sorted(skip_layers)}")
        if use_iterative_premasking:
            print(f"[Iterative Pruning] Pre-masking: enabled (pruned FFN neurons output=0 during forward)")
        if use_ntk_adjustment:
            print(f"[Iterative Pruning] NTK Adjustment: enabled (alpha={ntk_adjustment_alpha}, mode={ntk_adjustment_mode}, direction={ntk_sensitivity_direction})")
        if use_mode_connectivity:
            print(f"[Iterative Pruning] Mode Connectivity: enabled (threshold={barrier_threshold}, n_points={barrier_n_points}, action={barrier_action})")
            if adaptive_step_size:
                print(f"[Iterative Pruning] Adaptive Step Size: enabled (min={min_step_size:.2%})")
        print(f"{'='*60}")

    sens_sign = -1.0 if ntk_sensitivity_direction == "inverse" else 1.0

    current_masks = {}
    for layer_idx in range(num_layers):
        current_masks[layer_idx] = torch.ones(intermediate_size)

    layer_sensitivities = None  # {layer_idx: sensitivity_score}

    connectivity_results = {
        'enabled': use_mode_connectivity,
        'barrier_history': [],
        'step_history': [],
        'losses_history': [],
        'adaptive_steps': [],
        'threshold_exceeded_count': 0
    } if use_mode_connectivity else None

    current_step_size = step_size
    barrier_history = []
    prev_masks = None

    for step_idx, current_target in enumerate(steps):
        if verbose:
            print(f"\n[Iterative Step {step_idx + 1}/{len(steps)}] Pruning to {current_target:.1%}")

        if use_iterative_premasking and step_idx > 0:
            calculator = TaylorImportanceCalculator(
                model, device, verbose=False,
                ffn_mode=ffn_mode,
                importance_method=importance_method,
                ffn_masks=current_masks,
                use_gradient_checkpointing=use_gradient_checkpointing
            )
        else:
            calculator = TaylorImportanceCalculator(
                model, device, verbose=False,
                ffn_mode=ffn_mode,
                importance_method=importance_method,
                use_gradient_checkpointing=use_gradient_checkpointing
            )
        taylor_scores = calculator.compute_taylor_scores(dataloader, num_samples)

        diversity_scores = None
        if enable_svd_diversity and svd_diversity_alpha > 0:
            diversity_scores = calculator.compute_ffn_diversity_scores(
                dataloader=dataloader,
                num_samples=num_samples,
                block_size=dimension_multiple,
                method=svd_diversity_method
            )

        del calculator
        model.zero_grad(set_to_none=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if not use_iterative_premasking:
            for layer_idx in range(num_layers):
                taylor_scores[layer_idx] = taylor_scores[layer_idx] * current_masks[layer_idx].to(taylor_scores[layer_idx].device)

        if use_ntk_adjustment and ntk_adjustment_mode == "current":
            if verbose:
                print(f"  [NTK-current] 1st pruning (temporary)")
            temp_masks = _apply_pruning_from_scores(
                taylor_scores=taylor_scores,
                current_target=current_target,
                num_layers=num_layers,
                intermediate_size=intermediate_size,
                dimension_multiple=dimension_multiple,
                global_ffn_pruning=global_ffn_pruning,
                skip_layers=skip_layers,
                diversity_scores=diversity_scores,
                svd_diversity_alpha=svd_diversity_alpha,
                allow_full_block_pruning=allow_full_block_pruning
            )

            if step_idx == 0:
                temp_masks = {i: torch.ones(intermediate_size) for i in range(num_layers)}

            if global_ffn_pruning and ntk_granularity == "layer":
                if verbose:
                    print(f"  [NTK-current] Computing layer-level NTK (Global mode)")
                current_sensitivities = _compute_ntk_sensitivity(
                    model=model,
                    dataloader=dataloader,
                    current_masks=temp_masks,
                    prev_masks=current_masks,
                    mask_type="ffn",
                    num_samples=num_samples,
                    device=device,
                    verbose=verbose,
                    ntk_target_mode=ntk_target_mode,
                    skip_layers=skip_layers
                )
                if verbose:
                    print(f"  [NTK-current] Adjusting scores (alpha={ntk_adjustment_alpha})")
                adjusted_scores = _apply_ntk_adjustment_to_scores(
                    scores=taylor_scores,
                    sensitivities=current_sensitivities,
                    alpha=ntk_adjustment_alpha,
                    sens_sign=sens_sign,
                    granularity="layer",
                    unit_mode=ntk_unit_mode,
                    dimension_multiple=dimension_multiple,
                    intermediate_size=intermediate_size,
                    num_layers=num_layers,
                    verbose=verbose,
                    ntk_norm_method=ntk_norm_method,
                    skip_layers=skip_layers
                )
            elif global_ffn_pruning and ntk_granularity == "unit":
                if verbose:
                    print(f"  [NTK-current] Computing per-unit NTK (Global+Unit mode, unit_mode={ntk_unit_mode})")
                current_sensitivities = _compute_ntk_sensitivity_per_unit(
                    model=model,
                    dataloader=dataloader,
                    current_masks=temp_masks,
                    prev_masks=current_masks,
                    mask_type="ffn",
                    num_samples=num_samples,
                    device=device,
                    verbose=verbose,
                    dimension_multiple=dimension_multiple,
                    ntk_target_mode=ntk_target_mode,
                    ntk_unit_mode=ntk_unit_mode,
                    normalize_per_layer=False,
                    skip_layers=skip_layers
                )
                if ntk_norm_method == "legacy":
                    all_sens = torch.cat([current_sensitivities[l] for l in range(num_layers) if l in current_sensitivities])
                    nonzero_mask = all_sens > 0
                    global_mean = all_sens[nonzero_mask].mean() if nonzero_mask.any() else all_sens.mean()
                    if global_mean > 0:
                        for layer_idx in current_sensitivities:
                            current_sensitivities[layer_idx] = current_sensitivities[layer_idx] / global_mean
                    if verbose:
                        print(f"  [NTK-current] Global normalization (legacy): original_mean={global_mean:.4f}")

                if verbose:
                    print(f"  [NTK-current] Adjusting scores per-unit (alpha={ntk_adjustment_alpha})")
                adjusted_scores = _apply_ntk_adjustment_to_scores(
                    scores=taylor_scores,
                    sensitivities=current_sensitivities,
                    alpha=ntk_adjustment_alpha,
                    sens_sign=sens_sign,
                    granularity="unit",
                    unit_mode=ntk_unit_mode,
                    dimension_multiple=dimension_multiple,
                    intermediate_size=intermediate_size,
                    num_layers=num_layers,
                    verbose=verbose,
                    ntk_norm_method=ntk_norm_method,
                    skip_layers=skip_layers
                )
            else:
                if verbose:
                    print(f"  [NTK-current] Computing per-unit NTK (Layer-wise mode, unit_mode={ntk_unit_mode})")
                current_sensitivities = _compute_ntk_sensitivity_per_unit(
                    model=model,
                    dataloader=dataloader,
                    current_masks=temp_masks,
                    prev_masks=current_masks,
                    mask_type="ffn",
                    num_samples=num_samples,
                    device=device,
                    verbose=verbose,
                    dimension_multiple=dimension_multiple,
                    ntk_target_mode=ntk_target_mode,
                    ntk_unit_mode=ntk_unit_mode,
                    skip_layers=skip_layers
                )
                if verbose:
                    print(f"  [NTK-current] Adjusting scores per-unit (alpha={ntk_adjustment_alpha})")
                adjusted_scores = _apply_ntk_adjustment_to_scores(
                    scores=taylor_scores,
                    sensitivities=current_sensitivities,
                    alpha=ntk_adjustment_alpha,
                    sens_sign=sens_sign,
                    granularity="unit",
                    unit_mode=ntk_unit_mode,
                    dimension_multiple=dimension_multiple,
                    intermediate_size=intermediate_size,
                    num_layers=num_layers,
                    verbose=verbose,
                    ntk_norm_method=ntk_norm_method,
                    skip_layers=skip_layers,
                    layer_wise=True
                )

            if verbose:
                print(f"  [NTK-current] 2nd pruning (final, no forward)")
            current_masks = _apply_pruning_from_scores(
                taylor_scores=adjusted_scores,
                current_target=current_target,
                num_layers=num_layers,
                intermediate_size=intermediate_size,
                dimension_multiple=dimension_multiple,
                global_ffn_pruning=global_ffn_pruning,
                skip_layers=skip_layers,
                diversity_scores=diversity_scores,
                svd_diversity_alpha=svd_diversity_alpha,
                allow_full_block_pruning=allow_full_block_pruning
            )

        else:
            if use_ntk_adjustment and layer_sensitivities is not None:
                if verbose:
                    print(f"  [NTK-previous] Applying sensitivity adjustment (alpha={ntk_adjustment_alpha})")

                prev_granularity = "layer" if (global_ffn_pruning and ntk_granularity == "layer") else "unit"
                taylor_scores = _apply_ntk_adjustment_to_scores(
                    scores=taylor_scores,
                    sensitivities=layer_sensitivities,
                    alpha=ntk_adjustment_alpha,
                    sens_sign=sens_sign,
                    granularity=prev_granularity,
                    unit_mode=ntk_unit_mode,
                    dimension_multiple=dimension_multiple,
                    intermediate_size=intermediate_size,
                    num_layers=num_layers,
                    verbose=verbose,
                    ntk_norm_method=ntk_norm_method,
                    skip_layers=skip_layers,
                    layer_wise=not global_ffn_pruning
                )

            current_masks = _apply_pruning_from_scores(
                taylor_scores=taylor_scores,
                current_target=current_target,
                num_layers=num_layers,
                intermediate_size=intermediate_size,
                dimension_multiple=dimension_multiple,
                global_ffn_pruning=global_ffn_pruning,
                skip_layers=skip_layers,
                diversity_scores=diversity_scores,
                svd_diversity_alpha=svd_diversity_alpha,
                allow_full_block_pruning=allow_full_block_pruning
            )

            if use_ntk_adjustment and step_idx < len(steps) - 1:
                if global_ffn_pruning and ntk_granularity == "layer":
                    layer_sensitivities = _compute_ntk_sensitivity(
                        model=model,
                        dataloader=dataloader,
                        current_masks=current_masks,
                        prev_masks=None if step_idx == 0 else prev_masks,
                        mask_type="ffn",
                        num_samples=num_samples,
                        device=device,
                        verbose=verbose,
                        ntk_target_mode=ntk_target_mode,
                        skip_layers=skip_layers
                    )
                elif global_ffn_pruning and ntk_granularity == "unit":
                    layer_sensitivities = _compute_ntk_sensitivity_per_unit(
                        model=model,
                        dataloader=dataloader,
                        current_masks=current_masks,
                        prev_masks=None if step_idx == 0 else prev_masks,
                        mask_type="ffn",
                        num_samples=num_samples,
                        device=device,
                        verbose=verbose,
                        dimension_multiple=dimension_multiple,
                        ntk_target_mode=ntk_target_mode,
                        ntk_unit_mode=ntk_unit_mode,
                        normalize_per_layer=False,
                        skip_layers=skip_layers
                    )
                    if ntk_norm_method == "legacy":
                        all_sens = torch.cat([layer_sensitivities[l] for l in range(num_layers) if l in layer_sensitivities])
                        nonzero_mask = all_sens > 0
                        global_mean = all_sens[nonzero_mask].mean() if nonzero_mask.any() else all_sens.mean()
                        if global_mean > 0:
                            for layer_idx in layer_sensitivities:
                                layer_sensitivities[layer_idx] = layer_sensitivities[layer_idx] / global_mean
                        if verbose:
                            print(f"  [NTK-previous] Global normalization (legacy): original_mean={global_mean:.4f}")
                else:
                    layer_sensitivities = _compute_ntk_sensitivity_per_unit(
                        model=model,
                        dataloader=dataloader,
                        current_masks=current_masks,
                        prev_masks=None if step_idx == 0 else prev_masks,
                        mask_type="ffn",
                        num_samples=num_samples,
                        device=device,
                        verbose=verbose,
                        dimension_multiple=dimension_multiple,
                        ntk_target_mode=ntk_target_mode,
                        ntk_unit_mode=ntk_unit_mode,
                        skip_layers=skip_layers
                    )


        if verbose:
            total_kept = sum(mask.sum().item() for mask in current_masks.values())
            total_neurons = num_layers * intermediate_size
            actual_ratio = 1 - (total_kept / total_neurons)
            print(f"  → Actual pruning: {actual_ratio:.1%} ({int(total_neurons - total_kept)}/{total_neurons} pruned)")

        if use_mode_connectivity and step_idx > 0:
            if verbose:
                print(f"  [Mode Connectivity] Measuring barrier between step {step_idx} and {step_idx + 1}...")

            prev_masks_for_barrier = prev_masks if prev_masks is not None else {
                l: torch.ones(intermediate_size) for l in range(num_layers)
            }

            barrier, losses = _measure_mode_connectivity_barrier(
                model=model,
                dataloader=dataloader,
                masks_prev=prev_masks_for_barrier,
                masks_curr=current_masks,
                mask_type="ffn",
                n_points=barrier_n_points,
                num_samples=barrier_num_samples,
                device=device,
                verbose=verbose
            )

            barrier_history.append(barrier)

            if connectivity_results is not None:
                connectivity_results['barrier_history'].append(barrier)
                connectivity_results['step_history'].append(current_target)
                connectivity_results['losses_history'].append(losses)

            if verbose:
                print(f"  [Mode Connectivity] Barrier: {barrier:.4f} (threshold: {barrier_threshold})")

            if barrier > barrier_threshold:
                if connectivity_results is not None:
                    connectivity_results['threshold_exceeded_count'] += 1

                if verbose:
                    print(f"  [Mode Connectivity] ⚠️ Barrier exceeded threshold!")

                if barrier_action == "subdivide" and adaptive_step_size:
                    new_step_size = _get_adaptive_step_size(
                        base_step=base_step_size,
                        barrier_history=barrier_history,
                        min_step=min_step_size
                    )
                    if verbose:
                        print(f"  [Mode Connectivity] Reducing step size: {current_step_size:.2%} → {new_step_size:.2%}")
                    current_step_size = new_step_size
                    if connectivity_results is not None:
                        connectivity_results['adaptive_steps'].append({
                            'step_idx': step_idx,
                            'old_step': current_step_size,
                            'new_step': new_step_size,
                            'barrier': barrier
                        })
                elif barrier_action == "warn":
                    print(f"  [Mode Connectivity] Warning: High barrier detected at step {step_idx + 1}")

        if use_mode_connectivity or use_ntk_adjustment:
            prev_masks = {k: v.clone() for k, v in current_masks.items()}

    if verbose:
        print(f"\n[Iterative Pruning] Final mask distribution:")
        for layer_idx in range(num_layers):
            kept = current_masks[layer_idx].sum().item()
            total = current_masks[layer_idx].numel()
            skip_marker = " (skip)" if layer_idx in skip_layers else ""
            print(f"  Layer {layer_idx}: {int(kept)}/{total} neurons kept ({kept/total:.1%}){skip_marker}")

        if use_mode_connectivity and connectivity_results is not None:
            print(f"\n[Mode Connectivity] Summary:")
            print(f"  Total steps measured: {len(connectivity_results['barrier_history'])}")
            if connectivity_results['barrier_history']:
                avg_barrier = sum(connectivity_results['barrier_history']) / len(connectivity_results['barrier_history'])
                max_barrier = max(connectivity_results['barrier_history'])
                print(f"  Average barrier: {avg_barrier:.4f}")
                print(f"  Max barrier: {max_barrier:.4f}")
                print(f"  Threshold exceeded: {connectivity_results['threshold_exceeded_count']} times")
            if adaptive_step_size and connectivity_results['adaptive_steps']:
                print(f"  Adaptive adjustments: {len(connectivity_results['adaptive_steps'])}")

    model.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return current_masks, connectivity_results


def print_importance_scores(
    scores: Dict[int, torch.Tensor],
    score_type: str = "FFN",
    top_k: int = 10
):
    print("\n" + "="*80)
    print(f"{score_type} Importance Scores")
    print("="*80)

    num_layers = len(scores)

    all_scores = []
    for layer_idx in range(num_layers):
        all_scores.extend(scores[layer_idx].tolist())

    all_tensor = torch.tensor(all_scores)
    print(f"Total elements: {len(all_scores)}")
    print(f"Global stats: min={all_tensor.min():.6f}, max={all_tensor.max():.6f}, "
          f"mean={all_tensor.mean():.6f}, std={all_tensor.std():.6f}")

    print(f"\nPer-layer statistics:")
    for layer_idx in range(num_layers):
        s = scores[layer_idx]
        print(f"  Layer {layer_idx:2d}: min={s.min():.6f}, max={s.max():.6f}, "
              f"mean={s.mean():.6f}, std={s.std():.6f}")

    # Top-K / Bottom-K
    indexed = [(score, layer_idx, idx)
               for layer_idx in range(num_layers)
               for idx, score in enumerate(scores[layer_idx].tolist())]
    indexed.sort(key=lambda x: x[0], reverse=True)

    print(f"\nTop-{top_k} most important:")
    for rank, (score, layer_idx, idx) in enumerate(indexed[:top_k], 1):
        print(f"  {rank:2d}. Layer {layer_idx:2d}, Index {idx:4d}: {score:.6f}")

    print(f"\nBottom-{top_k} least important (pruning candidates):")
    for rank, (score, layer_idx, idx) in enumerate(indexed[-top_k:], 1):
        print(f"  {rank:2d}. Layer {layer_idx:2d}, Index {idx:4d}: {score:.6f}")

    print("="*80)


def compute_importance_scores_only(
    model: nn.Module,
    dataloader,
    importance_metric: str = "cett",
    num_samples: int = 128,
    device: str = "cuda",
    verbose: bool = True,
    save_path: str = None,
    importance_method: str = "ganda"
) -> Tuple[Dict[int, torch.Tensor], Optional[Dict[int, torch.Tensor]]]:
    import json
    from datetime import datetime

    print(f"\n[Score Only] Computing {importance_metric.upper()} importance scores...")
    if importance_metric == "taylor":
        print(f"[Score Only] Importance method: {importance_method}")
    print(f"[Score Only] Samples: {num_samples}, Device: {device}")
    print(f"[Score Only] NOTE: No model update, gradient computation only")

    if importance_metric == "cett":
        calculator = CETTImportanceCalculator(model, device, verbose)
        ffn_scores = calculator.compute_all_ffn_importance(dataloader, num_samples)
        calculator2 = CETTImportanceCalculator(model, device, verbose)
        head_scores = calculator2.compute_all_head_importance(dataloader, num_samples)
    elif importance_metric == "taylor":
        calculator = TaylorImportanceCalculator(model, device, verbose, importance_method=importance_method)
        ffn_scores = calculator.compute_taylor_scores(dataloader, num_samples)
        head_scores = None
    else:
        raise ValueError(f"Unknown importance metric: {importance_metric}")

    print_importance_scores(ffn_scores, f"{importance_metric.upper()} FFN")
    if head_scores:
        print_importance_scores(head_scores, f"{importance_metric.upper()} Head")

    if save_path:
        from pathlib import Path
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        data = {
            "metric": importance_metric,
            "num_samples": num_samples,
            "timestamp": timestamp,
            "ffn_scores": {str(k): v.tolist() for k, v in ffn_scores.items()}
        }
        if head_scores:
            data["head_scores"] = {str(k): v.tolist() for k, v in head_scores.items()}

        json_file = save_dir / f"{importance_metric}_scores_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\n[Score Only] Saved to: {json_file}")

        torch_file = save_dir / f"{importance_metric}_scores_{timestamp}.pt"
        torch.save({"ffn_scores": ffn_scores, "head_scores": head_scores}, torch_file)
        print(f"[Score Only] Torch format: {torch_file}")

    return ffn_scores, head_scores


def compute_layerwise_joint_masks(
    model: nn.Module,
    dataloader,
    ffn_pruning_ratio: float,
    head_pruning_ratio: float,
    num_samples: int = 128,
    dimension_multiple: int = 1,
    device: str = "cuda",
    verbose: bool = True,
    ffn_mode: str = "down",
    head_mode: str = "o",
    importance_method: str = "ganda",
    global_ffn_pruning: bool = True,
    global_head_pruning: bool = True,
    joint_ffn_head: bool = True,
    skip_first_layer_head: int = 0,
    skip_last_layer_head: int = 0,
    skip_first_layer_ffn: int = 0,
    skip_last_layer_ffn: int = 0,
    allow_full_block_pruning: bool = False
) -> Tuple[Optional[Dict[int, torch.Tensor]], Optional[Dict[int, torch.Tensor]]]:
    num_layers = model.config.num_hidden_layers
    intermediate_size = getattr(model.config, 'intermediate_size', model.config.hidden_size * 4)
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    num_skip_ffn = skip_first_layer_ffn + skip_last_layer_ffn
    num_skip_head = skip_first_layer_head + skip_last_layer_head

    if num_skip_ffn > 0 and num_skip_ffn < num_layers:
        adjusted_ffn_ratio = ffn_pruning_ratio * num_layers / (num_layers - num_skip_ffn)
        if adjusted_ffn_ratio > 1.0:
            print(f"[WARNING] Adjusted FFN ratio {adjusted_ffn_ratio:.2%} > 100%, clamping to 100%")
            adjusted_ffn_ratio = 1.0
    else:
        adjusted_ffn_ratio = ffn_pruning_ratio

    if num_skip_head > 0 and num_skip_head < num_layers:
        adjusted_head_ratio = head_pruning_ratio * num_layers / (num_layers - num_skip_head)
        if adjusted_head_ratio > 1.0:
            print(f"[WARNING] Adjusted Head ratio {adjusted_head_ratio:.2%} > 100%, clamping to 100%")
            adjusted_head_ratio = 1.0
    else:
        adjusted_head_ratio = head_pruning_ratio

    mode_str = "Joint (FFN+Head together)" if joint_ffn_head else "Separate (FFN, Head separately)"
    if verbose:
        print(f"\n{'='*70}")
        print(f"[Layerwise Calibration] Starting layer-by-layer pruning with propagation")
        print(f"[Layerwise Calibration] Mode: {mode_str}")
        print(f"[Layerwise Calibration] Target FFN ratio: {ffn_pruning_ratio:.2%}, Head ratio: {head_pruning_ratio:.2%}")
        if num_skip_ffn > 0 or num_skip_head > 0:
            print(f"[Layerwise Calibration] Skip layers (FFN): first={skip_first_layer_ffn}, last={skip_last_layer_ffn} ({num_skip_ffn} layers)")
            print(f"[Layerwise Calibration] Skip layers (HEAD): first={skip_first_layer_head}, last={skip_last_layer_head} ({num_skip_head} layers)")
            print(f"[Layerwise Calibration] Adjusted FFN ratio: {adjusted_ffn_ratio:.2%}, Head ratio: {adjusted_head_ratio:.2%}")
        print(f"[Layerwise Calibration] Num samples: {num_samples}")
        print(f"[Layerwise Calibration] FFN mode: {ffn_mode}, Head mode: {head_mode}")
        print(f"[Layerwise Calibration] Importance method: {importance_method}")
        print(f"[Layerwise Calibration] Process: Layer 0 → Layer 1 → ... → Layer {num_layers-1}")
        print(f"{'='*70}")

    ffn_masks = {i: torch.ones(intermediate_size) for i in range(num_layers)}
    head_masks = {i: torch.ones(num_heads) for i in range(num_layers)}

    group_size = dimension_multiple
    num_groups_per_layer = intermediate_size // group_size

    skip_layers_ffn = set()
    if skip_first_layer_ffn > 0:
        skip_layers_ffn.update(range(min(skip_first_layer_ffn, num_layers)))
    if skip_last_layer_ffn > 0:
        skip_layers_ffn.update(range(max(0, num_layers - skip_last_layer_ffn), num_layers))

    skip_layers_head = set()
    if skip_first_layer_head > 0:
        skip_layers_head.update(range(min(skip_first_layer_head, num_layers)))
    if skip_last_layer_head > 0:
        skip_layers_head.update(range(max(0, num_layers - skip_last_layer_head), num_layers))

    for layer_idx in range(num_layers):
        should_skip_ffn = layer_idx in skip_layers_ffn
        should_skip_head = layer_idx in skip_layers_head

        if should_skip_ffn and should_skip_head:
            if verbose:
                print(f"\n[Layer {layer_idx}/{num_layers-1}] SKIPPED (boundary layer protection for both FFN and Head)")
                print(f"  FFN: {intermediate_size}/{intermediate_size} neurons kept (100.0%)")
                print(f"  Head: {num_heads}/{num_heads} heads kept (100.0%)")
            continue

        if verbose:
            print(f"\n[Layer {layer_idx}/{num_layers-1}] Computing importance with propagated activations...")

        current_ffn_masks = {i: ffn_masks[i] for i in range(layer_idx)}
        current_head_masks = {i: head_masks[i] for i in range(layer_idx)}

        calculator = TaylorImportanceCalculator(
            model, device, verbose=False,
            ffn_mode=ffn_mode,
            importance_method=importance_method,
            ffn_masks=current_ffn_masks if layer_idx > 0 else None,
            head_masks=current_head_masks if layer_idx > 0 else None
        )

        if joint_ffn_head:
            ffn_scores, head_scores = calculator.compute_joint_scores(dataloader, num_samples, head_mode)
        else:
            ffn_scores = calculator.compute_taylor_scores(dataloader, num_samples) if adjusted_ffn_ratio > 0 else {}
            head_scores = calculator.compute_taylor_head_scores(dataloader, num_samples, head_mode) if adjusted_head_ratio > 0 else {}

        del calculator
        gc.collect()
        torch.cuda.empty_cache()

        if adjusted_ffn_ratio > 0:
            if should_skip_ffn:
                ffn_masks[layer_idx] = torch.ones(intermediate_size)
                if verbose:
                    print(f"  FFN: {intermediate_size}/{intermediate_size} neurons kept (100.0%) [SKIPPED]")
            else:
                scores = ffn_scores[layer_idx]

                ffn_masks[layer_idx] = _create_ffn_mask_sorted_grouping(
                    scores, adjusted_ffn_ratio, group_size,
                    allow_full_block_pruning=allow_full_block_pruning
                )

                if verbose:
                    kept = int(ffn_masks[layer_idx].sum().item())
                    print(f"  FFN: {kept}/{intermediate_size} neurons kept ({kept/intermediate_size:.1%})")

        if adjusted_head_ratio > 0:
            if should_skip_head:
                head_masks[layer_idx] = torch.ones(num_heads)
                if verbose:
                    print(f"  Head: {num_heads}/{num_heads} heads kept (100.0%) [SKIPPED]")
            else:
                scores = head_scores[layer_idx]

                min_heads = 0 if allow_full_block_pruning else 1
                num_to_keep = max(min_heads, int(num_heads * (1 - adjusted_head_ratio)))
                if num_to_keep > 0:
                    _, top_indices = torch.topk(scores, num_to_keep)
                    mask = torch.zeros(num_heads)
                    mask[top_indices] = 1.0
                else:
                    mask = torch.zeros(num_heads)
                head_masks[layer_idx] = mask

                if verbose:
                    kept = int(mask.sum().item())
                    print(f"  Head: {kept}/{num_heads} heads kept ({kept/num_heads:.1%})")

    if verbose:
        print(f"\n{'='*70}")
        print(f"[Layerwise Calibration] Completed!")

        total_ffn = num_layers * intermediate_size
        kept_ffn = sum(m.sum().item() for m in ffn_masks.values())
        total_heads = num_layers * num_heads
        kept_heads = sum(m.sum().item() for m in head_masks.values())

        print(f"[Layerwise Calibration] Final FFN: {int(kept_ffn)}/{total_ffn} neurons kept ({kept_ffn/total_ffn:.1%})")
        print(f"[Layerwise Calibration] Final Heads: {int(kept_heads)}/{total_heads} heads kept ({kept_heads/total_heads:.1%})")
        print(f"{'='*70}")

    return ffn_masks if ffn_pruning_ratio > 0 else None, head_masks if head_pruning_ratio > 0 else None
