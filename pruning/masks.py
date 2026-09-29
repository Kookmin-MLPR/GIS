
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np

class MaskManager:

    def __init__(self, model, config):
        """
        Args:
            model: Student model
            config: PruningConfig
        """
        self.model = model
        self.config = config

        model_type = getattr(model.config, 'model_type', 'llama')
        self.is_phi = (model_type == 'phi')

        self.llm_masks = {}

        self.lora_masks = {}

        self.masks = {}

    def _ffn_mask_keys(self, prefix: str):
        if self.is_phi:
            return {
                'up_outputs': [f"{prefix}.mlp.fc1.output"],
                'down_input': f"{prefix}.mlp.fc2.input",
                'up_inputs': [f"{prefix}.mlp.fc1.input"],
                'down_output': f"{prefix}.mlp.fc2.output",
            }
        else:
            return {
                'up_outputs': [f"{prefix}.mlp.gate_proj.output", f"{prefix}.mlp.up_proj.output"],
                'down_input': f"{prefix}.mlp.down_proj.input",
                'up_inputs': [f"{prefix}.mlp.gate_proj.input", f"{prefix}.mlp.up_proj.input"],
                'down_output': f"{prefix}.mlp.down_proj.output",
            }

    def _attn_output_key(self, prefix: str):
        if self.is_phi:
            return f"{prefix}.self_attn.dense.input"
        else:
            return f"{prefix}.self_attn.o_proj.input"

    def _attn_output_out_key(self, prefix: str):
        if self.is_phi:
            return f"{prefix}.self_attn.dense.output"
        else:
            return f"{prefix}.self_attn.o_proj.output"

    def create_initial_masks(self) -> Dict[str, torch.Tensor]:
        print("\n[Mask] Creating initial all-ones masks (no pruning)...")

        all_masks = {}
        num_layers = self.model.config.num_hidden_layers
        num_heads = self.model.config.num_attention_heads
        num_kv_heads = getattr(self.model.config, 'num_key_value_heads', num_heads)
        head_dim = self.model.config.hidden_size // num_heads
        hidden_size = self.model.config.hidden_size
        intermediate_size = self.model.config.intermediate_size

        for layer_idx in range(num_layers):
            prefix = f"base_model.model.layers.{layer_idx}"

            # Head masks (all ones)
            expanded_q_mask = torch.ones(num_heads * head_dim, device="cuda")
            expanded_kv_mask = torch.ones(num_kv_heads * head_dim, device="cuda")

            all_masks[f"{prefix}.self_attn.q_proj.output"] = expanded_q_mask.clone()
            all_masks[f"{prefix}.self_attn.k_proj.output"] = expanded_kv_mask.clone()
            all_masks[f"{prefix}.self_attn.v_proj.output"] = expanded_kv_mask.clone()
            all_masks[self._attn_output_key(prefix)] = expanded_q_mask.clone()

            # FFN masks (all ones)
            ffn_mask = torch.ones(intermediate_size, device="cuda")
            fkeys = self._ffn_mask_keys(prefix)
            for k in fkeys['up_outputs']:
                all_masks[k] = ffn_mask.clone()
            all_masks[fkeys['down_input']] = ffn_mask.clone()

        # Embedding mask (all ones)
        all_masks["embedding"] = torch.ones(hidden_size, device="cuda")

        # Dimension masks (all ones) if enabled
        if self.config.enable_dimension_pruning:
            for layer_idx in range(num_layers):
                prefix = f"base_model.model.layers.{layer_idx}"
                dim_mask = torch.ones(head_dim, device="cuda")
                all_masks[f"{prefix}.self_attn.qk_dim"] = dim_mask.clone()
                all_masks[f"{prefix}.self_attn.v_dim"] = dim_mask.clone()

        self.masks = all_masks
        print(f"[Mask] Created {len(all_masks)} initial masks")
        return all_masks

    def create_all_masks(
        self,
        importance_calculator,
        head_ratio: float = None,
        embedding_ratio: float = None,
        ffn_ratio: float = None,
        dimension_ratio: float = None,
        dataloader=None,
        calibration_samples: int = 64
    ) -> Dict[str, torch.Tensor]:
        head_ratio = head_ratio if head_ratio is not None else self.config.head_pruning_ratio
        embedding_ratio = embedding_ratio if embedding_ratio is not None else self.config.embedding_pruning_ratio
        ffn_ratio = ffn_ratio if ffn_ratio is not None else self.config.ffn_pruning_ratio
        dimension_ratio = dimension_ratio if dimension_ratio is not None else (
            self.config.dimension_pruning_ratio if self.config.enable_dimension_pruning else 0.0
        )

        print(f"\n[Mask] Creating masks with ratios:")
        print(f"  - Head: {head_ratio:.2f} ({'Global' if self.config.global_head_pruning else 'Layer-wise'}) [importance: {self.config.head_importance_mode}]")
        print(f"  - Embedding: {embedding_ratio:.2f}")
        print(f"  - FFN: {ffn_ratio:.2f} ({'Global' if self.config.global_ffn_pruning else 'Layer-wise'}) [importance: {self.config.ffn_importance_mode}]")
        if self.config.enable_dimension_pruning:
            print(f"  - Dimension: {dimension_ratio:.2f} (group_size: {self.config.dimension_group_size})")

        all_masks = {}

        if head_ratio > 0.0:
            head_importance = importance_calculator.compute_head_importance(mode=self.config.head_importance_mode)
            if self.config.global_head_pruning:
                head_masks = self.create_head_masks_global(head_importance, head_ratio)
            else:
                head_masks = self.create_head_masks(head_importance, head_ratio)
            all_masks.update(head_masks)
        else:
            head_masks = {}
            for layer_idx in range(self.model.config.num_hidden_layers):
                num_heads = self.model.config.num_attention_heads
                num_kv_heads = getattr(self.model.config, 'num_key_value_heads', num_heads)
                head_dim = self.model.config.hidden_size // num_heads

                q_head_mask = torch.ones(num_heads, device="cuda")
                expanded_q_mask = q_head_mask.repeat_interleave(head_dim)

                kv_head_mask = torch.ones(num_kv_heads, device="cuda")
                expanded_kv_mask = kv_head_mask.repeat_interleave(head_dim)

                layer_prefix = f"base_model.model.layers.{layer_idx}"
                prefix = f"{layer_prefix}.self_attn"
                head_masks[f"{prefix}.q_proj.output"] = expanded_q_mask.clone()
                head_masks[f"{prefix}.k_proj.output"] = expanded_kv_mask.clone()
                head_masks[f"{prefix}.v_proj.output"] = expanded_kv_mask.clone()
                head_masks[self._attn_output_key(layer_prefix)] = expanded_q_mask.clone()
            all_masks.update(head_masks)

        if embedding_ratio > 0.0:
            embedding_importance = importance_calculator.compute_embedding_importance()
            embedding_mask = self.create_embedding_masks(
                embedding_importance,
                embedding_ratio
            )
        else:
            hidden_size = self.model.config.hidden_size
            embedding_mask = torch.ones(hidden_size, device="cuda")

        all_masks["base_model.model.embed_tokens.output"] = embedding_mask.clone()

        for layer_idx in range(self.model.config.num_hidden_layers):
            prefix = f"base_model.model.layers.{layer_idx}"

            all_masks[f"{prefix}.input_layernorm.output"] = embedding_mask.clone()
            if not self.is_phi:
                all_masks[f"{prefix}.post_attention_layernorm.output"] = embedding_mask.clone()

            all_masks[f"{prefix}.self_attn.q_proj.input"] = embedding_mask.clone()
            all_masks[f"{prefix}.self_attn.k_proj.input"] = embedding_mask.clone()
            all_masks[f"{prefix}.self_attn.v_proj.input"] = embedding_mask.clone()
            all_masks[self._attn_output_out_key(prefix)] = embedding_mask.clone()

            fkeys = self._ffn_mask_keys(prefix)
            for k in fkeys['up_inputs']:
                all_masks[k] = embedding_mask.clone()
            all_masks[fkeys['down_output']] = embedding_mask.clone()

        # LLaMA: model.norm, Phi-2: model.final_layernorm
        if self.is_phi:
            all_masks["base_model.model.final_layernorm.output"] = embedding_mask.clone()
        else:
            all_masks["base_model.model.norm.output"] = embedding_mask.clone()

        if ffn_ratio > 0.0:
            if self.config.global_ffn_pruning:
                ffn_importance_all = {}
                for layer_idx in range(self.model.config.num_hidden_layers):
                    up_importance, down_importance = importance_calculator.compute_ffn_importance(layer_idx)
                    ffn_importance_all[layer_idx] = (up_importance, down_importance)

                diversity_scores = None
                if self.config.enable_svd_diversity and dataloader is not None:
                    if hasattr(importance_calculator, 'compute_ffn_diversity_scores'):
                        print(f"\n[Mask] Computing SVD diversity scores...")
                        diversity_scores = importance_calculator.compute_ffn_diversity_scores(
                            dataloader=dataloader,
                            num_samples=calibration_samples,
                            block_size=self.config.ffn_dimension_multiple,
                            method=self.config.svd_diversity_method
                        )
                    else:
                        print(f"[Mask] Warning: importance_calculator does not support SVD diversity")

                ffn_masks = self.create_ffn_masks_global(ffn_importance_all, ffn_ratio, diversity_scores)
                all_masks.update(ffn_masks)
            else:
                for layer_idx in range(self.model.config.num_hidden_layers):
                    up_importance, down_importance = importance_calculator.compute_ffn_importance(layer_idx)
                    up_mask, down_mask = self.create_ffn_masks(
                        up_importance,
                        down_importance,
                        ffn_ratio
                    )

                    layer_prefix = f"base_model.model.layers.{layer_idx}"
                    fkeys = self._ffn_mask_keys(layer_prefix)
                    for k in fkeys['up_outputs']:
                        all_masks[k] = up_mask.clone()
                    all_masks[fkeys['down_input']] = down_mask
        else:
            intermediate_size = self.model.config.intermediate_size
            hidden_size = self.model.config.hidden_size
            for layer_idx in range(self.model.config.num_hidden_layers):
                up_mask = torch.ones(intermediate_size, device="cuda")
                down_mask = torch.ones(intermediate_size, device="cuda")

                layer_prefix = f"base_model.model.layers.{layer_idx}"
                fkeys = self._ffn_mask_keys(layer_prefix)
                for k in fkeys['up_outputs']:
                    all_masks[k] = up_mask.clone()
                all_masks[fkeys['down_input']] = down_mask

        if self.config.enable_dimension_pruning and dimension_ratio > 0.0:
            qk_dim_importance, v_dim_importance = importance_calculator.compute_dimension_importance()
            dim_masks = self.create_dimension_masks_global(
                qk_dim_importance,
                v_dim_importance,
                dimension_ratio,
                self.config.dimension_group_size
            )
            all_masks.update(dim_masks)

            all_masks = self.combine_head_and_dim_masks(all_masks)

        all_masks = self.share_masks(all_masks)

        self.llm_masks, self.lora_masks = self.split_dual_masks(all_masks)

        self.masks = all_masks

        if hasattr(importance_calculator, 'importance_scores'):
            del importance_calculator.importance_scores

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return all_masks

    def create_head_masks(
        self,
        head_importance: Dict[int, torch.Tensor],
        head_pruning_ratio: float
    ) -> Dict[str, torch.Tensor]:
        head_masks = {}
        num_heads = self.model.config.num_attention_heads
        num_kv_heads = getattr(self.model.config, 'num_key_value_heads', num_heads)
        head_dim = self.model.config.hidden_size // num_heads
        num_queries_per_kv = num_heads // num_kv_heads
        is_gqa = (num_kv_heads != num_heads)

        for layer_idx, scores in head_importance.items():
            if is_gqa:
                num_groups_keep = max(
                    1,
                    int(num_kv_heads * (1 - head_pruning_ratio))
                )

                group_scores = scores[::num_queries_per_kv]

                _, groups_to_keep = torch.topk(group_scores, num_groups_keep, largest=True)

                head_mask = torch.zeros(num_heads, device=scores.device)
                for group_idx in groups_to_keep:
                    q_start = group_idx * num_queries_per_kv
                    q_end = q_start + num_queries_per_kv
                    head_mask[q_start:q_end] = 1.0

                kv_head_mask = torch.zeros(num_kv_heads, device=scores.device)
                kv_head_mask[groups_to_keep] = 1.0

                expanded_q_mask = head_mask.repeat_interleave(head_dim)
                expanded_kv_mask = kv_head_mask.repeat_interleave(head_dim)

            else:
                num_keep = max(
                    self.config.min_heads_per_layer,
                    int(num_heads * (1 - head_pruning_ratio))
                )

                _, heads_to_keep = torch.topk(scores, num_keep, largest=True)

                head_mask = torch.zeros(num_heads, device=scores.device)
                head_mask[heads_to_keep] = 1.0

                if head_mask.sum() == 0:
                    if self.config.allow_full_block_pruning:
                        print(f"[WARNING] Layer {layer_idx}: All heads pruned")
                    else:
                        best_head = torch.argmax(scores)
                        head_mask[best_head] = 1.0

                expanded_q_mask = head_mask.repeat_interleave(head_dim)
                expanded_kv_mask = expanded_q_mask.clone()

            layer_prefix = f"base_model.model.layers.{layer_idx}"
            prefix = f"{layer_prefix}.self_attn"
            head_masks[f"{prefix}.q_proj.output"] = expanded_q_mask.clone()
            head_masks[f"{prefix}.k_proj.output"] = expanded_kv_mask.clone()
            head_masks[f"{prefix}.v_proj.output"] = expanded_kv_mask.clone()
            head_masks[self._attn_output_key(layer_prefix)] = expanded_q_mask.clone()

        return head_masks

    def create_head_masks_global(
        self,
        head_importance: Dict[int, torch.Tensor],
        head_pruning_ratio: float
    ) -> Dict[str, torch.Tensor]:
        num_heads = self.model.config.num_attention_heads
        num_kv_heads = getattr(self.model.config, 'num_key_value_heads', num_heads)
        head_dim = self.model.config.hidden_size // num_heads
        num_queries_per_kv = num_heads // num_kv_heads
        is_gqa = (num_kv_heads != num_heads)
        num_layers = len(head_importance)

        if is_gqa:
            group_importance_list = []
            for layer_idx in sorted(head_importance.keys()):
                scores = head_importance[layer_idx]
                group_scores = scores[::num_queries_per_kv]
                group_importance_list.append(group_scores)

            all_group_importance = torch.cat(group_importance_list)
            total_groups = all_group_importance.size(0)

            num_groups_to_keep = max(
                num_layers,
                int(total_groups * (1 - head_pruning_ratio))
            )

            sorted_scores, sorted_global_indices = torch.sort(all_group_importance, descending=True)
            top_global_indices = sorted_global_indices[:num_groups_to_keep]

            top_layer_indices = top_global_indices // num_kv_heads
            top_group_indices = top_global_indices % num_kv_heads

            print(f"[Mask] Global Head Pruning (GQA): {total_groups} groups → {num_groups_to_keep} groups kept")
            print(f"[Mask] Each group = {num_queries_per_kv} Q heads + 1 KV head")

            head_masks = {}
            device = all_group_importance.device

            for layer_idx in range(num_layers):
                layer_mask = (top_layer_indices == layer_idx)
                selected_groups = top_group_indices[layer_mask]

                head_mask = torch.zeros(num_heads, device=device)
                kv_head_mask = torch.zeros(num_kv_heads, device=device)

                if selected_groups.numel() > 0:
                    for group_idx in selected_groups:
                        q_start = group_idx * num_queries_per_kv
                        q_end = q_start + num_queries_per_kv
                        head_mask[q_start:q_end] = 1.0
                        kv_head_mask[group_idx] = 1.0

                if head_mask.sum() == 0:
                    if self.config.allow_full_block_pruning:
                        print(f"[WARNING] Layer {layer_idx}: All groups pruned (global)")
                    else:
                        layer_start = layer_idx * num_kv_heads
                        layer_end = layer_start + num_kv_heads
                        layer_importance = all_group_importance[layer_start:layer_end]
                        best_group_idx = torch.argmax(layer_importance)
                        q_start = best_group_idx * num_queries_per_kv
                        q_end = q_start + num_queries_per_kv
                        head_mask[q_start:q_end] = 1.0
                        kv_head_mask[best_group_idx] = 1.0

                expanded_q_mask = head_mask.repeat_interleave(head_dim)
                expanded_kv_mask = kv_head_mask.repeat_interleave(head_dim)

                layer_prefix = f"base_model.model.layers.{layer_idx}"
                prefix = f"{layer_prefix}.self_attn"
                head_masks[f"{prefix}.q_proj.output"] = expanded_q_mask.clone()
                head_masks[f"{prefix}.k_proj.output"] = expanded_kv_mask.clone()
                head_masks[f"{prefix}.v_proj.output"] = expanded_kv_mask.clone()
                head_masks[self._attn_output_key(layer_prefix)] = expanded_q_mask.clone()

            del all_group_importance, sorted_scores, sorted_global_indices

        else:
            importance_list = []
            for layer_idx in sorted(head_importance.keys()):
                importance_list.append(head_importance[layer_idx])

            all_head_importance = torch.cat(importance_list)
            num_heads_per_layer = importance_list[0].size(0)
            total_heads = all_head_importance.size(0)

            sorted_scores, sorted_global_indices = torch.sort(all_head_importance, descending=True)

            num_heads_to_keep = max(
                self.config.min_heads_per_layer * num_layers,
                int(total_heads * (1 - head_pruning_ratio))
            )

            top_global_indices = sorted_global_indices[:num_heads_to_keep]
            top_layer_indices = top_global_indices // num_heads_per_layer
            top_head_indices = top_global_indices % num_heads_per_layer

            print(f"[Mask] Global Head Pruning (MHA): {total_heads} heads → {num_heads_to_keep} heads kept")

            head_masks = {}
            device = all_head_importance.device

            for layer_idx in range(num_layers):
                layer_mask = (top_layer_indices == layer_idx)
                selected_heads = top_head_indices[layer_mask]

                head_mask = torch.zeros(num_heads_per_layer, device=device)
                if selected_heads.numel() > 0:
                    head_mask[selected_heads] = 1.0

                if head_mask.sum() == 0:
                    if self.config.allow_full_block_pruning:
                        print(f"[WARNING] Layer {layer_idx}: All heads pruned (global)")
                    else:
                        layer_start = layer_idx * num_heads_per_layer
                        layer_end = layer_start + num_heads_per_layer
                        layer_importance = all_head_importance[layer_start:layer_end]
                        best_head_idx = torch.argmax(layer_importance)
                        head_mask[best_head_idx] = 1.0

                expanded_q_mask = head_mask.repeat_interleave(head_dim)
                expanded_kv_mask = expanded_q_mask.clone()

                layer_prefix = f"base_model.model.layers.{layer_idx}"
                prefix = f"{layer_prefix}.self_attn"
                head_masks[f"{prefix}.q_proj.output"] = expanded_q_mask.clone()
                head_masks[f"{prefix}.k_proj.output"] = expanded_kv_mask.clone()
                head_masks[f"{prefix}.v_proj.output"] = expanded_kv_mask.clone()
                head_masks[self._attn_output_key(layer_prefix)] = expanded_q_mask.clone()

            del all_head_importance, sorted_scores, sorted_global_indices
            del top_global_indices, top_layer_indices, top_head_indices

        return head_masks

    def create_embedding_masks(
        self,
        importance_scores: torch.Tensor,
        embedding_pruning_ratio: float
    ) -> torch.Tensor:
        embed_dim = importance_scores.size(0)
        group_size = 64

        sorted_scores, sorted_indices = torch.sort(
            importance_scores,
            descending=True
        )

        num_groups = embed_dim // group_size
        num_groups_to_keep = int(num_groups * (1 - embedding_pruning_ratio))
        num_groups_to_keep = max(1, num_groups_to_keep)

        num_keep = num_groups_to_keep * group_size
        dims_to_keep = sorted_indices[:num_keep]

        embed_mask = torch.zeros(embed_dim, device=importance_scores.device)
        embed_mask[dims_to_keep] = 1.0

        print(f"[Mask] Embedding: {embed_dim} dims → {num_keep} dims kept ({num_groups_to_keep} groups, sorted)")

        del sorted_scores, sorted_indices, dims_to_keep

        return embed_mask

    def create_ffn_masks(
        self,
        up_importance: torch.Tensor,
        down_importance: torch.Tensor,
        ffn_pruning_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_dim = up_importance.size(0)  # intermediate_size (11008)
        group_size = self.config.ffn_dimension_multiple  # 64

        if self.config.ffn_importance_mode == "down":
            importance_to_use = down_importance
        else:
            importance_to_use = up_importance

        sorted_scores, sorted_indices = torch.sort(
            importance_to_use,
            descending=True
        )

        num_groups = hidden_dim // group_size
        num_groups_to_keep = int(num_groups * (1 - ffn_pruning_ratio))
        min_groups = 0 if self.config.allow_full_block_pruning else 1
        num_groups_to_keep = max(min_groups, num_groups_to_keep)

        num_keep = num_groups_to_keep * group_size
        dims_to_keep = sorted_indices[:num_keep]

        up_out_mask = torch.zeros(hidden_dim, device=up_importance.device)
        if num_keep > 0:
            up_out_mask[dims_to_keep] = 1.0

        down_in_mask = up_out_mask.clone()

        print(f"[Mask] FFN: {hidden_dim} dims → {num_keep} dims kept ({num_groups_to_keep} groups, sorted)")

        del sorted_scores, sorted_indices, dims_to_keep

        return up_out_mask, down_in_mask

    def create_ffn_masks_global(
        self,
        ffn_importance_all: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        ffn_pruning_ratio: float,
        diversity_scores: Dict[int, torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        group_size = self.config.ffn_dimension_multiple  # 64
        num_layers = len(ffn_importance_all)
        device = next(iter(ffn_importance_all.values()))[0].device

        use_diversity = (
            diversity_scores is not None and
            self.config.enable_svd_diversity and
            self.config.svd_diversity_alpha > 0
        )

        if use_diversity:
            print(f"[Mask] SVD Diversity enabled: method={self.config.svd_diversity_method}, alpha={self.config.svd_diversity_alpha}")

        all_groups = []  # (layer_idx, group_idx, group_avg_importance, original_indices)

        for layer_idx in sorted(ffn_importance_all.keys()):
            up_importance, down_importance = ffn_importance_all[layer_idx]
            hidden_dim = up_importance.size(0)

            if self.config.ffn_importance_mode == "down":
                importance_to_use = down_importance
            else:
                importance_to_use = up_importance

            sorted_scores, sorted_indices = torch.sort(importance_to_use, descending=True)

            num_groups = hidden_dim // group_size
            for group_idx in range(num_groups):
                start_idx = group_idx * group_size
                end_idx = start_idx + group_size

                group_original_indices = sorted_indices[start_idx:end_idx]

                group_avg_importance = sorted_scores[start_idx:end_idx].mean().item()

                if use_diversity and layer_idx in diversity_scores:
                    layer_diversity = diversity_scores[layer_idx]
                    if group_idx < len(layer_diversity):
                        diversity_factor = layer_diversity[group_idx].item()
                        alpha = self.config.svd_diversity_alpha
                        group_avg_importance = group_avg_importance * (diversity_factor ** alpha)

                all_groups.append({
                    'layer_idx': layer_idx,
                    'group_idx': group_idx,
                    'avg_importance': group_avg_importance,
                    'original_indices': group_original_indices
                })

        all_groups.sort(key=lambda x: x['avg_importance'], reverse=True)

        total_groups = len(all_groups)
        num_groups_to_keep = int(total_groups * (1 - ffn_pruning_ratio))
        min_groups = 0 if self.config.allow_full_block_pruning else 1
        num_groups_to_keep = max(min_groups, num_groups_to_keep)

        selected_groups = all_groups[:num_groups_to_keep]

        total_dims = num_layers * next(iter(ffn_importance_all.values()))[0].size(0)
        num_dims_kept = num_groups_to_keep * group_size

        print(f"[Mask] Global FFN Pruning (Group-based): {total_dims} dims → {num_dims_kept} dims kept ({num_groups_to_keep} groups)")

        ffn_masks = {}
        hidden_dim = next(iter(ffn_importance_all.values()))[0].size(0)

        layer_groups = {i: [] for i in range(num_layers)}
        for group in selected_groups:
            layer_groups[group['layer_idx']].append(group)

        for layer_idx in range(num_layers):
            groups = layer_groups[layer_idx]

            up_out_mask = torch.zeros(hidden_dim, device=device)

            for group in groups:
                original_indices = group['original_indices']
                up_out_mask[original_indices] = 1.0

            num_kept = int(up_out_mask.sum().item())
            num_groups_kept = num_kept // group_size
            print(f"[Mask] Layer {layer_idx}: {hidden_dim} dims → {num_kept} dims ({num_groups_kept} groups)")

            down_in_mask = up_out_mask.clone()

            layer_prefix = f"base_model.model.layers.{layer_idx}"
            fkeys = self._ffn_mask_keys(layer_prefix)
            for k in fkeys['up_outputs']:
                ffn_masks[k] = up_out_mask.clone()
            ffn_masks[fkeys['down_input']] = down_in_mask

        return ffn_masks

    def share_masks(self, masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        shared_masks = {}
        num_heads = self.model.config.num_attention_heads
        num_kv_heads = getattr(self.model.config, 'num_key_value_heads', num_heads)
        head_dim = self.model.config.hidden_size // num_heads
        num_queries_per_kv = num_heads // num_kv_heads

        for layer_idx in range(self.model.config.num_hidden_layers):
            layer_prefix = f"base_model.model.layers.{layer_idx}"
            prefix = f"{layer_prefix}.self_attn"

            if self.config.share_qk_mask:
                q_mask = masks.get(f"{prefix}.q_proj.output")
                if q_mask is not None:
                    if num_kv_heads != num_heads:
                        q_head_mask = q_mask.view(num_heads, head_dim)[:, 0]
                        kv_head_mask = torch.zeros(num_kv_heads, device=q_mask.device)
                        for kv_idx in range(num_kv_heads):
                            q_start = kv_idx * num_queries_per_kv
                            q_end = q_start + num_queries_per_kv
                            if q_head_mask[q_start:q_end].sum() > 0:
                                kv_head_mask[kv_idx] = 1.0
                        shared_masks[f"{prefix}.k_proj.output"] = kv_head_mask.repeat_interleave(head_dim)
                    else:
                        shared_masks[f"{prefix}.k_proj.output"] = q_mask.clone()

            if self.config.share_v_proj_mask and num_kv_heads == num_heads:
                v_mask = masks.get(f"{prefix}.v_proj.output")
                if v_mask is not None:
                    shared_masks[self._attn_output_key(layer_prefix)] = v_mask.clone()

            if self.config.share_fc_mask:
                fkeys = self._ffn_mask_keys(layer_prefix)
                # Get the first up_output key to find the mask
                up_mask = masks.get(fkeys['up_outputs'][0])
                if up_mask is not None:
                    shared_masks[fkeys['down_input']] = up_mask.clone()


        masks.update(shared_masks)
        return masks

    def split_dual_masks(
        self,
        masks: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        llm_masks = {}
        lora_masks = {}

        for name, mask in masks.items():
            llm_masks[name] = mask.clone()
            lora_masks[name] = mask.clone()

        return llm_masks, lora_masks

    def get_mask_sparsity(self) -> Dict[str, float]:
        sparsity = {}

        for name, mask in self.masks.items():
            total = mask.numel()
            zeros = (mask == 0).sum().item()
            sparsity[name] = zeros / total if total > 0 else 0.0

        return sparsity

    def get_overall_sparsity(self) -> float:
        ffn_total = 0
        ffn_pruned = 0
        head_total = 0
        head_pruned = 0
        embedding_sparsity = 0.0

        for name, mask in self.masks.items():
            if '.q_proj.output' in name or \
               '.k_proj.output' in name or \
               '.v_proj.output' in name or \
               '.o_proj.input' in name or \
               '.dense.input' in name:
                head_total += mask.numel()
                head_pruned += (mask == 0).sum().item()

            elif '.up_proj.output' in name or \
                 '.gate_proj.output' in name or \
                 '.down_proj.input' in name or \
                 '.fc1.output' in name or \
                 '.fc2.input' in name:
                ffn_total += mask.numel()
                ffn_pruned += (mask == 0).sum().item()

            elif ('.up_proj.input' in name or '.fc1.input' in name) and embedding_sparsity == 0.0:
                embedding_sparsity = (mask == 0).sum().item() / mask.numel()

        total_params = ffn_total + head_total
        base_pruned = ffn_pruned + head_pruned

        remaining = total_params - base_pruned
        additional_pruned = remaining * embedding_sparsity

        total_pruned = base_pruned + additional_pruned

        return total_pruned / total_params if total_params > 0 else 0.0

    def create_dimension_masks_global(
        self,
        qk_dim_importance: Dict[int, torch.Tensor],
        v_dim_importance: Dict[int, torch.Tensor],
        dimension_pruning_ratio: float,
        dimension_group_size: int = 16
    ) -> Dict[str, torch.Tensor]:
        num_layers = len(qk_dim_importance)
        head_dim = qk_dim_importance[0].size(0)
        num_groups_per_layer = head_dim // dimension_group_size
        device = qk_dim_importance[0].device if qk_dim_importance[0].is_cuda else "cuda"

        all_groups = []  # (layer_idx, group_idx, combined_avg_importance, qk_original_indices, v_original_indices)

        for layer_idx in sorted(qk_dim_importance.keys()):
            qk_score = qk_dim_importance[layer_idx].to(device)
            v_score = v_dim_importance[layer_idx].to(device)

            qk_sorted_scores, qk_sorted_indices = torch.sort(qk_score, descending=True)

            v_sorted_scores, v_sorted_indices = torch.sort(v_score, descending=True)

            for group_idx in range(num_groups_per_layer):
                start_idx = group_idx * dimension_group_size
                end_idx = start_idx + dimension_group_size

                qk_group_indices = qk_sorted_indices[start_idx:end_idx]
                qk_group_avg = qk_sorted_scores[start_idx:end_idx].mean().item()

                v_group_indices = v_sorted_indices[start_idx:end_idx]
                v_group_avg = v_sorted_scores[start_idx:end_idx].mean().item()

                combined_avg = (qk_group_avg + v_group_avg) / 2

                all_groups.append({
                    'layer_idx': layer_idx,
                    'group_idx': group_idx,
                    'combined_avg': combined_avg,
                    'qk_original_indices': qk_group_indices,
                    'v_original_indices': v_group_indices
                })

        all_groups.sort(key=lambda x: x['combined_avg'], reverse=True)

        total_groups = len(all_groups)
        num_groups_to_keep = int(total_groups * (1 - dimension_pruning_ratio))
        num_groups_to_keep = max(num_layers, num_groups_to_keep)

        selected_groups = all_groups[:num_groups_to_keep]

        print(f"[Mask] Global Dimension Pruning: {total_groups} groups → {num_groups_to_keep} groups kept")
        print(f"[Mask] Dimension group size: {dimension_group_size}, head_dim: {head_dim}")

        dimension_masks = {}

        layer_groups = {i: [] for i in range(num_layers)}
        for group in selected_groups:
            layer_groups[group['layer_idx']].append(group)

        for layer_idx in range(num_layers):
            groups = layer_groups[layer_idx]

            qk_dim_mask = torch.zeros(head_dim, device=device)
            v_dim_mask = torch.zeros(head_dim, device=device)

            for group in groups:
                qk_dim_mask[group['qk_original_indices']] = 1.0
                v_dim_mask[group['v_original_indices']] = 1.0

            num_qk_kept = int(qk_dim_mask.sum().item())
            num_v_kept = int(v_dim_mask.sum().item())
            print(f"[Mask] Layer {layer_idx}: QK dims {num_qk_kept}/{head_dim}, V dims {num_v_kept}/{head_dim}")

            prefix = f"base_model.model.layers.{layer_idx}.self_attn"
            dimension_masks[f"{prefix}.qk_head_dim_mask"] = qk_dim_mask
            dimension_masks[f"{prefix}.v_head_dim_mask"] = v_dim_mask

        return dimension_masks

    def combine_head_and_dim_masks(
        self,
        masks: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        num_heads = self.model.config.num_attention_heads
        num_kv_heads = getattr(self.model.config, 'num_key_value_heads', num_heads)
        head_dim = self.model.config.hidden_size // num_heads
        num_layers = self.model.config.num_hidden_layers

        updated_masks = masks.copy()

        for layer_idx in range(num_layers):
            layer_prefix = f"base_model.model.layers.{layer_idx}"
            prefix = f"{layer_prefix}.self_attn"

            qk_dim_mask_key = f"{prefix}.qk_head_dim_mask"
            v_dim_mask_key = f"{prefix}.v_head_dim_mask"

            if qk_dim_mask_key not in masks or v_dim_mask_key not in masks:
                continue

            qk_dim_mask = masks[qk_dim_mask_key]
            v_dim_mask = masks[v_dim_mask_key]

            # Determine target device from model parameters
            target_device = next(self.model.parameters()).device

            # Move dimension masks to target device
            qk_dim_mask = qk_dim_mask.to(target_device)
            v_dim_mask = v_dim_mask.to(target_device)

            q_mask_key = f"{prefix}.q_proj.output"
            if q_mask_key in masks:
                q_mask = masks[q_mask_key]  # [num_heads * head_dim]
                q_mask_reshaped = q_mask.view(num_heads, head_dim)
                q_mask_combined = q_mask_reshaped * qk_dim_mask.unsqueeze(0)
                updated_masks[q_mask_key] = q_mask_combined.view(-1)

            k_mask_key = f"{prefix}.k_proj.output"
            if k_mask_key in masks:
                k_mask = masks[k_mask_key]  # [num_kv_heads * head_dim]
                k_mask_reshaped = k_mask.view(num_kv_heads, head_dim)
                k_mask_combined = k_mask_reshaped * qk_dim_mask.unsqueeze(0)
                updated_masks[k_mask_key] = k_mask_combined.view(-1)

            v_mask_key = f"{prefix}.v_proj.output"
            if v_mask_key in masks:
                v_mask = masks[v_mask_key]  # [num_kv_heads * head_dim]
                v_mask_reshaped = v_mask.view(num_kv_heads, head_dim)
                v_mask_combined = v_mask_reshaped * v_dim_mask.unsqueeze(0)
                updated_masks[v_mask_key] = v_mask_combined.view(-1)

            o_mask_key = self._attn_output_key(layer_prefix)
            if o_mask_key in masks:
                o_mask = masks[o_mask_key]
                o_mask_reshaped = o_mask.view(num_heads, head_dim)
                o_mask_combined = o_mask_reshaped * v_dim_mask.unsqueeze(0)
                updated_masks[o_mask_key] = o_mask_combined.view(-1)

        return updated_masks
