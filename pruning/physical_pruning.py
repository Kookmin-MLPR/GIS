
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import copy
import pickle
import os


class PhysicalPruner:

    def __init__(self, model, mask_tensors: Dict[str, torch.Tensor], mask_info: Dict):
        self.model = model
        self.mask_tensors = mask_tensors
        self.mask_info = mask_info
        self.pruned_model = None
        self.layer_compression_info = []

        print(f"[PhysicalPruner] Loaded {len(mask_tensors)} masks")

    def prune_linear_layer(
        self,
        layer: nn.Linear,
        input_mask: torch.Tensor = None,
        output_mask: torch.Tensor = None
    ) -> nn.Linear:
        weight = layer.weight.data.clone()
        bias = layer.bias.data.clone() if layer.bias is not None else None
        device = weight.device

        if output_mask is not None:
            output_indices = torch.where(output_mask.to(device) > 0)[0]
            weight = weight[output_indices]
            if bias is not None:
                bias = bias[output_indices]

        if input_mask is not None:
            input_indices = torch.where(input_mask.to(device) > 0)[0]
            weight = weight[:, input_indices]

        new_out_features = weight.size(0)
        new_in_features = weight.size(1)

        pruned_layer = nn.Linear(
            new_in_features,
            new_out_features,
            bias=(bias is not None),
            device=weight.device,
            dtype=weight.dtype
        )

        pruned_layer.weight.data = weight
        if bias is not None:
            pruned_layer.bias.data = bias

        return pruned_layer

    def prune_layer_norm(
        self,
        norm_layer,
        mask: torch.Tensor
    ):
        if mask is None:
            return norm_layer

        device = norm_layer.weight.device if hasattr(norm_layer, 'weight') and norm_layer.weight is not None else 'cuda'
        indices = torch.where(mask.to(device) > 0)[0]

        # Weight pruning
        if hasattr(norm_layer, 'weight') and norm_layer.weight is not None:
            new_weight = norm_layer.weight.data[indices].clone()
        else:
            new_weight = None

        if hasattr(norm_layer, 'bias') and norm_layer.bias is not None:
            new_bias = norm_layer.bias.data[indices].clone()
        else:
            new_bias = None

        norm_class = type(norm_layer)
        new_dim = len(indices)

        if 'RMS' in norm_class.__name__:
            pruned_norm = norm_class(
                new_dim,
                eps=norm_layer.variance_epsilon if hasattr(norm_layer, 'variance_epsilon') else 1e-6
            ).to(norm_layer.weight.device if hasattr(norm_layer, 'weight') else 'cuda')
        else:
            # LayerNorm
            pruned_norm = nn.LayerNorm(
                new_dim,
                eps=norm_layer.eps,
                elementwise_affine=norm_layer.elementwise_affine,
                device=norm_layer.weight.device if hasattr(norm_layer, 'weight') else 'cuda',
                dtype=norm_layer.weight.dtype if hasattr(norm_layer, 'weight') else torch.float16
            )

        if new_weight is not None:
            pruned_norm.weight.data = new_weight
        if new_bias is not None:
            pruned_norm.bias.data = new_bias

        return pruned_norm

    def prune_embedding_layer(
        self,
        embed_layer: nn.Embedding,
        embed_mask: torch.Tensor
    ) -> nn.Embedding:
        weight = embed_layer.weight.data.clone()

        indices = torch.where(embed_mask.to(weight.device) > 0)[0]
        weight = weight[:, indices]

        num_embeddings = weight.size(0)
        new_embedding_dim = weight.size(1)

        pruned_embed = nn.Embedding(
            num_embeddings,
            new_embedding_dim,
            padding_idx=embed_layer.padding_idx,
            device=weight.device,
            dtype=weight.dtype
        )

        pruned_embed.weight.data = weight

        return pruned_embed

    def get_mask(self, mask_name: str, device=None) -> torch.Tensor:
        mask = self.mask_tensors.get(mask_name)
        if mask is not None and device is not None:
            mask = mask.to(device)
        return mask

    def physically_prune_model(self) -> nn.Module:
        print("\n" + "=" * 70)
        print("Physical Pruning: Converting logical masks to physical pruning")
        print("=" * 70)

        print("[Pruning] Copying model...")
        pruned_model = copy.deepcopy(self.model)

        pruned_count = 0

        # Embedding layer pruning
        print("\n[Pruning] Pruning embedding layer...")
        embed_mask = None
        for key in self.mask_tensors.keys():
            if "embed_tokens.output" in key:
                embed_mask = self.mask_tensors[key]
                print(f"[Pruning] Found embedding mask: {key}")
                break

        if embed_mask is None:
            for key in self.mask_tensors.keys():
                if "layers.0.input_layernorm.output" in key:
                    embed_mask = self.mask_tensors[key]
                    print(f"[Pruning] Using fallback embedding mask: {key}")
                    break

        if embed_mask is not None and hasattr(pruned_model.model, 'embed_tokens'):
            pruned_model.model.embed_tokens = self.prune_embedding_layer(
                pruned_model.model.embed_tokens,
                embed_mask
            )
            new_dim = (embed_mask > 0).sum().item()
            print(f"[Pruning] Embedding: {pruned_model.config.hidden_size} → {new_dim} dims")

            pruned_model.config.hidden_size = new_dim

        num_layers = len(pruned_model.model.layers)
        print(f"\n[Pruning] Pruning {num_layers} transformer layers...")

        for layer_idx in range(num_layers):
            layer = pruned_model.model.layers[layer_idx]
            prefix = f"base_model.model.layers.{layer_idx}"

            layer_device = next(layer.parameters()).device

            layer_info = {
                'layer_idx': layer_idx,
                'original': {},
                'pruned': {}
            }

            if layer_idx % 8 == 0:
                print(f"[Pruning] Layer {layer_idx}/{num_layers}...")

            # LayerNorm pruning
            try:
                if embed_mask is not None:
                    # Input LayerNorm
                    if hasattr(layer, 'input_layernorm'):
                        layer.input_layernorm = self.prune_layer_norm(
                            layer.input_layernorm,
                            embed_mask
                        )

                    # Post-attention LayerNorm
                    if hasattr(layer, 'post_attention_layernorm'):
                        layer.post_attention_layernorm = self.prune_layer_norm(
                            layer.post_attention_layernorm,
                            embed_mask
                        )
            except Exception as e:
                print(f"[Warning] LayerNorm pruning failed for layer {layer_idx}: {e}")

            # Attention pruning
            try:
                attn = layer.self_attn

                qk_dim_mask = self.get_mask(f"{prefix}.self_attn.qk_dim", device=layer_device)
                if qk_dim_mask is None:
                    qk_dim_mask = self.get_mask(f"{prefix}.self_attn.qk_head_dim_mask", device=layer_device)
                q_output_mask_check = self.get_mask(f"{prefix}.self_attn.q_proj.output", device=layer_device)

                if layer_idx == 0 or layer_idx == num_layers - 1:
                    print(f"[DEBUG] Layer {layer_idx}: qk_dim_mask={'found' if qk_dim_mask is not None else 'NOT FOUND'}, "
                          f"q_output_mask={'found' if q_output_mask_check is not None else 'NOT FOUND'}")
                    if qk_dim_mask is not None:
                        print(f"[DEBUG] Layer {layer_idx}: qk_dim_mask shape={qk_dim_mask.shape}, sum={qk_dim_mask.sum().item()}")

                skip_attention = False
                if qk_dim_mask is not None and qk_dim_mask.sum().item() == 0:
                    skip_attention = True
                    print(f"  Layer {layer_idx}: Attention removed (head_dim=0)")
                elif q_output_mask_check is not None and q_output_mask_check.sum().item() == 0:
                    skip_attention = True
                    print(f"  Layer {layer_idx}: Attention removed (q_proj output=0)")

                if skip_attention:
                    layer.self_attn = None
                    layer.input_layernorm = None

                    layer_info['pruned']['skip_attention'] = True
                    layer_info['pruned']['qk_head_dim'] = 0
                    layer_info['pruned']['v_head_dim'] = 0

                else:
                    # Masks
                    # Head dimension masks (head pruning)
                    q_output_mask = self.get_mask(f"{prefix}.self_attn.q_proj.output", device=layer_device)
                    k_output_mask = self.get_mask(f"{prefix}.self_attn.k_proj.output", device=layer_device)
                    v_output_mask = self.get_mask(f"{prefix}.self_attn.v_proj.output", device=layer_device)
                    o_input_mask = self.get_mask(f"{prefix}.self_attn.o_proj.input", device=layer_device)
                    if o_input_mask is None:
                        o_input_mask = self.get_mask(f"{prefix}.self_attn.dense.input", device=layer_device)

                    # Dimension masks (dimension pruning within each head)
                    v_dim_mask = self.get_mask(f"{prefix}.self_attn.v_dim", device=layer_device)
                    if v_dim_mask is None:
                        v_dim_mask = self.get_mask(f"{prefix}.self_attn.v_head_dim_mask", device=layer_device)
                    # qk_dim_mask already loaded above

                    hidden_input_mask = embed_mask
                    hidden_output_mask = embed_mask

                    o_proj_module = getattr(attn, 'o_proj', None) or getattr(attn, 'dense', None)
                    o_proj_name = 'o_proj' if hasattr(attn, 'o_proj') else 'dense'

                    layer_info['original']['q_proj'] = (attn.q_proj.in_features, attn.q_proj.out_features)
                    layer_info['original']['k_proj'] = (attn.k_proj.in_features, attn.k_proj.out_features)
                    layer_info['original']['v_proj'] = (attn.v_proj.in_features, attn.v_proj.out_features)
                    layer_info['original'][o_proj_name] = (o_proj_module.in_features, o_proj_module.out_features)

                    # qk_dim_mask shape:
                    original_head_dim = getattr(
                        self.model.config, 'head_dim',
                        self.model.config.hidden_size // self.model.config.num_attention_heads
                    )
                    num_heads = attn.q_proj.out_features // original_head_dim
                    num_kv_heads = attn.k_proj.out_features // original_head_dim
                    expected_full_dim = num_heads * original_head_dim
                    expected_kv_dim = num_kv_heads * original_head_dim

                    if qk_dim_mask is not None and qk_dim_mask.numel() > 0:
                        if qk_dim_mask.numel() == original_head_dim:
                            pruned_head_dim = int(qk_dim_mask.sum().item())
                            qk_expanded_q_mask = qk_dim_mask.repeat(num_heads)
                            qk_expanded_kv_mask = qk_dim_mask.repeat(num_kv_heads)
                        elif qk_dim_mask.numel() == expected_full_dim:
                            qk_expanded_q_mask = qk_dim_mask
                            per_head_mask = qk_dim_mask[:original_head_dim]
                            qk_expanded_kv_mask = per_head_mask.repeat(num_kv_heads)
                            pruned_head_dim = int(per_head_mask.sum().item())
                        else:
                            print(f"  [Warning] Layer {layer_idx}: Unexpected qk_dim_mask shape {qk_dim_mask.shape}")
                            qk_expanded_q_mask = None
                            qk_expanded_kv_mask = None
                            pruned_head_dim = original_head_dim

                        if qk_expanded_q_mask is not None:
                            if layer_idx < 3 or layer_idx >= num_layers - 1:
                                print(f"  Layer {layer_idx}: Dimension pruning head_dim {original_head_dim} → {pruned_head_dim}")

                            if q_output_mask is not None:
                                q_output_mask = q_output_mask * qk_expanded_q_mask
                                k_output_mask = k_output_mask * qk_expanded_kv_mask if k_output_mask is not None else None
                            else:
                                q_output_mask = qk_expanded_q_mask
                                k_output_mask = qk_expanded_kv_mask

                    if v_dim_mask is not None and v_dim_mask.numel() > 0:
                        if v_dim_mask.numel() == original_head_dim:
                            v_expanded_q_mask = v_dim_mask.repeat(num_heads)
                            v_expanded_kv_mask = v_dim_mask.repeat(num_kv_heads)
                        elif v_dim_mask.numel() == expected_full_dim:
                            v_expanded_q_mask = v_dim_mask
                            per_head_mask = v_dim_mask[:original_head_dim]
                            v_expanded_kv_mask = per_head_mask.repeat(num_kv_heads)
                        else:
                            print(f"  [Warning] Layer {layer_idx}: Unexpected v_dim_mask shape {v_dim_mask.shape}")
                            v_expanded_q_mask = None
                            v_expanded_kv_mask = None

                        if v_expanded_kv_mask is not None:
                            if v_output_mask is not None:
                                v_output_mask = v_output_mask * v_expanded_kv_mask
                            else:
                                v_output_mask = v_expanded_kv_mask

                            if o_input_mask is not None:
                                o_input_mask = o_input_mask * v_expanded_q_mask
                            else:
                                o_input_mask = v_expanded_q_mask

                    if q_output_mask is not None:
                        attn.q_proj = self.prune_linear_layer(
                            attn.q_proj,
                            input_mask=hidden_input_mask,
                            output_mask=q_output_mask
                        )

                    if k_output_mask is not None:
                        attn.k_proj = self.prune_linear_layer(
                            attn.k_proj,
                            input_mask=hidden_input_mask,
                            output_mask=k_output_mask
                        )

                    if v_output_mask is not None:
                        attn.v_proj = self.prune_linear_layer(
                            attn.v_proj,
                            input_mask=hidden_input_mask,
                            output_mask=v_output_mask
                        )

                    # O projection pruning (LLaMA: o_proj, Phi-2: dense)
                    if o_input_mask is not None:
                        pruned_o = self.prune_linear_layer(
                            o_proj_module,
                            input_mask=o_input_mask,
                            output_mask=hidden_output_mask
                        )
                        setattr(attn, o_proj_name, pruned_o)

                    o_proj_module = getattr(attn, o_proj_name)
                    layer_info['pruned']['q_proj'] = (attn.q_proj.in_features, attn.q_proj.out_features)
                    layer_info['pruned']['k_proj'] = (attn.k_proj.in_features, attn.k_proj.out_features)
                    layer_info['pruned']['v_proj'] = (attn.v_proj.in_features, attn.v_proj.out_features)
                    layer_info['pruned'][o_proj_name] = (o_proj_module.in_features, o_proj_module.out_features)

                    if qk_dim_mask is not None:
                        qk_pruned_dim = int(qk_dim_mask.sum().item())
                        layer_info['pruned']['qk_head_dim'] = qk_pruned_dim

                    if v_dim_mask is not None:
                        v_pruned_dim = int(v_dim_mask.sum().item())
                        layer_info['pruned']['v_head_dim'] = v_pruned_dim

                    pruned_count += 1
            except Exception as e:
                print(f"[Warning] Attention pruning failed for layer {layer_idx}: {e}")

            # FFN/MLP pruning (LLaMA: gate_proj, up_proj, down_proj)
            try:
                mlp = layer.mlp

                up_output_mask = self.get_mask(f"{prefix}.mlp.up_proj.output", device=layer_device)
                gate_output_mask = self.get_mask(f"{prefix}.mlp.gate_proj.output", device=layer_device)
                down_input_mask = self.get_mask(f"{prefix}.mlp.down_proj.input", device=layer_device)

                if up_output_mask is None and gate_output_mask is None and down_input_mask is None:
                    up_output_mask = self.get_mask(f"{prefix}.mlp.fc1.output", device=layer_device)
                    down_input_mask = self.get_mask(f"{prefix}.mlp.fc2.input", device=layer_device)

                ffn_intermediate_mask = None
                if gate_output_mask is not None:
                    ffn_intermediate_mask = gate_output_mask
                elif up_output_mask is not None:
                    ffn_intermediate_mask = up_output_mask
                elif down_input_mask is not None:
                    ffn_intermediate_mask = down_input_mask

                skip_mlp = False
                if ffn_intermediate_mask is not None and ffn_intermediate_mask.sum().item() == 0:
                    skip_mlp = True
                    print(f"  Layer {layer_idx}: MLP removed (intermediate_size=0)")

                if skip_mlp:
                    layer.mlp = None
                    layer.post_attention_layernorm = None

                    layer_info['pruned']['skip_mlp'] = True
                    layer_info['pruned']['intermediate_size'] = 0
                    pruned_count += 1
                else:
                    hidden_input_mask = embed_mask
                    hidden_output_mask = embed_mask

                    # gate_proj pruning: Linear(hidden_size → intermediate_size)
                    if hasattr(mlp, 'gate_proj'):
                        gate_layer = mlp.gate_proj
                        layer_info['original']['gate_proj'] = (gate_layer.in_features, gate_layer.out_features)

                        if ffn_intermediate_mask is not None:
                            pruned_gate = self.prune_linear_layer(
                                gate_layer,
                                input_mask=hidden_input_mask,
                                output_mask=ffn_intermediate_mask
                            )
                            mlp.gate_proj = pruned_gate
                            layer_info['pruned']['gate_proj'] = (pruned_gate.in_features, pruned_gate.out_features)

                    # up_proj pruning: Linear(hidden_size → intermediate_size)
                    if hasattr(mlp, 'up_proj'):
                        up_layer = mlp.up_proj
                        layer_info['original']['up_proj'] = (up_layer.in_features, up_layer.out_features)

                        if ffn_intermediate_mask is not None:
                            pruned_up = self.prune_linear_layer(
                                up_layer,
                                input_mask=hidden_input_mask,
                                output_mask=ffn_intermediate_mask
                            )
                            mlp.up_proj = pruned_up
                            layer_info['pruned']['up_proj'] = (pruned_up.in_features, pruned_up.out_features)

                    # down_proj pruning: Linear(intermediate_size → hidden_size)
                    if hasattr(mlp, 'down_proj'):
                        down_layer = mlp.down_proj
                        layer_info['original']['down_proj'] = (down_layer.in_features, down_layer.out_features)

                        if ffn_intermediate_mask is not None:
                            pruned_down = self.prune_linear_layer(
                                down_layer,
                                input_mask=ffn_intermediate_mask,
                                output_mask=hidden_output_mask
                            )
                            mlp.down_proj = pruned_down
                            layer_info['pruned']['down_proj'] = (pruned_down.in_features, pruned_down.out_features)

                    # Phi-2: fc1 pruning: Linear(hidden_size → intermediate_size)
                    if hasattr(mlp, 'fc1'):
                        fc1_layer = mlp.fc1
                        layer_info['original']['fc1'] = (fc1_layer.in_features, fc1_layer.out_features)

                        if ffn_intermediate_mask is not None:
                            pruned_fc1 = self.prune_linear_layer(
                                fc1_layer,
                                input_mask=hidden_input_mask,
                                output_mask=ffn_intermediate_mask
                            )
                            mlp.fc1 = pruned_fc1
                            layer_info['pruned']['fc1'] = (pruned_fc1.in_features, pruned_fc1.out_features)

                    # Phi-2: fc2 pruning: Linear(intermediate_size → hidden_size)
                    if hasattr(mlp, 'fc2'):
                        fc2_layer = mlp.fc2
                        layer_info['original']['fc2'] = (fc2_layer.in_features, fc2_layer.out_features)

                        if ffn_intermediate_mask is not None:
                            pruned_fc2 = self.prune_linear_layer(
                                fc2_layer,
                                input_mask=ffn_intermediate_mask,
                                output_mask=hidden_output_mask
                            )
                            mlp.fc2 = pruned_fc2
                            layer_info['pruned']['fc2'] = (pruned_fc2.in_features, pruned_fc2.out_features)

                    if ffn_intermediate_mask is not None:
                        kept_neurons = int((ffn_intermediate_mask > 0).sum().item())
                        total_neurons = ffn_intermediate_mask.numel()
                        if kept_neurons < total_neurons:
                            pruned_count += 1
                            if layer_idx < 3 or layer_idx >= num_layers - 3:
                                print(f"[Pruning] Layer {layer_idx} FFN: {total_neurons} → {kept_neurons} neurons")

            except Exception as e:
                print(f"[Warning] FFN pruning failed for layer {layer_idx}: {e}")

            self.layer_compression_info.append(layer_info)

        print(f"\n[Pruning] Successfully pruned {pruned_count}/{num_layers} layers")

        # Final norm pruning (LLaMA: model.norm, Phi-2: model.final_layernorm)
        if embed_mask is not None:
            final_norm_name = None
            if hasattr(pruned_model.model, 'norm'):
                final_norm_name = 'norm'
            elif hasattr(pruned_model.model, 'final_layernorm'):
                final_norm_name = 'final_layernorm'

            if final_norm_name is not None:
                try:
                    final_norm = getattr(pruned_model.model, final_norm_name)
                    pruned_norm = self.prune_layer_norm(final_norm, embed_mask)
                    setattr(pruned_model.model, final_norm_name, pruned_norm)
                    print(f"[Pruning] Final norm ({final_norm_name}) pruned")
                except Exception as e:
                    print(f"[Warning] Final norm pruning failed: {e}")

        if embed_mask is not None and hasattr(pruned_model, 'lm_head'):
            try:
                pruned_model.lm_head = self.prune_linear_layer(
                    pruned_model.lm_head,
                    input_mask=embed_mask,
                    output_mask=None
                )
                print(f"[Pruning] LM head pruned")
            except Exception as e:
                print(f"[Warning] LM head pruning failed: {e}")

        self.pruned_model = pruned_model
        return pruned_model

    def get_model_size(self, model) -> Dict:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        memory_mb = total_params * 2 / (1024 ** 2)

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "memory_mb": memory_mb,
            "memory_gb": memory_mb / 1024
        }

    def print_layer_compression_details(self):
        print("\n" + "=" * 70)
        print("Layer-wise Compression Details")
        print("=" * 70)

        sample_indices = list(range(min(3, len(self.layer_compression_info)))) + \
                        list(range(max(0, len(self.layer_compression_info) - 3), len(self.layer_compression_info)))
        sample_indices = sorted(set(sample_indices))

        for idx in sample_indices:
            if idx >= len(self.layer_compression_info):
                continue

            info = self.layer_compression_info[idx]
            print(f"\nLayer {info['layer_idx']}:")

            # Attention
            if 'q_proj' in info['original'] and 'q_proj' in info['pruned']:
                orig = info['original']['q_proj']
                prun = info['pruned']['q_proj']
                print(f"  Q-proj: {orig[0]}x{orig[1]} → {prun[0]}x{prun[1]}")

            # Attention output (LLaMA: o_proj, Phi-2: dense)
            for o_name in ['o_proj', 'dense']:
                if o_name in info['original'] and o_name in info['pruned']:
                    orig = info['original'][o_name]
                    prun = info['pruned'][o_name]
                    print(f"  {o_name:6s}: {orig[0]}x{orig[1]} → {prun[0]}x{prun[1]}")

            # FFN (LLaMA: gate/up/down_proj, Phi-2: fc1/fc2)
            for proj_name in ['gate_proj', 'up_proj', 'down_proj', 'fc1', 'fc2']:
                if proj_name in info['original'] and proj_name in info['pruned']:
                    orig = info['original'][proj_name]
                    prun = info['pruned'][proj_name]
                    reduction = (1 - (prun[0] * prun[1]) / (orig[0] * orig[1])) * 100
                    label = proj_name.ljust(6)
                    print(f"  {label}: {orig[0]}x{orig[1]} → {prun[0]}x{prun[1]} ({reduction:.1f}% reduced)")

        if len(self.layer_compression_info) > 6:
            print(f"\n... ({len(self.layer_compression_info) - 6} more layers)")

    def compare_model_sizes(self):
        original_size = self.get_model_size(self.model)
        pruned_size = self.get_model_size(self.pruned_model)

        print("\n" + "=" * 70)
        print("Model Size Comparison")
        print("=" * 70)

        print(f"\nOriginal Model:")
        print(f"  Total Parameters:  {original_size['total_params']:,}")
        print(f"  Memory (FP16):     {original_size['memory_gb']:.2f} GB")

        print(f"\nPhysically Pruned Model:")
        print(f"  Total Parameters:  {pruned_size['total_params']:,}")
        print(f"  Memory (FP16):     {pruned_size['memory_gb']:.2f} GB")

        print(f"\nReduction:")
        param_reduction = (1 - pruned_size['total_params'] / original_size['total_params']) * 100
        memory_reduction = (1 - pruned_size['memory_gb'] / original_size['memory_gb']) * 100

        print(f"  Parameters:        {param_reduction:.2f}%")
        print(f"  Memory:            {memory_reduction:.2f}%")

        self.print_layer_compression_details()

        return {
            "original": original_size,
            "pruned": pruned_size,
            "reduction": {
                "params_percent": param_reduction,
                "memory_percent": memory_reduction
            }
        }


def load_masks(mask_path: str) -> Dict[str, torch.Tensor]:
    with open(mask_path, 'rb') as f:
        mask_tensors = pickle.load(f)

    print(f"[Load] Loaded {len(mask_tensors)} masks from {mask_path}")

    return mask_tensors
