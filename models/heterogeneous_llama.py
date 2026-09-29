
import os
import json
import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaAttention,
    LlamaRMSNorm,
    LlamaModel,
    LlamaRotaryEmbedding,
    create_causal_mask
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from typing import Optional, Tuple, Union, List


class HeterogeneousLlamaConfig(LlamaConfig):
    model_type = "heterogeneous_llama"

    def __init__(self, layer_configs=None, **kwargs):
        super().__init__(**kwargs)
        # layer_configs: {layer_idx: {'intermediate_size': ..., 'num_attention_heads': ..., 'head_dim': ...}}
        self.layer_configs = layer_configs or {}


class HeterogeneousLlamaMLP(LlamaMLP):
    def __init__(self, config, intermediate_size=None):
        if intermediate_size is not None:
            original_size = config.intermediate_size
            config.intermediate_size = intermediate_size

        super().__init__(config)

        if intermediate_size is not None:
            config.intermediate_size = original_size


class HeterogeneousLlamaAttention(LlamaAttention):
    def __init__(self, config, layer_idx=None, num_attention_heads=None, head_dim=None):
        original_heads = config.num_attention_heads
        original_kv_heads = config.num_key_value_heads
        original_head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)

        if num_attention_heads is not None:
            config.num_attention_heads = num_attention_heads
            config.num_key_value_heads = num_attention_heads

        if head_dim is not None:
            config.head_dim = head_dim

        super().__init__(config, layer_idx=layer_idx)

        self.actual_head_dim = head_dim if head_dim is not None else original_head_dim
        self.original_head_dim = original_head_dim

        config.num_attention_heads = original_heads
        config.num_key_value_heads = original_kv_heads
        if head_dim is not None:
            config.head_dim = original_head_dim

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        if position_embeddings is not None:
            cos, sin = position_embeddings
            rotary_dim = cos.size(-1)
            if rotary_dim > self.actual_head_dim:
                cos = cos[..., :self.actual_head_dim]
                sin = sin[..., :self.actual_head_dim]
                position_embeddings = (cos, sin)
            elif rotary_dim < self.actual_head_dim:
                # q_embed = (q * cos) + (rotate_half(q) * sin)
                pad_size = self.actual_head_dim - rotary_dim
                cos_pad = torch.ones(*cos.shape[:-1], pad_size, device=cos.device, dtype=cos.dtype)
                cos = torch.cat([cos, cos_pad], dim=-1)
                sin_pad = torch.zeros(*sin.shape[:-1], pad_size, device=sin.device, dtype=sin.dtype)
                sin = torch.cat([sin, sin_pad], dim=-1)
                position_embeddings = (cos, sin)

        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )


class MLPOnlyDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx=None, intermediate_size=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.mlp = HeterogeneousLlamaMLP(config, intermediate_size=intermediate_size)

        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        residual = hidden_states

        # LayerNorm → MLP
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        # Residual connection
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (None,)  # No attention weights

        if use_cache:
            outputs += (None,)  # No KV cache

        return outputs


class AttentionOnlyDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx=None, num_attention_heads=None, head_dim=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = HeterogeneousLlamaAttention(
            config=config,
            layer_idx=layer_idx,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim
        )

        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        residual = hidden_states

        # LayerNorm → Attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        # Residual connection
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (past_key_value,)

        return outputs


class IdentityDecoderLayer(nn.Module):
    def __init__(self, config=None, layer_idx=None):
        super().__init__()
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (None,)  # No attention weights

        if use_cache:
            outputs += (None,)  # No KV cache

        return outputs


class HeterogeneousLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx=None):
        intermediate_size = None
        num_attention_heads = None
        head_dim = None

        if (layer_idx is not None and
            hasattr(config, 'layer_configs') and
            str(layer_idx) in config.layer_configs):
            layer_config = config.layer_configs[str(layer_idx)]
            intermediate_size = layer_config.get('intermediate_size', None)
            num_attention_heads = layer_config.get('num_attention_heads', None)
            head_dim = layer_config.get('head_dim', None)

        super(LlamaDecoderLayer, self).__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = HeterogeneousLlamaAttention(
            config=config,
            layer_idx=layer_idx,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim
        )

        self.mlp = HeterogeneousLlamaMLP(config, intermediate_size=intermediate_size)

        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )


class HeterogeneousLlamaModel(LlamaModel):
    """
    Custom LlamaModel that properly handles mixed layer types (HeterogeneousLlamaDecoderLayer and MLPOnlyDecoderLayer)
    """

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Handle cache
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Create causal mask using the new API
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds

        # Create position embeddings
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Decoder layers
        for decoder_layer in self.layers[:self.config.num_hidden_layers]:
            # Call decoder layer
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            # Extract hidden_states from layer output (handle both tensor and tuple)
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class HeterogeneousLlamaForCausalLM(LlamaForCausalLM):
    config_class = HeterogeneousLlamaConfig

    def __init__(self, config):
        super().__init__(config)

        # Replace the standard LlamaModel with HeterogeneousLlamaModel
        # This ensures proper handling of mixed layer types
        old_model = self.model
        self.model = HeterogeneousLlamaModel(config)
        # Copy weights from old model
        self.model.load_state_dict(old_model.state_dict(), strict=False)

        if hasattr(config, 'layer_configs') and config.layer_configs:
            from torch.nn import ModuleList

            temp_layer_configs = {}
            for layer_idx, layer_cfg in config.layer_configs.items():
                skip_attention = layer_cfg.get('skip_attention', False)
                skip_mlp = layer_cfg.get('skip_mlp', False)
                head_dim = layer_cfg.get('head_dim', None)
                intermediate_size = layer_cfg.get('intermediate_size', None)

                if skip_attention or (head_dim is not None and head_dim == 0):
                    if skip_mlp or (intermediate_size is not None and intermediate_size == 0):
                        temp_layer_configs[layer_idx] = {
                            'skip_attention': True,
                            'skip_mlp': True
                        }
                    else:
                        temp_layer_configs[layer_idx] = {
                            'intermediate_size': layer_cfg.get('intermediate_size'),
                            'skip_attention': True
                        }
                elif skip_mlp or (intermediate_size is not None and intermediate_size == 0):
                    temp_layer_configs[layer_idx] = {
                        'num_attention_heads': layer_cfg.get('num_attention_heads'),
                        'head_dim': layer_cfg.get('head_dim'),
                        'skip_mlp': True
                    }
                else:
                    temp_layer_configs[layer_idx] = layer_cfg.copy()

            original_layer_configs = config.layer_configs
            config.layer_configs = temp_layer_configs

            layers = []
            for i in range(config.num_hidden_layers):
                layers.append(HeterogeneousLlamaDecoderLayer(config, layer_idx=i))

            self.model.layers = ModuleList(layers)

            config.layer_configs = original_layer_configs

    def replace_with_specialized_layers(self):
        if not hasattr(self.config, 'layer_configs') or not self.config.layer_configs:
            return []

        replaced_layers = []
        for i in range(self.config.num_hidden_layers):
            layer_config = self.config.layer_configs.get(str(i), {})
            head_dim = layer_config.get('head_dim', None)
            skip_attention = layer_config.get('skip_attention', False)
            skip_mlp = layer_config.get('skip_mlp', False)
            intermediate_size = layer_config.get('intermediate_size', None)
            num_attention_heads = layer_config.get('num_attention_heads', None)

            is_skip_attention = skip_attention or (head_dim is not None and head_dim == 0)
            is_skip_mlp = skip_mlp or (intermediate_size is not None and intermediate_size == 0)

            if is_skip_attention and is_skip_mlp:
                new_layer = IdentityDecoderLayer(config=self.config, layer_idx=i)

                old_layer = self.model.layers[i]
                old_params = list(old_layer.parameters())
                if old_params:
                    device = old_params[0].device
                    new_layer = new_layer.to(device=device)

                self.model.layers[i] = new_layer
                replaced_layers.append(i)

            elif is_skip_attention:
                old_layer = self.model.layers[i]

                new_layer = MLPOnlyDecoderLayer(
                    self.config,
                    layer_idx=i,
                    intermediate_size=intermediate_size
                )

                new_layer.post_attention_layernorm.load_state_dict(
                    old_layer.post_attention_layernorm.state_dict()
                )
                new_layer.mlp.load_state_dict(old_layer.mlp.state_dict())

                device = next(old_layer.parameters()).device
                dtype = next(old_layer.parameters()).dtype
                new_layer = new_layer.to(device=device, dtype=dtype)

                self.model.layers[i] = new_layer
                replaced_layers.append(i)

            elif is_skip_mlp:
                old_layer = self.model.layers[i]

                new_layer = AttentionOnlyDecoderLayer(
                    self.config,
                    layer_idx=i,
                    num_attention_heads=num_attention_heads,
                    head_dim=head_dim
                )

                new_layer.input_layernorm.load_state_dict(
                    old_layer.input_layernorm.state_dict()
                )
                new_layer.self_attn.load_state_dict(old_layer.self_attn.state_dict())

                device = next(old_layer.parameters()).device
                dtype = next(old_layer.parameters()).dtype
                new_layer = new_layer.to(device=device, dtype=dtype)

                self.model.layers[i] = new_layer
                replaced_layers.append(i)

        return replaced_layers

    def replace_with_mlp_only_layers(self):
        """Backward-compatible alias for replace_with_specialized_layers"""
        return self.replace_with_specialized_layers()


def load_heterogeneous_llama(model_path, device='cuda:0'):
    from transformers import AutoTokenizer
    from safetensors import safe_open

    layer_sizes_path = f"{model_path}/layer_sizes.json"
    if os.path.exists(layer_sizes_path):
        print("Reading layer sizes from layer_sizes.json...")
        with open(layer_sizes_path, 'r') as f:
            layer_sizes_list = json.load(f)

        layer_configs = {}
        skip_attention_layers = []
        skip_mlp_layers = []

        for layer_info in layer_sizes_list:
            layer_idx = str(layer_info['layer_idx'])
            head_dim = layer_info.get('head_dim')
            num_heads = layer_info.get('num_heads')
            intermediate_size = layer_info.get('intermediate_size')

            layer_configs[layer_idx] = {
                'num_attention_heads': num_heads,
                'intermediate_size': intermediate_size,
                'head_dim': head_dim
            }

            if (head_dim is not None and head_dim == 0) or (num_heads is not None and num_heads == 0):
                layer_configs[layer_idx]['skip_attention'] = True
                skip_attention_layers.append(int(layer_idx))

            if (intermediate_size is not None and intermediate_size == 0) or layer_info.get('skip_mlp', False):
                layer_configs[layer_idx]['skip_mlp'] = True
                skip_mlp_layers.append(int(layer_idx))

            layer_configs[layer_idx] = {
                k: v for k, v in layer_configs[layer_idx].items()
                if v is not None or k in ('skip_attention', 'skip_mlp')
            }

        print(f"Found {len(layer_configs)} layer-specific configurations from layer_sizes.json")

        safetensors_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
        if safetensors_files:
            from safetensors import safe_open
            mismatch_count = 0
            with safe_open(os.path.join(model_path, safetensors_files[0]), framework="pt") as sf:
                sf_keys = sf.keys()
                for layer_info in layer_sizes_list:
                    idx = layer_info['layer_idx']
                    gate_key = f"model.layers.{idx}.mlp.gate_proj.weight"
                    if gate_key in sf_keys:
                        actual_ffn = sf.get_slice(gate_key).get_shape()[0]
                        json_ffn = layer_info.get('intermediate_size')
                        if json_ffn is not None and json_ffn != actual_ffn:
                            print(f"  WARNING: Layer {idx} layer_sizes.json intermediate_size={json_ffn} "
                                  f"!= safetensors={actual_ffn}. Using safetensors value.")
                            layer_info['intermediate_size'] = actual_ffn
                            layer_configs[str(idx)]['intermediate_size'] = actual_ffn
                            mismatch_count += 1
                    q_key = f"model.layers.{idx}.self_attn.q_proj.weight"
                    if q_key in sf_keys:
                        actual_q_out = sf.get_slice(q_key).get_shape()[0]
                        json_head_dim = layer_info.get('head_dim', 128)
                        json_num_heads = layer_info.get('num_heads', 0)
                        if json_head_dim > 0 and json_num_heads > 0:
                            expected_q_out = json_head_dim * json_num_heads
                            if expected_q_out != actual_q_out:
                                actual_num_heads = actual_q_out // json_head_dim
                                print(f"  WARNING: Layer {idx} layer_sizes.json num_heads={json_num_heads} "
                                      f"!= safetensors={actual_num_heads}. Using safetensors value.")
                                layer_info['num_heads'] = actual_num_heads
                                layer_configs[str(idx)]['num_attention_heads'] = actual_num_heads
                                mismatch_count += 1
            if mismatch_count > 0:
                print(f"  Fixed {mismatch_count} mismatches between layer_sizes.json and safetensors")
            else:
                print(f"  layer_sizes.json validated against safetensors (OK)")

        head_dims = [cfg.get('head_dim') for cfg in layer_configs.values() if 'head_dim' in cfg]
        if head_dims:
            non_zero_dims = [d for d in head_dims if d > 0]
            if non_zero_dims:
                print(f"  Head dims: {min(non_zero_dims)} - {max(non_zero_dims)} (non-zero)")
            is_dimension_pruned = len(set(head_dims)) > 1 or (head_dims[0] != 128 if head_dims else False)
            if is_dimension_pruned:
                print(f"  ✓ Dimension pruning detected")

        if skip_attention_layers:
            print(f"  ✓ {len(skip_attention_layers)} layers with attention removed: {skip_attention_layers}")
        if skip_mlp_layers:
            print(f"  ✓ {len(skip_mlp_layers)} layers with MLP removed: {skip_mlp_layers}")
        both_skip = set(skip_attention_layers) & set(skip_mlp_layers)
        if both_skip:
            print(f"  ✓ {len(both_skip)} layers with both attention and MLP removed (identity): {sorted(both_skip)}")

        config = HeterogeneousLlamaConfig.from_pretrained(model_path)
        config.layer_configs = layer_configs

        needs_layer_replacement = skip_attention_layers or skip_mlp_layers
        print("Loading heterogeneous model...")
        if needs_layer_replacement:
            model = HeterogeneousLlamaForCausalLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype=torch.float16,
                device_map=None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            print(f"Replacing pruned layers with specialized layer types...")
            replaced = model.replace_with_specialized_layers()
            if replaced:
                print(f"  ✓ Replaced {len(replaced)} layers: {replaced}")

            if device == 'auto':
                actual_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                actual_device = device
            print(f"Moving model to {actual_device}...")
            model = model.to(actual_device)
        else:
            model = HeterogeneousLlamaForCausalLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True
            )

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"

        return model, tokenizer

    print("Reading layer sizes from saved weights...")
    layer_configs = {}

    config_path = f"{model_path}/config.json"
    config_head_dim = 128  # default
    try:
        with open(config_path, 'r') as f:
            config_json = json.load(f)
            config_head_dim = config_json.get('head_dim', 128)
            print(f"  Config head_dim: {config_head_dim}")
    except:
        pass

    meta_path = f"{model_path}/physical_pruning_meta.json"
    is_dimension_pruned = False
    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            meta_head_dim = meta.get('config', {}).get('head_dim')
            if meta_head_dim is not None:
                config_head_dim = meta_head_dim
                print(f"  Meta head_dim: {config_head_dim}")
            is_dimension_pruned = meta.get('config', {}).get('dimension_pruned', False)
            if is_dimension_pruned:
                print(f"  ✓ Dimension pruning was applied")
    except:
        pass

    index_path = f"{model_path}/model.safetensors.index.json"
    try:
        with open(index_path, 'r') as f:
            index = json.load(f)

        weight_map = index.get('weight_map', {})

        shard_weights = {}  # {shard_file: [weights]}
        for weight_name, shard_file in weight_map.items():
            if ('gate_proj.weight' in weight_name or 'q_proj.weight' in weight_name) and 'model.layers.' in weight_name:
                if shard_file not in shard_weights:
                    shard_weights[shard_file] = []
                shard_weights[shard_file].append(weight_name)

        for shard_file, weight_names in shard_weights.items():
            shard_path = f"{model_path}/{shard_file}"
            with safe_open(shard_path, framework="pt") as f:
                for weight_name in weight_names:
                    parts = weight_name.split('.')
                    if len(parts) >= 3 and parts[0] == 'model' and parts[1] == 'layers':
                        try:
                            layer_idx = int(parts[2])
                            layer_idx_str = str(layer_idx)

                            if layer_idx_str not in layer_configs:
                                layer_configs[layer_idx_str] = {}

                            tensor = f.get_tensor(weight_name)

                            if 'gate_proj.weight' in weight_name:
                                # gate_proj.weight shape: [intermediate_size, hidden_size]
                                intermediate_size = tensor.shape[0]
                                layer_configs[layer_idx_str]['intermediate_size'] = intermediate_size

                            elif 'q_proj.weight' in weight_name:
                                # q_proj.weight shape: [num_heads * head_dim, hidden_size]
                                q_proj_size = tensor.shape[0]
                                num_heads = q_proj_size // config_head_dim
                                layer_configs[layer_idx_str]['num_attention_heads'] = num_heads
                                layer_configs[layer_idx_str]['head_dim'] = config_head_dim

                        except (ValueError, IndexError):
                            continue

        print(f"Found {len(layer_configs)} layer-specific configurations")

        if layer_configs:
            head_counts = [cfg.get('num_attention_heads') for cfg in layer_configs.values() if 'num_attention_heads' in cfg]
            ffn_sizes = [cfg.get('intermediate_size') for cfg in layer_configs.values() if 'intermediate_size' in cfg]

            if head_counts:
                print(f"  Head counts: {min(head_counts)} - {max(head_counts)} (avg: {sum(head_counts)/len(head_counts):.1f})")
            else:
                print(f"  Head counts: Not found in weights")

            if ffn_sizes:
                print(f"  FFN sizes: {min(ffn_sizes)} - {max(ffn_sizes)} (avg: {sum(ffn_sizes)/len(ffn_sizes):.1f})")
            else:
                print(f"  FFN sizes: Not found in weights")

            is_heterogeneous_head = (len(head_counts) > 0 and len(set(head_counts)) > 1)
            is_heterogeneous_ffn = (len(ffn_sizes) > 0 and len(set(ffn_sizes)) > 1)

            if is_heterogeneous_head or is_heterogeneous_ffn:
                print(f"\n  ✓ Heterogeneous model detected:")
                if is_heterogeneous_head:
                    print(f"    - Heads: GLOBAL pruning (different per layer)")
                elif head_counts:
                    print(f"    - Heads: Layer-wise pruning (uniform: {head_counts[0]})")
                else:
                    print(f"    - Heads: Not available (will use config default)")

                if is_heterogeneous_ffn:
                    print(f"    - FFN: GLOBAL pruning (different per layer)")
                elif ffn_sizes:
                    print(f"    - FFN: Layer-wise pruning (uniform: {ffn_sizes[0]})")
                else:
                    print(f"    - FFN: Not available (will use config default)")

                if not is_heterogeneous_head:
                    for cfg in layer_configs.values():
                        cfg.pop('num_attention_heads', None)
                    if head_counts:
                        print(f"    → Using uniform head count from config")

                if not is_heterogeneous_ffn:
                    for cfg in layer_configs.values():
                        cfg.pop('intermediate_size', None)
                    if ffn_sizes:
                        print(f"    → Using uniform FFN size from config")
            else:
                print(f"\n  ✓ Uniform model detected (all layers identical)")
                if head_counts:
                    print(f"    - Heads: {head_counts[0]}")
                if ffn_sizes:
                    print(f"    - FFN: {ffn_sizes[0]}")
                print(f"    → No heterogeneous config needed, using standard model")
                layer_configs = {}

    except FileNotFoundError:
        print(f"Warning: {index_path} not found")
        layer_configs = {}
    except Exception as e:
        print(f"Warning: Could not read layer sizes: {e}")
        import traceback
        traceback.print_exc()
        layer_configs = {}

    config = HeterogeneousLlamaConfig.from_pretrained(model_path)
    config.layer_configs = layer_configs

    print("Loading heterogeneous model...")
    model = HeterogeneousLlamaForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    return model, tokenizer


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "./student_physically_pruned"

    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        output_path = model_path + "_heterogeneous"

    print("=" * 70)
    print("Heterogeneous LLaMA Model Compression")
    print("=" * 70)
    print(f"Input:  {model_path}")
    print(f"Output: {output_path}")
    print("=" * 70)
    print()

    print("Loading heterogeneous model...")
    model, tokenizer = load_heterogeneous_llama(model_path)

    print("\nModel structure (all layers):")
    print("-" * 70)
    mlp_only_layers = []
    attn_only_layers = []
    identity_layers = []
    for i, layer in enumerate(model.model.layers):
        layer_type = type(layer).__name__
        if isinstance(layer, IdentityDecoderLayer):
            identity_layers.append(i)
            print(f"Layer {i}: {layer_type} - Identity (pass-through)")
        elif isinstance(layer, MLPOnlyDecoderLayer):
            mlp_only_layers.append(i)
            gate_shape = layer.mlp.gate_proj.weight.shape
            print(f"Layer {i}: {layer_type} - MLP-only, intermediate_size = {gate_shape[0]}")
        elif isinstance(layer, AttentionOnlyDecoderLayer):
            attn_only_layers.append(i)
            q_shape = layer.self_attn.q_proj.weight.shape
            if hasattr(layer.self_attn, 'actual_head_dim'):
                head_dim = layer.self_attn.actual_head_dim
            else:
                head_dim = getattr(model.config, 'head_dim', 128)
            num_heads = q_shape[0] // head_dim if head_dim > 0 else 0
            print(f"Layer {i}: {layer_type} - Attention-only, {num_heads} heads (head_dim={head_dim})")
        else:
            has_mlp = hasattr(layer, 'mlp') and layer.mlp is not None
            has_attn = hasattr(layer, 'self_attn') and layer.self_attn is not None
            if has_attn:
                q_shape = layer.self_attn.q_proj.weight.shape
                if hasattr(layer.self_attn, 'actual_head_dim'):
                    head_dim = layer.self_attn.actual_head_dim
                else:
                    head_dim = getattr(model.config, 'head_dim', 128)
                num_heads = q_shape[0] // head_dim if head_dim > 0 else 0
                if has_mlp:
                    gate_shape = layer.mlp.gate_proj.weight.shape
                    print(f"Layer {i}: {layer_type} - {num_heads} heads (head_dim={head_dim}), intermediate_size = {gate_shape[0]}")
                else:
                    print(f"Layer {i}: {layer_type} - {num_heads} heads (head_dim={head_dim}), no MLP")
            else:
                if has_mlp:
                    gate_shape = layer.mlp.gate_proj.weight.shape
                    print(f"Layer {i}: {layer_type} - no self_attn, intermediate_size = {gate_shape[0]}")
                else:
                    print(f"Layer {i}: {layer_type} - no self_attn, no MLP")
    print("-" * 70)
    if mlp_only_layers:
        print(f"MLPOnlyDecoderLayer indices: {mlp_only_layers}")
    if attn_only_layers:
        print(f"AttentionOnlyDecoderLayer indices: {attn_only_layers}")
    if identity_layers:
        print(f"IdentityDecoderLayer indices: {identity_layers}")
    print()

    print("✅ Model loaded successfully!")
    print()

    print("Testing inference...")
    inputs = tokenizer("Hello, world!", return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)

    print(f"Output logits shape: {outputs.logits.shape}")
    print("✅ Inference test passed!")
    print()

    print("=" * 70)
    print("Saving heterogeneous model...")
    print("=" * 70)
    os.makedirs(output_path, exist_ok=True)

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    import shutil

    meta_src = os.path.join(model_path, "physical_pruning_meta.json")
    if os.path.exists(meta_src):
        meta_dst = os.path.join(output_path, "physical_pruning_meta.json")
        shutil.copy(meta_src, meta_dst)
        print(f"Copied: physical_pruning_meta.json")

    layer_sizes_src = os.path.join(model_path, "layer_sizes.json")
    if os.path.exists(layer_sizes_src):
        layer_sizes_dst = os.path.join(output_path, "layer_sizes.json")
        shutil.copy(layer_sizes_src, layer_sizes_dst)
        print(f"Copied: layer_sizes.json")

    print()
    print("=" * 70)
    print("Compression Complete!")
    print("=" * 70)
    print(f"Compressed model saved to: {output_path}")
    print()
    print("You can now load this model with:")
    print(f"  from models.heterogeneous_llama import load_heterogeneous_llama")
    print(f"  model, tokenizer = load_heterogeneous_llama('{output_path}')")
    print("=" * 70)
