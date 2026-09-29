
import os
import json
import torch
import torch.nn as nn
from transformers import PhiForCausalLM, PhiConfig
from transformers.models.phi.modeling_phi import (
    PhiDecoderLayer,
    PhiMLP,
    PhiAttention,
    PhiModel,
    PhiRotaryEmbedding,
    create_causal_mask
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from typing import Optional, Tuple, Union, List


class HeterogeneousPhiConfig(PhiConfig):
    model_type = "heterogeneous_phi"

    def __init__(self, layer_configs=None, **kwargs):
        super().__init__(**kwargs)
        # layer_configs: {layer_idx_str: {'intermediate_size': ..., 'num_attention_heads': ..., 'head_dim': ...}}
        self.layer_configs = layer_configs or {}


class HeterogeneousPhiMLP(PhiMLP):
    def __init__(self, config, intermediate_size=None):
        if intermediate_size is not None:
            original_size = config.intermediate_size
            config.intermediate_size = intermediate_size

        super().__init__(config)

        if intermediate_size is not None:
            config.intermediate_size = original_size


class HeterogeneousPhiAttention(PhiAttention):
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

        partial_rotary_factor = getattr(config, 'partial_rotary_factor', 0.5)
        self.rotary_ndims = int(self.actual_head_dim * partial_rotary_factor)

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
            expected_rotary = self.rotary_ndims

            if rotary_dim > expected_rotary:
                cos = cos[..., :expected_rotary]
                sin = sin[..., :expected_rotary]
                position_embeddings = (cos, sin)
            elif rotary_dim < expected_rotary:
                pad_size = expected_rotary - rotary_dim
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


class MLPOnlyPhiDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx=None, intermediate_size=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.mlp = HeterogeneousPhiMLP(config, intermediate_size=intermediate_size)

        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Dropout
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

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
        hidden_states = self.input_layernorm(hidden_states)

        feed_forward_hidden_states = self.resid_dropout(self.mlp(hidden_states))
        hidden_states = residual + feed_forward_hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (None,)
        return outputs


class AttentionOnlyPhiDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx=None, num_attention_heads=None, head_dim=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = HeterogeneousPhiAttention(
            config=config,
            layer_idx=layer_idx,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim
        )

        # LayerNorm
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Dropout
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

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
        hidden_states = self.input_layernorm(hidden_states)

        # Attention only
        attn_outputs, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        attn_outputs = self.resid_dropout(attn_outputs)
        hidden_states = residual + attn_outputs

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs


class IdentityPhiDecoderLayer(nn.Module):
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
            outputs += (None,)
        return outputs


class HeterogeneousPhiDecoderLayer(PhiDecoderLayer):
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

        super(PhiDecoderLayer, self).__init__()

        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = HeterogeneousPhiAttention(
            config=config,
            layer_idx=layer_idx,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim
        )

        self.mlp = HeterogeneousPhiMLP(config, intermediate_size=intermediate_size)

        # Phi-2: single LayerNorm (no post_attention_layernorm)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Dropout
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

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

        # Phi-2 parallel residual: output = residual + attn(norm(x)) + mlp(norm(x))
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


class HeterogeneousPhiModel(PhiModel):
    """
    Custom PhiModel that properly handles mixed layer types
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

        # Phi-2 has embed_dropout
        if hasattr(self, 'embed_dropout'):
            inputs_embeds = self.embed_dropout(inputs_embeds)

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

        # Create causal mask
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds

        # Create position embeddings (shared across layers)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Decoder layers
        for decoder_layer in self.layers[:self.config.num_hidden_layers]:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            # Extract hidden_states from layer output
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

        # Final layer norm (Phi-2: final_layernorm)
        hidden_states = self.final_layernorm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class HeterogeneousPhiForCausalLM(PhiForCausalLM):
    config_class = HeterogeneousPhiConfig

    def __init__(self, config):
        super().__init__(config)

        # Replace the standard PhiModel with HeterogeneousPhiModel
        old_model = self.model
        self.model = HeterogeneousPhiModel(config)
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
                layers.append(HeterogeneousPhiDecoderLayer(config, layer_idx=i))

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
                new_layer = IdentityPhiDecoderLayer(config=self.config, layer_idx=i)
                old_layer = self.model.layers[i]
                old_params = list(old_layer.parameters())
                if old_params:
                    device = old_params[0].device
                    new_layer = new_layer.to(device=device)
                self.model.layers[i] = new_layer
                replaced_layers.append(i)

            elif is_skip_attention:
                old_layer = self.model.layers[i]
                new_layer = MLPOnlyPhiDecoderLayer(
                    self.config,
                    layer_idx=i,
                    intermediate_size=intermediate_size
                )
                new_layer.input_layernorm.load_state_dict(
                    old_layer.input_layernorm.state_dict()
                )
                new_layer.mlp.load_state_dict(old_layer.mlp.state_dict())
                device = next(old_layer.parameters()).device
                dtype = next(old_layer.parameters()).dtype
                new_layer = new_layer.to(device=device, dtype=dtype)
                self.model.layers[i] = new_layer
                replaced_layers.append(i)

            elif is_skip_mlp:
                old_layer = self.model.layers[i]
                new_layer = AttentionOnlyPhiDecoderLayer(
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


def load_heterogeneous_phi(model_path, device='cuda:0'):
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
            mismatch_count = 0
            with safe_open(os.path.join(model_path, safetensors_files[0]), framework="pt") as sf:
                sf_keys = sf.keys()
                for layer_info in layer_sizes_list:
                    idx = layer_info['layer_idx']
                    # Phi-2: fc1 instead of gate_proj
                    fc1_key = f"model.layers.{idx}.mlp.fc1.weight"
                    if fc1_key in sf_keys:
                        actual_ffn = sf.get_slice(fc1_key).get_shape()[0]
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
                        json_head_dim = layer_info.get('head_dim', 80)  # Phi-2 default head_dim=80
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
            non_zero_dims = [d for d in head_dims if d and d > 0]
            if non_zero_dims:
                print(f"  Head dims: {min(non_zero_dims)} - {max(non_zero_dims)} (non-zero)")

        if skip_attention_layers:
            print(f"  {len(skip_attention_layers)} layers with attention removed: {skip_attention_layers}")
        if skip_mlp_layers:
            print(f"  {len(skip_mlp_layers)} layers with MLP removed: {skip_mlp_layers}")
        both_skip = set(skip_attention_layers) & set(skip_mlp_layers)
        if both_skip:
            print(f"  {len(both_skip)} layers with both removed (identity): {sorted(both_skip)}")

        config = HeterogeneousPhiConfig.from_pretrained(model_path)
        config.layer_configs = layer_configs

        needs_layer_replacement = skip_attention_layers or skip_mlp_layers
        print("Loading heterogeneous Phi model...")
        if needs_layer_replacement:
            model = HeterogeneousPhiForCausalLM.from_pretrained(
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
                print(f"  Replaced {len(replaced)} layers: {replaced}")

            if device == 'auto':
                actual_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                actual_device = device
            print(f"Moving model to {actual_device}...")
            model = model.to(actual_device)
        else:
            model = HeterogeneousPhiForCausalLM.from_pretrained(
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
    # Phi-2 default head_dim = hidden_size / num_attention_heads = 2560/32 = 80
    config_head_dim = 80
    try:
        with open(config_path, 'r') as f:
            config_json = json.load(f)
            hs = config_json.get('hidden_size', 2560)
            nh = config_json.get('num_attention_heads', 32)
            config_head_dim = config_json.get('head_dim', hs // nh)
            print(f"  Config head_dim: {config_head_dim}")
    except:
        pass

    index_path = f"{model_path}/model.safetensors.index.json"
    try:
        with open(index_path, 'r') as f:
            index = json.load(f)

        weight_map = index.get('weight_map', {})

        shard_weights = {}
        for weight_name, shard_file in weight_map.items():
            if ('fc1.weight' in weight_name or 'q_proj.weight' in weight_name) and 'model.layers.' in weight_name:
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

                            if 'fc1.weight' in weight_name:
                                # fc1.weight shape: [intermediate_size, hidden_size]
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
                print(f"  Head counts: {min(head_counts)} - {max(head_counts)}")
            if ffn_sizes:
                print(f"  FFN sizes: {min(ffn_sizes)} - {max(ffn_sizes)}")

            is_heterogeneous_head = (len(head_counts) > 0 and len(set(head_counts)) > 1)
            is_heterogeneous_ffn = (len(ffn_sizes) > 0 and len(set(ffn_sizes)) > 1)

            if not is_heterogeneous_head and not is_heterogeneous_ffn:
                print(f"  Uniform model detected (all layers identical)")
                layer_configs = {}

    except FileNotFoundError:
        # Single safetensors file (not sharded)
        safetensors_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
        if safetensors_files:
            with safe_open(os.path.join(model_path, safetensors_files[0]), framework="pt") as sf:
                sf_keys = sf.keys()
                for key in sf_keys:
                    if 'fc1.weight' in key and 'model.layers.' in key:
                        parts = key.split('.')
                        layer_idx = parts[2]
                        if layer_idx not in layer_configs:
                            layer_configs[layer_idx] = {}
                        layer_configs[layer_idx]['intermediate_size'] = sf.get_slice(key).get_shape()[0]
                    elif 'q_proj.weight' in key and 'model.layers.' in key:
                        parts = key.split('.')
                        layer_idx = parts[2]
                        if layer_idx not in layer_configs:
                            layer_configs[layer_idx] = {}
                        q_size = sf.get_slice(key).get_shape()[0]
                        layer_configs[layer_idx]['num_attention_heads'] = q_size // config_head_dim
                        layer_configs[layer_idx]['head_dim'] = config_head_dim
            print(f"Found {len(layer_configs)} layer-specific configurations from safetensors")
        else:
            print(f"Warning: No safetensors files found in {model_path}")
            layer_configs = {}
    except Exception as e:
        print(f"Warning: Could not read layer sizes: {e}")
        import traceback
        traceback.print_exc()
        layer_configs = {}

    config = HeterogeneousPhiConfig.from_pretrained(model_path)
    config.layer_configs = layer_configs

    print("Loading heterogeneous Phi model...")
    model = HeterogeneousPhiForCausalLM.from_pretrained(
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
