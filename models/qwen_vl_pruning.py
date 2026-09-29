"""VLM (Qwen2.5-VL / Qwen3-VL) pruning bridge.

The existing pruning pipeline (test_depth_pruning.py + pruning/cett_importance.py)
expects a CausalLM with the following module layout:

    model.model.embed_tokens
    model.model.layers[i].self_attn.{q,k,v,o}_proj
    model.model.layers[i].mlp.{gate,up,down}_proj
    model.model.norm
    model.lm_head
    model.config.{hidden_size, num_hidden_layers, intermediate_size, ...}

Qwen2.5-VL / Qwen3-VL store the same modules at a deeper path:

    vlm.model.language_model.embed_tokens
    vlm.model.language_model.layers[i].{self_attn, mlp}.*
    vlm.model.language_model.norm
    vlm.lm_head
    vlm.config.text_config  ←  the LLM-side config

This module provides a *facade* that exposes the language sub-tree as if it
were a flat CausalLM, sharing tensors (no copy). Mutations to the facade —
deleting layers, slicing q/k/v/o_proj weights, replacing modules — apply
directly to the underlying VLM.

Usage:
    vlm = AutoModelForImageTextToText.from_pretrained(...)
    causal_lm = VLMCausalLMFacade(vlm)
    # Pass `causal_lm` to existing pruning code; it sees the LLM only.
    # When done, save the original `vlm`:
    sync_vlm_config_after_pruning(vlm)
    vlm.save_pretrained(out_dir)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast


class VLMCausalLMFacade(nn.Module):
    """Make a Qwen2.5-VL / Qwen3-VL look like a Qwen2/Llama CausalLM.

    Shares all tensors with the underlying VLM. Mutations propagate.
    """

    def __init__(self, vlm: nn.Module):
        super().__init__()
        # Bypass nn.Module's __setattr__ so `_vlm` is NOT registered as a
        # submodule — otherwise `facade.parameters()` would double-count
        # the LLM weights (once via `_vlm.model.language_model.*` and again
        # via `model.*`), corrupting all pruning param-count math.
        object.__setattr__(self, "_vlm", vlm)
        self.config = vlm.config.text_config
        # Mirror the CausalLM hierarchy. These ARE registered as submodules
        # so parameters() / modules() walks find them — and since they share
        # tensors with the underlying VLM, mutations propagate.
        self.model = vlm.model.language_model  # has .embed_tokens, .layers, .norm
        self.lm_head = vlm.lm_head
        # Useful for downstream code that checks dtype / device
        self.dtype = next(self.model.parameters()).dtype
        self.device = next(self.model.parameters()).device

    @property
    def vlm(self) -> nn.Module:
        return self._vlm

    def __deepcopy__(self, memo):
        """Deep-copy the underlying VLM, then build a fresh facade over it.

        The pruning pipeline does `pruned_model = copy.deepcopy(model)` so it
        can mutate without affecting the original. A naive deepcopy of the
        facade would copy `_vlm` and `self.model` independently, breaking the
        weight-sharing invariant. We must deep-copy `_vlm` and rebuild the
        facade to point at the new VLM's language_model + lm_head.
        """
        import copy as _copy
        new_vlm = _copy.deepcopy(self._vlm, memo)
        return VLMCausalLMFacade(new_vlm)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def gradient_checkpointing_enable(self, *args, **kwargs):
        return self.model.gradient_checkpointing_enable(*args, **kwargs)

    def gradient_checkpointing_disable(self, *args, **kwargs):
        return self.model.gradient_checkpointing_disable(*args, **kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs,
    ):
        """Pure text forward through the language model + lm_head.

        Vision inputs are ignored — calibration / fine-tuning data is text-only,
        and the LLM treats vision projector outputs as embedding vectors,
        which is the same form as text token embeddings here.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values if use_cache else None,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )

    def save_pretrained(self, save_directory, **kwargs):
        """Save the *underlying VLM* (not the facade) so vision tower is preserved."""
        sync_vlm_config_after_pruning(self._vlm)
        return self._vlm.save_pretrained(save_directory, **kwargs)


def sync_vlm_config_after_pruning(vlm: nn.Module) -> None:
    """Update vlm.config.text_config to reflect the actual pruned LLM shape.

    Call this AFTER any depth/width pruning has mutated the language_model in
    place, BEFORE save_pretrained, so the saved config matches the saved
    state_dict and the model can later be reloaded with from_pretrained.
    """
    lm = vlm.model.language_model
    text_cfg = vlm.config.text_config

    # Depth: number of decoder layers
    text_cfg.num_hidden_layers = len(lm.layers)

    # Width: intermediate_size from any surviving layer.
    # If width pruning produced HETEROGENEOUS sizes (different per layer),
    # we cannot represent that in a flat config field. We pick the MAX so
    # that from_pretrained allocates enough capacity; the actual Linear
    # weights stored per-layer determine what runs at inference.
    if len(lm.layers) > 0:
        sizes = []
        for layer in lm.layers:
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and hasattr(mlp, "down_proj"):
                # down_proj.in_features == intermediate_size
                sizes.append(mlp.down_proj.in_features)
        if sizes:
            unique = sorted(set(sizes))
            text_cfg.intermediate_size = max(sizes)
            if len(unique) > 1:
                # Heterogeneous result. Standard from_pretrained will fail to
                # load. Caller should be aware.
                print(
                    f"  [WARN] Heterogeneous FFN widths detected: {unique}. "
                    f"config.text_config.intermediate_size set to max ({max(sizes)}) "
                    f"but standard AutoModelForImageTextToText.from_pretrained "
                    f"will not reload this correctly without a custom loader."
                )

        # Hidden size and head counts: depth/width FFN pruning does not
        # change these in the existing pipeline, so leave them alone.

    # Mirror to top-level config attributes that some inference code reads.
    # (Qwen2.5-VL config exposes both forms.)
    if hasattr(vlm.config, "num_hidden_layers"):
        vlm.config.num_hidden_layers = text_cfg.num_hidden_layers
    if hasattr(vlm.config, "intermediate_size"):
        vlm.config.intermediate_size = text_cfg.intermediate_size

    # If the original config carries `layer_types` (Qwen2.5-VL / Qwen3), it
    # must shrink in lockstep with `num_hidden_layers`. test_depth_pruning's
    # physically_prune_model already updates pruned_model.config (which is
    # text_config for VLM via the facade), but only IF layer_types lived
    # there. Mirror both directions to be safe.
    if hasattr(text_cfg, "layer_types") and text_cfg.layer_types is not None:
        if len(text_cfg.layer_types) > text_cfg.num_hidden_layers:
            text_cfg.layer_types = text_cfg.layer_types[: text_cfg.num_hidden_layers]
    if hasattr(vlm.config, "layer_types") and vlm.config.layer_types is not None:
        if len(vlm.config.layer_types) > text_cfg.num_hidden_layers:
            vlm.config.layer_types = vlm.config.layer_types[: text_cfg.num_hidden_layers]


def load_vlm(model_name: str, dtype: torch.dtype = torch.bfloat16,
             device_map: str = "cuda:0", trust_remote_code: bool = True):
    """Load a Qwen2.5-VL / Qwen3-VL model and its processor."""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if hasattr(processor, "tokenizer") and processor.tokenizer.padding_side != "left":
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
    vlm = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    return vlm, processor


def tokenizer_for_calibration(vlm_or_path):
    """Return a plain text tokenizer compatible with the VLM's LLM.

    The pruning pipeline's calibration loader expects a HF tokenizer with
    apply_chat_template / encode behavior. We return processor.tokenizer.
    """
    from transformers import AutoTokenizer
    if isinstance(vlm_or_path, str):
        return AutoTokenizer.from_pretrained(vlm_or_path, trust_remote_code=True)
    # Fallback: build a tokenizer from the VLM's processor path
    return AutoTokenizer.from_pretrained(vlm_or_path.config._name_or_path,
                                         trust_remote_code=True)


def lora_target_modules_vlm() -> str:
    """LoRA target_modules pattern for VLM fine-tuning, scoped to the LLM.

    PEFT's `target_modules` argument is matched as suffix when given a list
    of strings, but as a full regex when given a single string. We must
    return a regex STRING so that the pattern actually constrains matches
    to `model.language_model.layers.*.{self_attn,mlp}.*_proj` and excludes
    the vision tower's q_proj / k_proj / v_proj / o_proj.
    """
    return (
        r".*language_model\.layers\.\d+\."
        r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
        r"|mlp\.(gate_proj|up_proj|down_proj))$"
    )
