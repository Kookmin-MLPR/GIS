
import os
import json
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    TrainerCallback
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

from config import LoRAConfig, FineTuningConfig, DataConfig, TrainingConfig, get_model_name, LLAMA_MODELS
from pruning.distributed_utils import (
    setup_distributed, cleanup_distributed,
    is_main_process, is_distributed, get_local_rank,
    get_device_map_for_loading, barrier
)
from utils import (
    load_alpaca_dataset,
    load_c4_dataset,
    format_alpaca_prompt,
    print_trainable_parameters
)

# Soft Orthogonal Regularization
from pruning.soft_orthogonal import (
    SoftOrthogonalConfig,
    compute_svd_cache,
    load_svd_cache,
    save_svd_cache,
    register_svd_buffers_to_model,
    compute_soft_orthogonal_loss_for_model,
    STRATEGIES
)

# KD Mode: Masked Student approach (like Stage 1)
from models.student_lora import (
    create_student_model,
    apply_masks_to_student,
    _set_student_forward_context
)
from pruning.physical_pruning import load_masks, PhysicalPruner

# Momentum LoRA
from models.momentum_lora import (
    MomentumLoRALinear,
    collect_momentum_layers,
    update_all_momentum_buffers,
    compute_total_soft_orth_loss as compute_momentum_soft_orth_loss,
)
from models.lora_utils import (
    add_lora_to_model,
    apply_svd_cache_to_momentum_layers,
    MomentumLoRAConfig,
)
from config import MomentumLoRAConfig as MomentumLoRAConfigDataclass

# PiSSA and Orthogonal LoRA Initialization
from pruning.pissa_init import (
    apply_pissa_init,
    apply_orthogonal_init,
)

from torch.optim import AdamW
from transformers import get_scheduler


def create_optimizer_with_separate_lr(
    model,
    base_lr: float,
    attn_lr: float,
    ffn_lr: float,
    weight_decay: float = 0.0
):
    attn_params = []
    ffn_params = []
    other_params = []

    attn_keywords = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'self_attn']
    ffn_keywords = ['gate_proj', 'up_proj', 'down_proj', 'mlp']

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_attn = any(kw in name for kw in attn_keywords)
        is_ffn = any(kw in name for kw in ffn_keywords)

        if is_attn and not is_ffn:
            attn_params.append(param)
        elif is_ffn and not is_attn:
            ffn_params.append(param)
        else:
            other_params.append(param)

    param_groups = []

    if attn_params:
        param_groups.append({
            'params': attn_params,
            'lr': attn_lr,
            'name': 'attention'
        })

    if ffn_params:
        param_groups.append({
            'params': ffn_params,
            'lr': ffn_lr,
            'name': 'ffn'
        })

    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': base_lr,
            'name': 'other'
        })

    print(f"\n[Optimizer] Parameter groups with separate learning rates:")
    print(f"  Attention params: {len(attn_params)} tensors, lr={attn_lr}")
    print(f"  FFN params: {len(ffn_params)} tensors, lr={ffn_lr}")
    print(f"  Other params: {len(other_params)} tensors, lr={base_lr}")

    optimizer = AdamW(param_groups, weight_decay=weight_decay)

    return optimizer


class DetailedLoggingCallback(TrainerCallback):

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and state.is_local_process_zero:
            if 'loss' in logs:
                loss_val = logs['loss']
                grad_norm_val = logs.get('grad_norm', None)
                lr_val = logs.get('learning_rate', None)
                epoch_val = logs.get('epoch', None)

                log_str = f"[Step {state.global_step}]"
                log_str += f" loss: {loss_val:.8f}"
                if grad_norm_val is not None:
                    log_str += f", grad_norm: {grad_norm_val:.8e}"
                if lr_val is not None:
                    log_str += f", lr: {lr_val:.2e}"
                if epoch_val is not None:
                    log_str += f", epoch: {epoch_val:.2f}"

                print(log_str)


class MomentumLoRAUpdateCallback(TrainerCallback):

    def __init__(self, model):
        self.model = model
        self.momentum_layers = collect_momentum_layers(model)
        print(f"[MomentumLoRA] Callback initialized with {len(self.momentum_layers)} layers")

    def on_step_end(self, args, state, control, **kwargs):
        update_all_momentum_buffers(self.model)


class Stage2Trainer(Trainer):
    """
    Custom Trainer for Stage 2 Fine-tuning.

    Supports:
    1. Standard Fine-tuning (use_kd=False): Simple supervised learning
    2. KD Fine-tuning (use_kd=True): Knowledge Distillation with masked student
    3. Soft Orthogonal Regularization: Optional regularization for LoRA
    4. Momentum LoRA: Advanced LoRA with momentum-based gradient accumulation
    """

    def __init__(self, use_kd=False, temperature=2.0, kd_weight=0.5,
                 layer_kd_weight=0.9, logit_kd_weight=0.1,
                 use_momentum_lora=False,
                 regularization="none", reg_lambda=1e-4,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        # KD settings
        self.use_kd = use_kd
        self.temperature = temperature
        self.kd_weight = kd_weight
        self.ce_weight = 1.0 - kd_weight
        self.layer_kd_weight = layer_kd_weight
        self.logit_kd_weight = logit_kd_weight

        # PEFT model reference (for disabling adapters during teacher forward)
        self.peft_model = None

        # Regularization (L1/L2)
        self.regularization = regularization
        self.reg_lambda = reg_lambda

        # Soft Orthogonal Regularization
        self.soft_orth_config = None

        # Momentum LoRA
        self.use_momentum_lora = use_momentum_lora
        self.momentum_soft_orth_lambda_U = 0.01
        self.momentum_soft_orth_lambda_V = 0.01

        if self.regularization != "none":
            print(f"\n[Stage2Trainer] {self.regularization.upper()} Regularization enabled (lambda={self.reg_lambda})")

    def set_peft_model(self, peft_model):
        """Store reference to PEFT model for KD teacher forward."""
        self.peft_model = peft_model

    def set_soft_orthogonal(self, config: SoftOrthogonalConfig):
        """
        Enable soft orthogonal regularization.

        Args:
            config: SoftOrthogonalConfig with lambda_U, lambda_V, etc.
        """
        self.soft_orth_config = config
        if config.use_soft_orthogonal:
            print(f"\n[Stage2Trainer] Soft Orthogonal Regularization enabled")
            print(f"  Strategy: {config.strategy}")
            print(f"  lambda_U: {config.lambda_U}")
            print(f"  lambda_V: {config.lambda_V}")

    def set_momentum_soft_orthogonal(self, lambda_U: float, lambda_V: float):
        """
        Set soft orthogonal regularization parameters for Momentum LoRA.

        Args:
            lambda_U: Regularization strength for output subspace
            lambda_V: Regularization strength for input subspace
        """
        self.momentum_soft_orth_lambda_U = lambda_U
        self.momentum_soft_orth_lambda_V = lambda_V
        print(f"\n[Stage2Trainer] Momentum LoRA Soft Orthogonal enabled")
        print(f"  lambda_U: {lambda_U}")
        print(f"  lambda_V: {lambda_V}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss.

        If use_kd=True:
            Loss = α * KD_loss + (1-α) * CE_loss
            KD_loss = layer_kd_weight * Layer_KD + logit_kd_weight * Logit_KD
        If use_kd=False:
            Loss = CE_loss only
        """
        labels = inputs.get("labels")

        if self.use_kd:
            # KD Mode: Masked Student with Teacher
            loss = self._compute_kd_loss(model, inputs, labels)
        else:
            # Standard Mode: Simple supervised learning
            outputs = model(**inputs)
            loss = outputs.loss

        # Add soft orthogonal regularization if enabled
        if (self.soft_orth_config is not None and
            self.soft_orth_config.use_soft_orthogonal and
            not self.use_momentum_lora):  # Standard LoRA soft orth

            soft_orth_loss, num_layers = compute_soft_orthogonal_loss_for_model(
                model=model,
                lambda_U=self.soft_orth_config.lambda_U,
                lambda_V=self.soft_orth_config.lambda_V
            )

            if soft_orth_loss is not None and num_layers > 0:
                loss = loss + soft_orth_loss

        # Add Momentum LoRA soft orthogonal regularization if enabled
        if self.use_momentum_lora and self.momentum_soft_orth_lambda_U > 0:
            momentum_soft_orth_loss, num_layers = compute_momentum_soft_orth_loss(
                model=model,
                lambda_U=self.momentum_soft_orth_lambda_U,
                lambda_V=self.momentum_soft_orth_lambda_V
            )

            if momentum_soft_orth_loss is not None and num_layers > 0:
                loss = loss + momentum_soft_orth_loss

        # Add L1/L2 regularization on LoRA parameters if enabled
        if self.regularization != "none" and self.reg_lambda > 0:
            reg_loss = torch.tensor(0.0, device=loss.device)
            for name, param in model.named_parameters():
                if param.requires_grad and ("lora_" in name):
                    if self.regularization == "l1":
                        reg_loss = reg_loss + param.abs().sum()
                    elif self.regularization == "l2":
                        reg_loss = reg_loss + param.pow(2).sum()
            loss = loss + self.reg_lambda * reg_loss

        return (loss, {"loss": loss}) if return_outputs else loss

    def _compute_kd_loss(self, model, inputs, labels):
        """
        Compute Knowledge Distillation loss.

        Teacher: Base model without masks (clean output)
        Student: Base model with masks applied (pruned output)
        """
        # Remove labels from inputs for manual loss computation
        inputs_no_labels = {k: v for k, v in inputs.items() if k != "labels"}

        # Student forward (with masks applied)
        with _set_student_forward_context(True):
            student_outputs = model(**inputs_no_labels, output_hidden_states=True)
            student_logits = student_outputs.logits

        # Cross-Entropy Loss
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )

        # Teacher forward (without masks - clean output)
        with torch.no_grad():
            with _set_student_forward_context(False):
                # Disable LoRA adapters for teacher forward
                if self.peft_model is not None:
                    from peft import PeftModel
                    peft_model = self.peft_model

                    # Find PeftModel if wrapped
                    if not isinstance(peft_model, PeftModel):
                        if hasattr(peft_model, 'base_model') and isinstance(peft_model.base_model, PeftModel):
                            peft_model = peft_model.base_model

                    # Temporarily set LoRA scaling to 0
                    original_scalings = {}
                    for name, module in peft_model.named_modules():
                        if hasattr(module, 'scaling'):
                            scaling = module.scaling
                            if isinstance(scaling, dict):
                                original_scalings[name] = dict(scaling)
                                for adapter_name in scaling:
                                    module.scaling[adapter_name] = 0.0
                            else:
                                original_scalings[name] = scaling
                                module.scaling = 0.0

                    try:
                        teacher_outputs = self.peft_model(**inputs_no_labels, output_hidden_states=True)
                    finally:
                        # Restore LoRA scaling
                        for name, module in peft_model.named_modules():
                            if name in original_scalings:
                                module.scaling = original_scalings[name]
                else:
                    teacher_outputs = model(**inputs_no_labels, output_hidden_states=True)

                teacher_logits = teacher_outputs.logits
                teacher_hidden_states = teacher_outputs.hidden_states

        student_hidden_states = student_outputs.hidden_states

        # Layer KD Loss (hidden states alignment)
        layer_kd_loss = 0.0
        num_layers = len(teacher_hidden_states) - 1  # Exclude embedding

        for layer_idx in range(num_layers):
            teacher_hidden = teacher_hidden_states[layer_idx + 1]
            student_hidden = student_hidden_states[layer_idx + 1]
            layer_kd_loss += F.mse_loss(student_hidden.float(), teacher_hidden.float())

        layer_kd_loss = layer_kd_loss / num_layers

        # Logit KD Loss (output distribution alignment)
        logit_kd_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # Combined KD Loss
        kd_loss = self.layer_kd_weight * layer_kd_loss + self.logit_kd_weight * logit_kd_loss

        # Total Loss
        total_loss = self.kd_weight * kd_loss + self.ce_weight * ce_loss

        return total_loss


def tokenize_function(examples, tokenizer, max_seq_length):
    text_key = "text" if "text" in examples else "sentence"
    outputs = tokenizer(
        examples[text_key],
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
        return_tensors=None
    )
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs


def load_physically_pruned_model(pruned_model_path: str):
    print("\n" + "=" * 70)
    print("Loading Physically Pruned Model")
    print("=" * 70)

    print(f"[Load] Path: {pruned_model_path}")

    config_path = os.path.join(pruned_model_path, "config.json")
    meta_path = os.path.join(pruned_model_path, "physical_pruning_meta.json")
    layer_sizes_path = os.path.join(pruned_model_path, "layer_sizes.json")
    is_heterogeneous = False

    model_type = ""
    is_vlm_model = False
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            model_type = config_data.get("model_type", "")
            is_heterogeneous = (model_type in ("heterogeneous_llama", "heterogeneous_phi"))
            # VLM: Qwen2.5-VL / Qwen3-VL
            is_vlm_model = model_type in ("qwen2_5_vl", "qwen3_vl")

    if not is_heterogeneous and os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            mask_info = meta.get('mask_info', {})
            global_pruning = mask_info.get('global_pruning', {})
            is_global = global_pruning.get('ffn', False) or global_pruning.get('head', False)
            if is_global:
                is_heterogeneous = True
                print("[Load] Auto-detected: Global pruning in meta, will use heterogeneous loader")

    if not is_heterogeneous and os.path.exists(layer_sizes_path):
        with open(layer_sizes_path, 'r') as f:
            layer_sizes = json.load(f)
            if len(layer_sizes) > 1:
                ffn_sizes = [layer.get('intermediate_size', 0) for layer in layer_sizes]
                unique_ffn_sizes = set(ffn_sizes)
                if len(unique_ffn_sizes) > 1:
                    is_heterogeneous = True
                    print(f"[Load] Auto-detected: Different FFN sizes across layers ({len(unique_ffn_sizes)} unique sizes)")
                    print(f"[Load] FFN size range: {min(ffn_sizes)} ~ {max(ffn_sizes)}")

                head_counts = [layer.get('num_heads', 0) for layer in layer_sizes]
                unique_head_counts = set(head_counts)
                if len(unique_head_counts) > 1:
                    is_heterogeneous = True
                    print(f"[Load] Auto-detected: Different head counts across layers ({len(unique_head_counts)} unique counts)")

    if not is_heterogeneous and os.path.exists(layer_sizes_path) and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        with open(layer_sizes_path, 'r') as f:
            layer_sizes = json.load(f)

        hidden_size = cfg.get('hidden_size', 0)
        num_heads = cfg.get('num_attention_heads', 1)
        config_head_dim = cfg.get('head_dim', hidden_size // num_heads if num_heads > 0 else 0)
        actual_head_dim = 0
        for ls in layer_sizes:
            if ls.get('num_heads', 0) > 0 and ls.get('head_dim', 0) > 0:
                actual_head_dim = ls['head_dim']
                break
        if actual_head_dim > 0 and config_head_dim != actual_head_dim:
            is_heterogeneous = True
            print(f"[Load] Auto-detected: head_dim mismatch (config={config_head_dim}, actual={actual_head_dim})")
            print(f"[Load] This happens when head pruning changes num_heads but model computes head_dim=hidden_size//num_heads")

    if is_vlm_model:
        # VLM (Qwen2.5-VL / Qwen3-VL): load full multimodal model.
        # The pruned LLM weights are stored under model.language_model.*
        # and the vision tower is preserved unchanged.
        print("[Load] Using VLM loader (AutoModelForImageTextToText)")
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            pruned_model_path,
            torch_dtype=torch.float16,
            device_map=get_device_map_for_loading(),
            trust_remote_code=True,
            local_files_only=True,
        )
    elif is_heterogeneous:
        is_phi_model = (model_type in ("phi", "heterogeneous_phi"))
        if not is_phi_model and model_type == "":
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                    if 'partial_rotary_factor' in cfg:
                        is_phi_model = True

        if is_phi_model:
            print("[Load] Using heterogeneous Phi model loader")
            from models.heterogeneous_phi import load_heterogeneous_phi
            model, _ = load_heterogeneous_phi(pruned_model_path, device='auto')
        else:
            print("[Load] Using heterogeneous LLaMA model loader")
            from models.heterogeneous_llama import load_heterogeneous_llama
            model, _ = load_heterogeneous_llama(pruned_model_path, device='auto')
    else:
        print("[Load] Using standard model loader (layer-wise pruning)")
        model = AutoModelForCausalLM.from_pretrained(
            pruned_model_path,
            torch_dtype=torch.float16,
            device_map=get_device_map_for_loading(),
            trust_remote_code=True,
            local_files_only=True
        )

    from pruning.distributed_utils import is_distributed, get_local_rank
    if is_distributed():
        local_device = torch.device(f"cuda:{get_local_rank()}")
        model = model.to(local_device)
        print(f"[Load] Moved to {local_device}")

    print("[Load] Model loaded successfully")

    meta_path = os.path.join(pruned_model_path, "physical_pruning_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta_info = json.load(f)

        size_comp = meta_info.get('size_comparison', {})
        print(f"\n[Model Info]")
        print(f"  Parameters: {size_comp.get('pruned', {}).get('total_params', 'N/A'):,}")
        print(f"  Memory: {size_comp.get('pruned', {}).get('memory_gb', 'N/A'):.2f} GB")
        print(f"  Reduction: {size_comp.get('reduction', {}).get('params_percent', 'N/A'):.2f}%")
    else:
        print("\n[Warning] No physical_pruning_meta.json found")
        meta_info = None

    return model, meta_info


def parse_args():
    parser = argparse.ArgumentParser(description="Stage-2: Fine-tuning Physically Pruned Model")

    # Model selection (for tokenizer)
    parser.add_argument(
        "--model",
        type=str,
        default="llama1-7b",
        choices=list(LLAMA_MODELS.keys()) + ["custom"],
        help="Base model to use for tokenizer: llama1-7b (default), llama1-13b, llama2-7b, llama2-13b, or custom"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Local model path (overrides HuggingFace download). For offline environments."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Local dataset file path (JSON/JSONL). For offline environments."
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: from config.py)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (default: from config.py)"
    )
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=None,
        help="Micro batch size per GPU (default: from config.py). "
             "gradient_accumulation_steps = batch_size // micro_batch_size"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Gradient accumulation steps (default: batch_size // micro_batch_size)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (default: from config.py)"
    )
    parser.add_argument(
        "--attn_lr",
        type=float,
        default=None,
        help="Learning rate for attention layers (if None, uses learning_rate)"
    )
    parser.add_argument(
        "--ffn_lr",
        type=float,
        default=None,
        help="Learning rate for FFN layers (if None, uses learning_rate)"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=None,
        help="Warmup steps (in iterations; if set, used instead of warmup_ratio)"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="Maximum sequence length for tokenization (default: from config.py)"
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default=None,
        choices=["cosine", "linear"],
        help="LR scheduler type: cosine or linear (default: from config.py)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature (default: from config.py)"
    )

    # LoRA settings
    parser.add_argument(
        "--lora_r",
        type=int,
        default=None,
        help="LoRA rank (default: from config.py)"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=None,
        help="LoRA alpha (default: from config.py)"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=None,
        help="LoRA dropout (default: from config.py)"
    )

    # LoRA Initialization Method
    parser.add_argument(
        "--lora_init_method",
        type=str,
        default="default",
        choices=["default", "pissa", "orthogonal"],
        help="LoRA initialization method: default (kaiming), pissa (SVD-based), orthogonal (random orthogonal to top-r)"
    )
    parser.add_argument(
        "--pissa_niter",
        type=int,
        default=4,
        help="Number of iterations for randomized SVD in PiSSA (default: 4)"
    )
    parser.add_argument(
        "--pissa_modify_base_weight",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Whether to modify base weight in PiSSA (W = W - B @ A) (default: true)"
    )
    parser.add_argument(
        "--orthogonal_scale",
        type=float,
        default=0.01,
        help="Scale for orthogonal initialization (default: 0.01)"
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["alpaca", "alpaca_cleaned", "c4"],
        help="Dataset to use: alpaca (original), alpaca_cleaned (cleaned), c4"
    )

    # Paths
    parser.add_argument(
        "--input_model_path",
        type=str,
        default=None,
        help="Input model path (default: from config.py)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: from config.py)"
    )

    # ==================== Knowledge Distillation ====================
    parser.add_argument(
        "--use_kd",
        action="store_true",
        help="Enable KD mode: Use masked student approach like Stage 1"
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default=None,
        help="Path to masks.pkl file from Stage 1 (required for KD mode)"
    )
    parser.add_argument(
        "--kd_temperature",
        type=float,
        default=2.0,
        help="Temperature for KD softmax (default: 2.0)"
    )
    parser.add_argument(
        "--kd_weight",
        type=float,
        default=0.5,
        help="Weight for KD loss vs CE loss (default: 0.5)"
    )
    parser.add_argument(
        "--layer_kd_weight",
        type=float,
        default=0.9,
        help="Weight for layer KD within KD loss (default: 0.9)"
    )
    parser.add_argument(
        "--logit_kd_weight",
        type=float,
        default=0.1,
        help="Weight for logit KD within KD loss (default: 0.1)"
    )
    parser.add_argument(
        "--physical_prune_after",
        action="store_true",
        default=True,
        help="Physically prune the model after KD training (default: True)"
    )

    # ==================== Soft Orthogonal Regularization ====================
    parser.add_argument(
        "--use_soft_orthogonal",
        action="store_true",
        help="Enable soft orthogonal regularization for LoRA"
    )
    parser.add_argument(
        "--soft_orth_strategy",
        type=str,
        default=None,
        choices=["recover", "balanced", "orthogonal"],
        help="Preset strategy for soft orthogonal: recover (λ=0.001), balanced (λ=0.01), orthogonal (λ=0.1)"
    )
    parser.add_argument(
        "--soft_orth_lambda_u",
        type=float,
        default=0.01,
        help="Lambda for output subspace regularization (default: 0.01)"
    )
    parser.add_argument(
        "--soft_orth_lambda_v",
        type=float,
        default=0.01,
        help="Lambda for input subspace regularization (default: 0.01)"
    )
    parser.add_argument(
        "--soft_orth_svd_rank",
        type=int,
        default=64,
        help="Number of top singular vectors to use (default: 64)"
    )
    parser.add_argument(
        "--svd_cache_path",
        type=str,
        default=None,
        help="Path to pre-computed SVD cache file (optional)"
    )

    # ==================== Momentum LoRA ====================
    parser.add_argument(
        "--use_momentum_lora",
        action="store_true",
        help="Enable Momentum LoRA instead of standard PEFT LoRA"
    )
    parser.add_argument(
        "--momentum_rank1",
        type=int,
        default=16,
        help="Velocity LoRA rank (A1, B1) - periodically reset (default: 16)"
    )
    parser.add_argument(
        "--momentum_rank2",
        type=int,
        default=32,
        help="Momentum LoRA rank (A2, B2) - never reset (default: 32)"
    )
    parser.add_argument(
        "--momentum_beta",
        type=float,
        default=0.9,
        help="Momentum decay rate (default: 0.9, use 0.95 for high pruning)"
    )
    parser.add_argument(
        "--momentum_reset_interval",
        type=int,
        default=100,
        help="Steps between velocity LoRA resets (default: 100)"
    )

    # ==================== Regularization ====================
    parser.add_argument(
        "--regularization",
        type=str,
        default="none",
        choices=["none", "l1", "l2"],
        help="Regularization type for LoRA parameters: none (default), l1, l2"
    )
    parser.add_argument(
        "--reg_lambda",
        type=float,
        default=1e-4,
        help="Regularization strength (default: 1e-4)"
    )

    # Seed for reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Initialize distributed environment (multi-GPU DDP)
    # Must be called BEFORE any CUDA operations or model loading
    setup_distributed()

    # Parse command line arguments
    args = parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)
    print(f"[Config] Random Seed: {args.seed}")

    print("=" * 70)
    if args.use_kd:
        print("Stage-2: Fine-tuning with Knowledge Distillation (KD Mode)")
        print("=" * 70)
        print("\nObjective: Train masked student with KD, then physically prune")
    else:
        print("Stage-2: Fine-tuning Physically Pruned Model")
        print("=" * 70)
        print("\nObjective: Fine-tune the physically compressed model")
    print("=" * 70)

    # Configuration
    lora_config = LoRAConfig()
    finetuning_config = FineTuningConfig()
    data_config = DataConfig()
    training_config = TrainingConfig()

    # Model selection (--model_path overrides HuggingFace download)
    if args.model_path:
        lora_config.model_name = args.model_path
    elif args.model == "custom":
        raise ValueError("--model_path is required when using --model custom")
    else:
        lora_config.model_name = get_model_name(args.model)

    # Dataset path (offline support)
    if hasattr(args, 'dataset_path') and args.dataset_path:
        data_config.dataset_path = args.dataset_path

    # Override configs with command line arguments
    if args.num_epochs is not None:
        finetuning_config.num_epochs = args.num_epochs
    if args.micro_batch_size is not None:
        finetuning_config.micro_batch_size = args.micro_batch_size
    if args.batch_size is not None:
        finetuning_config.batch_size = args.batch_size
    # Gradient accumulation: explicit > computed from batch_size/micro_batch_size
    if args.gradient_accumulation_steps is not None:
        finetuning_config.gradient_accumulation_steps = args.gradient_accumulation_steps
    else:
        finetuning_config.gradient_accumulation_steps = finetuning_config.batch_size // finetuning_config.micro_batch_size
    if args.max_seq_length is not None:
        finetuning_config.max_seq_length = args.max_seq_length
    if args.learning_rate is not None:
        finetuning_config.learning_rate = args.learning_rate
    if args.warmup_steps is not None:
        finetuning_config.warmup_steps = args.warmup_steps
        finetuning_config.warmup_ratio = 0.0
    else:
        finetuning_config.warmup_steps = 0
    if args.lr_scheduler_type is not None:
        finetuning_config.lr_scheduler_type = args.lr_scheduler_type

    # Separate learning rates for attention/FFN
    attn_lr = args.attn_lr if args.attn_lr is not None else finetuning_config.learning_rate
    ffn_lr = args.ffn_lr if args.ffn_lr is not None else finetuning_config.learning_rate
    use_separate_lr = (args.attn_lr is not None or args.ffn_lr is not None)

    # LoRA settings
    if args.lora_r is not None:
        lora_config.lora_r = args.lora_r
    if args.lora_alpha is not None:
        lora_config.lora_alpha = args.lora_alpha
    if args.lora_dropout is not None:
        lora_config.lora_dropout = args.lora_dropout

    # Paths
    if args.input_model_path is not None:
        lora_config.student_physically_pruned_path = args.input_model_path
    if args.output_dir is not None:
        lora_config.student_finetuned_path = args.output_dir

    # Dataset
    if args.dataset == "c4":
        data_config.use_c4(num_samples=20000, max_length=512)
    elif args.dataset is not None:
        data_config.dataset_name = args.dataset

    print(f"\n[Config] Output: {lora_config.student_finetuned_path}")
    print(f"[Config] Dataset: {data_config.dataset_name}")
    print(f"[Config] LoRA r: {lora_config.lora_r}, alpha: {lora_config.lora_alpha}, dropout: {lora_config.lora_dropout}")
    print(f"[Config] LoRA Init: {args.lora_init_method}")
    print(f"[Config] Learning Rate: {finetuning_config.learning_rate}")
    if use_separate_lr:
        print(f"  → Attention LR: {attn_lr}")
        print(f"  → FFN LR: {ffn_lr}")
    print(f"[Config] Epochs: {finetuning_config.num_epochs}")
    print(f"[Config] Batch Size: {finetuning_config.batch_size} (micro={finetuning_config.micro_batch_size} x accum={finetuning_config.gradient_accumulation_steps})")
    if finetuning_config.warmup_steps > 0:
        print(f"[Config] Warmup: {finetuning_config.warmup_steps} steps")
    else:
        print(f"[Config] Warmup: {finetuning_config.warmup_ratio*100:.0f}% ratio")
    print(f"[Config] LR Scheduler: {finetuning_config.lr_scheduler_type}")

    # KD Mode configuration
    use_kd = args.use_kd
    mask_tensors = None  # Will be loaded if KD mode
    mask_info = None     # Will be loaded if KD mode

    if use_kd:
        print(f"\n[KD Mode] Enabled")
        print(f"[KD Mode] Method: Masked Student + Teacher KD (like Stage 1)")
        print(f"[KD Mode] Base Model: {lora_config.model_name}")
        print(f"[KD Mode] Temperature: {args.kd_temperature}")
        print(f"[KD Mode] KD Weight: {args.kd_weight}")
        print(f"[KD Mode] Layer KD: {args.layer_kd_weight}, Logit KD: {args.logit_kd_weight}")

        # Find mask path
        mask_path = args.mask_path
        if mask_path is None:
            # Try default locations
            default_paths = [
                "./student_lora_output/masks.pkl",
                os.path.join(lora_config.student_physically_pruned_path, "..", "masks.pkl"),
            ]
            for path in default_paths:
                if os.path.exists(path):
                    mask_path = path
                    break

        if mask_path is None or not os.path.exists(mask_path):
            raise FileNotFoundError(
                f"Mask file not found. KD mode requires masks from Stage 1.\n"
                f"Please specify --mask_path or ensure masks.pkl exists in ./student_lora_output/"
            )

        print(f"[KD Mode] Mask Path: {mask_path}")
        mask_tensors = load_masks(mask_path)
        print(f"[KD Mode] Loaded {len(mask_tensors)} masks")

        mask_info_path = mask_path.replace("masks.pkl", "mask_info.json")
        if os.path.exists(mask_info_path):
            with open(mask_info_path, 'r') as f:
                mask_info = json.load(f)
            print(f"[KD Mode] Loaded mask_info from {mask_info_path}")
        else:
            mask_info = {
                "num_heads": 32,
                "num_layers": 32,
                "hidden_size": 4096,
                "intermediate_size": 11008
            }
            print(f"[KD Mode] mask_info.json not found, using defaults")
    else:
        print(f"\n[Config] Input: {lora_config.student_physically_pruned_path}")
        print(f"[Config] Method: Standard LoRA Fine-tuning (physically pruned model)")

    # ==================== Soft Orthogonal Configuration ====================
    soft_orth_config = SoftOrthogonalConfig(
        use_soft_orthogonal=args.use_soft_orthogonal,
        svd_rank=args.soft_orth_svd_rank,
        lambda_U=args.soft_orth_lambda_u,
        lambda_V=args.soft_orth_lambda_v,
        strategy=args.soft_orth_strategy,
        svd_cache_path=args.svd_cache_path
    )

    if soft_orth_config.use_soft_orthogonal:
        print(f"\n[Soft Orthogonal] Enabled")
        print(f"[Soft Orthogonal] Strategy: {soft_orth_config.strategy}")
        print(f"[Soft Orthogonal] lambda_U: {soft_orth_config.lambda_U}")
        print(f"[Soft Orthogonal] lambda_V: {soft_orth_config.lambda_V}")
        print(f"[Soft Orthogonal] SVD Rank: {soft_orth_config.svd_rank}")

    # ==================== Momentum LoRA Configuration ====================
    use_momentum_lora = args.use_momentum_lora

    if use_momentum_lora:
        print(f"\n[Momentum LoRA] Enabled")
        print(f"[Momentum LoRA] rank1 (velocity): {args.momentum_rank1}")
        print(f"[Momentum LoRA] rank2 (momentum): {args.momentum_rank2}")
        print(f"[Momentum LoRA] beta: {args.momentum_beta}")
        print(f"[Momentum LoRA] reset_interval: {args.momentum_reset_interval}")
        if soft_orth_config.use_soft_orthogonal:
            print(f"[Momentum LoRA] Soft Orth lambda_U: {soft_orth_config.lambda_U}")
            print(f"[Momentum LoRA] Soft Orth lambda_V: {soft_orth_config.lambda_V}")

    # ==================== 1. Load Tokenizer ====================
    _offline = os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
    if use_kd:
        # KD Mode: Load tokenizer from original pretrained model
        tokenizer = AutoTokenizer.from_pretrained(lora_config.model_name, local_files_only=_offline)
    else:
        # Standard Mode: Load tokenizer from physically pruned model
        tokenizer = AutoTokenizer.from_pretrained(lora_config.student_physically_pruned_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ==================== 2. Load Model ====================
    meta_info = None

    if use_kd:
        # KD Mode: Load original pretrained model and apply masks
        print("\n" + "=" * 70)
        print("Loading Original Pretrained Model (KD Mode)")
        print("=" * 70)

        base_model = AutoModelForCausalLM.from_pretrained(
            lora_config.model_name,
            torch_dtype=torch.float16,
            device_map=get_device_map_for_loading(),
            trust_remote_code=True,
            local_files_only=_offline
        )
        if is_distributed():
            local_device = torch.device(f"cuda:{get_local_rank()}")
            base_model = base_model.to(local_device)
            print(f"[Model] Moved to {local_device}")
        print(f"[Model] Loaded: {lora_config.model_name}")

        # Apply LoRA and masks using create_student_model
        print("\n[LoRA] Applying LoRA with masks (Student model)...")
        model = create_student_model(base_model, lora_config, masks=mask_tensors)
        print("[LoRA] LoRA + Masks applied successfully")
        print("[KD Mode] Teacher = Base model (no masks), Student = Base model + Masks + LoRA")

    else:
        # Standard Mode: Load physically pruned model
        model, meta_info = load_physically_pruned_model(
            lora_config.student_physically_pruned_path
        )

        # VLM detection: if the loaded model is a Vision-Language Model,
        # restrict LoRA target_modules to the language_model subtree only,
        # so that the vision tower stays frozen and untouched.
        _is_vlm_loaded = type(model).__name__ in (
            "Qwen2_5_VLForConditionalGeneration",
            "Qwen3VLForConditionalGeneration",
        )
        if _is_vlm_loaded:
            from models.qwen_vl_pruning import lora_target_modules_vlm
            print("[LoRA] VLM detected — restricting target_modules to language_model.*")
            lora_config.target_modules = lora_target_modules_vlm()

        if use_momentum_lora:
            print("\n[LoRA] Applying Momentum LoRA to pruned model...")

            momentum_lora_config = {
                'type': 'momentum',
                'momentum': {
                    'rank1': args.momentum_rank1,
                    'rank2': args.momentum_rank2,
                    'beta': args.momentum_beta,
                    'reset_interval': args.momentum_reset_interval,
                },
                'target_modules': lora_config.target_modules,
                'soft_orthogonal': {
                    'enabled': soft_orth_config.use_soft_orthogonal,
                    'lambda_U': soft_orth_config.lambda_U,
                    'lambda_V': soft_orth_config.lambda_V,
                    'svd_rank': soft_orth_config.svd_rank,
                },
            }

            model = add_lora_to_model(model, momentum_lora_config)
            print("[LoRA] Momentum LoRA applied successfully")

        else:
            print("\n[LoRA] Applying standard LoRA to pruned model...")
            print(f"[LoRA] Init Method: {args.lora_init_method}")

            peft_config = LoraConfig(
                r=lora_config.lora_r,
                lora_alpha=lora_config.lora_alpha,
                lora_dropout=lora_config.lora_dropout,
                target_modules=lora_config.target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )

            model = get_peft_model(model, peft_config)

            # Apply LoRA initialization method
            if args.lora_init_method == "pissa":
                print(f"[LoRA Init] Applying PiSSA initialization...")
                print(f"  niter: {args.pissa_niter}")
                print(f"  modify_base_weight: {args.pissa_modify_base_weight}")
                modify_base = args.pissa_modify_base_weight.lower() == "true"
                apply_pissa_init(
                    model=model,
                    lora_config=lora_config,
                    niter=args.pissa_niter,
                    modify_base_weight=modify_base
                )
                print("[LoRA Init] PiSSA initialization applied")

            elif args.lora_init_method == "orthogonal":
                print(f"[LoRA Init] Applying Orthogonal initialization...")
                print(f"  scale: {args.orthogonal_scale}")
                apply_orthogonal_init(
                    model=model,
                    lora_config=lora_config,
                    scale=args.orthogonal_scale,
                    niter=args.pissa_niter
                )
                print("[LoRA Init] Orthogonal initialization applied")

            else:
                print("[LoRA Init] Using default (Kaiming) initialization")

            print("[LoRA] Standard LoRA applied successfully")

    print_trainable_parameters(model)

    # ==================== Soft Orthogonal: SVD Cache & Buffer Registration ====================
    svd_cache = None
    if soft_orth_config.use_soft_orthogonal:
        # Try to find SVD cache from Stage 1
        svd_cache_candidates = []

        # 1. User-specified path
        if soft_orth_config.svd_cache_path:
            svd_cache_candidates.append(soft_orth_config.svd_cache_path)

        # 2. Default location: Stage 1 output directory (student_lora_output/svd_cache.pt)
        stage1_default_path = "./student_lora_output/svd_cache.pt"
        svd_cache_candidates.append(stage1_default_path)

        # 3. Same directory as input model
        input_dir_cache = os.path.join(lora_config.student_physically_pruned_path, "..", "svd_cache.pt")
        svd_cache_candidates.append(input_dir_cache)

        # Try to load from candidates
        svd_cache_loaded = False
        for cache_path in svd_cache_candidates:
            if os.path.exists(cache_path):
                print(f"\n[SVD Cache] Loading from Stage 1: {cache_path}")
                print("[SVD Cache] Using original pretrained model's SVD (not pruned model)")
                svd_cache = load_svd_cache(cache_path)
                svd_cache_loaded = True
                break

        if not svd_cache_loaded:
            print("\n" + "=" * 70)
            print("[ERROR] SVD Cache not found!")
            print("=" * 70)
            print("\nSoft Orthogonal Regularization requires SVD cache from Stage 1.")
            print("The SVD must be computed from the ORIGINAL pretrained model,")
            print("not from the pruned model.")
            print("\nPlease either:")
            print("  1. Run Stage 1 with --use_soft_orthogonal to generate svd_cache.pt")
            print("  2. Specify the path with --svd_cache_path <path_to_svd_cache.pt>")
            print("\nSearched locations:")
            for path in svd_cache_candidates:
                print(f"  - {path}")
            print("=" * 70)
            raise FileNotFoundError("SVD cache from Stage 1 is required for Soft Orthogonal in Stage 2")

        # Register SVD buffers to model for efficient GPU computation
        if svd_cache is not None:
            if use_momentum_lora:
                apply_svd_cache_to_momentum_layers(model, svd_cache)
                print(f"[SVD] Applied SVD cache to Momentum LoRA layers")
            else:
                num_registered = register_svd_buffers_to_model(model, svd_cache)
                print(f"[SVD] Registered {num_registered} SVD buffers")

    # ==================== 3. Prepare Dataset ====================
    print("\n" + "=" * 70)
    print("Loading Dataset")
    print("=" * 70)

    if data_config.dataset_name == "c4":
        train_dataset, eval_dataset = load_c4_dataset(
            data_config,
            num_samples=data_config.num_samples or 20000,
            max_length=data_config.max_length or 512
        )
        print(f"[Dataset] Using C4 with {data_config.num_samples or 20000} samples")
    else:
        train_dataset, eval_dataset = load_alpaca_dataset(data_config)
        train_dataset = train_dataset.map(
            format_alpaca_prompt,
            remove_columns=train_dataset.column_names
        )
        if eval_dataset:
            eval_dataset = eval_dataset.map(
                format_alpaca_prompt,
                remove_columns=eval_dataset.column_names
            )

    print(f"[Dataset] Train: {len(train_dataset)}")
    if eval_dataset:
        print(f"[Dataset] Eval: {len(eval_dataset)}")

    # Tokenize
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, finetuning_config.max_seq_length),
        batched=True,
        remove_columns=train_dataset.column_names
    )

    if eval_dataset:
        eval_dataset = eval_dataset.map(
            lambda x: tokenize_function(x, tokenizer, finetuning_config.max_seq_length),
            batched=True,
            remove_columns=eval_dataset.column_names
        )

    # ==================== 4. Training Arguments ====================
    training_args = TrainingArguments(
        output_dir=lora_config.student_finetuned_path,
        num_train_epochs=finetuning_config.num_epochs,
        per_device_train_batch_size=finetuning_config.micro_batch_size,
        per_device_eval_batch_size=finetuning_config.micro_batch_size,
        gradient_accumulation_steps=finetuning_config.gradient_accumulation_steps,
        learning_rate=finetuning_config.learning_rate,
        warmup_steps=finetuning_config.warmup_steps if finetuning_config.warmup_steps > 0 else 0,
        warmup_ratio=finetuning_config.warmup_ratio if finetuning_config.warmup_steps == 0 else 0.0,
        weight_decay=finetuning_config.weight_decay,
        max_grad_norm=finetuning_config.max_grad_norm,
        lr_scheduler_type=finetuning_config.lr_scheduler_type,
        fp16=finetuning_config.fp16,
        optim=finetuning_config.optim,
        logging_steps=finetuning_config.logging_steps,
        save_strategy="no",
        save_total_limit=1,
        eval_strategy="no",
        eval_steps=None,
        load_best_model_at_end=False,
        metric_for_best_model=None,
        report_to="none",
        seed=42
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    callbacks = [DetailedLoggingCallback()]

    if use_momentum_lora:
        momentum_callback = MomentumLoRAUpdateCallback(model)
        callbacks.append(momentum_callback)
        print(f"[Callback] MomentumLoRAUpdateCallback added")

    # if eval_dataset and finetuning_config.early_stopping_patience > 0:
    #     callbacks.append(
    #         EarlyStoppingCallback(
    #             early_stopping_patience=finetuning_config.early_stopping_patience,
    #             early_stopping_threshold=finetuning_config.early_stopping_threshold
    #         )
    #     )

    # ==================== 5. Custom Optimizer (Separate LR for Attention/FFN) ====================
    custom_optimizer = None
    custom_scheduler = None

    if use_separate_lr:
        print(f"\n[Optimizer] Creating custom optimizer with separate learning rates")
        custom_optimizer = create_optimizer_with_separate_lr(
            model=model,
            base_lr=finetuning_config.learning_rate,
            attn_lr=attn_lr,
            ffn_lr=ffn_lr,
            weight_decay=finetuning_config.weight_decay
        )

        # Create scheduler
        num_training_steps = (
            len(train_dataset) // finetuning_config.batch_size
        ) * finetuning_config.num_epochs
        if finetuning_config.warmup_steps > 0:
            num_warmup_steps = finetuning_config.warmup_steps
        else:
            num_warmup_steps = int(num_training_steps * finetuning_config.warmup_ratio)

        custom_scheduler = get_scheduler(
            name=finetuning_config.lr_scheduler_type,
            optimizer=custom_optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        print(f"[Optimizer] Scheduler: {finetuning_config.lr_scheduler_type}, warmup={num_warmup_steps}, total={num_training_steps}")

    # ==================== 6. Trainer ====================
    trainer = Stage2Trainer(
        use_kd=use_kd,
        temperature=args.kd_temperature,
        kd_weight=args.kd_weight,
        layer_kd_weight=args.layer_kd_weight,
        logit_kd_weight=args.logit_kd_weight,
        use_momentum_lora=use_momentum_lora,
        regularization=args.regularization,
        reg_lambda=args.reg_lambda,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=callbacks,
        optimizers=(custom_optimizer, custom_scheduler) if use_separate_lr else (None, None)
    )

    # Set PEFT model reference for KD mode (needed for teacher forward)
    if use_kd:
        trainer.set_peft_model(model)

    # Enable soft orthogonal regularization if configured
    if soft_orth_config.use_soft_orthogonal and not use_momentum_lora:
        trainer.set_soft_orthogonal(soft_orth_config)

    # Enable Momentum LoRA soft orthogonal if configured
    if use_momentum_lora and soft_orth_config.use_soft_orthogonal:
        trainer.set_momentum_soft_orthogonal(
            soft_orth_config.lambda_U,
            soft_orth_config.lambda_V
        )

    # ==================== 6. Fine-tune ====================
    print("\n" + "=" * 70)
    print("Starting Fine-tuning")
    print("=" * 70)

    train_result = trainer.train()

    print(f"\n[Training] Completed!")
    print(f"[Training] Final loss: {train_result.training_loss:.4f}")

    # ==================== 7. Physical Pruning (KD Mode only) ====================
    if use_kd and args.physical_prune_after:
        print("\n" + "=" * 70)
        print("Physical Pruning (KD Mode)")
        print("=" * 70)
        print("[Prune] Converting masked model to physically pruned model...")

        # Merge LoRA into base model first
        from peft import PeftModel
        if isinstance(model, PeftModel):
            model = model.merge_and_unload()
            print("[Prune] LoRA merged into base model")

        # Physical pruning
        pruner = PhysicalPruner(model, mask_tensors, mask_info)
        pruned_model, pruning_stats = pruner.prune()

        print(f"[Prune] Original params: {pruning_stats['size_comparison']['original']['total_params']:,}")
        print(f"[Prune] Pruned params: {pruning_stats['size_comparison']['pruned']['total_params']:,}")
        print(f"[Prune] Reduction: {pruning_stats['size_comparison']['reduction']['params_percent']:.2f}%")

        # Save physically pruned model
        print("\n" + "=" * 70)
        print("Saving Physically Pruned Model")
        print("=" * 70)

        if is_main_process():
            pruned_model.save_pretrained(lora_config.student_finetuned_path)
            tokenizer.save_pretrained(lora_config.student_finetuned_path)
        barrier()

        # Update meta_info with pruning stats
        meta_info = pruning_stats

    else:
        # ==================== 7. Save (Standard Mode) ====================
        print("\n" + "=" * 70)
        print("Saving Fine-tuned Model")
        print("=" * 70)

        if is_main_process():
            trainer.save_model(lora_config.student_finetuned_path)
            tokenizer.save_pretrained(lora_config.student_finetuned_path)
        barrier()

    # Save metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Save meta info
    finetuning_meta = {
        "stage": "stage2_finetuning",
        "method": "KD Fine-tuning" if use_kd else "LoRA Fine-tuning",
        "input_model": lora_config.model_name if use_kd else lora_config.student_physically_pruned_path,
        "output_model": lora_config.student_finetuned_path,
        "dataset": data_config.dataset_name,
        "lora_config": {
            "r": lora_config.lora_r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "target_modules": lora_config.target_modules,
            "init_method": args.lora_init_method
        },
        "finetuning_config": {
            "learning_rate": finetuning_config.learning_rate,
            "attn_lr": attn_lr if use_separate_lr else None,
            "ffn_lr": ffn_lr if use_separate_lr else None,
            "use_separate_lr": use_separate_lr,
            "num_epochs": finetuning_config.num_epochs,
            "batch_size": finetuning_config.batch_size,
            "weight_decay": finetuning_config.weight_decay,
            "lr_scheduler": finetuning_config.lr_scheduler_type
        },
        "kd_config": {
            "enabled": use_kd,
            "temperature": args.kd_temperature if use_kd else None,
            "kd_weight": args.kd_weight if use_kd else None,
            "layer_kd_weight": args.layer_kd_weight if use_kd else None,
            "logit_kd_weight": args.logit_kd_weight if use_kd else None,
            "physical_prune_after": args.physical_prune_after if use_kd else None
        },
        "soft_orthogonal_config": {
            "enabled": soft_orth_config.use_soft_orthogonal,
            "strategy": soft_orth_config.strategy if soft_orth_config.use_soft_orthogonal else None,
            "lambda_U": soft_orth_config.lambda_U if soft_orth_config.use_soft_orthogonal else None,
            "lambda_V": soft_orth_config.lambda_V if soft_orth_config.use_soft_orthogonal else None,
            "svd_rank": soft_orth_config.svd_rank if soft_orth_config.use_soft_orthogonal else None
        },
        "training_results": {
            "final_loss": float(train_result.training_loss),
            "total_steps": train_result.global_step
        },
        "physical_pruning_meta": meta_info
    }

    if is_main_process():
        meta_path = os.path.join(lora_config.student_finetuned_path, "finetuning_meta.json")
        with open(meta_path, 'w') as f:
            json.dump(finetuning_meta, f, indent=2)

    print(f"[Save] Model: {lora_config.student_finetuned_path}")
    print(f"[Save] Meta: {meta_path}")

    print("\n" + "=" * 70)
    print("Stage-2 Fine-tuning Completed!")
    print("=" * 70)
    print("\nNext: python evaluate_finetuned_model.py")


if __name__ == "__main__":
    main()
