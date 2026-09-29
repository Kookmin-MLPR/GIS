from dataclasses import dataclass
from typing import List, Dict

SUPPORTED_MODELS: Dict[str, str] = {
    "llama1-7b": "huggyllama/llama-7b",
    "llama1-13b": "huggyllama/llama-13b",
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "llama2-13b": "meta-llama/Llama-2-13b-hf",
    "llama3-8b": "meta-llama/Meta-Llama-3-8B",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B",
    "phi-2": "microsoft/phi-2",
    "qwen3-8b": "Qwen/Qwen3-8B",
    # VLM models — pruned via LLM-only path (see models/qwen_vl_pruning.py)
    "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
}
# Backward compatibility alias
LLAMA_MODELS = SUPPORTED_MODELS

# Vision-Language Models — load via AutoModelForImageTextToText, prune
# only the inner language_model submodule.
VLM_MODELS = {"qwen2.5-vl-7b", "qwen3-vl-8b"}


def is_vlm_key(model_key: str) -> bool:
    """Return True if model_key (or its HF path) is a registered VLM."""
    return model_key in VLM_MODELS

def get_model_name(model_key: str = "llama1-7b") -> str:
    if model_key in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[model_key]
    else:
        return model_key

@dataclass
class LoRAConfig:
    # Model
    # - "llama1-13b": LLaMA 1 13B (huggyllama/llama-13b)
    # - "llama2-7b": LLaMA 2 7B (meta-llama/Llama-2-7b-hf)
    # - "llama2-13b": LLaMA 2 13B (meta-llama/Llama-2-13b-hf)
    model_name: str = "huggyllama/llama-7b"  # Default: LLaMA 1 7B


    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: List[str] = None  # ["q_proj", "v_proj", "up_proj", "gate_proj"]

    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 64
    micro_batch_size: int = 32
    learning_rate: float = 3e-4
    warmup_steps: int = 0
    max_seq_length: int = 128

    # Optimization
    gradient_accumulation_steps: int = 2  # 128 / 4

    fp16: bool = True
    optim: str = "adamw_torch"
    weight_decay: float = 0.0

    # DataLoader
    num_workers: int = 8

    # Paths
    student_output_dir: str = "./student_lora_output"
    student_physically_pruned_path: str = "./student_physically_pruned"
    student_finetuned_path: str = "./student_finetuned_output"  # Stage-2 fine-tuned
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3

    def __post_init__(self):
        #
        # Attention:
        #   - q_proj: Query projection ✅
        #   - k_proj: Key projection ✅
        #   - v_proj: Value projection ✅
        #   - o_proj: Output projection ✅
        #
        # FFN (MLP):
        #   - gate_proj: Gate projection ✅
        #   - up_proj: Up projection ✅
        #   - down_proj: Down projection ✅
        self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "dense", "gate_proj", "up_proj", "down_proj", "fc1", "fc2"]
        self.gradient_accumulation_steps = self.batch_size // self.micro_batch_size

@dataclass
class DataConfig:
    dataset_name: str = "tatsu-lab/alpaca"
    dataset_path: str = None
    dataset_config: str = None
    validation_split: float = 0.0
    streaming: bool = False  # C4 streaming mode
    num_samples: int = None
    max_length: int = 128

    def use_c4(self, num_samples=20000, max_length=128):
        self.dataset_name = "c4"
        self.dataset_config = "en"
        self.streaming = True
        self.num_samples = num_samples
        self.max_length = max_length

    def use_c4_shard(self, num_samples=20000, max_length=128, shard_id=0):
        self.dataset_name = "c4_shard"
        self.dataset_config = "en"
        self.streaming = False
        self.num_samples = num_samples
        self.max_length = max_length
        self.shard_id = shard_id

    def use_c4_local(self, num_samples=20000, max_length=128, data_path="./data/c4/c4_train_20k.jsonl"):
        self.dataset_name = "c4_local"
        self.dataset_config = None
        self.streaming = False
        self.num_samples = num_samples
        self.max_length = max_length
        self.data_path = data_path

@dataclass
class EvalConfig:
    # Benchmark datasets
    eval_datasets: List[str] = None

    # Evaluation parameters
    eval_batch_size: int = 16
    max_eval_samples: int = None
    num_fewshot: int = 0  # 0-shot evaluation

    # lm_eval specific settings
    use_lm_eval: bool = True
    device: str = "cuda:0"
    limit: int = None

    # Output
    eval_output_dir: str = "./eval_results"
    student_eval_output_dir: str = "./eval_results/student"

    def __post_init__(self):
        self.eval_datasets = [
            "winogrande",
            "hellaswag",
            "arc_easy",
            "arc_challenge",
            "piqa",
        ]

    def get_tasks_string(self) -> str:
        return ",".join(self.eval_datasets)

@dataclass
class PruningConfig:
    # Pruning ratios for ~50% overall pruning
    head_pruning_ratio: float = 0.30      # 30% heads pruned (22/32 heads kept)
    embedding_pruning_ratio: float = 0.10  # 10% embedding dims pruned (58/64 groups kept)
    ffn_pruning_ratio: float = 0.50       # 50% FFN dims pruned (86/172 groups kept)

    # Pruning strategy (Layer-wise for standard architecture)
    global_head_pruning: bool = False
    global_ffn_pruning: bool = False

    # Mask mode for LoRA
    mask_mode: str = "dynamic"  # "dynamic" or "static"

    mask_application_timing: str = "after"  # "before" or "after"

    # Gradual pruning
    initial_sparsity: float = 0.0
    final_sparsity: float = 0.5
    pruning_steps: int = 1000
    pruning_frequency: int = 100

    # LoRA Merging (Optional)
    enable_lora_merging: bool = False
    lora_merging_frequency: int = 500

    # - "weight": |W_frozen × (LoRA_A @ LoRA_B)|
    # - "fisher": |W_frozen × (LoRA_B.grad² @ LoRA_A.grad²)| (Empirical Fisher Information)
    importance_metric: str = "gradient"

    # - "weight_norm" / "wn": ||W_gate[i,:]|| + ||W_up[i,:]|| + ||W_down[:,i]||
    # - "weight_taylor" / "wt": |W⊙∇W| for gate, up, down projections
    # - "weight_ganda" / "wg": |∇gate×gate| + |∇up×up| + |∇down_in×down_in|
    # - "weight_wanda" / "ww": |W_gate×X| + |W_up×X| + |W_down×X| (WANDA style: weight × activation)
    ffn_importance_mode: str = "down"  # "down", "up", "all", "weight_norm/wn", "weight_taylor/wt", "weight_ganda/wg", "weight_wanda/ww"

    # - "qk_attention": Q×K importance attention (Q_imp @ K_imp^T)
    # - "weight_norm" / "wn": ||W_q[:,i]|| + ||W_k[:,i]|| + ||W_v[:,i]|| + ||W_o[i,:]||
    # - "weight_taylor" / "wt": |W⊙∇W| for Q, K, V, O projections
    # - "weight_ganda" / "wg": |∇Q[i]×Q[i]| + |∇K[i]×K[i]| + |∇V[i]×V[i]| + |∇O[i]×O[i]|
    # - "weight_wanda" / "ww": |W_q×X| + |W_k×X| + |W_v×X| + |W_o×X| (WANDA style: weight × activation)
    head_importance_mode: str = "o"  # "o", "v", "qk", "qk_element", "qk_attention", "weight_norm/wn", "weight_taylor/wt", "weight_ganda/wg", "weight_wanda/ww"

    # - "taylor": |gradient × weight|
    # - "wanda": |weight × activation|
    # - "cett": Contribution to hidden state (CETT)
    # - "ganda_ntk_dyn": |grad × activation| × dynamics (gradient norm²)
    # - "ganda_ntk_coupling": |grad × activation| × coupling (sample interaction)
    # - "ganda_ntk_combined": |grad × activation| × dynamics × coupling
    importance_method: str = "ganda"

    # Mask sharing rules
    share_qk_mask: bool = True
    share_v_proj_mask: bool = True
    share_fc_mask: bool = True

    # Constraints
    min_heads_per_layer: int = 1
    ffn_dimension_multiple: int = 64  # FFN dims must be multiple of 64
    embedding_dimension_multiple: int = 64  # Embedding dims must be multiple of 64
    allow_full_block_pruning: bool = False

    enable_dimension_pruning: bool = False
    dimension_group_size: int = 16
    dimension_pruning_ratio: float = 0.25
    global_dimension_pruning: bool = True
    separate_qkv_dimension: bool = True

    enable_svd_diversity: bool = False
    svd_diversity_method: str = "dominant"
    svd_diversity_alpha: float = 1.0

    enable_iterative_pruning: bool = False
    pruning_step_size: float = 0.05

    use_ntk_adjustment: bool = False
    ntk_adjustment_alpha: float = 0.3

    ntk_adjustment_mode: str = "previous"     # "previous" or "current"

    ntk_sensitivity_direction: str = "normal"  # "normal" or "inverse"

    ntk_target_mode: str = "default"  # "default" or "all"

    ntk_granularity: str = "layer"  # "layer" or "unit"

    ntk_unit_mode: str = "group"  # "group" or "neuron"

    ntk_method: str = "frobenius"  # "frobenius", "eigenvalue", "delta", "quadform"

    ntk_eigenvalue_k: int = 5

    use_ntk_embedding: bool = False

    use_iterative_premasking: bool = False

    # Logging
    log_sparsity_steps: int = 100

@dataclass
class TrainingConfig:
    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 64
    micro_batch_size: int = 32
    learning_rate: float = 3e-4
    warmup_steps: int = 100
    warmup_ratio: float = None
    max_seq_length: int = 128

    # Optimization
    gradient_accumulation_steps: int = 2
    fp16: bool = True
    optim: str = "adamw_torch"
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    # DataLoader
    num_workers: int = 16

    # Checkpointing
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = None
    save_total_limit: int = 3

    def __post_init__(self):
        self.gradient_accumulation_steps = self.batch_size // self.micro_batch_size

@dataclass
class FineTuningConfig:
    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 64
    micro_batch_size: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    max_seq_length: int = 512

    # Optimization
    gradient_accumulation_steps: int = 8
    fp16: bool = True
    optim: str = "adamw_torch"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"

    # DataLoader
    num_workers: int = 16

    # Checkpointing
    logging_steps: int = 200
    save_steps: int = 500
    eval_steps: int = None
    save_total_limit: int = 3

    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001

    def __post_init__(self):
        self.gradient_accumulation_steps = self.batch_size // self.micro_batch_size


@dataclass
class MomentumLoRAConfig:

    # Enable/disable Momentum LoRA
    # - False: Use standard PEFT LoRA (default, simpler and faster)
    # - True: Use custom Momentum LoRA (better recovery, more complex)
    use_momentum_lora: bool = False

    # Velocity LoRA rank (A1, B1) - periodically reset
    rank1: int = 16

    # Momentum LoRA rank (A2, B2) - never reset, learns long-term patterns
    # Should be >= rank1 for better capacity
    rank2: int = 32

    # Momentum decay rate
    # - Higher beta (0.95-0.99): More stable, better for high pruning rates
    # - Lower beta (0.85-0.9): Faster adaptation, better for low pruning rates
    beta: float = 0.9

    # Steps between velocity LoRA (A1, B1) resets
    # - Lower values: More frequent resets, faster short-term adaptation
    # - Higher values: Longer learning periods before reset
    reset_interval: int = 100

    # LoRA scaling factor (alpha / rank)
    # Standard LoRA uses alpha = rank
    alpha: float = None  # None means alpha = rank1

    # Soft orthogonal regularization settings
    # Works with both standard and momentum LoRA
    use_soft_orthogonal: bool = True
    lambda_U: float = 0.01  # Output subspace regularization strength
    lambda_V: float = 0.01  # Input subspace regularization strength
    svd_rank: int = 64  # Number of singular vectors to use
    svd_cache_path: str = None  # Path to pre-computed SVD cache

    # Target modules (same as standard LoRA)
    target_modules: List[str] = None

    def __post_init__(self):
        # Default target modules for LLaMA-style models
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

        # Default alpha = rank1
        if self.alpha is None:
            self.alpha = float(self.rank1)

    def get_config_dict(self) -> Dict:
        """Convert to dictionary format for add_lora_to_model()."""
        if self.use_momentum_lora:
            return {
                'type': 'momentum',
                'momentum': {
                    'rank1': self.rank1,
                    'rank2': self.rank2,
                    'beta': self.beta,
                    'reset_interval': self.reset_interval,
                },
                'target_modules': self.target_modules,
                'soft_orthogonal': {
                    'enabled': self.use_soft_orthogonal,
                    'lambda_U': self.lambda_U,
                    'lambda_V': self.lambda_V,
                    'svd_rank': self.svd_rank,
                },
            }
        else:
            return {
                'type': 'standard',
                'rank': self.rank1,  # Use rank1 as standard LoRA rank
                'lora_alpha': int(self.alpha),
                'dropout': 0.0,
                'target_modules': self.target_modules,
                'soft_orthogonal': {
                    'enabled': self.use_soft_orthogonal,
                    'lambda_U': self.lambda_U,
                    'lambda_V': self.lambda_V,
                    'svd_rank': self.svd_rank,
                },
            }

    @classmethod
    def for_high_pruning(cls, pruning_ratio: float = 0.7) -> "MomentumLoRAConfig":
        """
        Create config optimized for high pruning rates (70%+).

        High pruning needs:
        - Higher beta for stability
        - Larger momentum rank
        - Less frequent resets
        - Lower soft orth lambda (more focus on recovery)
        """
        return cls(
            use_momentum_lora=True,
            rank1=16,
            rank2=48,  # Larger momentum rank
            beta=0.95,  # Higher beta for stability
            reset_interval=150,  # Less frequent resets
            use_soft_orthogonal=True,
            lambda_U=0.005,  # Lower lambda for more recovery focus
            lambda_V=0.005,
            svd_rank=64,
        )

    @classmethod
    def for_moderate_pruning(cls, pruning_ratio: float = 0.5) -> "MomentumLoRAConfig":
        """
        Create config optimized for moderate pruning rates (40-70%).
        This is the default balanced configuration.
        """
        return cls(
            use_momentum_lora=True,
            rank1=16,
            rank2=32,
            beta=0.9,
            reset_interval=100,
            use_soft_orthogonal=True,
            lambda_U=0.01,
            lambda_V=0.01,
            svd_rank=64,
        )

    @classmethod
    def for_low_pruning(cls, pruning_ratio: float = 0.3) -> "MomentumLoRAConfig":
        """
        Create config optimized for low pruning rates (20-40%).

        Low pruning can use:
        - Lower beta for faster adaptation
        - Smaller momentum rank (less capacity needed)
        - More frequent resets
        """
        return cls(
            use_momentum_lora=True,
            rank1=16,
            rank2=24,  # Smaller momentum rank
            beta=0.85,  # Lower beta
            reset_interval=50,  # More frequent resets
            use_soft_orthogonal=True,
            lambda_U=0.01,
            lambda_V=0.01,
            svd_rank=64,
        )


# Hyperparameter guidelines for Momentum LoRA
MOMENTUM_LORA_GUIDELINES = """
Momentum LoRA Hyperparameter Guidelines
=======================================

| Pruning Rate | beta | reset_interval | rank1 | rank2 | lambda |
|--------------|------|----------------|-------|-------|--------|
| 20-40%       | 0.85 | 50             | 16    | 24    | 0.01   |
| 40-70%       | 0.9  | 100            | 16    | 32    | 0.01   |
| 70-90%       | 0.95 | 150            | 16    | 48    | 0.005  |

When to use Momentum LoRA:
- High pruning rates (>50%): Significant improvement
- Complex recovery tasks: Better long-term learning
- Sufficient compute budget: ~20% slower than standard LoRA

When to use Standard LoRA:
- Low pruning rates (<30%): Simpler is better
- Fast iteration needed: Standard LoRA is faster
- Limited compute budget: Less memory overhead

Expected Performance (LLaMA-7B, 50% pruning):
- Standard LoRA:          PPL increase ~8.5
- Momentum LoRA:          PPL increase ~6.5 (24% better)
- Momentum + Soft Orth:   PPL increase ~5.8 (32% better)
"""
