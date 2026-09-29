
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FFNDescriptor:
    has_gate: bool                  # True=SwiGLU(LLaMA), False=Standard(Phi-2)
    up_proj_names: List[str]        # ["gate_proj", "up_proj"] or ["fc1"]
    down_proj_name: str             # "down_proj" or "fc2"

    @property
    def all_proj_names(self) -> List[str]:
        return self.up_proj_names + [self.down_proj_name]


@dataclass
class AttnDescriptor:
    qkv_proj_names: List[str]       # ["q_proj", "k_proj", "v_proj"]
    output_proj_name: str           # "o_proj" or "dense"

    @property
    def all_proj_names(self) -> List[str]:
        return self.qkv_proj_names + [self.output_proj_name]


@dataclass
class ModelDescriptor:
    model_type: str
    ffn: FFNDescriptor
    attn: AttnDescriptor
    lora_target_modules: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.lora_target_modules:
            self.lora_target_modules = self.attn.all_proj_names + self.ffn.all_proj_names


LLAMA_DESCRIPTOR = ModelDescriptor(
    model_type="llama",
    ffn=FFNDescriptor(
        has_gate=True,
        up_proj_names=["gate_proj", "up_proj"],
        down_proj_name="down_proj",
    ),
    attn=AttnDescriptor(
        qkv_proj_names=["q_proj", "k_proj", "v_proj"],
        output_proj_name="o_proj",
    ),
)

PHI_DESCRIPTOR = ModelDescriptor(
    model_type="phi",
    ffn=FFNDescriptor(
        has_gate=False,
        up_proj_names=["fc1"],
        down_proj_name="fc2",
    ),
    attn=AttnDescriptor(
        qkv_proj_names=["q_proj", "k_proj", "v_proj"],
        output_proj_name="dense",
    ),
)

_DESCRIPTOR_REGISTRY = {
    "llama": LLAMA_DESCRIPTOR,
    "mistral": LLAMA_DESCRIPTOR,
    "qwen3": LLAMA_DESCRIPTOR,
    "phi": PHI_DESCRIPTOR,
}


def get_model_descriptor(model_or_config) -> ModelDescriptor:
    if isinstance(model_or_config, str):
        model_type = model_or_config
    else:
        config = getattr(model_or_config, 'config', model_or_config)
        model_type = getattr(config, 'model_type', 'llama')

    descriptor = _DESCRIPTOR_REGISTRY.get(model_type)
    if descriptor is None:
        print(f"[ModelDescriptor] Unknown model_type '{model_type}', defaulting to LLaMA")
        return LLAMA_DESCRIPTOR

    return descriptor


def get_down_proj_module(mlp) -> Optional:
    return getattr(mlp, 'down_proj', None) or getattr(mlp, 'fc2', None)


def get_up_proj_modules(mlp) -> List:
    modules = []
    # LLaMA SwiGLU
    if hasattr(mlp, 'gate_proj'):
        modules.append(mlp.gate_proj)
        if hasattr(mlp, 'up_proj'):
            modules.append(mlp.up_proj)
    # Phi-2 Standard MLP
    elif hasattr(mlp, 'fc1'):
        modules.append(mlp.fc1)
    return modules


def get_attn_output_proj(attn) -> Optional:
    return getattr(attn, 'o_proj', None) or getattr(attn, 'dense', None)


def is_swiglu_ffn(mlp) -> bool:
    return hasattr(mlp, 'gate_proj') and hasattr(mlp, 'down_proj')


def get_ffn_mask_keys(prefix: str, mlp) -> dict:
    if hasattr(mlp, 'gate_proj'):
        return {
            'up_output': [
                f"{prefix}.mlp.gate_proj.output",
                f"{prefix}.mlp.up_proj.output",
            ],
            'down_input': f"{prefix}.mlp.down_proj.input",
        }
    elif hasattr(mlp, 'fc1'):
        return {
            'up_output': [
                f"{prefix}.mlp.fc1.output",
            ],
            'down_input': f"{prefix}.mlp.fc2.input",
        }
    else:
        # fallback to LLaMA naming
        return {
            'up_output': [
                f"{prefix}.mlp.gate_proj.output",
                f"{prefix}.mlp.up_proj.output",
            ],
            'down_input': f"{prefix}.mlp.down_proj.input",
        }


def get_attn_output_mask_key(prefix: str, attn) -> str:
    if hasattr(attn, 'o_proj'):
        return f"{prefix}.self_attn.o_proj.input"
    elif hasattr(attn, 'dense'):
        return f"{prefix}.self_attn.dense.input"
    else:
        return f"{prefix}.self_attn.o_proj.input"
