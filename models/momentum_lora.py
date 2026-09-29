"""
Momentum LoRA for LLM Pruning Recovery

Momentum LoRA is an advanced LoRA variant that uses momentum accumulation
to improve pruning recovery performance.

Mathematical Formulation:
    K_t = β * K_{t-1} + (B1_t @ A1_t - B1_{t-1} @ A1_{t-1})  # Momentum accumulation
    W_t = W_0 + K_t + B2_t @ A2_t

where:
    - W_0: Original frozen weight (frozen)
    - K_t: Momentum buffer - accumulates velocity changes with exponential decay
    - A1_t, B1_t: Velocity LoRA (periodically reset) - drives momentum updates
    - A2_t, B2_t: Auxiliary LoRA (never reset) - captures residual patterns

Physical Intuition (like a ball rolling):
    W_0:      Initial position (frozen)
    K:        Accumulated momentum from velocity changes (β controls decay)
    A1*B1:    Current velocity (periodically reset to explore new directions)
    A2*B2:    Fine adjustment term (captures patterns not in momentum)

The key insight is that K accumulates the "changes in velocity" over time,
creating a momentum effect that helps escape local minima and smooth optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math


class MomentumLoRALayer(nn.Module):
    """
    Momentum LoRA Layer with Full Momentum Accumulation

    K_t = β * K_{t-1} + (B1_t @ A1_t - B1_{t-1} @ A1_{t-1})
    W = W_0 + K + B2 @ A2

    K buffer accumulates velocity changes over time, creating true momentum effect.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank1: Rank for velocity LoRA (A1, B1) - periodically reset
        rank2: Rank for auxiliary LoRA (A2, B2) - never reset
        beta: Momentum decay rate (default: 0.9)
        reset_interval: Steps between A1/B1 resets (default: 100)
        alpha: Scaling factor for LoRA (default: 1.0)
        bias: Whether the original layer has bias (default: False)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank1: int = 16,
        rank2: int = 32,
        beta: float = 0.9,
        reset_interval: int = 100,
        alpha: float = 1.0,
        bias: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank1 = rank1
        self.rank2 = rank2
        self.beta = beta
        self.reset_interval = reset_interval
        self.alpha = alpha
        self.scaling1 = alpha / rank1  # Scaling for velocity LoRA
        self.scaling2 = alpha / rank2  # Scaling for auxiliary LoRA

        # Frozen original weight (will be set later)
        self.register_buffer('W_0', torch.zeros(out_features, in_features))

        # Optional bias from original layer
        if bias:
            self.register_buffer('bias', torch.zeros(out_features))
        else:
            self.register_buffer('bias', None)

        # K: Momentum buffer (full-rank, accumulates velocity changes)
        # Use float16 to save memory since it's not directly trained
        self.register_buffer('K', torch.zeros(out_features, in_features, dtype=torch.float16))

        # A1, B1: Velocity LoRA (trainable, periodically reset)
        # IMPORTANT: Use float32 for trainable params (required for mixed precision training)
        self.A1 = nn.Parameter(torch.empty(rank1, in_features, dtype=torch.float32))
        self.B1 = nn.Parameter(torch.zeros(out_features, rank1, dtype=torch.float32))

        # A1_prev, B1_prev: Previous velocity for momentum computation
        self.register_buffer('A1_prev', torch.zeros(rank1, in_features, dtype=torch.float16))
        self.register_buffer('B1_prev', torch.zeros(out_features, rank1, dtype=torch.float16))

        # A2, B2: Auxiliary LoRA (trainable, never reset)
        self.A2 = nn.Parameter(torch.empty(rank2, in_features, dtype=torch.float32))
        self.B2 = nn.Parameter(torch.zeros(out_features, rank2, dtype=torch.float32))

        # Step counter
        self.register_buffer('step_count', torch.tensor(0, dtype=torch.long))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize LoRA weights using Kaiming uniform for A and zeros for B."""
        nn.init.kaiming_uniform_(self.A1, a=math.sqrt(5))
        nn.init.zeros_(self.B1)

        nn.init.kaiming_uniform_(self.A2, a=math.sqrt(5))
        nn.init.zeros_(self.B2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: y = x @ W^T where W = W_0 + K + scaling2*(B2@A2)

        Note: Velocity term (A1*B1) contributes through K buffer after update_momentum(),
        not directly in forward pass. This is because K accumulates velocity changes.

        Args:
            x: Input tensor [..., in_features]

        Returns:
            Output tensor [..., out_features]
        """
        # Base output: x @ W_0^T
        base_output = F.linear(x, self.W_0, None)

        # Momentum term: x @ K^T (accumulated velocity changes)
        # K is stored in float16, convert to input dtype for computation
        momentum_output = F.linear(x, self.K.to(x.dtype))

        # Auxiliary term (A2, B2): low-rank computation for residual patterns
        auxiliary_output = F.linear(F.linear(x, self.A2), self.B2) * self.scaling2

        # Current velocity term (A1, B1): needed for gradient flow
        # This allows A1/B1 to receive gradients, which then get accumulated into K
        velocity_output = F.linear(F.linear(x, self.A1), self.B1) * self.scaling1

        # Combine all terms
        output = base_output + momentum_output + auxiliary_output + velocity_output

        if self.bias is not None:
            output = output + self.bias

        return output

    def update_momentum(self):
        """
        Update momentum buffer K after optimizer step.
        Should be called after optimizer.step()

        K_t = β * K_{t-1} + scaling1 * (B1_t @ A1_t - B1_{t-1} @ A1_{t-1})
        """
        with torch.no_grad():
            # Increment step counter
            self.step_count += 1

            # Compute current velocity: B1 @ A1
            current_velocity = self.B1.data @ self.A1.data  # [out_features, in_features]

            # Compute previous velocity: B1_prev @ A1_prev
            prev_velocity = self.B1_prev.to(current_velocity.dtype) @ self.A1_prev.to(current_velocity.dtype)

            # Compute velocity change (delta)
            velocity_delta = self.scaling1 * (current_velocity - prev_velocity)

            # Update K with momentum: K = β * K + velocity_delta
            self.K = (self.beta * self.K.to(velocity_delta.dtype) + velocity_delta).to(torch.float16)

            # Store current velocity as previous for next step
            self.A1_prev.copy_(self.A1.data.to(torch.float16))
            self.B1_prev.copy_(self.B1.data.to(torch.float16))

            # Periodic reset of A1, B1 (velocity)
            if self.step_count % self.reset_interval == 0:
                self.reset_velocity()

    def reset_velocity(self):
        """Reset velocity LoRA (A1, B1) and prev buffers to initial state."""
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.A1, a=math.sqrt(5))
            nn.init.zeros_(self.B1)
            # Also reset prev buffers to match
            self.A1_prev.copy_(self.A1.data.to(torch.float16))
            nn.init.zeros_(self.B1_prev)
            print(f"[MomentumLoRA] Reset velocity at step {self.step_count.item()}")

    def set_original_weight(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """
        Set the frozen original weight W_0 and optional bias.

        Args:
            weight: Original weight matrix [out_features, in_features]
            bias: Optional bias vector [out_features]
        """
        assert weight.shape == (self.out_features, self.in_features), \
            f"Weight shape mismatch: expected {(self.out_features, self.in_features)}, got {weight.shape}"

        self.W_0.copy_(weight)

        if bias is not None:
            if self.bias is None:
                self.register_buffer('bias', bias.clone())
            else:
                self.bias.copy_(bias)

    def get_effective_weight(self) -> torch.Tensor:
        """Get the effective weight W = W_0 + K + scaling1*(B1@A1) + scaling2*(B2@A2)."""
        velocity = self.B1 @ self.A1  # [out_features, in_features]
        auxiliary = self.B2 @ self.A2  # [out_features, in_features]
        return self.W_0 + self.K.to(self.W_0.dtype) + self.scaling1 * velocity + self.scaling2 * auxiliary

    def merge_and_unload(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute the effective weight and return it for merging into a regular Linear layer.

        Returns:
            Tuple of (effective_weight, bias)
        """
        effective_weight = self.get_effective_weight()
        return effective_weight, self.bias

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank1={self.rank1}, rank2={self.rank2}, "
            f"beta={self.beta}, reset_interval={self.reset_interval}"
        )


class SVDMomentumLoRALayer(MomentumLoRALayer):
    """
    Momentum LoRA with SVD-based Soft Orthogonal Regularization.

    This extends MomentumLoRALayer with the ability to compute soft orthogonal
    regularization loss based on pre-computed SVD components of the original weight.

    Additional Args:
        use_soft_orth: Enable soft orthogonal regularization
        svd_rank: Number of top singular vectors to use for regularization (default: 64)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank1: int = 16,
        rank2: int = 32,
        beta: float = 0.9,
        reset_interval: int = 100,
        alpha: float = 1.0,
        bias: bool = False,
        use_soft_orth: bool = True,
        svd_rank: int = 64,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            rank1=rank1,
            rank2=rank2,
            beta=beta,
            reset_interval=reset_interval,
            alpha=alpha,
            bias=bias,
        )

        self.use_soft_orth = use_soft_orth
        self.svd_rank = svd_rank

        # SVD components (will be set later)
        # U_k: [out_features, k], V_k: [in_features, k]
        self.register_buffer('U_k', None)
        self.register_buffer('V_k', None)

    def set_svd_cache(self, U_k: torch.Tensor, V_k: torch.Tensor):
        """
        Set pre-computed SVD components.

        Args:
            U_k: Top-k left singular vectors [out_features, k]
            V_k: Top-k right singular vectors - can be [in_features, k] or [k, in_features]
                 (will be transposed if needed to get [in_features, k])

        Note: The existing SVD cache from pruning/soft_orthogonal.py stores V_k
        with shape [k, in_features], so we handle both formats.
        """
        assert U_k.shape[0] == self.out_features, \
            f"U_k shape mismatch: expected out_features={self.out_features}, got {U_k.shape[0]}"

        # Handle V_k which might be transposed
        # We need [in_features, k] but might get [k, in_features]
        if V_k.shape[0] == self.in_features:
            # Already correct: [in_features, k]
            V_k_fixed = V_k.clone()
        elif V_k.shape[1] == self.in_features:
            # Need to transpose: [k, in_features] -> [in_features, k]
            V_k_fixed = V_k.t().clone()
        else:
            raise ValueError(
                f"V_k shape mismatch: expected one dimension to be in_features={self.in_features}, "
                f"got shape {V_k.shape}"
            )

        # Register as buffers (will move with model)
        if self.U_k is None:
            self.register_buffer('U_k', U_k.clone())
        else:
            self.U_k = U_k.clone()

        if self.V_k is None:
            self.register_buffer('V_k', V_k_fixed)
        else:
            self.V_k = V_k_fixed

    def compute_soft_orth_loss(
        self,
        lambda_U: float = 0.01,
        lambda_V: float = 0.01
    ) -> torch.Tensor:
        """
        Compute soft orthogonal regularization loss.

        L_orth = lambda_U * ||U_k^T B1||^2_F + lambda_V * ||A1 V_k||^2_F
               + (lambda_U/2) * ||U_k^T B2||^2_F + (lambda_V/2) * ||A2 V_k||^2_F

        The momentum LoRA (A2, B2) has reduced regularization strength (half)
        since it should have more freedom to learn new directions.

        Args:
            lambda_U: Regularization strength for output subspace
            lambda_V: Regularization strength for input subspace

        Returns:
            Soft orthogonal loss (scalar tensor)
        """
        if not self.use_soft_orth or self.U_k is None or self.V_k is None:
            return torch.tensor(0.0, device=self.A1.device, requires_grad=False)

        # Move SVD tensors to correct device and dtype
        U_k = self.U_k.to(self.A1.device, dtype=self.A1.dtype)
        V_k = self.V_k.to(self.A1.device, dtype=self.A1.dtype)

        loss = torch.tensor(0.0, device=self.A1.device)

        # Regularize velocity LoRA (A1, B1) - full strength
        # B1: [out_features, rank1], U_k: [out_features, k]
        # proj_B1: [k, rank1]
        proj_B1 = U_k.t() @ self.B1  # [k, rank1]
        loss = loss + lambda_U * torch.sum(proj_B1 ** 2)

        # A1: [rank1, in_features], V_k: [in_features, k]
        # proj_A1: [rank1, k]
        proj_A1 = self.A1 @ V_k  # [rank1, k]
        loss = loss + lambda_V * torch.sum(proj_A1 ** 2)

        # Regularize momentum LoRA (A2, B2) - half strength
        # Momentum should have more freedom to learn new directions
        proj_B2 = U_k.t() @ self.B2  # [k, rank2]
        loss = loss + (lambda_U * 0.5) * torch.sum(proj_B2 ** 2)

        proj_A2 = self.A2 @ V_k  # [rank2, k]
        loss = loss + (lambda_V * 0.5) * torch.sum(proj_A2 ** 2)

        return loss

    def extra_repr(self) -> str:
        base_repr = super().extra_repr()
        return f"{base_repr}, use_soft_orth={self.use_soft_orth}, svd_rank={self.svd_rank}"


class MomentumLoRALinear(nn.Module):
    """
    A drop-in replacement for nn.Linear with Momentum LoRA.

    This class wraps an existing Linear layer and adds Momentum LoRA on top.
    It's designed to be compatible with existing model architectures.

    Args:
        base_layer: The original nn.Linear layer
        rank1: Rank for velocity LoRA (default: 16)
        rank2: Rank for momentum LoRA (default: 32)
        beta: Momentum decay rate (default: 0.9)
        reset_interval: Steps between resets (default: 100)
        alpha: LoRA scaling factor (default: 1.0)
        use_soft_orth: Enable soft orthogonal regularization (default: True)
        svd_rank: SVD rank for regularization (default: 64)
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank1: int = 16,
        rank2: int = 32,
        beta: float = 0.9,
        reset_interval: int = 100,
        alpha: float = 1.0,
        use_soft_orth: bool = True,
        svd_rank: int = 64,
    ):
        super().__init__()

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        # Create the Momentum LoRA layer
        self.momentum_lora = SVDMomentumLoRALayer(
            in_features=self.in_features,
            out_features=self.out_features,
            rank1=rank1,
            rank2=rank2,
            beta=beta,
            reset_interval=reset_interval,
            alpha=alpha,
            bias=base_layer.bias is not None,
            use_soft_orth=use_soft_orth,
            svd_rank=svd_rank,
        )

        # Copy original weights
        self.momentum_lora.set_original_weight(
            base_layer.weight.data,
            base_layer.bias.data if base_layer.bias is not None else None
        )

        # Freeze original weight (it's already stored as buffer)
        # The trainable parameters are A1, B1, A2, B2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.momentum_lora(x)

    def update_momentum(self):
        """Update momentum buffers after optimizer step."""
        self.momentum_lora.update_momentum()

    def set_svd_cache(self, U_k: torch.Tensor, V_k: torch.Tensor):
        """Set SVD cache for soft orthogonal regularization."""
        self.momentum_lora.set_svd_cache(U_k, V_k)

    def compute_soft_orth_loss(
        self,
        lambda_U: float = 0.01,
        lambda_V: float = 0.01
    ) -> torch.Tensor:
        """Compute soft orthogonal loss."""
        return self.momentum_lora.compute_soft_orth_loss(lambda_U, lambda_V)

    @property
    def weight(self) -> torch.Tensor:
        """Return effective weight for compatibility."""
        return self.momentum_lora.get_effective_weight()

    @property
    def bias(self) -> Optional[torch.Tensor]:
        """Return bias for compatibility."""
        return self.momentum_lora.bias


def collect_momentum_layers(model: nn.Module) -> list:
    """
    Collect all Momentum LoRA modules from a model.

    This collects MomentumLoRALinear wrappers (preferred) or standalone
    MomentumLoRALayer/SVDMomentumLoRALayer instances.

    To avoid double-counting (since MomentumLoRALinear contains a
    SVDMomentumLoRALayer internally), we only collect the wrapper
    when it's present.

    Args:
        model: Model to search

    Returns:
        List of momentum LoRA layers/wrappers
    """
    layers = []
    # Track modules that are children of MomentumLoRALinear to avoid double-counting
    skip_modules = set()

    # First pass: find all MomentumLoRALinear and mark their children
    for module in model.modules():
        if isinstance(module, MomentumLoRALinear):
            layers.append(module)
            # Mark the inner momentum_lora to skip it
            skip_modules.add(id(module.momentum_lora))

    # Second pass: find standalone MomentumLoRALayer/SVDMomentumLoRALayer
    for module in model.modules():
        if id(module) in skip_modules:
            continue
        if isinstance(module, (MomentumLoRALayer, SVDMomentumLoRALayer)):
            # Only add if not a MomentumLoRALinear (already added) and not skipped
            if not isinstance(module, type(None)):  # Always true, just for clarity
                if module not in layers:  # Avoid duplicates
                    layers.append(module)

    return layers


def update_all_momentum_buffers(model: nn.Module):
    """
    Update momentum buffers for all Momentum LoRA layers in the model.
    Should be called after each optimizer.step().

    Args:
        model: Model containing Momentum LoRA layers
    """
    # Use collect_momentum_layers to avoid double-updating
    layers = collect_momentum_layers(model)
    for layer in layers:
        if isinstance(layer, MomentumLoRALinear):
            layer.update_momentum()
        elif isinstance(layer, MomentumLoRALayer):
            layer.update_momentum()


def compute_total_soft_orth_loss(
    model: nn.Module,
    lambda_U: float = 0.01,
    lambda_V: float = 0.01
) -> Tuple[torch.Tensor, int]:
    """
    Compute total soft orthogonal loss for all Momentum LoRA layers.

    Args:
        model: Model containing Momentum LoRA layers
        lambda_U: Regularization strength for output subspace
        lambda_V: Regularization strength for input subspace

    Returns:
        Tuple of (total_loss, num_layers)
    """
    device = next(model.parameters()).device
    total_loss = torch.tensor(0.0, device=device)
    num_layers = 0

    # Use collect_momentum_layers to avoid double-counting
    layers = collect_momentum_layers(model)
    for layer in layers:
        if isinstance(layer, MomentumLoRALinear):
            layer_loss = layer.compute_soft_orth_loss(lambda_U, lambda_V)
            total_loss = total_loss + layer_loss
            num_layers += 1
        elif isinstance(layer, SVDMomentumLoRALayer):
            layer_loss = layer.compute_soft_orth_loss(lambda_U, lambda_V)
            total_loss = total_loss + layer_loss
            num_layers += 1

    return total_loss, num_layers


def get_momentum_lora_state_dict(model: nn.Module) -> Dict[str, Any]:
    """
    Get a state dict containing all Momentum LoRA specific states.
    This includes K buffers, step counts, and A1_prev/B1_prev states.

    Args:
        model: Model containing Momentum LoRA layers

    Returns:
        Dictionary with momentum LoRA states
    """
    state = {}

    for name, module in model.named_modules():
        if isinstance(module, MomentumLoRALayer):
            state[name] = {
                'K': module.K.cpu().clone(),
                'step_count': module.step_count.cpu().clone(),
                'A1_prev': module.A1_prev.cpu().clone(),
                'B1_prev': module.B1_prev.cpu().clone(),
            }
        elif isinstance(module, MomentumLoRALinear):
            state[name] = {
                'K': module.momentum_lora.K.cpu().clone(),
                'step_count': module.momentum_lora.step_count.cpu().clone(),
                'A1_prev': module.momentum_lora.A1_prev.cpu().clone(),
                'B1_prev': module.momentum_lora.B1_prev.cpu().clone(),
            }

    return state


def load_momentum_lora_state_dict(model: nn.Module, state: Dict[str, Any]):
    """
    Load Momentum LoRA states from a state dict.

    Args:
        model: Model containing Momentum LoRA layers
        state: State dict from get_momentum_lora_state_dict
    """
    for name, module in model.named_modules():
        if name in state:
            module_state = state[name]

            if isinstance(module, MomentumLoRALayer):
                module.K.copy_(module_state['K'].to(module.K.device))
                module.step_count.copy_(module_state['step_count'].to(module.step_count.device))
                module.A1_prev.copy_(module_state['A1_prev'].to(module.A1_prev.device))
                module.B1_prev.copy_(module_state['B1_prev'].to(module.B1_prev.device))
            elif isinstance(module, MomentumLoRALinear):
                lora = module.momentum_lora
                lora.K.copy_(module_state['K'].to(lora.K.device))
                lora.step_count.copy_(module_state['step_count'].to(lora.step_count.device))
                lora.A1_prev.copy_(module_state['A1_prev'].to(lora.A1_prev.device))
                lora.B1_prev.copy_(module_state['B1_prev'].to(lora.B1_prev.device))


def merge_momentum_lora_to_base(model: nn.Module) -> nn.Module:
    """
    Merge all MomentumLoRALinear layers into regular nn.Linear layers.

    This converts:
        MomentumLoRALinear (W_0 + K + A1*B1 + A2*B2)
    into:
        nn.Linear (effective_weight)

    This is useful for saving the model in a standard format that can be
    loaded without the Momentum LoRA classes.

    Args:
        model: Model containing MomentumLoRALinear layers

    Returns:
        Modified model with nn.Linear layers (in-place modification)
    """
    modules_to_replace = []

    # First pass: find all MomentumLoRALinear modules
    for name, module in model.named_modules():
        if isinstance(module, MomentumLoRALinear):
            modules_to_replace.append((name, module))

    print(f"[MomentumLoRA] Merging {len(modules_to_replace)} layers to base...")

    # Second pass: replace with nn.Linear
    for name, module in modules_to_replace:
        # Get effective weight and bias
        effective_weight, bias = module.momentum_lora.merge_and_unload()

        # Create new Linear layer
        new_linear = nn.Linear(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=(bias is not None),
            device=effective_weight.device,
            dtype=effective_weight.dtype
        )

        # Copy weights
        new_linear.weight.data.copy_(effective_weight)
        if bias is not None:
            new_linear.bias.data.copy_(bias)

        # Replace in parent module
        _set_module_by_name(model, name, new_linear)

    print(f"[MomentumLoRA] ✓ Merged {len(modules_to_replace)} layers")
    return model


def _set_module_by_name(model: nn.Module, name: str, new_module: nn.Module):
    """
    Set a module by its name (e.g., 'model.layers.0.self_attn.q_proj').

    Args:
        model: Root model
        name: Dot-separated module path
        new_module: Module to set
    """
    parts = name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)
