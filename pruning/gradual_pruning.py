
import math

class GradualPruningScheduler:

    def __init__(
        self,
        initial_sparsity: float = 0.0,
        final_sparsity: float = 0.5,
        pruning_steps: int = 1000,
        pruning_frequency: int = 100,
        warmup_steps: int = 0
    ):
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.pruning_steps = pruning_steps
        self.pruning_frequency = pruning_frequency
        self.warmup_steps = warmup_steps

        self.start_step = warmup_steps
        self.end_step = warmup_steps + pruning_steps

    def should_prune(self, current_step: int) -> bool:
        if current_step < self.start_step:
            return False

        if current_step > self.end_step:
            return False

        if (current_step - self.start_step) % self.pruning_frequency == 0:
            return True

        return False

    def get_target_sparsity(self, current_step: int) -> float:
        if current_step < self.start_step:
            return self.initial_sparsity

        if current_step >= self.end_step:
            return self.final_sparsity

        # Progress ratio [0, 1]
        progress = (current_step - self.start_step) / (self.end_step - self.start_step)

        # Reach final sparsity around 80% of pruning steps
        target_progress = 0.8
        adjusted_progress = min(progress / target_progress, 1.0)

        sparsity_range = self.final_sparsity - self.initial_sparsity
        current_sparsity = self.initial_sparsity + sparsity_range * adjusted_progress

        return current_sparsity

    def get_pruning_ratio(self, current_step: int, component: str = "all") -> float:
        target_sparsity = self.get_target_sparsity(current_step)

        return target_sparsity

    def get_schedule_info(self) -> dict:
        return {
            "initial_sparsity": self.initial_sparsity,
            "final_sparsity": self.final_sparsity,
            "pruning_steps": self.pruning_steps,
            "pruning_frequency": self.pruning_frequency,
            "warmup_steps": self.warmup_steps,
            "start_step": self.start_step,
            "end_step": self.end_step
        }
