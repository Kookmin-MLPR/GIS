
import torch
import torch.nn as nn
from typing import Dict, Tuple, List

class LoRAWeightImportanceCalculator:

    def __init__(self, model, device, metric="weight", activation_stats: Dict[str, torch.Tensor] = None):
        self.model = model
        self.device = device
        self.metric = metric
        self.activation_stats = activation_stats if activation_stats is not None else {}

    def compute_importance(self) -> Dict[str, torch.Tensor]:
        if self.metric == "gradient":
            return self._compute_importance_gradient()
        elif self.metric == "lora_grad_only":
            return self._compute_importance_lora_grad_only()
        elif self.metric == "lora_grad_weight":
            return self._compute_importance_lora_grad_weight()
        elif self.metric == "fisher":
            return self._compute_importance_fisher()
        elif self.metric == "wanda":
            return self._compute_importance_wanda()
        else:
            return self._compute_importance_weight()

    def _compute_importance_weight(self) -> Dict[str, torch.Tensor]:
        importance_scores = {}

        for name, module in self.model.named_modules():
            # Check for MomentumLoRALinear (has momentum_lora attribute with A1, B1, A2, B2)
            if hasattr(module, 'momentum_lora') and hasattr(module.momentum_lora, 'A1'):
                try:
                    momentum_lora = module.momentum_lora
                    base_weight = momentum_lora.W_0.data

                    accumulated_importance = None

                    # Process A1/B1 (velocity LoRA)
                    lora_delta = momentum_lora.B1.data @ momentum_lora.A1.data
                    importance = torch.abs(base_weight * lora_delta)
                    accumulated_importance = importance

                    # Process A2/B2 (momentum LoRA)
                    lora_delta = momentum_lora.B2.data @ momentum_lora.A2.data
                    importance = torch.abs(base_weight * lora_delta)
                    accumulated_importance += importance

                    importance_scores[name] = accumulated_importance.detach()

                except Exception as e:
                    self._log_debug_once(f"momentum_lora_error:{name}", f"[ImportanceDebug] MomentumLoRA weight error for {name}: {e}")
                    continue

            # Standard PEFT LoRA check
            elif hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                try:
                    adapter_names = self._get_adapter_names(module)
                    if not adapter_names:
                        self._log_debug_once(f"no_adapter:{name}", f"[ImportanceDebug] No adapter names found for {name}")
                        continue

                    base_weight = module.base_layer.weight.data
                    accumulated_importance = None
                    fallback_importance = None

                    for adapter_name in adapter_names:
                        try:
                            lora_A_module = module.lora_A[adapter_name]
                            lora_B_module = module.lora_B[adapter_name]
                        except (KeyError, AttributeError):
                            continue

                        if not lora_A_module.weight.requires_grad and not lora_B_module.weight.requires_grad:
                            continue

                        try:
                            lora_A = lora_A_module.weight.data
                            lora_B = lora_B_module.weight.data
                        except AttributeError:
                            continue

                        # LoRA delta
                        lora_delta = lora_B @ lora_A

                        importance = torch.abs(base_weight * lora_delta)

                        if accumulated_importance is None:
                            accumulated_importance = importance
                        else:
                            accumulated_importance += importance

                    if accumulated_importance is not None:
                        importance_scores[name] = accumulated_importance.detach()

                except Exception as e:
                    print(f"Warning: Could not compute importance for {name}: {e}")
                    continue

        return importance_scores

    def _compute_importance_gradient(self) -> Dict[str, torch.Tensor]:
        importance_scores = {}

        for name, module in self.model.named_modules():
            # Check for MomentumLoRALinear (has momentum_lora attribute with A1, B1, A2, B2)
            if hasattr(module, 'momentum_lora') and hasattr(module.momentum_lora, 'A1'):
                try:
                    momentum_lora = module.momentum_lora
                    base_weight = momentum_lora.W_0.data

                    accumulated_importance = None

                    # Process A1/B1 (velocity LoRA)
                    if momentum_lora.A1.grad is not None and momentum_lora.B1.grad is not None:
                        lora_delta = momentum_lora.B1.grad.data @ momentum_lora.A1.grad.data
                        importance = torch.abs(base_weight * lora_delta)
                        accumulated_importance = importance

                    # Process A2/B2 (momentum LoRA)
                    if momentum_lora.A2.grad is not None and momentum_lora.B2.grad is not None:
                        lora_delta = momentum_lora.B2.grad.data @ momentum_lora.A2.grad.data
                        importance = torch.abs(base_weight * lora_delta)
                        if accumulated_importance is None:
                            accumulated_importance = importance
                        else:
                            accumulated_importance += importance

                    if accumulated_importance is not None:
                        importance_scores[name] = accumulated_importance.detach()

                except Exception as e:
                    self._log_debug_once(f"momentum_lora_error:{name}", f"[ImportanceDebug] MomentumLoRA error for {name}: {e}")
                    continue

            # Standard PEFT LoRA check
            elif hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                try:
                    adapter_names = self._get_adapter_names(module)
                    if not adapter_names:
                        continue

                    base_weight = module.base_layer.weight.data
                    accumulated_importance = None

                    for adapter_name in adapter_names:
                        try:
                            lora_A_module = module.lora_A[adapter_name]
                            lora_B_module = module.lora_B[adapter_name]
                        except (KeyError, AttributeError):
                            self._log_debug_once(f"missing_adapter:{name}:{adapter_name}", f"[ImportanceDebug] Adapter '{adapter_name}' missing for {name}")
                            continue

                        if not lora_A_module.weight.requires_grad and not lora_B_module.weight.requires_grad:
                            self._log_debug_once(f"frozen_adapter:{name}:{adapter_name}", f"[ImportanceDebug] Adapter '{adapter_name}' frozen for {name}")
                            continue

                        try:
                            lora_A_weight = lora_A_module.weight
                            lora_B_weight = lora_B_module.weight
                        except AttributeError:
                            self._log_debug_once(f"no_weight_attr:{name}:{adapter_name}", f"[ImportanceDebug] Adapter '{adapter_name}' missing weight attr for {name}")
                            continue

                        if lora_A_weight.grad is None or lora_B_weight.grad is None:
                            self._log_debug_once(f"no_grad:{name}:{adapter_name}", f"[ImportanceDebug] No grad for adapter '{adapter_name}' in {name}, skipping")
                            continue

                        lora_A_grad = lora_A_weight.grad.data
                        lora_B_grad = lora_B_weight.grad.data

                        lora_delta = lora_B_grad @ lora_A_grad

                        importance = torch.abs(base_weight * lora_delta)

                        if accumulated_importance is None:
                            accumulated_importance = importance
                        else:
                            accumulated_importance += importance

                    if accumulated_importance is not None:
                        importance_scores[name] = accumulated_importance.detach()
                    else:
                        self._log_debug_once(f"no_importance:{name}", f"[ImportanceDebug] Could not compute importance for {name}")

                except Exception as e:
                    print(f"Warning: Could not compute importance for {name}: {e}")
                    continue

        return importance_scores

    def _compute_importance_lora_grad_only(self) -> Dict[str, torch.Tensor]:
        importance_scores = {}

        for name, module in self.model.named_modules():
            # Check for MomentumLoRALinear (has momentum_lora attribute with A1, B1, A2, B2)
            if hasattr(module, 'momentum_lora') and hasattr(module.momentum_lora, 'A1'):
                try:
                    momentum_lora = module.momentum_lora
                    accumulated_importance = None

                    # Process A1/B1 (velocity LoRA)
                    if momentum_lora.A1.grad is not None and momentum_lora.B1.grad is not None:
                        lora_delta = momentum_lora.B1.grad.data @ momentum_lora.A1.grad.data
                        importance = torch.abs(lora_delta)
                        accumulated_importance = importance

                    # Process A2/B2 (momentum LoRA)
                    if momentum_lora.A2.grad is not None and momentum_lora.B2.grad is not None:
                        lora_delta = momentum_lora.B2.grad.data @ momentum_lora.A2.grad.data
                        importance = torch.abs(lora_delta)
                        if accumulated_importance is None:
                            accumulated_importance = importance
                        else:
                            accumulated_importance += importance

                    if accumulated_importance is not None:
                        importance_scores[name] = accumulated_importance.detach()

                except Exception as e:
                    self._log_debug_once(f"momentum_lora_error:{name}", f"[ImportanceDebug] MomentumLoRA lora_grad_only error for {name}: {e}")
                    continue

            # Standard PEFT LoRA check
            elif hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                try:
                    adapter_names = self._get_adapter_names(module)
                    if not adapter_names:
                        continue

                    accumulated_importance = None

                    for adapter_name in adapter_names:
                        try:
                            lora_A_module = module.lora_A[adapter_name]
                            lora_B_module = module.lora_B[adapter_name]
                        except (KeyError, AttributeError):
                            self._log_debug_once(f"missing_adapter:{name}:{adapter_name}", f"[ImportanceDebug] Adapter '{adapter_name}' missing for {name}")
                            continue

                        if not lora_A_module.weight.requires_grad and not lora_B_module.weight.requires_grad:
                            self._log_debug_once(f"frozen_adapter:{name}:{adapter_name}", f"[ImportanceDebug] Adapter '{adapter_name}' frozen for {name}")
                            continue

                        try:
                            lora_A_weight = lora_A_module.weight
                            lora_B_weight = lora_B_module.weight
                        except AttributeError:
                            self._log_debug_once(f"no_weight_attr:{name}:{adapter_name}", f"[ImportanceDebug] Adapter '{adapter_name}' missing weight attr for {name}")
                            continue

                        if lora_A_weight.grad is None or lora_B_weight.grad is None:
                            self._log_debug_once(f"no_grad:{name}:{adapter_name}", f"[ImportanceDebug] No grad for adapter '{adapter_name}' in {name}, skipping")
                            continue

                        lora_A_grad = lora_A_weight.grad.data
                        lora_B_grad = lora_B_weight.grad.data

                        lora_delta = lora_B_grad @ lora_A_grad

                        importance = torch.abs(lora_delta)

                        if accumulated_importance is None:
                            accumulated_importance = importance
                        else:
                            accumulated_importance += importance

                    if accumulated_importance is not None:
                        importance_scores[name] = accumulated_importance.detach()
                    else:
                        self._log_debug_once(f"no_importance:{name}", f"[ImportanceDebug] Could not compute importance for {name}")

                except Exception as e:
                    print(f"Warning: Could not compute importance for {name}: {e}")
                    continue

        return importance_scores

    def _compute_importance_fisher(self) -> Dict[str, torch.Tensor]:
        importance_scores = {}

        for name, module in self.model.named_modules():
            # Check for MomentumLoRALinear (has momentum_lora attribute with A1, B1, A2, B2)
            if hasattr(module, 'momentum_lora') and hasattr(module.momentum_lora, 'A1'):
                try:
                    momentum_lora = module.momentum_lora
                    base_weight = momentum_lora.W_0.data

                    accumulated_importance = None

                    # Process A1/B1 (velocity LoRA)
                    if momentum_lora.A1.grad is not None and momentum_lora.B1.grad is not None:
                        # Fisher: gradient squared
                        lora_A_fisher = momentum_lora.A1.grad.data ** 2
                        lora_B_fisher = momentum_lora.B1.grad.data ** 2
                        lora_delta = lora_B_fisher @ lora_A_fisher
                        importance = torch.abs(base_weight * lora_delta)
                        accumulated_importance = importance

                    # Process A2/B2 (momentum LoRA)
                    if momentum_lora.A2.grad is not None and momentum_lora.B2.grad is not None:
                        lora_A_fisher = momentum_lora.A2.grad.data ** 2
                        lora_B_fisher = momentum_lora.B2.grad.data ** 2
                        lora_delta = lora_B_fisher @ lora_A_fisher
                        importance = torch.abs(base_weight * lora_delta)
                        if accumulated_importance is None:
                            accumulated_importance = importance
                        else:
                            accumulated_importance += importance

                    if accumulated_importance is not None:
                        importance_scores[name] = accumulated_importance.detach()

                except Exception as e:
                    self._log_debug_once(f"momentum_lora_error:{name}", f"[ImportanceDebug] MomentumLoRA fisher error for {name}: {e}")
                    continue

            # Standard PEFT LoRA check
            elif hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                try:
                    adapter_names = self._get_adapter_names(module)
                    if not adapter_names:
                        continue

                    base_weight = module.base_layer.weight.data
                    accumulated_importance = None

                    for adapter_name in adapter_names:
                        try:
                            lora_A_module = module.lora_A[adapter_name]
                            lora_B_module = module.lora_B[adapter_name]
                        except (KeyError, AttributeError):
                            self._log_debug_once(f"missing_adapter:{name}:{adapter_name}", f"[ImportanceDebug] Adapter '{adapter_name}' missing for {name}")
                            continue

                        if not lora_A_module.weight.requires_grad and not lora_B_module.weight.requires_grad:
                            self._log_debug_once(f"frozen_adapter:{name}:{adapter_name}", f"[ImportanceDebug] Adapter '{adapter_name}' frozen for {name}")
                            continue

                        try:
                            lora_A_weight = lora_A_module.weight
                            lora_B_weight = lora_B_module.weight
                        except AttributeError:
                            self._log_debug_once(f"no_weight_attr:{name}:{adapter_name}", f"[ImportanceDebug] Adapter '{adapter_name}' missing weight attr for {name}")
                            continue

                        if lora_A_weight.grad is None or lora_B_weight.grad is None:
                            self._log_debug_once(f"no_grad:{name}:{adapter_name}", f"[ImportanceDebug] No grad for adapter '{adapter_name}' in {name}, skipping")
                            continue

                        # Empirical Fisher: gradient squared
                        lora_A_fisher = lora_A_weight.grad.data ** 2
                        lora_B_fisher = lora_B_weight.grad.data ** 2

                        lora_delta = lora_B_fisher @ lora_A_fisher

                        importance = torch.abs(base_weight * lora_delta)

                        if accumulated_importance is None:
                            accumulated_importance = importance
                        else:
                            accumulated_importance += importance

                    if accumulated_importance is not None:
                        importance_scores[name] = accumulated_importance.detach()
                    else:
                        self._log_debug_once(f"no_importance:{name}", f"[ImportanceDebug] Could not compute fisher importance for {name}")

                except Exception as e:
                    print(f"Warning: Could not compute fisher importance for {name}: {e}")
                    continue

        return importance_scores

    def _compute_importance_wanda(self) -> Dict[str, torch.Tensor]:
        importance_scores = {}

        if not self.activation_stats:
            print("[WANDA Warning] No activation stats provided. Using weight-only importance.")
            return self._compute_importance_weight()

        for name, module in self.model.named_modules():
            # Check for MomentumLoRALinear (has momentum_lora attribute with A1, B1, A2, B2)
            if hasattr(module, 'momentum_lora') and hasattr(module.momentum_lora, 'A1'):
                try:
                    momentum_lora = module.momentum_lora
                    base_weight = momentum_lora.W_0.data

                    lora_delta = momentum_lora.B1.data @ momentum_lora.A1.data
                    lora_delta += momentum_lora.B2.data @ momentum_lora.A2.data

                    # W_total = W_frozen + LoRA_delta
                    W_total = base_weight + lora_delta

                    act_stat = self._resolve_activation_stat(name)
                    if act_stat is None:
                        importance = torch.abs(W_total)
                    else:
                        # WANDA: |W| × |X|
                        # W_total shape: [out_dim, in_dim]
                        importance = torch.abs(W_total) * act_stat.unsqueeze(0).to(W_total.device)

                    importance_scores[name] = importance.detach()

                except Exception as e:
                    self._log_debug_once(f"momentum_lora_error:{name}", f"[ImportanceDebug] MomentumLoRA wanda error for {name}: {e}")
                    continue

            # Standard PEFT LoRA check
            elif hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                try:
                    adapter_names = self._get_adapter_names(module)
                    if not adapter_names:
                        continue

                    base_weight = module.base_layer.weight.data

                    lora_delta = None
                    scaling = getattr(module, 'scaling', {})

                    for adapter_name in adapter_names:
                        try:
                            lora_A_module = module.lora_A[adapter_name]
                            lora_B_module = module.lora_B[adapter_name]
                        except (KeyError, AttributeError):
                            continue

                        try:
                            lora_A = lora_A_module.weight.data
                            lora_B = lora_B_module.weight.data
                        except AttributeError:
                            continue

                        # LoRA scaling factor
                        scale = scaling.get(adapter_name, 1.0) if isinstance(scaling, dict) else scaling

                        delta = (lora_B @ lora_A) * scale
                        if lora_delta is None:
                            lora_delta = delta
                        else:
                            lora_delta += delta

                    # W_total = W_frozen + LoRA_delta
                    if lora_delta is not None:
                        W_total = base_weight + lora_delta
                    else:
                        W_total = base_weight

                    act_stat = self._resolve_activation_stat(name)
                    if act_stat is None:
                        importance = torch.abs(W_total)
                    else:
                        # WANDA: |W| × |X|
                        # act_stat shape: [input_dim]
                        # W_total shape: [out_dim, in_dim]
                        importance = torch.abs(W_total) * act_stat.unsqueeze(0).to(W_total.device)

                    importance_scores[name] = importance.detach()

                except Exception as e:
                    print(f"Warning: Could not compute wanda importance for {name}: {e}")
                    continue

        return importance_scores

    def _resolve_activation_stat(self, target_name: str):
        if target_name in self.activation_stats:
            return self.activation_stats[target_name]

        if ".model.layers." in target_name:
            alt = target_name.replace(".model.layers.", ".model.model.layers.")
            if alt in self.activation_stats:
                return self.activation_stats[alt]

        if ".model.model.layers." in target_name:
            alt = target_name.replace(".model.model.layers.", ".model.layers.")
            if alt in self.activation_stats:
                return self.activation_stats[alt]

        if target_name.startswith("base_model.model."):
            alt = target_name.replace("base_model.model.", "model.", 1)
            if alt in self.activation_stats:
                return self.activation_stats[alt]
            alt = target_name.replace("base_model.", "", 1)
            if alt in self.activation_stats:
                return self.activation_stats[alt]

        return None

    def _compute_importance_lora_grad_weight(self) -> Dict[str, torch.Tensor]:
        importance_scores = {}

        for name, module in self.model.named_modules():
            # Check for MomentumLoRALinear (has momentum_lora attribute with A1, B1, A2, B2)
            if hasattr(module, 'momentum_lora') and hasattr(module.momentum_lora, 'A1'):
                try:
                    momentum_lora = module.momentum_lora
                    accumulated_importance = None

                    # Process A1/B1 (velocity LoRA)
                    if momentum_lora.A1.grad is not None and momentum_lora.B1.grad is not None:
                        A1_combined = momentum_lora.A1.grad.data * momentum_lora.A1.data
                        B1_combined = momentum_lora.B1.grad.data * momentum_lora.B1.data
                        lora_delta = B1_combined @ A1_combined
                        importance = torch.abs(lora_delta)
                        accumulated_importance = importance

                    # Process A2/B2 (momentum LoRA)
                    if momentum_lora.A2.grad is not None and momentum_lora.B2.grad is not None:
                        A2_combined = momentum_lora.A2.grad.data * momentum_lora.A2.data
                        B2_combined = momentum_lora.B2.grad.data * momentum_lora.B2.data
                        lora_delta = B2_combined @ A2_combined
                        importance = torch.abs(lora_delta)
                        if accumulated_importance is None:
                            accumulated_importance = importance
                        else:
                            accumulated_importance += importance

                    if accumulated_importance is not None:
                        importance_scores[name] = accumulated_importance.detach()

                except Exception as e:
                    self._log_debug_once(f"momentum_lora_error:{name}", f"[ImportanceDebug] MomentumLoRA lora_grad_weight error for {name}: {e}")
                    continue

            # Standard PEFT LoRA check
            elif hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                try:
                    adapter_names = self._get_adapter_names(module)
                    if not adapter_names:
                        continue

                    accumulated_importance = None

                    for adapter_name in adapter_names:
                        try:
                            lora_A_module = module.lora_A[adapter_name]
                            lora_B_module = module.lora_B[adapter_name]
                        except (KeyError, AttributeError):
                            self._log_debug_once(f"missing_adapter:{name}:{adapter_name}", f"[ImportanceDebug] Adapter '{adapter_name}' missing for {name}")
                            continue

                        if not lora_A_module.weight.requires_grad and not lora_B_module.weight.requires_grad:
                            self._log_debug_once(f"frozen_adapter:{name}:{adapter_name}", f"[ImportanceDebug] Adapter '{adapter_name}' frozen for {name}")
                            continue

                        try:
                            lora_A_weight = lora_A_module.weight
                            lora_B_weight = lora_B_module.weight
                        except AttributeError:
                            self._log_debug_once(f"no_weight_attr:{name}:{adapter_name}", f"[ImportanceDebug] Adapter '{adapter_name}' missing weight attr for {name}")
                            continue

                        if lora_A_weight.grad is None or lora_B_weight.grad is None:
                            self._log_debug_once(f"no_grad:{name}:{adapter_name}", f"[ImportanceDebug] No grad for adapter '{adapter_name}' in {name}, skipping")
                            continue

                        lora_A_grad = lora_A_weight.grad.data
                        lora_B_grad = lora_B_weight.grad.data
                        lora_A_data = lora_A_weight.data
                        lora_B_data = lora_B_weight.data

                        # Element-wise multiplication: gradient × weight
                        lora_A_combined = lora_A_grad * lora_A_data
                        lora_B_combined = lora_B_grad * lora_B_data

                        lora_delta = lora_B_combined @ lora_A_combined

                        importance = torch.abs(lora_delta)

                        if accumulated_importance is None:
                            accumulated_importance = importance
                        else:
                            accumulated_importance += importance

                    if accumulated_importance is not None:
                        importance_scores[name] = accumulated_importance.detach()
                    else:
                        self._log_debug_once(f"no_importance:{name}", f"[ImportanceDebug] Could not compute importance for {name}")

                except Exception as e:
                    print(f"Warning: Could not compute importance for {name}: {e}")
                    continue

        return importance_scores

    def _get_adapter_names(self, module) -> List[str]:
        """
        Retrieve adapter names for a LoRA module, prioritizing the active adapter.
        """
        adapter_names = []

        active = getattr(module, "active_adapter", None)
        if active:
            if isinstance(active, str):
                adapter_names.append(active)
            elif isinstance(active, (list, tuple, set)):
                adapter_names.extend(list(active))

        if hasattr(module.lora_A, "keys"):
            for name in module.lora_A.keys():
                if name not in adapter_names:
                    adapter_names.append(name)
        elif isinstance(module.lora_A, dict):
            for name in module.lora_A:
                if name not in adapter_names:
                    adapter_names.append(name)

        return adapter_names

    def _log_debug_once(self, key: str, message: str):
        """Log a debug message only once to avoid flooding output."""
        if not hasattr(self, "_debug_messages"):
            self._debug_messages = set()
        if key not in self._debug_messages:
            print(message)
            self._debug_messages.add(key)

    def _resolve_importance_name(self, importance_scores: Dict[str, torch.Tensor], target_name: str):
        """
        Resolve module names that may differ by an extra '.model' segment
        or have MomentumLoRALinear wrappers.
        """
        if target_name in importance_scores:
            return importance_scores[target_name]

        if ".model.layers." in target_name:
            alt = target_name.replace(".model.layers.", ".model.model.layers.")
            if alt in importance_scores:
                return importance_scores[alt]

        if ".model.model.layers." in target_name:
            alt = target_name.replace(".model.model.layers.", ".model.layers.")
            if alt in importance_scores:
                return importance_scores[alt]

        # Try without 'base_model.model.' prefix (for MomentumLoRALinear)
        if target_name.startswith("base_model.model."):
            alt = target_name.replace("base_model.model.", "model.", 1)
            if alt in importance_scores:
                return importance_scores[alt]
            # Also try just 'model.layers.X...'
            alt = target_name.replace("base_model.", "", 1)
            if alt in importance_scores:
                return importance_scores[alt]

        # Try with 'model.layers.X...' pattern for Momentum LoRA
        # Importance might be stored as 'model.layers.X.self_attn.v_proj'
        for key in importance_scores.keys():
            # Extract layer and proj info from target_name
            # e.g., target: 'base_model.model.layers.0.self_attn.v_proj'
            # key might be: 'model.layers.0.self_attn.v_proj'
            if target_name.endswith(key.split("model.")[-1]) if "model." in key else False:
                return importance_scores[key]
            # Direct suffix match
            target_suffix = target_name.split("layers.")[-1] if "layers." in target_name else ""
            key_suffix = key.split("layers.")[-1] if "layers." in key else ""
            if target_suffix and key_suffix and target_suffix == key_suffix:
                return importance_scores[key]

        return None

    def compute_head_importance(
        self,
        mode: str = "o",
        qk_activation_importance: Dict[int, torch.Tensor] = None
    ) -> Dict[int, torch.Tensor]:
        importance_scores = self.compute_importance()
        head_importance = {}

        num_layers = self.model.config.num_hidden_layers
        num_heads = self.model.config.num_attention_heads
        num_kv_heads = getattr(self.model.config, 'num_key_value_heads', num_heads)
        head_dim = self.model.config.hidden_size // num_heads
        num_queries_per_kv = num_heads // num_kv_heads
        is_gqa = (num_kv_heads != num_heads)

        if is_gqa:
            print(f"[Importance] GQA detected: {num_heads} Q heads, {num_kv_heads} KV heads, ratio {num_queries_per_kv}:1")
            print(f"[Importance] Head pruning will be done in groups of {num_queries_per_kv} Q heads + 1 KV head")
            print(f"[Importance] Group importance = average of Q head importances in group")

        print(f"[Importance] Head importance mode: {mode}")

        for layer_idx in range(num_layers):
            if mode == "v":
                proj_name = f"base_model.model.layers.{layer_idx}.self_attn.v_proj"
                proj_importance = self._resolve_importance_name(importance_scores, proj_name)

                if proj_importance is None:
                    print(f"Warning: No importance for {proj_name}")
                    head_importance[layer_idx] = torch.zeros(num_heads)
                    continue

                kv_head_scores = []
                for kv_head_i in range(num_kv_heads):
                    start_idx = kv_head_i * head_dim
                    end_idx = (kv_head_i + 1) * head_dim
                    kv_score = proj_importance[start_idx:end_idx, :].sum().item()
                    kv_head_scores.append(kv_score)

                head_scores = []
                for kv_idx, kv_score in enumerate(kv_head_scores):
                    for _ in range(num_queries_per_kv):
                        head_scores.append(kv_score)

            elif mode == "qk":
                if qk_activation_importance is not None and layer_idx in qk_activation_importance:
                    head_scores = qk_activation_importance[layer_idx]
                else:
                    print(f"Warning: No QK activation importance for layer {layer_idx}, falling back to qk_element")
                    head_scores = self._compute_qk_element_importance(
                        importance_scores, layer_idx,
                        num_heads, num_kv_heads, head_dim, num_queries_per_kv, is_gqa
                    )

            elif mode == "qk_element":
                head_scores = self._compute_qk_element_importance(
                    importance_scores, layer_idx,
                    num_heads, num_kv_heads, head_dim, num_queries_per_kv, is_gqa
                )

            elif mode == "qk_attention":
                head_scores = self._compute_qk_attention_importance(
                    importance_scores, layer_idx,
                    num_heads, num_kv_heads, head_dim, num_queries_per_kv, is_gqa
                )

            else:
                proj_name = f"base_model.model.layers.{layer_idx}.self_attn.q_proj"
                proj_importance = self._resolve_importance_name(importance_scores, proj_name)

                if proj_importance is None:
                    print(f"Warning: No importance for {proj_name}")
                    head_importance[layer_idx] = torch.zeros(num_heads)
                    continue

                if is_gqa:
                    head_scores = []
                    for kv_idx in range(num_kv_heads):
                        group_score = 0.0
                        for q_offset in range(num_queries_per_kv):
                            q_head_i = kv_idx * num_queries_per_kv + q_offset
                            start_idx = q_head_i * head_dim
                            end_idx = (q_head_i + 1) * head_dim
                            group_score += proj_importance[start_idx:end_idx, :].sum().item()

                        avg_group_score = group_score / num_queries_per_kv

                        for _ in range(num_queries_per_kv):
                            head_scores.append(avg_group_score)
                else:
                    head_scores = []
                    for head_i in range(num_heads):
                        start_idx = head_i * head_dim
                        end_idx = (head_i + 1) * head_dim
                        head_score = proj_importance[start_idx:end_idx, :].sum().item()
                        head_scores.append(head_score)

            head_importance[layer_idx] = torch.tensor(head_scores) if isinstance(head_scores, list) else head_scores

        return head_importance

    def _compute_qk_element_importance(
        self,
        importance_scores: Dict[str, torch.Tensor],
        layer_idx: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        num_queries_per_kv: int,
        is_gqa: bool
    ) -> torch.Tensor:
        q_proj_name = f"base_model.model.layers.{layer_idx}.self_attn.q_proj"
        k_proj_name = f"base_model.model.layers.{layer_idx}.self_attn.k_proj"

        q_importance = self._resolve_importance_name(importance_scores, q_proj_name)
        k_importance = self._resolve_importance_name(importance_scores, k_proj_name)

        if q_importance is None or k_importance is None:
            print(f"Warning: No Q or K importance for layer {layer_idx}")
            return torch.zeros(num_heads)

        # Q importance: [num_heads * head_dim, hidden_size] -> [num_heads, head_dim]
        q_per_head = torch.zeros(num_heads, head_dim)
        for head_i in range(num_heads):
            start_idx = head_i * head_dim
            end_idx = (head_i + 1) * head_dim
            q_per_head[head_i] = q_importance[start_idx:end_idx, :].sum(dim=1).cpu()

        # K importance: [num_kv_heads * head_dim, hidden_size] -> [num_kv_heads, head_dim]
        k_per_kv_head = torch.zeros(num_kv_heads, head_dim)
        for kv_head_i in range(num_kv_heads):
            start_idx = kv_head_i * head_dim
            end_idx = (kv_head_i + 1) * head_dim
            k_per_kv_head[kv_head_i] = k_importance[start_idx:end_idx, :].sum(dim=1).cpu()

        # [num_kv_heads, head_dim] -> [num_heads, head_dim]
        k_expanded = k_per_kv_head.repeat_interleave(num_queries_per_kv, dim=0)

        # [num_heads, head_dim] × [num_heads, head_dim] -> [num_heads, head_dim]
        qk_element_imp = q_per_head * k_expanded

        head_scores = qk_element_imp.sum(dim=1)

        if is_gqa:
            head_scores = self._aggregate_to_gqa_groups(head_scores, num_heads, num_kv_heads, num_queries_per_kv)

        return head_scores

    def _compute_qk_attention_importance(
        self,
        importance_scores: Dict[str, torch.Tensor],
        layer_idx: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        num_queries_per_kv: int,
        is_gqa: bool
    ) -> torch.Tensor:
        q_proj_name = f"base_model.model.layers.{layer_idx}.self_attn.q_proj"
        k_proj_name = f"base_model.model.layers.{layer_idx}.self_attn.k_proj"

        q_importance = self._resolve_importance_name(importance_scores, q_proj_name)
        k_importance = self._resolve_importance_name(importance_scores, k_proj_name)

        if q_importance is None or k_importance is None:
            print(f"Warning: No Q or K importance for layer {layer_idx}")
            return torch.zeros(num_heads)

        # Q importance: [num_heads * head_dim, hidden_size] -> [num_heads, head_dim]
        q_per_head = torch.zeros(num_heads, head_dim)
        for head_i in range(num_heads):
            start_idx = head_i * head_dim
            end_idx = (head_i + 1) * head_dim
            q_per_head[head_i] = q_importance[start_idx:end_idx, :].sum(dim=1).cpu()

        # K importance: [num_kv_heads * head_dim, hidden_size] -> [num_kv_heads, head_dim]
        k_per_kv_head = torch.zeros(num_kv_heads, head_dim)
        for kv_head_i in range(num_kv_heads):
            start_idx = kv_head_i * head_dim
            end_idx = (kv_head_i + 1) * head_dim
            k_per_kv_head[kv_head_i] = k_importance[start_idx:end_idx, :].sum(dim=1).cpu()

        # [num_kv_heads, head_dim] -> [num_heads, head_dim]
        k_expanded = k_per_kv_head.repeat_interleave(num_queries_per_kv, dim=0)

        # Importance Attention: Q_imp @ K_imp^T
        # [num_heads, head_dim] @ [num_heads, head_dim]^T
        # Q_imp[i] @ K_imp[i]^T = sum(Q_imp[i] * K_imp[i]) (dot product)

        # Q_imp: [num_heads, head_dim]
        # K_imp: [num_heads, head_dim]

        # A_imp[h] = Q_imp[h] @ K_imp[h]^T (scalar, inner product)

        # head_scores = (q_per_head * k_expanded).sum(dim=1)

        # A = Q_per_head @ K_expanded^T  # [num_heads, num_heads]


        # Q_imp: [num_heads, head_dim], K_imp: [num_heads, head_dim]

        # head_score[h] = ||Q_imp[h]|| * ||K_imp[h]|| * cos(Q_imp[h], K_imp[h])
        #              = Q_imp[h] · K_imp[h]


        head_scores = torch.zeros(num_heads)

        for head_i in range(num_heads):
            q_vec = q_per_head[head_i]  # [head_dim]
            k_vec = k_expanded[head_i]  # [head_dim]

            # Inner product (attention-like score)
            attn_score = torch.dot(q_vec, k_vec)

            # attn_score = attn_score / (torch.norm(q_vec) * torch.norm(k_vec) + 1e-8)

            head_scores[head_i] = attn_score.item()

        if is_gqa:
            head_scores = self._aggregate_to_gqa_groups(head_scores, num_heads, num_kv_heads, num_queries_per_kv)

        return head_scores

    def _aggregate_to_gqa_groups(
        self,
        head_scores: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
        num_queries_per_kv: int
    ) -> torch.Tensor:
        # [num_heads] -> [num_kv_heads, num_queries_per_kv]
        grouped = head_scores.view(num_kv_heads, num_queries_per_kv)

        group_avg = grouped.mean(dim=1)  # [num_kv_heads]

        aggregated = group_avg.repeat_interleave(num_queries_per_kv)

        return aggregated

    def compute_embedding_importance(self) -> torch.Tensor:
        importance_scores = self.compute_importance()

        hidden_size = self.model.config.hidden_size
        embed_importance = torch.zeros(hidden_size)

        for name, scores in importance_scores.items():
            if 'weight' in name and scores.dim() == 2:
                if scores.size(1) == hidden_size:
                    embed_importance += scores.sum(dim=0).cpu()

                if scores.size(0) == hidden_size:
                    embed_importance += scores.sum(dim=1).cpu()

        return embed_importance

    def compute_ffn_importance(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        importance_scores = self.compute_importance()

        down_proj_name = f"base_model.model.layers.{layer_idx}.mlp.down_proj"
        down_scores = self._resolve_importance_name(importance_scores, down_proj_name)
        # Phi-2 fallback: fc2
        if down_scores is None:
            fc2_name = f"base_model.model.layers.{layer_idx}.mlp.fc2"
            down_scores = self._resolve_importance_name(importance_scores, fc2_name)

        if down_scores is None:
            print(f"Warning: No down_proj/fc2 importance for layer {layer_idx}")
            # Fallback: uniform importance
            neuron_importance = torch.ones(self.model.config.intermediate_size)
        else:
            # down_proj shape: [hidden_size, intermediate_size]
            neuron_importance = down_scores.sum(dim=0).cpu()

        return neuron_importance, neuron_importance

    def compute_dimension_importance(self) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        importance_scores = self.compute_importance()

        num_layers = self.model.config.num_hidden_layers
        num_heads = self.model.config.num_attention_heads
        head_dim = self.model.config.hidden_size // num_heads

        qk_dim_importance = {}
        v_dim_importance = {}

        for layer_idx in range(num_layers):
            q_proj_name = f"base_model.model.layers.{layer_idx}.self_attn.q_proj"
            q_importance = self._resolve_importance_name(importance_scores, q_proj_name)

            k_proj_name = f"base_model.model.layers.{layer_idx}.self_attn.k_proj"
            k_importance = self._resolve_importance_name(importance_scores, k_proj_name)

            v_proj_name = f"base_model.model.layers.{layer_idx}.self_attn.v_proj"
            v_importance = self._resolve_importance_name(importance_scores, v_proj_name)

            if q_importance is not None:
                # q_importance shape: [num_heads * head_dim, hidden_size]
                q_per_dim = q_importance.sum(dim=1)
                q_per_dim = q_per_dim.view(num_heads, head_dim)
                q_dim_score = q_per_dim.mean(dim=0).cpu()
            else:
                q_dim_score = torch.ones(head_dim)

            if k_importance is not None:
                k_per_dim = k_importance.sum(dim=1)
                k_per_dim = k_per_dim.view(num_heads, head_dim)
                k_dim_score = k_per_dim.mean(dim=0).cpu()
            else:
                k_dim_score = torch.ones(head_dim)

            if v_importance is not None:
                v_per_dim = v_importance.sum(dim=1)
                v_per_dim = v_per_dim.view(num_heads, head_dim)
                v_dim_score = v_per_dim.mean(dim=0).cpu()
            else:
                v_dim_score = torch.ones(head_dim)

            qk_dim_score = (q_dim_score + k_dim_score) / 2

            qk_dim_importance[layer_idx] = qk_dim_score
            v_dim_importance[layer_idx] = v_dim_score

        return qk_dim_importance, v_dim_importance


class WandaActivationCollector:

    def __init__(self, model, target_modules: List[str] = None):
        self.model = model
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "dense", "gate_proj", "up_proj", "down_proj", "fc1", "fc2"]
        self.target_modules = target_modules

        self.activation_sums = {}  # {name: sum of squared activations}
        self.activation_counts = {}  # {name: count}
        self.hooks = []

    def _make_hook(self, name):
        def hook(module, input, output):
            if len(input) > 0 and input[0] is not None:
                # input[0] shape: [batch, seq_len, hidden_dim]
                inp = input[0].detach()
                if inp.dim() == 3:
                    # Flatten batch and seq dimensions
                    inp = inp.view(-1, inp.size(-1))  # [batch*seq, hidden_dim]

                # Squared L2 norm per channel
                squared_norm = (inp ** 2).sum(dim=0)  # [hidden_dim]

                if name not in self.activation_sums:
                    self.activation_sums[name] = squared_norm.cpu()
                    self.activation_counts[name] = inp.size(0)
                else:
                    self.activation_sums[name] += squared_norm.cpu()
                    self.activation_counts[name] += inp.size(0)
        return hook

    def register_hooks(self):
        for name, module in self.model.named_modules():
            # Check if this is a target module
            is_target = any(target in name for target in self.target_modules)
            if not is_target:
                continue

            # Check for LoRA modules (PEFT or Momentum)
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # PEFT LoRA - hook on base_layer
                if hasattr(module, 'base_layer'):
                    hook = module.base_layer.register_forward_hook(self._make_hook(name))
                    self.hooks.append(hook)
            elif hasattr(module, 'momentum_lora'):
                # MomentumLoRA - hook on the module itself
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)

    def get_activation_stats(self) -> Dict[str, torch.Tensor]:
        activation_stats = {}
        for name, squared_sum in self.activation_sums.items():
            count = self.activation_counts[name]
            if count > 0:
                # RMS (Root Mean Square) per channel
                activation_stats[name] = torch.sqrt(squared_sum / count)
        return activation_stats

    def reset(self):
        self.activation_sums = {}
        self.activation_counts = {}

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def collect_activation_stats(
    model,
    dataloader,
    device,
    num_samples: int = 128,
    target_modules: List[str] = None
) -> Dict[str, torch.Tensor]:
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "dense", "gate_proj", "up_proj", "down_proj", "fc1", "fc2"]

    activation_sums = {}  # {name: sum of squared activations}
    activation_counts = {}  # {name: count}
    hooks = []

    def make_hook(name):
        def hook(module, input, output):
            if len(input) > 0 and input[0] is not None:
                # input[0] shape: [batch, seq_len, hidden_dim]
                inp = input[0].detach()
                if inp.dim() == 3:
                    # Flatten batch and seq dimensions
                    inp = inp.view(-1, inp.size(-1))  # [batch*seq, hidden_dim]

                # Squared L2 norm per channel (dim=0: average over tokens)
                squared_norm = (inp ** 2).sum(dim=0)  # [hidden_dim]

                if name not in activation_sums:
                    activation_sums[name] = squared_norm.cpu()
                    activation_counts[name] = inp.size(0)
                else:
                    activation_sums[name] += squared_norm.cpu()
                    activation_counts[name] += inp.size(0)
        return hook

    # Register hooks
    for name, module in model.named_modules():
        # Check if this is a target module
        is_target = any(target in name for target in target_modules)
        if not is_target:
            continue

        # Check for LoRA modules (PEFT or Momentum)
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            # PEFT LoRA - hook on base_layer
            if hasattr(module, 'base_layer'):
                hook = module.base_layer.register_forward_hook(make_hook(name))
                hooks.append(hook)
        elif hasattr(module, 'momentum_lora'):
            # MomentumLoRA - hook on the module itself
            hook = module.register_forward_hook(make_hook(name))
            hooks.append(hook)

    # Collect activations
    model.eval()
    samples_collected = 0

    with torch.no_grad():
        for batch in dataloader:
            if samples_collected >= num_samples:
                break

            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                input_ids = batch.get('input_ids')
                attention_mask = batch.get('attention_mask')
            else:
                input_ids = batch[0].to(device) if isinstance(batch[0], torch.Tensor) else batch[0]
                attention_mask = batch[1].to(device) if len(batch) > 1 and isinstance(batch[1], torch.Tensor) else None

            # Forward pass
            try:
                if attention_mask is not None:
                    model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    model(input_ids=input_ids)
            except Exception as e:
                print(f"[WANDA] Forward pass error: {e}")
                continue

            samples_collected += input_ids.size(0)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Compute mean activation norms
    activation_stats = {}
    for name, squared_sum in activation_sums.items():
        count = activation_counts[name]
        # RMS (Root Mean Square) per channel
        activation_stats[name] = torch.sqrt(squared_sum / count)

    print(f"[WANDA] Collected activation stats for {len(activation_stats)} modules from {samples_collected} samples")

    return activation_stats
