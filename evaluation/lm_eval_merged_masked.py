
import os
import pickle
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_and_merge_lora_with_masks(base_model_path, lora_path, device="cuda:0"):
    print("=" * 70)
    print("Loading Model with Merge-then-Mask Approach")
    print("=" * 70)
    print(f"Base model: {base_model_path}")
    print(f"LoRA path: {lora_path}")
    print(f"Device: {device}")
    print()

    print("[1/5] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    print("✓ Base model loaded")

    print("[2/5] Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    print("✓ LoRA adapter loaded")

    print("[3/5] Merging LoRA into base weights...")
    print("   This creates: W_merged = W_base + LoRA_B @ LoRA_A")

    model = model.to(torch.float32)
    model = model.merge_and_unload()
    model = model.to(torch.float16)

    print("✓ LoRA merged into base weights")

    mask_path = os.path.join(lora_path, "masks.pkl")
    if not os.path.exists(mask_path):
        print(f"\n❌ Error: Masks not found at {mask_path}")
        return None, None

    print("[4/5] Loading and applying masks to merged weights...")
    print("   Applying: W_masked = W_merged × mask")
    print("   This is equivalent to physical pruning!")

    with open(mask_path, 'rb') as f:
        masks = pickle.load(f)

    print(f"✓ Loaded {len(masks)} masks")

    mask_info_path = os.path.join(lora_path, "mask_info.json")
    if os.path.exists(mask_info_path):
        import json
        with open(mask_info_path, 'r') as f:
            mask_info = json.load(f)

        print(f"\nMask Info:")
        print(f"  Overall Sparsity: {mask_info.get('overall_sparsity', 'N/A'):.4f}")
        ratios = mask_info.get('pruning_ratios', {})
        print(f"  Pruning Ratios:")
        print(f"    - Head:      {ratios.get('head', 'N/A'):.4f}")
        print(f"    - Embedding: {ratios.get('embedding', 'N/A'):.4f}")
        print(f"    - FFN:       {ratios.get('ffn', 'N/A'):.4f}")
        print()

    masks_applied = apply_masks_to_merged_weights(model, masks)

    print(f"✓ Applied {masks_applied} masks to merged weights")

    print("[5/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("✓ Tokenizer loaded")

    print()
    print("=" * 70)
    print("Model Ready for Evaluation (Merge-then-Mask)")
    print("=" * 70)
    print()
    print("✅ Weights are masked at weight-level (NOT output-level)")
    print("✅ This matches physical pruning exactly")
    print("✅ Evaluation will be accurate and consistent")
    print()

    model.eval()
    return model, tokenizer


def apply_masks_to_merged_weights(model, masks):
    masks_applied = 0


    for mask_name, mask_tensor in masks.items():
        # "base_model.model.layers.0.self_attn.q_proj.output"
        # "base_model.model.embed_tokens.output"

        parts = mask_name.split('.')
        mask_type = parts[-1]  # "output" or "input"
        module_path_parts = parts[:-1]

        if module_path_parts[0] == "base_model" and module_path_parts[1] == "model":
            # "base_model.model.layers.0..." → "model.layers.0..."
            module_path_parts = ["model"] + module_path_parts[2:]

        module_path = '.'.join(module_path_parts)

        try:
            module = model
            for attr in module_path.split('.'):
                module = getattr(module, attr)
        except AttributeError:
            continue

        if isinstance(module, nn.Linear):
            if mask_type == "output":
                # W: [out_features, in_features]
                # mask: [out_features]
                mask = mask_tensor.to(module.weight.device, dtype=module.weight.dtype)
                with torch.no_grad():
                    # Broadcasting: [out_features, in_features] * [out_features, 1]
                    module.weight.data = module.weight.data * mask.unsqueeze(1)
                    if module.bias is not None:
                        module.bias.data = module.bias.data * mask
                masks_applied += 1

            elif mask_type == "input":
                # W: [out_features, in_features]
                # mask: [in_features]
                mask = mask_tensor.to(module.weight.device, dtype=module.weight.dtype)
                with torch.no_grad():
                    # Broadcasting: [out_features, in_features] * [1, in_features]
                    module.weight.data = module.weight.data * mask.unsqueeze(0)
                masks_applied += 1

        elif isinstance(module, nn.Embedding):
            if mask_type == "output":
                mask = mask_tensor.to(module.weight.device, dtype=module.weight.dtype)
                with torch.no_grad():
                    # Broadcasting: [vocab_size, embedding_dim] * [1, embedding_dim]
                    module.weight.data = module.weight.data * mask.unsqueeze(0)
                masks_applied += 1

        elif isinstance(module, (nn.LayerNorm, nn.modules.normalization.LayerNorm)):
            if mask_type == "output":
                mask = mask_tensor.to(module.weight.device, dtype=module.weight.dtype)
                with torch.no_grad():
                    if module.weight is not None:
                        module.weight.data = module.weight.data * mask
                    if module.bias is not None:
                        module.bias.data = module.bias.data * mask
                masks_applied += 1

        try:
            from transformers.models.llama.modeling_llama import LlamaRMSNorm
            if isinstance(module, LlamaRMSNorm):
                if mask_type == "output":
                    mask = mask_tensor.to(module.weight.device, dtype=module.weight.dtype)
                    with torch.no_grad():
                        module.weight.data = module.weight.data * mask
                    masks_applied += 1
        except ImportError:
            pass

    return masks_applied


def evaluate_merged_masked_model(base_model_path, lora_path, tasks, batch_size=8, device="cuda:0", output_dir=None):
    model, tokenizer = load_and_merge_lora_with_masks(base_model_path, lora_path, device)

    if model is None:
        return None

    print("=" * 70)
    print("Running lm_eval with Merged-Masked Weights")
    print("=" * 70)
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Batch size: {batch_size}")
    print()

    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM

        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            device=device
        )

        results = evaluator.simple_evaluate(
            model=lm,
            tasks=tasks,
            batch_size=batch_size,
            device=device
        )

        if output_dir is not None:
            import os
            import json
            from datetime import datetime

            os.makedirs(output_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = os.path.join(output_dir, f"results_{timestamp}.json")

            import numpy as np

            def make_json_serializable(obj):
                if obj is None or isinstance(obj, (bool, int, float, str)):
                    return obj
                elif isinstance(obj, dict):
                    return {k: make_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [make_json_serializable(item) for item in obj]
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif hasattr(obj, 'item'):  # torch tensor
                    try:
                        return obj.item()
                    except:
                        return str(obj)
                elif isinstance(obj, type):
                    return str(obj)
                else:
                    return str(obj)

            results_clean = make_json_serializable(results)

            with open(result_file, 'w') as f:
                json.dump(results_clean, f, indent=2)

            print(f"\n✓ Results saved to: {result_file}")

        return results

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    from config import LoRAConfig, EvalConfig

    lora_config = LoRAConfig()
    eval_config = EvalConfig()

    print("=" * 70)
    print("Comparing Evaluation Methods")
    print("=" * 70)
    print()
    print("Method 1: Forward Hook (Current)")
    print("  - Applies mask to output: ((W + LoRA)·x) × mask")
    print("  - LoRA output also masked")
    print()
    print("Method 2: Merge-then-Mask (Improved)")
    print("  - Applies mask to weight: (W + LoRA) × mask")
    print("  - Same as physical pruning")
    print()
    print("=" * 70)
    print()

    # Test on a single task
    test_tasks = ["boolq"]

    print("Testing on task: boolq")
    print()

    # Method 2: Merge-then-Mask
    print("\n" + "=" * 70)
    print("Evaluating with Merge-then-Mask")
    print("=" * 70)

    results = evaluate_merged_masked_model(
        base_model_path=lora_config.model_name,
        lora_path=lora_config.student_output_dir,
        tasks=test_tasks,
        batch_size=eval_config.eval_batch_size,
        device="cuda:0"
    )

    if results:
        print("\n" + "=" * 70)
        print("Results (Merge-then-Mask)")
        print("=" * 70)
        print(results)
