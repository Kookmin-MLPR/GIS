
import subprocess
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Union
import tempfile


def run_lm_eval(
    model_path: str,
    tasks: Union[str, List[str]],
    output_dir: str = "./lm_eval_results",
    batch_size: int = 8,
    device: str = "cuda:0",
    num_fewshot: int = 0,
    limit: Optional[int] = None,
    peft_path: Optional[str] = None,
    model_args: Optional[Dict[str, str]] = None,
    verbose: bool = True
) -> Dict:

    try:
        subprocess.run(
            ["lm_eval", "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("lm_eval not installed, installing...")
        subprocess.run(["pip", "install", "lm-eval"], check=True)

    if isinstance(tasks, list):
        tasks_str = ",".join(tasks)
    else:
        tasks_str = tasks

    os.makedirs(output_dir, exist_ok=True)

    model_args_dict = {"pretrained": model_path}
    if peft_path:
        model_args_dict["peft"] = peft_path
    if model_args:
        model_args_dict.update(model_args)

    model_args_str = ",".join([f"{k}={v}" for k, v in model_args_dict.items()])

    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", model_args_str,
        "--tasks", tasks_str,
        "--batch_size", str(batch_size),
        "--device", device,
        "--output_path", output_dir
    ]

    if num_fewshot > 0:
        cmd.extend(["--num_fewshot", str(num_fewshot)])

    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    if verbose:
        print("=" * 70)
        print("Running lm_eval")
        print("=" * 70)
        print(f"Model: {model_path}")
        if peft_path:
            print(f"PEFT: {peft_path}")
        print(f"Tasks: {tasks_str}")
        print(f"Batch size: {batch_size}")
        print(f"Device: {device}")
        print(f"Output: {output_dir}")
        print("=" * 70)
        print()

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE if not verbose else None,
            stderr=subprocess.STDOUT if not verbose else None,
            text=True,
            check=True
        )

        if verbose and result.stdout:
            print(result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"lm_eval execution failed!")
        if e.stdout:
            print(e.stdout)
        raise

    result_file = find_latest_result(output_dir)

    if result_file is None:
        raise FileNotFoundError(f"Result file not found: {output_dir}")

    with open(result_file, 'r') as f:
        results = json.load(f)

    if verbose:
        print("\n" + "=" * 70)
        print("Evaluation Results")
        print("=" * 70)
        print_results(results)
        print("=" * 70)

    return results


def find_latest_result(output_dir: str) -> Optional[str]:
    output_path = Path(output_dir)

    result_files = list(output_path.rglob("results_*.json"))

    if not result_files:
        return None

    latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
    return str(latest_file)


def print_results(results: Dict, show_stderr: bool = False):
    if "results" not in results:
        print("Failed to parse results.")
        print(json.dumps(results, indent=2)[:500])
        return

    task_results = results["results"]

    accuracies = []
    best_accuracies = []
    perplexities = []

    accuracy_tasks = ["winogrande", "hellaswag", "arc_easy", "arc_challenge", "piqa"]
    perplexity_tasks = ["wikitext"]

    print("\n[Accuracy Tasks]")
    for task_name in accuracy_tasks:
        if task_name not in task_results:
            continue

        metrics = task_results[task_name]
        if not isinstance(metrics, dict):
            continue

        acc = metrics.get("acc,none", metrics.get("acc", None))
        acc_norm = metrics.get("acc_norm,none", metrics.get("acc_norm", None))

        output_parts = []

        if acc is not None:
            if show_stderr:
                acc_stderr = metrics.get("acc_stderr,none", metrics.get("acc_stderr", None))
                if acc_stderr is not None:
                    output_parts.append(f"acc={acc:.4f}±{acc_stderr:.4f}")
                else:
                    output_parts.append(f"acc={acc:.4f}")
            else:
                output_parts.append(f"acc={acc:.4f}")
            accuracies.append(acc)

        if acc_norm is not None:
            if show_stderr:
                acc_norm_stderr = metrics.get("acc_norm_stderr,none", metrics.get("acc_norm_stderr", None))
                if acc_norm_stderr is not None:
                    output_parts.append(f"acc_norm={acc_norm:.4f}±{acc_norm_stderr:.4f}")
                else:
                    output_parts.append(f"acc_norm={acc_norm:.4f}")
            else:
                output_parts.append(f"acc_norm={acc_norm:.4f}")

        if acc is not None and acc_norm is not None:
            best_accuracies.append(max(acc, acc_norm))
        elif acc is not None:
            best_accuracies.append(acc)
        elif acc_norm is not None:
            best_accuracies.append(acc_norm)

        if output_parts:
            print(f"{task_name:20s}: {', '.join(output_parts)}")

    if accuracies:
        avg_acc = sum(accuracies) / len(accuracies)
        print(f"\n{'Average (acc)':20s}: {avg_acc:.4f}")

    if best_accuracies:
        avg_best = sum(best_accuracies) / len(best_accuracies)
        print(f"{'Average (best)':20s}: {avg_best:.4f}")

    print("\n[Perplexity Tasks]")
    for task_name in perplexity_tasks:
        if task_name not in task_results:
            continue

        metrics = task_results[task_name]
        if not isinstance(metrics, dict):
            continue

        word_ppl = metrics.get("word_perplexity,none", metrics.get("word_perplexity", None))
        byte_ppl = metrics.get("byte_perplexity,none", metrics.get("byte_perplexity", None))
        bits_per_byte = metrics.get("bits_per_byte,none", metrics.get("bits_per_byte", None))

        output_parts = []

        if word_ppl is not None:
            output_parts.append(f"ppl={word_ppl:.2f}")
            perplexities.append(word_ppl)

        if byte_ppl is not None:
            output_parts.append(f"byte_ppl={byte_ppl:.2f}")

        if bits_per_byte is not None:
            output_parts.append(f"bpb={bits_per_byte:.4f}")

        if output_parts:
            display_name = "WikiText2" if task_name == "wikitext" else "PTB"
            print(f"{display_name:20s}: {', '.join(output_parts)}")

    if perplexities:
        avg_ppl = sum(perplexities) / len(perplexities)
        print(f"\n{'Average (ppl)':20s}: {avg_ppl:.2f}")


def evaluate_model_with_lm_eval(
    model_path: str,
    output_dir: str = "./lm_eval_results",
    tasks: Optional[List[str]] = None,
    batch_size: int = 8,
    device: str = "cuda:0",
    peft_path: Optional[str] = None
) -> Dict:

    if tasks is None:
        tasks = [
            "winogrande",
            "hellaswag",
            "arc_easy",
            "arc_challenge",
            "piqa",
        ]

    return run_lm_eval(
        model_path=model_path,
        tasks=tasks,
        output_dir=output_dir,
        batch_size=batch_size,
        device=device,
        peft_path=peft_path,
        verbose=True
    )


def evaluate_lora_model(
    base_model: str,
    lora_path: str,
    output_dir: str = "./lm_eval_results",
    tasks: Optional[List[str]] = None,
    batch_size: int = 8,
    device: str = "cuda:0"
) -> Dict:

    return evaluate_model_with_lm_eval(
        model_path=base_model,
        output_dir=output_dir,
        tasks=tasks,
        batch_size=batch_size,
        device=device,
        peft_path=lora_path
    )


def evaluate_merged_model(
    model_path: str,
    output_dir: str = "./lm_eval_results",
    tasks: Optional[List[str]] = None,
    batch_size: int = 8,
    device: str = "cuda:0"
) -> Dict:
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        print("lm_eval not installed, installing...")
        subprocess.run(["pip", "install", "lm-eval"], check=True)
        import lm_eval
        from lm_eval.models.huggingface import HFLM

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if tasks is None:
        tasks = [
            "winogrande",
            "hellaswag",
            "arc_easy",
            "arc_challenge",
            "piqa",
        ]

    model_path = os.path.abspath(model_path)

    print("=" * 70)
    print("Evaluating Model with lm_eval Python API")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print("=" * 70)
    print()

    print("[Load] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True
    )
    model.eval()
    print("[Load] Model loaded successfully")
    print()

    print("=" * 70)
    print("Creating lm_eval model wrapper")
    print("=" * 70)

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device
    )

    print("\n" + "=" * 70)
    print("Running lm_eval tasks")
    print("=" * 70)
    print()

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        batch_size=batch_size,
        device=device
    )

    os.makedirs(output_dir, exist_ok=True)

    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(output_dir, f"results_{timestamp}.json")

    def make_json_serializable(obj):
        import numpy as np
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # scalar tensor
            return obj.item()
        elif isinstance(obj, type) or str(type(obj).__name__) == 'dtype':
            return str(obj)
        elif hasattr(obj, 'dtype') and not isinstance(obj, (int, float, str, bool)):
            try:
                return float(obj)
            except:
                return str(obj)
        elif callable(obj):
            return str(obj)
        else:
            return obj

    serializable_results = make_json_serializable(results)

    with open(result_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print_results(results)
    print("=" * 70)
    print(f"\nResults saved to: {result_file}")

    return results


def evaluate_lora_model_with_masks(
    base_model: str,
    lora_path: str,
    output_dir: str = "./lm_eval_results",
    tasks: Optional[List[str]] = None,
    batch_size: int = 8,
    device: str = "cuda:0"
) -> Dict:
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        print("lm_eval not installed, installing...")
        subprocess.run(["pip", "install", "lm-eval"], check=True)
        import lm_eval
        from lm_eval.models.huggingface import HFLM

    from evaluation.lm_eval_masked_wrapper import load_masked_lora_for_lm_eval

    if tasks is None:
        tasks = [
            "winogrande",
            "hellaswag",
            "arc_easy",
            "arc_challenge",
            "piqa",
        ]

    print("=" * 70)
    print("Evaluating Masked LoRA with lm_eval Python API")
    print("=" * 70)
    print(f"Base model: {base_model}")
    print(f"LoRA path: {lora_path}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print("=" * 70)
    print()

    model, tokenizer = load_masked_lora_for_lm_eval(base_model, lora_path, device)

    print("\n" + "=" * 70)
    print("Creating lm_eval model wrapper")
    print("=" * 70)

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device
    )

    print("\n" + "=" * 70)
    print("Running lm_eval tasks")
    print("=" * 70)
    print()

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        batch_size=batch_size,
        device=device
    )

    os.makedirs(output_dir, exist_ok=True)

    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(output_dir, f"results_{timestamp}.json")

    def make_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(item) for item in obj]
        elif hasattr(obj, 'dtype'):
            return str(obj)
        elif isinstance(obj, type):
            return str(obj)
        else:
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)

    serializable_results = make_json_serializable(results)

    with open(result_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print_results(results)
    print("=" * 70)
    print(f"\nResults saved to: {result_file}")

    return results


def evaluate_lora_model_with_heterogeneous(
    base_model_path: str,
    lora_path: str,
    output_dir: str = "./lm_eval_results",
    tasks: Optional[List[str]] = None,
    batch_size: int = 8,
    device: str = "cuda:0"
) -> Dict:
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        print("lm_eval not installed, installing...")
        subprocess.run(["pip", "install", "lm-eval"], check=True)
        import lm_eval
        from lm_eval.models.huggingface import HFLM

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    if tasks is None:
        tasks = [
            "winogrande",
            "hellaswag",
            "arc_easy",
            "arc_challenge",
            "piqa",
        ]

    print("=" * 70)
    print("Evaluating Heterogeneous Model + LoRA with lm_eval Python API")
    print("=" * 70)
    print(f"Base model: {base_model_path}")
    print(f"LoRA path: {lora_path}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print("=" * 70)
    print()

    print("[Load] Loading base model...")

    base_model_path = os.path.abspath(base_model_path)
    lora_path = os.path.abspath(lora_path)

    config_path = os.path.join(base_model_path, "config.json")
    meta_path = os.path.join(base_model_path, "physical_pruning_meta.json")
    layer_sizes_path = os.path.join(base_model_path, "layer_sizes.json")
    is_heterogeneous = False

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            model_type = config_data.get("model_type", "")
            is_heterogeneous = (model_type == "heterogeneous_llama")

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

                head_counts = [layer.get('num_heads', 0) for layer in layer_sizes]
                unique_head_counts = set(head_counts)
                if len(unique_head_counts) > 1:
                    is_heterogeneous = True
                    print(f"[Load] Auto-detected: Different head counts across layers ({len(unique_head_counts)} unique counts)")

    if is_heterogeneous:
        print("[Load] Using heterogeneous model loader (supports global pruning)")
        from models.heterogeneous_llama import load_heterogeneous_llama
        base_model, tokenizer = load_heterogeneous_llama(base_model_path, device='auto')
    else:
        print("[Load] Using standard model loader (layer-wise pruning)")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            local_files_only=True
        )

    print("[Load] Base model loaded successfully")

    print("[Load] Applying LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, lora_path, local_files_only=True)
    model.eval()

    print("[Load] LoRA adapter applied successfully")
    print()

    print("=" * 70)
    print("Creating lm_eval model wrapper")
    print("=" * 70)

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device
    )

    print("\n" + "=" * 70)
    print("Running lm_eval tasks")
    print("=" * 70)
    print()

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        batch_size=batch_size,
        device=device
    )

    os.makedirs(output_dir, exist_ok=True)

    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(output_dir, f"results_{timestamp}.json")

    def make_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(item) for item in obj]
        elif hasattr(obj, 'dtype'):
            return str(obj)
        elif isinstance(obj, type):
            return str(obj)
        else:
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)

    serializable_results = make_json_serializable(results)

    with open(result_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print_results(results)
    print("=" * 70)
    print(f"\nResults saved to: {result_file}")

    return results


def evaluate_heterogeneous_model(
    model_path: str,
    output_dir: str = "./lm_eval_results",
    tasks: Optional[List[str]] = None,
    batch_size: int = 8,
    device: str = "cuda:0"
) -> Dict:
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        print("lm_eval not installed, installing...")
        subprocess.run(["pip", "install", "lm-eval"], check=True)
        import lm_eval
        from lm_eval.models.huggingface import HFLM

    import torch

    if tasks is None:
        tasks = [
            "winogrande",
            "hellaswag",
            "arc_easy",
            "arc_challenge",
            "piqa",
        ]

    model_path = os.path.abspath(model_path)

    print("=" * 70)
    print("Evaluating Heterogeneous Model with lm_eval Python API")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print("=" * 70)
    print()

    print("[Load] Loading heterogeneous model...")
    from models.heterogeneous_llama import load_heterogeneous_llama
    model, tokenizer = load_heterogeneous_llama(model_path, device='auto')
    model.eval()
    print("[Load] Model loaded successfully")
    print()

    print("=" * 70)
    print("Creating lm_eval model wrapper")
    print("=" * 70)

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device
    )

    print("\n" + "=" * 70)
    print("Running lm_eval tasks")
    print("=" * 70)
    print()

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        batch_size=batch_size,
        device=device
    )

    os.makedirs(output_dir, exist_ok=True)

    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(output_dir, f"results_{timestamp}.json")

    def make_json_serializable(obj):
        import numpy as np
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # scalar tensor
            return obj.item()
        elif isinstance(obj, type) or str(type(obj).__name__) == 'dtype':
            return str(obj)
        elif hasattr(obj, 'dtype') and not isinstance(obj, (int, float, str, bool)):
            try:
                return float(obj)
            except:
                return str(obj)
        elif callable(obj):
            return str(obj)
        else:
            return obj

    serializable_results = make_json_serializable(results)

    with open(result_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print_results(results)
    print("=" * 70)
    print(f"\nResults saved to: {result_file}")

    return results


if __name__ == "__main__":
    print("lm_eval_utils module.")
    print("Call evaluate_lora_model() or evaluate_merged_model() to run evaluation.")
