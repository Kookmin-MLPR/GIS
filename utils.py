import torch
from datasets import load_dataset
from transformers import AutoTokenizer

def load_alpaca_dataset(config):
    import os

    dataset_name = config.dataset_name

    dataset_path = getattr(config, 'dataset_path', None)
    if dataset_path and os.path.exists(dataset_path):
        print(f"[Alpaca] Loading from local file: {dataset_path}")
        dataset = load_dataset("json", data_files=dataset_path)
    else:
        if dataset_name == "alpaca":
            dataset_name = "tatsu-lab/alpaca"
        elif dataset_name == "alpaca_cleaned":
            dataset_name = "yahma/alpaca-cleaned"

        dataset = load_dataset(dataset_name)

    if config.validation_split > 0:
        train_val = dataset["train"].train_test_split(
            test_size=config.validation_split, seed=42
        )
        return train_val["train"], train_val["test"]
    else:
        return dataset["train"], None

def format_alpaca_prompt(example):
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output = example["output"]

    if input_text:
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    else:
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

    return {"text": prompt + output}

def tokenize_function(examples, tokenizer, max_length):
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    outputs["labels"] = outputs["input_ids"].clone()
    return outputs

def load_c4_dataset(config, num_samples=20000, max_length=512):
    print(f"[C4] Loading {num_samples} samples with max_length={max_length}")

    try:
        dataset = load_dataset(
            "allenai/c4",
            "en",
            split="train",
            streaming=False
        )
    except Exception as e:
        print(f"[C4] Warning: Could not load allenai/c4, trying legacy method: {e}")
        print("[C4] Using fallback: generating sample data for testing")
        from datasets import Dataset
        samples = [{"text": f"This is sample text number {i} for C4 dataset testing."} for i in range(num_samples)]
        dataset = Dataset.from_list(samples)

        if config.validation_split > 0:
            train_val = dataset.train_test_split(
                test_size=config.validation_split,
                seed=42
            )
            return train_val["train"], train_val["test"]
        else:
            return dataset, None

    samples = []
    print(f"[C4] Streaming and collecting {num_samples} samples...")
    for i, example in enumerate(dataset):
        if i >= num_samples:
            break
        samples.append({"text": example["text"]})

        if (i + 1) % 5000 == 0:
            print(f"[C4] Collected {i + 1}/{num_samples} samples...")

    from datasets import Dataset
    dataset = Dataset.from_list(samples)

    print(f"[C4] Successfully loaded {len(dataset)} samples")

    if config.validation_split > 0:
        train_val = dataset.train_test_split(
            test_size=config.validation_split,
            seed=42
        )
        return train_val["train"], train_val["test"]
    else:
        return dataset, None


def load_c4_shard_dataset(config, num_samples=20000, max_length=512, shard_id=0):
    print(f"[C4-Shard] Loading shard {shard_id:05d} with {num_samples} samples, max_length={max_length}")
    print(f"[C4-Shard] This matches other implementations using first shard only")
    print(f"[C4-Shard] NOTE: No train/val split (eval_dataset=None, same as other papers)")

    try:
        shard_file = f"en/c4-train.{shard_id:05d}-of-01024.json.gz"
        print(f"[C4-Shard] Loading specific shard file: {shard_file}")

        dataset = load_dataset(
            "allenai/c4",
            data_files={"train": shard_file},
            split="train",
            streaming=False
        )

        print(f"[C4-Shard] Shard loaded successfully, total samples in shard: {len(dataset)}")

    except Exception as e:
        print(f"[C4-Shard] Warning: Could not load specific shard, error: {e}")
        print("[C4-Shard] Falling back to regular C4 loading...")
        return load_c4_dataset(config, num_samples, max_length)

    if len(dataset) > num_samples:
        dataset = dataset.select(range(num_samples))
        print(f"[C4-Shard] Selected first {num_samples} samples from shard")
    else:
        print(f"[C4-Shard] Using all {len(dataset)} samples from shard (less than requested {num_samples})")

    dataset = dataset.select_columns(["text"])

    print(f"[C4-Shard] Successfully loaded {len(dataset)} samples from shard {shard_id:05d}")

    return dataset, None


def load_c4_local_dataset(config, num_samples=20000, max_length=512, data_path="./data/c4/c4_train_20k.jsonl"):
    import json
    from pathlib import Path
    from datasets import Dataset

    data_file = Path(data_path)

    print(f"[C4-Local] Loading local dataset from: {data_file}")

    if not data_file.exists():
        print(f"[C4-Local] ERROR: File not found: {data_file}")
        print(f"[C4-Local] Please run: python scripts/download_c4.py")
        print(f"[C4-Local] Falling back to online loading...")
        return load_c4_shard_dataset(config, num_samples, max_length)

    samples = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            data = json.loads(line.strip())
            samples.append({"text": data["text"]})

    dataset = Dataset.from_list(samples)

    print(f"[C4-Local] Successfully loaded {len(dataset)} samples")

    if len(dataset) > 0:
        preview = dataset[0]["text"][:100] + "..." if len(dataset[0]["text"]) > 100 else dataset[0]["text"]
        print(f"[C4-Local] Sample preview: {preview}")

    return dataset, None


def load_wikitext2_dataset(config, split="train", num_samples=None, max_length=512):
    from datasets import Dataset

    print(f"[WikiText2] Loading {split} split with max_length={max_length}")

    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        print(f"[WikiText2] Loaded {len(dataset)} samples from {split}")

    except Exception as e:
        print(f"[WikiText2] Warning: Could not load wikitext-2-raw-v1: {e}")
        print("[WikiText2] Trying alternative method...")

        try:
            dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
            print(f"[WikiText2] Loaded {len(dataset)} samples from Salesforce/wikitext")
        except Exception as e2:
            print(f"[WikiText2] ERROR: Could not load WikiText2: {e2}")
            print("[WikiText2] Using fallback: generating sample data")
            samples = [{"text": f"This is sample text {i} for WikiText2 testing."} for i in range(1000)]
            dataset = Dataset.from_list(samples)
            return dataset, None

    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 10)
    print(f"[WikiText2] After filtering empty lines: {len(dataset)} samples")

    if num_samples is not None and len(dataset) > num_samples:
        dataset = dataset.select(range(num_samples))
        print(f"[WikiText2] Selected first {num_samples} samples")

    print(f"[WikiText2] Final dataset size: {len(dataset)} samples")

    if len(dataset) > 0:
        preview = dataset[0]["text"][:100] + "..." if len(dataset[0]["text"]) > 100 else dataset[0]["text"]
        print(f"[WikiText2] Sample preview: {preview}")

    return dataset, None


def load_wikitext2_train_dataset(config, num_samples=None, max_length=512):
    return load_wikitext2_dataset(config, split="train", num_samples=num_samples, max_length=max_length)


def load_wikitext2_val_dataset(config, num_samples=None, max_length=512):
    return load_wikitext2_dataset(config, split="validation", num_samples=num_samples, max_length=max_length)


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(
        f"trainable params: {trainable_params:,} || "
        f"all params: {all_param:,} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}%"
    )
