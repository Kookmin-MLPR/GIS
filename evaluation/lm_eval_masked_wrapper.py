
from evaluation.lm_eval_merged_masked import load_and_merge_lora_with_masks


def load_masked_lora_for_lm_eval(base_model, lora_path, device="cuda:0"):
    return load_and_merge_lora_with_masks(base_model, lora_path, device)
