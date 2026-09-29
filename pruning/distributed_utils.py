
import os
import torch
import torch.distributed as dist
from typing import List, Optional


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1 and dist.is_available() and not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        print(f"[Distributed] Initialized: rank={dist.get_rank()}, "
              f"world_size={dist.get_world_size()}, local_rank={local_rank}")
        return True
    return False


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def get_rank() -> int:
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if is_distributed():
        return dist.get_world_size()
    return 1


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    return get_rank() == 0


def barrier():
    if is_distributed():
        dist.barrier()


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    if not is_distributed():
        return tensor

    if tensor.is_cuda:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    else:
        device = torch.device(f"cuda:{get_local_rank()}")
        tensor_gpu = tensor.to(device)
        dist.all_reduce(tensor_gpu, op=dist.ReduceOp.SUM)
        tensor.copy_(tensor_gpu.cpu())
    return tensor


def all_reduce_dict(score_dict: dict) -> dict:
    if not is_distributed():
        return score_dict
    for key in score_dict:
        if isinstance(score_dict[key], torch.Tensor):
            score_dict[key] = all_reduce_sum(score_dict[key])
    return score_dict


def all_gather_cat(local_tensor: torch.Tensor) -> torch.Tensor:
    if not is_distributed():
        return local_tensor

    device = torch.device(f"cuda:{get_local_rank()}")
    local_gpu = local_tensor.to(device)

    gathered = [torch.zeros_like(local_gpu) for _ in range(get_world_size())]
    dist.all_gather(gathered, local_gpu)

    result = torch.cat(gathered, dim=0)
    if not local_tensor.is_cuda:
        result = result.cpu()
    return result


def all_gather_cat_variable(local_tensor: torch.Tensor) -> torch.Tensor:
    if not is_distributed():
        return local_tensor

    device = torch.device(f"cuda:{get_local_rank()}")
    local_gpu = local_tensor.to(device)

    local_size = torch.tensor([local_gpu.shape[0]], dtype=torch.long, device=device)
    all_sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(get_world_size())]
    dist.all_gather(all_sizes, local_size)
    all_sizes = [s.item() for s in all_sizes]
    max_size = max(all_sizes)

    if local_gpu.shape[0] < max_size:
        pad_shape = list(local_gpu.shape)
        pad_shape[0] = max_size - local_gpu.shape[0]
        padding = torch.zeros(pad_shape, dtype=local_gpu.dtype, device=device)
        local_padded = torch.cat([local_gpu, padding], dim=0)
    else:
        local_padded = local_gpu

    gathered = [torch.zeros_like(local_padded) for _ in range(get_world_size())]
    dist.all_gather(gathered, local_padded)

    trimmed = [g[:s] for g, s in zip(gathered, all_sizes)]
    result = torch.cat(trimmed, dim=0)

    if not local_tensor.is_cuda:
        result = result.cpu()
    return result


def broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    if not is_distributed():
        return tensor

    device = torch.device(f"cuda:{get_local_rank()}")
    tensor_gpu = tensor.to(device)
    dist.broadcast(tensor_gpu, src=src)

    if not tensor.is_cuda:
        return tensor_gpu.cpu()
    return tensor_gpu


def get_device_map_for_loading():
    if is_distributed():
        return None
    return "auto"


def print_rank0(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)
