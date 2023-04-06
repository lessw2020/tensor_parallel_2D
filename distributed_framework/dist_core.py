import torch.distributed as dist
import torch
import torch.nn as nn
import os
from typing import Optional

# import torch.distributed


def dist_setup():
    dist.init_process_group(backend="nccl")
    gpu_id = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)


def get_rank() -> Optional[int]:
    try:
        rank = dist.get_rank()
    except RuntimeError:
        rank = None
    return rank


def apply_ddp(model: nn.Module) -> nn.Module:
    """
    Applies DDP to ``model``.
    We enable ``gradient_as_bucket_view`` as a performance optimization.
    """
    model = DDP(model, device_ids=[dist.get_rank()], gradient_as_bucket_view=True)
    return model
