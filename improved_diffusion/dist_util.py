"""
Helpers for distributed training (single-GPU version).
"""

import io
import os
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist

# Disable MPI completely
MPI = None

# Single-GPU setup
GPUS_PER_NODE = 1  # changed from 8

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a (fake) distributed process group for single GPU.
    """
    if dist.is_initialized():
        return

    # Single-process fake distributed env
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    backend = "gloo"
    if th.cuda.is_available():
        backend = "nccl"

    dist.init_process_group(
        backend=backend,
        init_method="env://",
        rank=0,
        world_size=1,
    )


def dev():
    """
    Always return cuda:0 (single GPU) if available.
    """
    if th.cuda.is_available():
        return th.device("cuda:0")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Simply load a PyTorch checkpoint (no MPI sync).
    """
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    No-op for single GPU.
    """
    return


def _find_free_port():
    """
    Find a free port for single-node training.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
