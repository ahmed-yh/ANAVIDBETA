"""
GPU/CUDA device detection shared by DwellTimeTracker and the standalone
diagnostic scripts (check_cuda_setup.py, test_cuda.py).
"""

import logging
import subprocess

logger = logging.getLogger(__name__)


def nvidia_smi_available() -> bool:
    """Check whether nvidia-smi runs successfully (NVIDIA drivers installed)."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def resolve_device():
    """
    Determine which device Ultralytics/PyTorch should run inference on.

    Returns:
        0 (first CUDA GPU) if available and usable, otherwise 'cpu'.
    """
    import torch

    if torch.cuda.is_available():
        logger.info(
            "CUDA available: using GPU 0 (%s, CUDA %s)",
            torch.cuda.get_device_name(0),
            torch.version.cuda,
        )
        return 0

    if nvidia_smi_available() and torch.version.cuda:
        logger.warning(
            "NVIDIA drivers detected but PyTorch cannot access CUDA "
            "(likely a CPU-only or mismatched torch build). Falling back to CPU."
        )
    else:
        logger.info("CUDA not available. Using CPU (inference will be slower).")

    return 'cpu'


def describe_device(device) -> str:
    """Human-readable label for a device value returned by resolve_device()."""
    if isinstance(device, int):
        import torch
        return f"GPU {device} ({torch.cuda.get_device_name(device)})"
    return "CPU"
