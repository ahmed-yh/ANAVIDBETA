"""
CUDA Setup Diagnostic Tool
Checks whether GPU acceleration is available and working, and explains why
if it isn't. This is the one script to run when tracking seems slow or
you're not sure whether the GPU is being used.
"""

import subprocess

from gpu_utils import nvidia_smi_available, resolve_device, describe_device


def print_header(title: str):
    print("\n" + "="*60)
    print(title)
    print("="*60)


def check_nvidia_drivers():
    print("\n1️⃣ Checking NVIDIA Drivers...")
    if nvidia_smi_available():
        print("   ✅ NVIDIA drivers are installed")
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            for line in result.stdout.split('\n'):
                if ('NVIDIA' in line and 'Driver Version' in line) or 'CUDA Version' in line:
                    print(f"   {line.strip()}")
        except Exception as e:
            print(f"   ⚠️  Could not read nvidia-smi details: {e}")
    else:
        print("   ❌ nvidia-smi not working - NVIDIA drivers may not be installed")
        print("   💡 Install NVIDIA drivers from: https://www.nvidia.com/drivers")


def check_pytorch():
    print("\n2️⃣ Checking PyTorch Installation...")
    try:
        import torch
    except ImportError:
        print("   ❌ PyTorch not installed")
        print("   💡 Install with: pip install torch")
        return

    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"   ✅ CUDA is working!")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   Number of GPUs: {torch.cuda.device_count()}")
        print(f"   cuDNN: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'Not available'}")
        return

    print(f"   ❌ CUDA not available in PyTorch")
    if torch.version.cuda:
        print(f"   ⚠️  PyTorch was built with CUDA {torch.version.cuda} but cannot access the GPU")
        print(f"   💡 Possible issues: CUDA runtime version mismatch, GPU drivers too old, GPU not accessible")
    else:
        print(f"   ❌ PyTorch was installed WITHOUT CUDA support (CPU-only)")
        print(f"   💡 SOLUTION: Reinstall PyTorch with CUDA, e.g.:")
        print(f"      pip uninstall torch torchvision torchaudio")
        print(f"      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")


def check_cuda_toolkit():
    print("\n3️⃣ Checking CUDA Toolkit...")
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("   ✅ CUDA Toolkit installed")
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    print(f"   {line.strip()}")
        else:
            print("   ⚠️  nvcc not found (CUDA Toolkit may not be in PATH)")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("   ⚠️  CUDA Toolkit not found in PATH (may still work if installed)")


def print_summary():
    print_header("📋 SUMMARY")
    device = resolve_device()
    print(f"\n{'✅' if isinstance(device, int) else '⚠️ '} Tracking will run on: {describe_device(device)}")
    if device == 'cpu':
        print("\n   To enable GPU acceleration:")
        print("   pip uninstall torch torchvision torchaudio")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("   See: https://pytorch.org/get-started/locally/")


if __name__ == "__main__":
    print_header("🔍 CUDA SETUP DIAGNOSTIC TOOL")
    check_nvidia_drivers()
    check_pytorch()
    check_cuda_toolkit()
    print_summary()
    print()
