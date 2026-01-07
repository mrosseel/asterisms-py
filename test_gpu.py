import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"ROCm available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"ROCm version: {torch.version.hip}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device capability: {torch.cuda.get_device_capability(0)}")

    # Test GPU memory
    try:
        x = torch.randn(100, 100, device='cuda')
        print(f"✓ GPU tensor creation successful")
        print(f"  GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    except Exception as e:
        print(f"✗ GPU tensor creation failed: {e}")
        sys.exit(1)
else:
    print("✗ ROCm not available")
    sys.exit(1)

print("\n✓ All GPU verification tests passed!")
