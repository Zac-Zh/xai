"""
Fix CUDNN Warnings

This module provides utilities to handle CUDNN warnings that may occur during training.

The warning: "CUDNN_STATUS_NOT_SUPPORTED" typically occurs when:
1. CUDNN tries to use an unsupported execution plan
2. The GPU architecture doesn't support certain CUDNN features
3. There are incompatibilities between PyTorch and CUDNN versions

This is usually harmless - PyTorch will fall back to a slower but supported implementation.
"""

import torch


def configure_cudnn_for_stability():
    """
    Configure CUDNN settings to reduce warnings and improve stability.

    This function:
    - Enables CUDNN benchmarking for better performance
    - Allows CUDNN to fall back to alternative algorithms
    - Configures deterministic behavior when needed
    """
    if torch.cuda.is_available():
        # Allow CUDNN to benchmark and select the best algorithms
        # This may cause the warning but improves performance
        torch.backends.cudnn.benchmark = True

        # Allow CUDNN to use alternative algorithms if the optimal one fails
        torch.backends.cudnn.allow_tf32 = True

        # Only enable deterministic mode if reproducibility is critical
        # (This can significantly slow down training)
        # torch.backends.cudnn.deterministic = True

        print("CUDNN Configuration:")
        print(f"  - CUDNN version: {torch.backends.cudnn.version()}")
        print(f"  - CUDNN enabled: {torch.backends.cudnn.enabled}")
        print(f"  - CUDNN benchmark: {torch.backends.cudnn.benchmark}")
        print(f"  - TF32 enabled: {torch.backends.cudnn.allow_tf32}")


def check_gpu_compatibility():
    """Check GPU and CUDNN compatibility."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print("\nGPU Information:")
    print(f"  - PyTorch version: {torch.__version__}")
    print(f"  - CUDA version: {torch.version.cuda}")
    print(f"  - Number of GPUs: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  - Compute capability: {props.major}.{props.minor}")
        print(f"  - Total memory: {props.total_memory / 1e9:.2f} GB")
        print(f"  - Multi-processor count: {props.multi_processor_count}")


def suppress_cudnn_warnings():
    """
    Suppress CUDNN warnings in Python.

    Note: This only suppresses Python warnings, not C++ warnings.
    C++ warnings from CUDNN will still appear in the console.
    """
    import warnings

    # Filter out specific CUDNN-related warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='torch.autograd')


def test_cudnn_configuration():
    """Test CUDNN configuration with a simple convolution."""
    if not torch.cuda.is_available():
        print("CUDA not available - skipping test")
        return

    print("\nTesting CUDNN configuration...")

    try:
        # Create a simple convolution operation
        conv = torch.nn.Conv2d(3, 64, 3, padding=1).cuda()
        x = torch.randn(1, 3, 224, 224).cuda()

        # Forward pass
        y = conv(x)

        # Backward pass (where the warning typically occurs)
        loss = y.sum()
        loss.backward()

        print("✓ CUDNN test passed successfully")

    except Exception as e:
        print(f"✗ CUDNN test failed: {e}")


def apply_cudnn_fixes():
    """
    Apply all CUDNN fixes and configurations.

    Call this function at the start of your training script.
    """
    configure_cudnn_for_stability()
    check_gpu_compatibility()
    suppress_cudnn_warnings()
    test_cudnn_configuration()


if __name__ == "__main__":
    print("="*60)
    print("CUDNN Configuration and Diagnostics")
    print("="*60)
    apply_cudnn_fixes()
    print("\n" + "="*60)
    print("About the CUDNN_STATUS_NOT_SUPPORTED warning:")
    print("="*60)
    print("""
This warning is typically harmless. It means:
- CUDNN tried to use an optimized execution plan
- The plan is not supported on your GPU/CUDNN version
- PyTorch automatically falls back to a slower but working method

Solutions:
1. Update PyTorch and CUDA to the latest compatible versions
2. Update CUDNN to the latest version
3. Use torch.backends.cudnn.benchmark = True (done above)
4. If the warning is annoying but training works, you can ignore it

The warning does NOT mean:
- Your code is wrong
- Training will fail
- Results will be incorrect

It only means a minor performance optimization is unavailable.
    """)
