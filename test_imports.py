#!/usr/bin/env python3
"""
Simple test script to verify the DDPG environment is working.
"""
import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    from tensordict import TensorDictBase
    print("✓ tensordict imported successfully")
except ImportError as e:
    print(f"✗ tensordict import failed: {e}")
    sys.exit(1)

try:
    from torchrl.collectors import SyncDataCollector
    print("✓ torchrl collectors imported successfully")
except ImportError as e:
    print(f"✗ torchrl collectors import failed: {e}")
    sys.exit(1)

try:
    import torch
    print(f"✓ torch imported successfully (version: {torch.__version__})")
except ImportError as e:
    print(f"✗ torch import failed: {e}")
    sys.exit(1)

print("\n🎉 All imports successful! Environment is ready.")
