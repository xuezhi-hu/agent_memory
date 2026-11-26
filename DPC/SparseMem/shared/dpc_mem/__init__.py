"""TFM implementation; shared utilities retain the dpc_mem package name."""
import os

# This implementation uses PyTorch; avoid loading optional TensorFlow backends.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
__version__ = "0.1.0"
