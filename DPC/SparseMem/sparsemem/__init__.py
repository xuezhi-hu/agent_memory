"""SparseMem and its bundled shared components."""

from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1] / "shared"
sys.path.insert(0, str(BASE))
