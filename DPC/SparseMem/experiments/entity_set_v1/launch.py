#!/usr/bin/env python3
"""Run an experiment with model connection settings from the environment."""
import os, sys
from pathlib import Path
root=Path(__file__).resolve().parents[2]
if len(sys.argv) < 2:
    raise SystemExit('Usage: python launch.py <command> [args ...]')
env=os.environ.copy(); env.update(
    PYTHONPATH=os.pathsep.join(filter(None, [str(root), str(root/'shared'), env.get('PYTHONPATH', '')])),
    USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
os.execvpe(sys.argv[1],sys.argv[1:],env)
