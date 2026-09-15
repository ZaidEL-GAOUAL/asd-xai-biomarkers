"""Locate the repository and launch its shared implementation."""
from pathlib import Path
import os
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]


def run_script(filename):
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / 'src'))
    runpy.run_path(str(ROOT / 'src' / filename), run_name='__main__')


def run_module(module):
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / 'src'))
    runpy.run_module(module, run_name='__main__')
