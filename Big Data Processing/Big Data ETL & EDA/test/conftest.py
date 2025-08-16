import sys, os, pathlib

# Ensure ./src is on the path during tests
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))