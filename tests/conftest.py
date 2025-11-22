import sys
from pathlib import Path

# Ensure the src directory is on the path for test imports
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
