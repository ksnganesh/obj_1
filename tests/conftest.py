import sys
from pathlib import Path

# Make ``diqds`` importable when the tests run without installing the package (``pytest`` from the repository root).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
