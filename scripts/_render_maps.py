"""DELEGATED: scripts/_render_maps.py now forwards to run_all_maps.py
(Phase 5 single entry point). Kept for backward compatibility."""
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
rc = subprocess.call([sys.executable, str(root / "run_all_maps.py")],
                     cwd=str(root))
sys.exit(rc)
