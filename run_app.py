from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
APP_PATH = PROJECT_ROOT / "app.py"
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements-app.txt"


def _missing_modules() -> list[str]:
    required = ["streamlit", "altair", "pandas", "numpy"]
    missing = []
    for module in required:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError:
            missing.append(module)
    return missing


def main() -> int:
    missing = _missing_modules()
    if missing:
        print("Missing required packages for the NativeHarvest app:")
        for module in missing:
            print(f" - {module}")
        print("")
        print("Install them with:")
        print(f"  python -m pip install -r \"{REQUIREMENTS_PATH}\"")
        return 1

    command = [sys.executable, "-m", "streamlit", "run", str(APP_PATH)]
    return subprocess.call(command, cwd=str(PROJECT_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
