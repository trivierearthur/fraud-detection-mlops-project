# pyright: strict
"""Remove generated cache directories. Cross-platform (used by `just clean`)."""

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CACHE_DIR_NAMES = {"__pycache__", ".pytest_cache", ".ruff_cache"}


def main() -> None:
    removed = 0

    for path in PROJECT_ROOT.rglob("*"):
        if path.is_dir() and path.name in CACHE_DIR_NAMES:
            shutil.rmtree(path, ignore_errors=True)
            print(f"Removed {path.relative_to(PROJECT_ROOT)}")
            removed += 1

    if removed == 0:
        print("Nothing to clean.")


if __name__ == "__main__":
    main()
