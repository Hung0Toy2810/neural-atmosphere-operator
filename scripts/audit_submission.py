"""Fail when generated artifacts, secrets, hard-coded host paths, or extra checkpoints leak into the package."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", "wandb", "tensorboard", "notebook"}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo", ".log", ".tmp", ".ipynb"}
TEXT_SUFFIXES = {".py", ".sh", ".md", ".txt", ".json", ".toml", ".yml", ".yaml"}
SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?i)(api[_-]?key|password|secret)\s*[:=]\s*['\"][^'\"]+"),
)
HOST_PATH = re.compile(r"/(?:Users|home|workspace|local-data)/")


def main() -> None:
    failures: list[str] = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if path.is_dir() and path.name in FORBIDDEN_DIRS:
            failures.append(f"forbidden directory: {relative}")
            continue
        if not path.is_file():
            continue
        if path.suffix in FORBIDDEN_SUFFIXES:
            failures.append(f"forbidden file: {relative}")
        if path.suffix in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
            if HOST_PATH.search(text):
                failures.append(f"host-specific path: {relative}")
            if any(pattern.search(text) for pattern in SECRET_PATTERNS):
                failures.append(f"possible secret: {relative}")
    checkpoints = sorted((ROOT / "checkpoints").glob("stage*/best.pt"))
    expected = [ROOT / "checkpoints" / f"stage{stage}" / "best.pt" for stage in (1, 2)]
    all_checkpoints = sorted((ROOT / "checkpoints").rglob("*.pt"))
    if checkpoints != expected or all_checkpoints != expected:
        failures.append(f"expected two stage-best checkpoints, found: {all_checkpoints}")
    if failures:
        raise SystemExit("\n".join(failures))
    print("Submission audit passed: clean tree, no detected secrets, two stage-best checkpoints.")


if __name__ == "__main__":
    main()
