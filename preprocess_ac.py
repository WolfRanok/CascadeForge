"""Compatibility entry point for CascadeForge preprocessing."""

from pathlib import Path

from cascadeforge.preprocess import run_preprocess


if __name__ == "__main__":
    raise SystemExit(run_preprocess(Path("100_CUT"), Path("IMAGE_MASK"), workers=100))
