"""Compatibility entry point for CascadeForge VLM selection."""

from pathlib import Path

from cascadeforge.select import run_selection


if __name__ == "__main__":
    raise SystemExit(run_selection(Path("IMAGE_MASK"), None))
