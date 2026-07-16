"""Compatibility entry point for CascadeForge image editing."""

from pathlib import Path

from cascadeforge.editor import run_editor


if __name__ == "__main__":
    raise SystemExit(run_editor(Path("IMAGE_MASK"), None, watch=True))
