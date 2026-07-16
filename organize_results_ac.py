"""Compatibility entry point for CascadeForge result organization."""

from pathlib import Path

from cascadeforge.organize import run_organize


if __name__ == "__main__":
    raise SystemExit(
        run_organize(
            Path("IMAGE_MASK") / "EDITED_4K",
            Path("OUTPUT") / "AC_multi_object",
            Path("IMAGE_MASK") / "IMAGE_2",
        )
    )
