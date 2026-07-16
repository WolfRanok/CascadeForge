"""Unified command-line interface for CascadeForge."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cascadeforge",
        description="多目标级联图像编辑数据流水线",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    preprocess = sub.add_parser("preprocess", help="解析标注并生成候选目标")
    preprocess.add_argument("--input", type=Path, required=True, help="原图与同名 JSON 所在目录")
    preprocess.add_argument("--output", type=Path, required=True, help="中间产物目录")
    preprocess.add_argument("--workers", type=int, default=4)

    select = sub.add_parser("select", help="选择四个分离目标并生成四轮局部编辑")
    select.add_argument("--input", type=Path, default=Path("IMAGE_MASK"))
    select.add_argument("--config", type=Path)
    select.add_argument("--concurrency", type=int, default=8)

    edit = sub.add_parser("edit", help="提交四宫格累计编辑任务")
    edit.add_argument("--input", type=Path, default=Path("IMAGE_MASK"))
    edit.add_argument("--config", type=Path)
    edit.add_argument("--concurrency", type=int, default=4)
    edit.add_argument("--watch", action="store_true", help="持续监听新增任务")

    organize = sub.add_parser("organize", help="拆分四宫格结果")
    organize.add_argument("--input", type=Path, required=True, help="编辑结果所在目录")
    organize.add_argument("--source", type=Path, help="标准化原图目录")
    organize.add_argument("--output", type=Path, required=True)
    organize.add_argument("--workers", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preprocess":
        from .preprocess import run_preprocess

        return run_preprocess(args.input, args.output, args.workers)
    if args.command == "select":
        from .select import run_selection

        return run_selection(args.input, args.config, args.concurrency)
    if args.command == "edit":
        from .editor import run_editor

        return run_editor(args.input, args.config, args.concurrency, args.watch)
    if args.command == "organize":
        from .organize import run_organize

        return run_organize(args.input, args.output, args.source, args.workers)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
