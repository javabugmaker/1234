from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .logging_utils import configure_logging
from .model import ModelRegistry
from .pipeline import NeuralAlphaPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neural-alpha", description="TickFlow.free() point-in-time neural alpha pipeline"
    )
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    update = sub.add_parser("update", help="TickFlow historical/full or daily increment")
    update.add_argument("--full", action="store_true")

    features = sub.add_parser("features", help="build PIT features and mature-label cache")
    features.add_argument("--years", nargs="*", type=int)

    train = sub.add_parser("train", help="train a champion/challenger MLP")
    train.add_argument(
        "--allow-degraded-survivorship",
        action="store_true",
        help="research-only override; result must not be labelled strict OOS",
    )

    walk = sub.add_parser("walk-forward", help="expanding purged walk-forward")
    walk.add_argument("--max-folds", type=int)
    walk.add_argument("--allow-degraded-survivorship", action="store_true")

    sub.add_parser("backtest", help="backtest historical OOS predictions")
    daily = sub.add_parser("daily", help="increment, infer, rank and publish daily pages")
    daily.add_argument("--skip-update", action="store_true")
    sub.add_parser("weekly", help="publish the professional weekly research report")
    sub.add_parser("gui", help="launch the non-blocking Tkinter desktop dashboard")
    sub.add_parser("models", help="print the model registry")
    promote = sub.add_parser("promote", help="promote a challenger to champion")
    promote.add_argument("model_version")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    configure_logging(config.paths.logs_dir, args.verbose)
    if args.command == "gui":
        from .gui import launch_gui

        launch_gui(args.config)
        return 0
    pipeline = NeuralAlphaPipeline(config)
    if args.command == "update":
        pipeline.update_tickflow(full=args.full)
    elif args.command == "features":
        pipeline.build_derived(years=args.years)
    elif args.command == "train":
        pipeline.train(allow_degraded_survivorship=args.allow_degraded_survivorship)
    elif args.command == "walk-forward":
        pipeline.walk_forward(
            max_folds=args.max_folds,
            allow_degraded_survivorship=args.allow_degraded_survivorship,
        )
    elif args.command == "backtest":
        pipeline.run_backtest()
    elif args.command == "daily":
        pipeline.daily(skip_update=args.skip_update)
    elif args.command == "weekly":
        pipeline.weekly()
    elif args.command == "models":
        print(json.dumps(ModelRegistry(config.paths.models_dir).read(), ensure_ascii=False, indent=2))
    elif args.command == "promote":
        ModelRegistry(config.paths.models_dir).promote(args.model_version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
