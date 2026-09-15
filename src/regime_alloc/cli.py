"""Command registration; computational modules have no CLI dependency."""

import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="거시경제 레짐 자산배분 연구")
    parser.add_argument("--version", action="version", version="regime-alloc 0.1.0")
    commands = parser.add_subparsers(dest="command")
    data = commands.add_parser("data", help="원자료 취득 및 검증")
    actions = data.add_subparsers(dest="action", required=True)
    for action in ("acquire", "validate"):
        sub = actions.add_parser(action)
        sub.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    from .config import load_config
    from .contracts import ResearchError, canonical_json

    try:
        from . import data as providers

        config = load_config(args.config)
        result = getattr(providers, args.action)(config)
        print(canonical_json(result))
        return 0
    except ResearchError as exc:
        print(canonical_json(exc.to_dict()), file=sys.stderr)
        return exc.exit_code
    except Exception as exc:
        # A provider/library failure remains a failed command, with its cause.
        print(canonical_json({"error_code": "missing_data", "reason": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 3
