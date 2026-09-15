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
    for name, help_text in [('run', '고정 원자료로 독립 백테스트 실행'), ('suite', '두 자료 방식의 대조·민감도 실험 묶음')]:
        run = commands.add_parser(name, help=help_text)
        run.add_argument('--config', required=True)
        run.add_argument('--output', required=True)
        run.add_argument('--smoke', action='store_true', help='최초·2020-04·마지막 평가월의 독립 통합 검사')
        run.add_argument('--no-cache', action='store_true', help='이전 실행 캐시를 읽지 않음 (항상 적용되는 기본 정책)')
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    from .config import load_config
    from .contracts import ResearchError, canonical_json

    try:
        config = load_config(args.config)
        if args.command == 'run':
            from .backtest.engine import run_research
            result = run_research(config, args.output, smoke=args.smoke, use_prior_cache=False)
        elif args.command == 'suite':
            from .backtest.experiments import run_suite
            result = run_suite(config, args.output, smoke=args.smoke)
        else:
            from . import data as providers
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
