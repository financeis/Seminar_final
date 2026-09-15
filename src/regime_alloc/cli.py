"""Command registration; computational modules have no CLI dependency."""

import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description="거시경제 레짐 자산배분 연구")
    parser.add_argument("--version", action="version", version="regime-alloc 0.1.0")
    parser.parse_args(argv)
    parser.print_help()
    return 0
