"""Run offline verification; full mode rejects silently skipped runtime tests."""
import argparse
from pathlib import Path
import sys
import unittest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--require-runtime',action='store_true');args=parser.parse_args()
    if args.require_runtime:
        from autoresearch_budget.docker_runtime import check_runtime
        check_runtime()
    suite=unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if args.require_runtime and result.skipped:
        print(f'Full runtime verification unexpectedly skipped {len(result.skipped)} tests.',file=sys.stderr)
        return 1
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())
