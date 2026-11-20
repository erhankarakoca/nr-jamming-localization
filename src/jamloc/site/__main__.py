"""Command line interface for the showcase exporter."""

import argparse

from jamloc.site.cases import CASES
from jamloc.site.export import export


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("export", help="write example slots and their manifest")
    command.add_argument("--cases", choices=CASES, nargs="+", help="export these cases in order")
    args = parser.parse_args(argv)
    export(CASES[name] for name in dict.fromkeys(args.cases or CASES))


if __name__ == "__main__":
    main()
