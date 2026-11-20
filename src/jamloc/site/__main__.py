"""Export, build, preview, and publish the static showcase."""

import argparse
import subprocess
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from jamloc.paths import results_dir
from jamloc.site.cases import CASES


def serve(port: int) -> None:
    directory = results_dir() / "site" / "build"
    if not (directory / "index.html").is_file():
        raise FileNotFoundError(
            f"No showcase build at {directory}. Run python -m jamloc.site build."
        )
    handler = partial(SimpleHTTPRequestHandler, directory=directory)
    with ThreadingHTTPServer(("127.0.0.1", port), handler) as server:
        print(f"Serving {directory} at http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("export", help="write example slots and their manifest")
    command.add_argument("--cases", choices=CASES, nargs="+", help="export these cases in order")
    for name, help_text in (
        ("build", "assemble and check the static site"),
        ("deploy", "build and publish the site to gh-pages"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument(
            "--results-from", type=Path, help="directory containing paper result CSVs"
        )
    command = commands.add_parser("serve", help="serve an existing build on the loopback interface")
    command.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            from jamloc.site.export import export

            export(CASES[name] for name in dict.fromkeys(args.cases or CASES))
        elif args.command == "build":
            from jamloc.site.build import build_site

            print(build_site(results_from=args.results_from))
        elif args.command == "deploy":
            from jamloc.site.deploy import deploy_site

            deploy_site(results_from=args.results_from)
        else:
            serve(args.port)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    except subprocess.CalledProcessError as exc:
        parser.error(exc.stderr.strip() or str(exc))


if __name__ == "__main__":
    main()
