"""Keep the paper's Overleaf project in step with the experiments.

The paper lives in its own Overleaf project. Overleaf's Git integration exposes it as a Git
repository; clone it once per machine, next to this repository. The clone asks for a Git
authentication token (Overleaf: Account Settings -> Git integration), which the system's
credential manager stores (macOS Keychain, Git Credential Manager on Windows); never put it in a
file or in the URL::

    git clone https://git.overleaf.com/<project-id> nr-jamming-paper

Overleaf is the home of the text (``main.tex``, ``references.bib``); the figures are generated
here. This command copies every figure that ``main.tex`` includes from ``results_dir()`` into the
clone's ``figures/`` folder::

    python -m jamloc.paper --dest ../nr-jamming-paper          # or set JAMLOC_PAPER_DIR
    python -m jamloc.paper --dest ../nr-jamming-paper --init   # first time: main.tex and .bib too

It never commits or pushes. Pull in the clone before running it, check ``git status`` and
``git diff`` there, then commit and push.
"""

import argparse
import os
import re
import shutil
from pathlib import Path

from jamloc.paths import results_dir

REPO = Path(__file__).resolve().parents[2]
INCLUDE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}")
# The draft in this repository reads figures from results/ and the bibliography from docs/;
# the Overleaf project keeps both next to main.tex
_PATHS = (
    ("\\graphicspath{{../results/}}", "\\graphicspath{{figures/}}"),
    ("\\bibliography{../docs/references}", "\\bibliography{references}"),
)


def init_sources(dest: Path, force: bool = False) -> None:
    """Write ``main.tex`` (paths rewritten) and ``references.bib`` into ``dest``.

    Refuses to overwrite a ``main.tex`` that is already this paper, unless ``force``: after the
    first push, the Overleaf copy is the one people edit.
    """
    target = dest / "main.tex"
    if target.is_file() and "IEEEtran" in target.read_text(encoding="utf-8") and not force:
        raise SystemExit(
            f"{target} already holds the paper; edit it in Overleaf (or pass --force to overwrite)."
        )
    tex = (REPO / "paper" / "main.tex").read_text(encoding="utf-8")
    for old, new in _PATHS:
        if old not in tex:
            raise SystemExit(f"paper/main.tex has no {old!r}; update jamloc.paper")
        tex = tex.replace(old, new)
    target.write_text(tex, encoding="utf-8")
    shutil.copy2(REPO / "docs" / "references.bib", dest / "references.bib")


def figure_paths(tex: str) -> list[str]:
    """Graphics files included by a LaTeX source (``.pdf`` added when there is no extension)."""
    paths = []
    for name in INCLUDE.findall(tex):
        name = name.strip()
        paths.append(name if Path(name).suffix else f"{name}.pdf")
    return list(dict.fromkeys(paths))


def sync_figures(dest: Path) -> tuple[list[str], list[str]]:
    """Copy the figures ``dest/main.tex`` includes from ``results_dir()`` to ``dest/figures``.

    Returns the copied and the missing paths (relative to ``results/``).
    """
    tex = (dest / "main.tex").read_text(encoding="utf-8")
    copied, missing = [], []
    for rel in figure_paths(tex):
        src = results_dir() / Path(rel)
        if not src.is_file():
            missing.append(rel)
            continue
        target = dest / "figures" / Path(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        copied.append(rel)
    return copied, missing


def main(argv=None):
    p = argparse.ArgumentParser(description="Copy the paper's figures into its Overleaf clone.")
    p.add_argument(
        "--dest",
        type=Path,
        default=os.environ.get("JAMLOC_PAPER_DIR"),
        help="local clone of the Overleaf project (default: $JAMLOC_PAPER_DIR)",
    )
    p.add_argument("--init", action="store_true", help="also write main.tex and references.bib")
    p.add_argument("--force", action="store_true", help="with --init: overwrite the paper")
    args = p.parse_args(argv)
    if args.dest is None:
        raise SystemExit("Give --dest or set JAMLOC_PAPER_DIR to the Overleaf clone.")
    dest = Path(args.dest).expanduser().resolve()
    if not dest.is_dir():
        raise SystemExit(f"{dest} does not exist: clone the Overleaf project first.")
    if not (dest / ".git").exists():
        print(f"Warning: {dest} is not a Git clone; nothing here will reach Overleaf.")

    if args.init:
        init_sources(dest, args.force)
        print(f"Wrote {dest / 'main.tex'} and {dest / 'references.bib'}")
    copied, missing = sync_figures(dest)
    for rel in copied:
        print(f"Copied figures/{rel}")
    for rel in missing:
        print(f"Missing in results/: {rel} (run the experiment that makes it)")
    print("Now, in the clone: git status, git diff, then commit and push.")
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
