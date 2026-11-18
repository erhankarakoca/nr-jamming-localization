"""Export the editable paper diagrams as vector PDFs and three-times-scale PNG previews.

Edit the sources in ``diagrams/`` and run ``python -m jamloc.diagrams`` to refresh the copies
in ``results/figures/``. The desktop application's command line renders the same shapes and
labels as the editor. The PDF is cropped around the diagram for inclusion in the paper.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from jamloc.paths import REPO_ROOT, results_dir


def find_drawio() -> Path | None:
    """Find a desktop executable on PATH or in the platform's usual install locations."""
    executable = shutil.which("drawio") or shutil.which("draw.io.exe")
    if executable:
        return Path(executable)

    if sys.platform == "darwin":
        candidates = [Path("/Applications/draw.io.app/Contents/MacOS/draw.io")]
    elif sys.platform == "win32":
        candidates = [
            Path(os.environ[var]) / "draw.io" / "draw.io.exe"
            for var in ("ProgramFiles", "ProgramFiles(x86)")
            if var in os.environ
        ]
        if "LOCALAPPDATA" in os.environ:
            candidates.append(Path(os.environ["LOCALAPPDATA"]) / "Programs/draw.io/draw.io.exe")
    else:
        candidates = [
            Path(path)
            for path in (
                "/usr/bin/drawio",
                "/usr/local/bin/drawio",
                "/opt/drawio/drawio",
                "/snap/bin/drawio",
            )
        ]
    return next((path for path in candidates if path.is_file() and os.access(path, os.X_OK)), None)


def main() -> None:
    executable = find_drawio()
    if executable is None:
        raise SystemExit(
            "Install draw.io desktop from https://www.drawio.com "
            "(macOS: brew install --cask drawio), then run python -m jamloc.diagrams again."
        )

    out = results_dir() / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for source in sorted((REPO_ROOT / "diagrams").glob("*.drawio")):
        # Publish the pair only after both exports succeed, so a failed preview leaves the
        # previous paper figure and preview together.
        with TemporaryDirectory(dir=out) as staging:
            files = []
            for extension, options in (("pdf", ["--crop"]), ("png", ["--scale", "3"])):
                target = Path(staging) / f"{source.stem}.{extension}"
                command = [
                    str(executable),
                    "-x",
                    "-f",
                    extension,
                    *options,
                    "-o",
                    str(target),
                    str(source),
                ]
                try:
                    subprocess.run(command, check=True)
                except subprocess.CalledProcessError as error:
                    raise SystemExit(
                        f"draw.io failed to export {source.name} as {extension.upper()} "
                        f"(return code {error.returncode})."
                    ) from error
                except OSError as error:
                    raise SystemExit(f"Could not start draw.io at {executable}: {error}") from error
                if not target.is_file() or not target.stat().st_size:
                    raise SystemExit(f"draw.io did not write {target.name}.")
                files.append(target)
            for target in files:
                target.replace(out / target.name)
        print(f"Wrote {out / source.stem}.pdf/.png")


if __name__ == "__main__":
    main()
