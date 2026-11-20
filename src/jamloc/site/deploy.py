"""Publish a checked build to the repository's dedicated Pages branch."""

import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from jamloc.paths import REPO_ROOT
from jamloc.site.build import build_site, check_build

BRANCH = "gh-pages"
REF = f"refs/heads/{BRANCH}"


def _git(*args, cwd=REPO_ROOT, check=True):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=check, text=True, encoding="utf-8", capture_output=True
    )


def _require_clean_main():
    record = _git("worktree", "list", "--porcelain", "-z").stdout.split("\0", 1)[0]
    main = Path(record.removeprefix("worktree "))
    if _git(
        "--no-optional-locks",
        "status",
        "--porcelain",
        "--untracked-files=normal",
        "--ignore-submodules=none",
        cwd=main,
    ).stdout:
        raise ValueError(f"The main checkout is dirty: {main}")
    return main


def _remote_tip(main):
    remote = _git("ls-remote", "--exit-code", "origin", REF, cwd=main, check=False)
    if remote.returncode == 2:  # ls-remote exits 2 when the requested ref does not exist.
        return None
    remote.check_returncode()
    oid = remote.stdout.split()[0]
    _git("fetch", "--no-tags", "origin", oid, cwd=main)
    return oid


def deploy_site(*, results_from: Path | None = None) -> None:
    """Build and publish without touching the main checkout's working files or force-pushing."""
    main = _require_clean_main()
    output = build_site(results_from=results_from)
    check_build(output)
    _require_clean_main()
    remote = _remote_tip(main)
    local = _git("show-ref", "--verify", "--quiet", REF, cwd=main, check=False)
    if local.returncode not in (0, 1):
        local.check_returncode()
    has_branch = local.returncode == 0
    with TemporaryDirectory(prefix="jamloc-pages-") as temp:
        worktree = Path(temp) / BRANCH
        args = ("worktree", "add", "--no-checkout")
        if has_branch:
            _git(*args, str(worktree), BRANCH, cwd=main)
        elif remote:
            _git(*args, "-b", BRANCH, str(worktree), remote, cwd=main)
        else:
            _git(*args, "--detach", str(worktree), "HEAD", cwd=main)
        try:
            if not has_branch and not remote:
                _git("switch", "--orphan", BRANCH, cwd=worktree)
            elif has_branch and remote:
                behind = _git(
                    "merge-base", "--is-ancestor", "HEAD", remote, cwd=worktree, check=False
                )
                if behind.returncode == 0:
                    _git("reset", "--soft", remote, cwd=worktree)
                else:
                    ahead = _git(
                        "merge-base", "--is-ancestor", remote, "HEAD", cwd=worktree, check=False
                    )
                    if ahead.returncode != 0:
                        raise ValueError("Local and origin gh-pages have diverged.")
            _git("read-tree", "--empty", cwd=worktree)
            shutil.copytree(output, worktree, dirs_exist_ok=True)
            _git("add", "-A", "--force", "--", ".", cwd=worktree)
            changed = _git("diff", "--cached", "--quiet", cwd=worktree, check=False)
            if changed.returncode == 1:
                _git("commit", "-m", "Update the site", cwd=worktree)
            else:
                changed.check_returncode()
            _require_clean_main()
            _git("push", "origin", f"HEAD:{REF}", cwd=worktree)
        finally:
            _git("worktree", "remove", "--force", str(worktree), cwd=main)
