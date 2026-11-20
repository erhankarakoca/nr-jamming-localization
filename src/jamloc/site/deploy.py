"""Publish a checked build to the repository's dedicated Pages branch."""

import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from jamloc.paths import REPO_ROOT
from jamloc.site.build import build_site, check_build

BRANCH = "gh-pages"
REF = f"refs/heads/{BRANCH}"


def _git(*args, cwd, check=True):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=check, text=True, encoding="utf-8", capture_output=True
    )


def _require_clean_source(source: Path):
    if _git(
        "--no-optional-locks",
        "status",
        "--porcelain",
        "--untracked-files=normal",
        "--ignore-submodules=none",
        cwd=source,
    ).stdout:
        raise ValueError(f"The source checkout is dirty: {source}")


def _remote_tip(source):
    remote = _git("ls-remote", "--exit-code", "origin", REF, cwd=source, check=False)
    if remote.returncode == 2:  # ls-remote exits 2 when the requested ref does not exist.
        return None
    remote.check_returncode()
    oid = remote.stdout.split()[0]
    _git("fetch", "--no-tags", "origin", oid, cwd=source)
    return oid


def deploy_site(*, results_from: Path | None = None) -> None:
    """Build from a clean source checkout and publish from a temporary worktree."""
    source = REPO_ROOT
    _require_clean_source(source)
    output = build_site(results_from=results_from)
    check_build(output)
    _require_clean_source(source)
    remote = _remote_tip(source)
    local = _git("show-ref", "--verify", "--quiet", REF, cwd=source, check=False)
    if local.returncode not in (0, 1):
        local.check_returncode()
    has_branch = local.returncode == 0
    with TemporaryDirectory(prefix="jamloc-pages-") as temp:
        worktree = Path(temp) / BRANCH
        args = ("worktree", "add", "--no-checkout")
        if has_branch:
            _git(*args, str(worktree), BRANCH, cwd=source)
        elif remote:
            _git(*args, "-b", BRANCH, str(worktree), remote, cwd=source)
        else:
            _git(*args, "--detach", str(worktree), "HEAD", cwd=source)
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
            _require_clean_source(source)
            _git("push", "origin", f"HEAD:{REF}", cwd=worktree)
        finally:
            _git("worktree", "remove", "--force", str(worktree), cwd=source)
