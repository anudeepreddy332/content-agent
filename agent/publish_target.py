"""Fail-closed publish-target policy for demo vs production.

Default deny. Local merge (git_node) and remote push (/ui/runs/{id}/publish)
must call the same validator. Inspects each remote's fetch URLs and effective
push URLs (not path strings, not fetch-only). Does not contact the network.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

DEMO_GITHUB = ("anudeepreddy332", "themachinist-website-fork")
PRODUCTION_GITHUB = ("anudeepreddy332", "themachinist-website")
DEMO_SITE_CANON = ("https", "tmw-demo-site.netlify.app", "/")
PRODUCTION_CONFIRM_VALUE = "I_UNDERSTAND"

_SCP_GITHUB = re.compile(
    r"^(?:ssh://)?(?:git@)?github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/]+?)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PublishTargetDecision:
    allowed: bool
    target: str
    reason: str


def _deny(reason: str, target: str = "") -> PublishTargetDecision:
    return PublishTargetDecision(allowed=False, target=target, reason=reason)


def _allow(target: str, reason: str) -> PublishTargetDecision:
    return PublishTargetDecision(allowed=True, target=target, reason=reason)


def github_owner_repo(url: str) -> tuple[str, str] | None:
    """Return (owner, repo) for a GitHub remote URL, or None if unparseable."""
    if not url or not isinstance(url, str):
        return None
    raw = url.strip()
    if not raw:
        return None
    scp = _SCP_GITHUB.match(raw)
    if scp:
        repo = scp.group("repo").lower().removesuffix(".git").rstrip("/")
        return scp.group("owner").lower(), repo
    try:
        parts = urlsplit(raw)
    except Exception:
        return None
    host = (parts.hostname or "").lower()
    if host == "www.github.com":
        host = "github.com"
    if host != "github.com":
        return None
    segs = [s for s in (parts.path or "").split("/") if s]
    if len(segs) < 2:
        return None
    owner = segs[0].lower()
    repo = segs[1].lower().removesuffix(".git")
    if not owner or not repo:
        return None
    return owner, repo


def canonicalize_https_base(url: str) -> tuple[str, str, str] | None:
    """Canonical (scheme, host, path) for an HTTPS origin/base URL."""
    if not url or not isinstance(url, str):
        return None
    trimmed = url.strip()
    if not trimmed:
        return None
    try:
        parts = urlsplit(trimmed)
    except Exception:
        return None
    scheme = (parts.scheme or "").lower()
    if scheme != "https":
        return None
    try:
        if parts.username is not None or parts.password is not None:
            return None
        host = (parts.hostname or "").lower()
        port = parts.port
    except ValueError:
        return None
    if not host:
        return None
    if port not in (None, 443):
        return None
    if parts.query or parts.fragment:
        return None
    path = parts.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/") or "/"
    return scheme, host, path


def _git_push_enabled(value: str | None) -> bool:
    raw = (value if value is not None else os.environ.get("GIT_PUSH_ENABLED", "false"))
    return str(raw).strip().lower() == "true"


def _split_git_urls(raw: str) -> list[str]:
    return [line.strip() for line in str(raw).splitlines() if line.strip()]


def _remote_url_sets(repo_path: str) -> dict[str, tuple[list[str], list[str]]]:
    """Map remote name -> (fetch URLs, effective push URLs). No network.

    Uses `git remote get-url --all` and `git remote get-url --push --all` so a
    distinct pushurl is visible. GitPython `remote.urls` is fetch-only and must
    not be used here.
    """
    import git

    repo = git.Repo(repo_path)
    found: dict[str, tuple[list[str], list[str]]] = {}
    for remote in repo.remotes:
        fetch = _split_git_urls(repo.git.remote("get-url", "--all", remote.name))
        push = _split_git_urls(
            repo.git.remote("get-url", "--push", "--all", remote.name)
        )
        found[remote.name] = (fetch, push)
    return found


def evaluate_publish_target(
    *,
    repo_path: str | None = None,
    publish_remote: str | None = None,
    netlify_base_url: str | None = None,
    publish_target: str | None = None,
    confirm_production: str | None = None,
    git_push_enabled: str | None = None,
    require_push_enabled: bool = False,
) -> PublishTargetDecision:
    """Decide whether local merge / remote push may proceed. Default deny."""
    if require_push_enabled and not _git_push_enabled(git_push_enabled):
        return _deny("GIT_PUSH_ENABLED is not true; remote publish disabled")

    target = (
        publish_target
        if publish_target is not None
        else os.environ.get("PUBLISH_TARGET", "")
    ).strip().lower()
    if not target:
        return _deny("PUBLISH_TARGET is unset; publication disabled")

    if target == "production":
        confirm = (
            confirm_production
            if confirm_production is not None
            else os.environ.get("CONFIRM_PRODUCTION_PUBLISH", "")
        )
        if confirm != PRODUCTION_CONFIRM_VALUE:
            return _deny(
                "production publish requires CONFIRM_PRODUCTION_PUBLISH="
                f"{PRODUCTION_CONFIRM_VALUE}",
                target="production",
            )
        return _allow(
            "production",
            "production confirmation present; Git SHA/parent/non-force checks still apply",
        )

    if target != "demo":
        return _deny(
            f"unknown PUBLISH_TARGET {target!r}; publication disabled",
            target=target,
        )

    path = (
        repo_path
        if repo_path is not None
        else os.environ.get("THEMACHINIST_REPO_PATH", "")
    )
    if not path or not str(path).strip():
        return _deny("THEMACHINIST_REPO_PATH is unset; demo publication disabled", "demo")

    remote_name = (
        publish_remote
        if publish_remote is not None
        else os.environ.get("PUBLISH_REMOTE", "origin")
    ).strip() or "origin"

    site = (
        netlify_base_url
        if netlify_base_url is not None
        else os.environ.get("NETLIFY_BASE_URL", "")
    )
    site_canon = canonicalize_https_base(site)
    if site_canon != DEMO_SITE_CANON:
        return _deny(
            "NETLIFY_BASE_URL is not the approved demo site "
            "https://tmw-demo-site.netlify.app",
            "demo",
        )

    try:
        remotes = _remote_url_sets(str(path).strip())
    except Exception as exc:
        return _deny(f"THEMACHINIST_REPO_PATH is not a readable git repo: {exc}", "demo")

    if not remotes:
        return _deny("demo fork clone has no git remotes", "demo")

    for name, (fetch_urls, push_urls) in remotes.items():
        for url in [*fetch_urls, *push_urls]:
            if github_owner_repo(url) == PRODUCTION_GITHUB:
                return _deny(
                    f"remote {name!r} points at the production website repository",
                    "demo",
                )

    chosen = remotes.get(remote_name)
    if not chosen:
        return _deny(
            f"PUBLISH_REMOTE {remote_name!r} is not configured on the demo fork clone",
            "demo",
        )
    fetch_urls, push_urls = chosen
    if not push_urls:
        return _deny(
            f"PUBLISH_REMOTE {remote_name!r} has no effective push URL",
            "demo",
        )
    identities = [github_owner_repo(url) for url in [*fetch_urls, *push_urls]]
    if not identities or any(ident != DEMO_GITHUB for ident in identities):
        return _deny(
            f"PUBLISH_REMOTE {remote_name!r} does not resolve to "
            "anudeepreddy332/themachinist-website-fork",
            "demo",
        )
    return _allow("demo", "demo allowlist matched")
