"""
Trusted rendering and publication security boundary.

Untrusted model/retrieval/reviewer-influenced HTML exists only as a temporary
local value, is sanitized immediately, then discarded. Only TrustedFragment /
TrustedArticle values may enter state, interrupts, preview, archive, or Git.
"""

from __future__ import annotations

import base64
import hashlib
import html as html_module
import ipaddress
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit, urlunsplit

import nh3
from markdown_it import MarkdownIt

HTML_POLICY_VERSION = "p0-1-v1"

ALLOWED_TAGS = {
    "h2", "h3", "h4", "p", "ul", "ol", "li", "pre", "code", "strong", "em",
    "blockquote", "table", "thead", "tbody", "tr", "th", "td", "hr", "br",
    "div", "span", "sup", "sub",
}

CLASS_TAGS = {"div", "span", "pre", "code"}
ALLOWED_CLASSES = {
    "callout",
    "callout-info",
    "callout-key",
    "callout-label",
    "sl-definition",
    "sl-code-block",
    "sl-code-header",
    "sl-code-label",
    "sl-code-lang",
    "language-python",
    "language-text",
    "language-json",
    "language-bash",
    "language-sql",
}

NH3_ATTRIBUTES = {tag: set() for tag in ALLOWED_TAGS}
NH3_ATTRIBUTES.update({
    "div": {"class"},
    "span": {"class"},
    "pre": {"class"},
    "code": {"class"},
    "th": {"scope"},
})
NH3_TAG_ATTRIBUTE_VALUES = {"th": {"scope": {"row", "col"}}}
NH3_CLEAN_CONTENT_TAGS = {"script", "style"}

_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")


class PolicyError(Exception):
    """Fail-closed security/structural failure."""


@dataclass(frozen=True)
class TrustedFragment:
    html: str
    policy_version: str = HTML_POLICY_VERSION


@dataclass(frozen=True)
class TrustedArticle:
    html: str
    body_html: str
    sha256: str
    policy_version: str = HTML_POLICY_VERSION


def sha256_utf8(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _csp_style_hash(css: str) -> str:
    digest = hashlib.sha256(css.encode("utf-8")).digest()
    return base64.b64encode(digest).decode("ascii")


def _attribute_filter(tag: str, attr: str, value: str) -> str | None:
    if tag in CLASS_TAGS and attr == "class":
        kept = [c for c in value.split() if c in ALLOWED_CLASSES]
        return " ".join(kept) if kept else None
    if tag == "th" and attr == "scope" and value in {"row", "col"}:
        return value
    return None


def sanitize_fragment(untrusted_html: str) -> TrustedFragment:
    """Parser-based sanitizer. Raw input must not be retained by the caller."""
    if untrusted_html is None:
        raise PolicyError("missing HTML fragment")
    try:
        cleaned = nh3.clean(
            untrusted_html,
            tags=set(ALLOWED_TAGS),
            attributes=NH3_ATTRIBUTES,
            attribute_filter=_attribute_filter,
            tag_attribute_values=NH3_TAG_ATTRIBUTE_VALUES,
            clean_content_tags=set(NH3_CLEAN_CONTENT_TAGS),
            strip_comments=True,
            link_rel=None,
            url_schemes=set(),
            generic_attribute_prefixes=set(),
        )
        again = nh3.clean(
            cleaned,
            tags=set(ALLOWED_TAGS),
            attributes=NH3_ATTRIBUTES,
            attribute_filter=_attribute_filter,
            tag_attribute_values=NH3_TAG_ATTRIBUTE_VALUES,
            clean_content_tags=set(NH3_CLEAN_CONTENT_TAGS),
            strip_comments=True,
            link_rel=None,
            url_schemes=set(),
            generic_attribute_prefixes=set(),
        )
    except PolicyError:
        raise
    except Exception as exc:
        raise PolicyError(f"sanitizer exception: {exc}") from exc
    if again != cleaned:
        raise PolicyError("sanitizer is not idempotent")
    return TrustedFragment(html=cleaned, policy_version=HTML_POLICY_VERSION)


def _markdown_engine() -> MarkdownIt:
    md = MarkdownIt(
        "js-default",
        {
            "html": False,
            "linkify": False,
            "typographer": False,
        },
    )
    md.disable(["image", "strikethrough"])
    return md


_MD = _markdown_engine()


def render_markdown_fragment(markdown: str) -> TrustedFragment:
    raw = _MD.render(markdown or "")
    fragment = sanitize_fragment(raw)
    del raw
    return fragment


REVIEW_CSS = (
    "body{margin:0;padding:20px 22px;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;"
    "line-height:1.7;font-size:14.5px;color:#111827;background:#fff}"
    "h1,h2,h3,h4{font-size:16px;margin:22px 0 10px;line-height:1.4}"
    "h1:first-child,h2:first-child,h3:first-child{margin-top:0}"
    "p{margin:0 0 14px}ul,ol{margin:0 0 14px;padding-left:22px}li{margin-bottom:6px}"
    "pre{background:#0a0e14;color:#e6edf3;padding:12px 14px;border-radius:6px;overflow:auto;margin:0 0 14px}"
    "code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px}"
    "blockquote{margin:0 0 14px;padding-left:12px;border-left:3px solid #d1d5db;color:#4b5563}"
    "table{border-collapse:collapse;width:100%}th,td{border:1px solid #e5e7eb;padding:6px 8px;text-align:left}"
)

REVIEW_CSS_HASH = _csp_style_hash(REVIEW_CSS)

ARTICLE_CSS = """\
html{scroll-behavior:smooth}
:root{
  --orange:#f97316;--orange-subtle:rgba(249,115,22,0.08);--orange-border:rgba(249,115,22,0.25);
  --muted:#6b7280;--border:#e5e7eb;--card:#f9fafb;--text-color:#111827;--text-secondary:#4b5563;
  --section-bg:#f3f4f6;--spacing-xl:64px;--spacing-md:24px;--bg:#ffffff;
}
@media (prefers-color-scheme:dark){
  :root{
    --orange-subtle:rgba(249,115,22,0.1);--orange-border:rgba(249,115,22,0.3);
    --muted:#9ca3af;--border:#27272a;--card:#18181b;--text-color:#f4f4f5;--text-secondary:#d4d4d8;
    --section-bg:#18181b;--bg:#09090b;
  }
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text-color);font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;line-height:1.5}
.skip-link{position:absolute;left:-999px;top:0}
.skip-link:focus{left:12px;top:12px;background:#111;color:#fff;padding:8px}
nav#nav{border-bottom:1px solid var(--border);background:var(--bg)}
nav#nav .container{max-width:1100px;margin:0 auto;padding:12px 24px}
nav#nav ul{list-style:none;margin:0;padding:0;display:flex;gap:18px;flex-wrap:wrap}
nav#nav a{color:var(--text-color);text-decoration:none;font-size:0.92rem}
nav#nav a:hover{text-decoration:underline}
.sl-header{text-align:left;padding:calc(var(--spacing-xl) + 24px) var(--spacing-md) var(--spacing-lg,32px);background:var(--section-bg);border-bottom:1px solid var(--border)}
.sl-header .container{max-width:720px;margin:0 auto}
.sl-breadcrumb{display:flex;align-items:center;gap:6px;flex-wrap:wrap;font-size:0.78rem;color:var(--muted);margin-bottom:20px}
.sl-breadcrumb a{color:var(--muted);text-decoration:none}
.sl-breadcrumb-sep{opacity:0.4}
.sl-series-label{font-size:0.72rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:var(--muted);margin-bottom:12px}
.sl-header h1{font-size:clamp(1.6rem,4vw,2.4rem);font-weight:800;letter-spacing:-0.025em;line-height:1.2;margin:0 0 14px}
.sl-subtitle{font-size:0.98rem;color:var(--text-secondary);line-height:1.75;max-width:560px;margin:0 0 24px}
.sl-meta{display:flex;gap:18px;flex-wrap:wrap;align-items:center;padding-top:18px;border-top:1px solid var(--border);font-size:0.78rem;color:var(--muted)}
.meta-badge{font-size:0.68rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;padding:3px 8px;border-radius:3px;background:var(--orange-subtle);border:1px solid var(--orange-border);color:var(--orange)}
.sl-body{max-width:720px;margin:0 auto;padding:52px 24px 80px}
.sl-body p{font-size:0.96rem;line-height:1.82;color:var(--text-color);margin-bottom:18px}
.sl-body h2{font-size:1.2rem;font-weight:700;margin:44px 0 14px;letter-spacing:-0.01em}
.sl-body h3{font-size:1rem;font-weight:700;margin:28px 0 10px}
.sl-body strong{color:var(--text-color)}
.sl-body ul,.sl-body ol{padding-left:20px;margin-bottom:18px}
.sl-body li{font-size:0.96rem;line-height:1.78;margin-bottom:5px}
.sl-definition{border-left:3px solid var(--text-color);background:var(--section-bg);border-radius:0 6px 6px 0;padding:18px 22px;margin:22px 0 32px}
.sl-definition p{margin:0;font-size:0.95rem;line-height:1.78}
.callout{border-radius:6px;padding:16px 20px;margin:22px 0;border:1px solid}
.callout-label{font-size:0.68rem;font-weight:800;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:6px}
.callout p{margin:0;font-size:0.875rem;line-height:1.72}
.callout-key{background:var(--orange-subtle);border-color:var(--orange-border);border-left:3px solid var(--orange)}
.callout-key .callout-label{color:var(--orange)}
.callout-info{background:rgba(139,92,246,0.06);border-color:rgba(139,92,246,0.2);border-left:3px solid #8b5cf6}
.callout-info .callout-label{color:#8b5cf6}
.sl-code-block{background:#1a1a1a;border-radius:7px;margin:22px 0 28px;overflow:hidden}
.sl-code-label{background:#111;padding:7px 16px;font-size:0.68rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#6b7280;border-bottom:1px solid #2a2a2a}
.sl-code-block pre{margin:0;padding:18px 20px;overflow-x:auto;white-space:pre}
.sl-code-block code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:0.82rem;line-height:1.7;color:#d4d4d4}
.sl-body code:not(.sl-code-block code){background:var(--section-bg);border:1px solid var(--border);border-radius:3px;padding:1px 5px;font-size:0.85em;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.sl-hr{border:none;border-top:1px solid var(--border);margin:38px 0}
.sl-sources{border-top:1px solid var(--border);margin-top:52px;padding-top:28px}
.sl-sources-label{font-size:0.72rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:var(--muted);margin-bottom:14px}
.sl-sources ol{padding-left:20px;margin:0}
.sl-sources li{font-size:0.82rem;color:var(--muted);line-height:1.65;margin-bottom:7px}
.sl-sources a{color:var(--muted);text-decoration:underline;text-underline-offset:2px}
.back-wrap{margin-top:52px}
.back-link{color:var(--text-color)}
footer{border-top:1px solid var(--border);padding:28px 24px}
footer .container{max-width:1100px;margin:0 auto}
.footer-links{display:flex;gap:14px;align-items:center}
.footer-links a{color:var(--text-color)}
.footer-links svg{width:20px;height:20px;fill:none;stroke:currentColor;stroke-width:2}
.copyright{color:var(--muted);font-size:0.82rem}
@media (max-width:640px){.sl-body{padding:36px 16px 60px}.sl-header{padding:calc(var(--spacing-xl) + 24px) 16px 36px}}
"""

ARTICLE_CSS_HASH = _csp_style_hash(ARTICLE_CSS)

ARTICLE_CSP = (
    "default-src 'none'; "
    f"style-src 'sha256-{ARTICLE_CSS_HASH}'; "
    "script-src 'none'; "
    "img-src 'none'; "
    "font-src 'none'; "
    "connect-src 'none'; "
    "media-src 'none'; "
    "object-src 'none'; "
    "frame-src 'none'; "
    "worker-src 'none'; "
    "base-uri 'none'; "
    "form-action 'none'"
)

REVIEW_IFRAME_CSP = (
    "default-src 'none'; "
    f"style-src 'sha256-{REVIEW_CSS_HASH}'; "
    "script-src 'none'; "
    "img-src 'none'; "
    "font-src 'none'; "
    "connect-src 'none'; "
    "media-src 'none'; "
    "object-src 'none'; "
    "frame-src 'none'; "
    "worker-src 'none'; "
    "base-uri 'none'; "
    "form-action 'none'"
)

REVIEWER_APP_CSP = (
    "default-src 'none'; "
    "script-src 'self'; "
    "style-src 'self'; "
    "connect-src 'self'; "
    "frame-src 'self'; "
    "img-src 'self' data:; "
    "font-src 'self'; "
    "object-src 'none'; "
    "base-uri 'none'; "
    "form-action 'self'; "
    "frame-ancestors 'none'; "
    "worker-src 'none'; "
    "media-src 'none'"
)


def serialize_json_ld(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    return (
        raw.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def wrap_review_document(fragment: TrustedFragment) -> str:
    if fragment.policy_version != HTML_POLICY_VERSION:
        raise PolicyError("unexpected policy version")
    csp = REVIEW_IFRAME_CSP
    return (
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\"/>"
        f"<meta http-equiv=\"Content-Security-Policy\" content=\"{csp}\"/>"
        f"<style>{REVIEW_CSS}</style></head><body>{fragment.html}</body></html>"
    )


def render_markdown_review_html(markdown: str) -> str:
    return wrap_review_document(render_markdown_fragment(markdown))


def _ip_is_public(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if ip.is_global and not ip.is_multicast:
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
            return _ip_is_public(ip.ipv4_mapped)
        return True
    return False


def _parse_ipv4_number(part: str) -> int | None:
    """WHATWG IPv4-number parser. None means the part is not a valid IPv4 number."""
    if not part:
        return None
    if len(part) >= 2 and part[:2].lower() == "0x":
        radix, body, digits = 16, part[2:], "0123456789abcdefABCDEF"
    elif len(part) >= 2 and part[0] == "0":
        radix, body, digits = 8, part[1:], "01234567"
    else:
        radix, body, digits = 10, part, "0123456789"
    if not body:
        return 0
    if any(c not in digits for c in body):
        return None
    return int(body, radix)


def _ipv4_parts(host: str) -> list[str]:
    parts = host.split(".")
    if parts and parts[-1] == "" and len(parts) > 1:
        parts.pop()
    return parts


def _ends_in_a_number(host: str) -> bool:
    """WHATWG ends-in-a-number checker. True forces IPv4 parse (failure rejects the host)."""
    parts = _ipv4_parts(host)
    if not parts:
        return False
    last = parts[-1]
    if last and all("0" <= c <= "9" for c in last):
        return True
    return last != "" and _parse_ipv4_number(last) is not None


def _parse_whatwg_ipv4(host: str) -> ipaddress.IPv4Address | None:
    """Parse browser-recognized dotted/octal/hex/decimal IPv4 forms. None if invalid."""
    parts = _ipv4_parts(host)
    if not parts or len(parts) > 4:
        return None
    nums: list[int] = []
    for part in parts:
        value = _parse_ipv4_number(part)
        if value is None:
            return None
        nums.append(value)
    last_max = {1: 0xFFFFFFFF, 2: 0xFFFFFF, 3: 0xFFFF, 4: 0xFF}[len(nums)]
    if nums[-1] > last_max:
        return None
    if any(v > 255 for v in nums[:-1]):
        return None
    n = len(nums)
    if n == 1:
        ip_int = nums[0]
    elif n == 2:
        ip_int = (nums[0] << 24) | nums[1]
    elif n == 3:
        ip_int = (nums[0] << 24) | (nums[1] << 16) | nums[2]
    else:
        ip_int = (nums[0] << 24) | (nums[1] << 16) | (nums[2] << 8) | nums[3]
    return ipaddress.IPv4Address(ip_int)


def _dns_hostname_allowed(host: str) -> bool:
    if not host or len(host) > 253:
        return False
    labels = host.split(".")
    if any(len(label) == 0 or len(label) > 63 for label in labels):
        return False
    for label in labels:
        if label.startswith("-") or label.endswith("-"):
            return False
        if any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in label):
            return False
    return True


def _canonical_host(host: str) -> str | None:
    """Validate host and return the form written into a citation URL, or None."""
    if not host or host.startswith("[") or "\\" in host:
        return None
    try:
        host_idna = host.encode("idna").decode("ascii").lower().rstrip(".")
    except Exception:
        return None
    if not host_idna:
        return None
    if host_idna == "localhost" or host_idna.endswith(".localhost") or host_idna.endswith(".local"):
        return None
    try:
        ip = ipaddress.ip_address(host_idna)
    except ValueError:
        ip = None
    if isinstance(ip, ipaddress.IPv6Address):
        if not _ip_is_public(ip):
            return None
        return f"[{ip.compressed}]"
    if _ends_in_a_number(host_idna):
        ipv4 = _parse_whatwg_ipv4(host_idna)
        if ipv4 is None or not _ip_is_public(ipv4):
            return None
        return str(ipv4)
    if isinstance(ip, ipaddress.IPv4Address):
        if not _ip_is_public(ip):
            return None
        return str(ip)
    if not _dns_hostname_allowed(host_idna):
        return None
    return host_idna


def _percent_encoding_valid(text: str) -> bool:
    i = 0
    while i < len(text):
        if text[i] == "%":
            hexpart = text[i + 1:i + 3]
            if len(hexpart) < 2 or any(c not in "0123456789abcdefABCDEF" for c in hexpart):
                return False
            i += 3
        else:
            i += 1
    return True


def normalize_citation_url(candidate: str) -> str | None:
    if candidate is None:
        return None
    if not isinstance(candidate, str):
        return None
    if len(candidate) > 2048:
        return None
    if "\\" in candidate or _CTRL_RE.search(candidate):
        return None
    trimmed = candidate.strip(" \t\n\r\f\v")
    if not trimmed or len(trimmed) > 2048:
        return None
    if not _percent_encoding_valid(trimmed):
        return None
    try:
        parts = urlsplit(trimmed)
    except Exception:
        return None
    try:
        username = parts.username
        password = parts.password
        hostname = parts.hostname
        port = parts.port
    except ValueError:
        return None
    scheme = (parts.scheme or "").lower()
    if scheme != "https":
        return None
    if username is not None or password is not None:
        return None
    host = hostname
    if not host:
        return None
    host_canon = _canonical_host(host)
    if not host_canon:
        return None
    if port not in (None, 443):
        return None
    path = parts.path if parts.path else "/"
    query = parts.query
    normalized = urlunsplit(("https", host_canon, path, query, ""))
    if len(normalized) > 2048:
        return None
    return normalized


def canonical_source_urls(web_sources: Iterable[Mapping[str, Any]] | None) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for source in web_sources or []:
        url = normalize_citation_url(str(source.get("url") or ""))
        if not url:
            continue
        out[url] = source
    return out


def resolve_verifier_url(candidate: str, canonical: Mapping[str, Mapping[str, Any]]) -> str | None:
    normalized = normalize_citation_url(candidate)
    if not normalized:
        return None
    if normalized in canonical:
        return normalized
    return None


def _label_for_url(url: str, source: Mapping[str, Any] | None) -> str:
    title = (source or {}).get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    host = urlsplit(url).hostname or url
    return host


def build_citation_items(
    grounding_report: Iterable[Mapping[str, Any]] | None,
    web_sources: Iterable[Mapping[str, Any]] | None,
    kb_results: Iterable[Mapping[str, Any]] | None = None,
) -> list[dict[str, str | None]]:
    canonical = canonical_source_urls(web_sources)
    items: list[dict[str, str | None]] = []
    seen: set[str] = set()

    for entry in grounding_report or []:
        status = str(entry.get("status") or "")
        if status not in {"verified", "weak"}:
            continue
        kind = str(entry.get("source_kind") or "")
        ref = entry.get("source_ref")
        claim = str(entry.get("claim") or "")
        if kind.lower() == "kb":
            label = str(ref or "KB")
            if label.startswith("kb:"):
                label = label[3:]
            key = f"kb:{label}"
            if key in seen:
                continue
            seen.add(key)
            items.append({"kind": "kb", "label": label, "url": None, "claim": claim})
            continue
        resolved = resolve_verifier_url(str(ref or ""), canonical)
        if resolved:
            if resolved in seen:
                continue
            seen.add(resolved)
            items.append({
                "kind": "web",
                "label": _label_for_url(resolved, canonical.get(resolved)),
                "url": resolved,
                "claim": claim,
            })
        else:
            key = f"unresolved:{claim[:80]}"
            if key in seen:
                continue
            seen.add(key)
            items.append({"kind": "unresolved", "label": "Unresolved source", "url": None, "claim": claim})

    if not items:
        ranked = sorted(
            canonical.items(),
            key=lambda kv: float((kv[1] or {}).get("score") or 0),
            reverse=True,
        )
        for url, source in ranked[:3]:
            if url in seen:
                continue
            seen.add(url)
            items.append({
                "kind": "web",
                "label": _label_for_url(url, source),
                "url": url,
                "claim": "",
            })

    if not items:
        for kb in list(kb_results or [])[:3]:
            label = str(kb.get("source") or "KB")
            key = f"kb:{label}"
            if key in seen:
                continue
            seen.add(key)
            items.append({"kind": "kb", "label": label, "url": None, "claim": ""})

    return items


def render_citations_html(items: list[dict[str, str | None]]) -> str:
    if not items:
        return "<li>Sources retrieved via Tavily web search.</li>"
    lis = []
    for item in items:
        label = html_module.escape(str(item.get("label") or "source"))
        url = item.get("url")
        if item.get("kind") == "web" and isinstance(url, str) and normalize_citation_url(url) == url:
            href = html_module.escape(url, quote=True)
            lis.append(
                f'<li><a href="{href}" target="_blank" rel="noopener noreferrer nofollow">{label}</a></li>'
            )
        else:
            lis.append(f"<li>{label}</li>")
    return "\n".join(lis)


def safe_grounding_rows(
    grounding_report: Iterable[Mapping[str, Any]] | None,
    web_sources: Iterable[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    canonical = canonical_source_urls(web_sources)
    rows: list[dict[str, Any]] = []
    for entry in grounding_report or []:
        kind = str(entry.get("source_kind") or "")
        ref = entry.get("source_ref")
        resolved = None
        clickable = False
        if kind.lower() == "kb":
            label = str(ref or "KB")
            if label.startswith("kb:"):
                label = label[3:]
        else:
            resolved = resolve_verifier_url(str(ref or ""), canonical)
            if resolved:
                label = _label_for_url(resolved, canonical.get(resolved))
                clickable = True
            elif isinstance(ref, str) and ref.strip() and not str(ref).lower().startswith("http"):
                label = ref.strip()
            else:
                label = "Unresolved source"
        conf = entry.get("confidence")
        try:
            confidence = float(conf)
        except (TypeError, ValueError):
            confidence = 0.0
        rows.append({
            "claim": str(entry.get("claim") or ""),
            "status": str(entry.get("status") or ""),
            "confidence": confidence,
            "source_kind": kind,
            "source_label": label,
            "source_url": resolved if clickable else None,
        })
    return rows


def _clock_icon() -> str:
    return (
        '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="2" aria-hidden="true">'
        '<circle cx="12" cy="12" r="10"/>'
        '<polyline points="12 6 12 12 16 14"/>'
        "</svg>"
    )


def _footer_icons() -> str:
    return """\
<div class="footer-links">
<a href="mailto:anudeepreddy332@gmail.com" aria-label="Email">
<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/><polyline points="22,6 12,13 2,6"/></svg>
</a>
<a href="https://linkedin.com/in/anudeep-reddy-mutyala" target="_blank" rel="noopener noreferrer nofollow" aria-label="LinkedIn">
<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"/><rect x="2" y="9" width="4" height="12"/><circle cx="4" cy="4" r="2"/></svg>
</a>
<a href="https://github.com/anudeepreddy332" target="_blank" rel="noopener noreferrer nofollow" aria-label="GitHub">
<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2C6.477 2 2 6.477 2 12c0 4.42 2.865 8.17 6.839 9.49.5.092.682-.217.682-.482 0-.237-.008-.866-.013-1.7-2.782.603-3.369-1.34-3.369-1.34-.454-1.156-1.11-1.464-1.11-1.464-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.578 9.578 0 0112 6.836a9.59 9.59 0 012.504.337c1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.578.688.48C19.138 20.167 22 16.418 22 12c0-5.523-4.477-10-10-10z"/></svg>
</a>
</div>"""


def assemble_trusted_article(
    *,
    topic: str,
    slug: str,
    meta_description: str,
    series_label: str,
    breadcrumb_section: str,
    read_time: str,
    problem_framing: TrustedFragment,
    technical_dive: TrustedFragment,
    code_snippets: TrustedFragment,
    takeaways: TrustedFragment,
    citations_html: str,
) -> TrustedArticle:
    if not _SLUG_RE.fullmatch(slug or ""):
        raise PolicyError("invalid slug")
    for name, frag in (
        ("problem_framing", problem_framing),
        ("technical_dive", technical_dive),
        ("code_snippets", code_snippets),
        ("takeaways", takeaways),
    ):
        if not isinstance(frag, TrustedFragment):
            raise PolicyError(f"{name} is not a TrustedFragment")
        if frag.policy_version != HTML_POLICY_VERSION:
            raise PolicyError(f"{name} policy version mismatch")
        if not frag.html.strip():
            raise PolicyError(f"{name} sanitized to empty")

    topic_e = html_module.escape(topic)
    topic_short_e = html_module.escape(topic.lower())
    slug_e = html_module.escape(slug)
    meta_e = html_module.escape(meta_description)
    series_e = html_module.escape(series_label)
    crumb_e = html_module.escape(breadcrumb_section)
    read_e = html_module.escape(str(read_time))
    canonical = f"https://themachinist.org/{slug}.html"
    canonical_e = html_module.escape(canonical, quote=True)
    csp_e = ARTICLE_CSP

    json_ld = serialize_json_ld({
        "@context": "https://schema.org",
        "@type": "Article",
        "headline": f"{topic} | The Machinist",
        "author": {
            "@type": "Person",
            "name": "Anudeep Reddy",
            "url": "https://themachinist.org",
        },
        "publisher": {
            "@type": "Organization",
            "name": "The Machinist",
            "url": "https://themachinist.org",
        },
        "url": canonical,
    })

    body_html = (
        f"<h2>What is {topic_short_e} and why does it matter?</h2>\n"
        f'<div class="sl-definition">\n{problem_framing.html}\n</div>\n'
        f'<hr class="sl-hr"/>\n'
        f"<h2>How it works</h2>\n{technical_dive.html}\n"
        f'<hr class="sl-hr"/>\n'
        f"<h2>In practice</h2>\n{code_snippets.html}\n"
        f'<hr class="sl-hr"/>\n'
        f"<h2>Key takeaways</h2>\n"
        f'<div class="callout callout-key">\n'
        f'<div class="callout-label">What you now know</div>\n'
        f"{takeaways.html}\n"
        f"</div>\n"
    )
    if "{{" in body_html or "}}" in body_html:
        raise PolicyError("unreplaced placeholders in article body")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<meta http-equiv="Content-Security-Policy" content="{csp_e}"/>
<title>{topic_e} | The Machinist</title>
<meta name="description" content="{meta_e}"/>
<meta name="author" content="Anudeep Reddy"/>
<meta property="og:type" content="website"/>
<meta property="og:url" content="{canonical_e}"/>
<meta property="og:title" content="{topic_e} | The Machinist"/>
<meta property="og:description" content="{meta_e}"/>
<meta name="twitter:card" content="summary"/>
<meta name="twitter:title" content="{topic_e} | The Machinist"/>
<meta name="twitter:description" content="{meta_e}"/>
<link rel="canonical" href="{canonical_e}"/>
<style>{ARTICLE_CSS}</style>
</head>
<body>
<a href="#main" class="skip-link">Skip to main content</a>
<nav id="nav" role="navigation" aria-label="Main navigation">
<div class="container"><ul>
<li><a href="index.html#projects">Projects</a></li>
<li><a href="index.html#about">About</a></li>
<li><a href="index.html#skills">Skills</a></li>
<li><a href="index.html#learning-log">Learning Log</a></li>
<li><a href="index.html#contact">Contact</a></li>
</ul></div>
</nav>
<header class="sl-header">
<div class="container">
<nav class="sl-breadcrumb" aria-label="Breadcrumb">
<a href="index.html#learning-log">Learning Log</a>
<span class="sl-breadcrumb-sep">/</span>
<span>{crumb_e}</span>
<span class="sl-breadcrumb-sep">/</span>
<span>{topic_e}</span>
</nav>
<div class="sl-series-label">{series_e}</div>
<h1>{topic_e}</h1>
<p class="sl-subtitle">{meta_e}</p>
<div class="sl-meta">
<span>{_clock_icon()} {read_e} min read</span>
<span class="meta-badge">Intermediate</span>
</div>
</div>
</header>
<main id="main">
<div class="sl-body">
{body_html}
<div class="sl-sources">
<div class="sl-sources-label">Sources</div>
<ol>
{citations_html}
</ol>
</div>
<div class="back-wrap"><a href="index.html#learning-log" class="back-link">← Back to Learning Log</a></div>
</div>
</main>
<footer>
<div class="container">
{_footer_icons()}
<p class="copyright">&copy; 2026 The Machinist. All rights reserved.</p>
</div>
</footer>
<script type="application/ld+json">{json_ld}</script>
</body>
</html>
"""
    if "{{" in html:
        raise PolicyError("unreplaced placeholders")
    if "<script>" in html.lower().replace(" ", "") and "application/ld+json" not in html:
        raise PolicyError("executable script in trusted article")
    if "shared.js" in html or "shared.css" in html or "fonts.googleapis.com" in html:
        raise PolicyError("forbidden external subresource in trusted article")
    if "javascript:" in html.lower():
        raise PolicyError("javascript: URL in trusted article")
    digest = sha256_utf8(html)
    return TrustedArticle(
        html=html,
        body_html=body_html,
        sha256=digest,
        policy_version=HTML_POLICY_VERSION,
    )


def reassemble_from_body(
    *,
    topic: str,
    slug: str,
    meta_description: str,
    series_label: str,
    breadcrumb_section: str,
    read_time: str,
    body_html: str,
    citations_html: str,
) -> TrustedArticle:
    """Rebuild the immutable shell around a sanitized body fragment. Citations stay server-owned."""
    fragment = sanitize_fragment(body_html)
    # Body contains trusted wrappers (h2, sl-definition, callout). Re-parse as a
    # single fragment then split is lossy; instead sanitize then place as inner body.
    if not fragment.html.strip():
        raise PolicyError("revised body sanitized to empty")
    dummy = TrustedFragment(html="<p>.</p>")
    article = assemble_trusted_article(
        topic=topic,
        slug=slug,
        meta_description=meta_description,
        series_label=series_label,
        breadcrumb_section=breadcrumb_section,
        read_time=read_time,
        problem_framing=dummy,
        technical_dive=dummy,
        code_snippets=dummy,
        takeaways=dummy,
        citations_html=citations_html,
    )
    html = article.html
    start = html.find('<div class="sl-body">')
    sources = html.find('<div class="sl-sources">')
    if start < 0 or sources < 0:
        raise PolicyError("trusted shell missing body markers")
    inner_start = start + len('<div class="sl-body">')
    rebuilt = html[:inner_start] + "\n" + fragment.html + "\n" + html[sources:]
    if "{{" in rebuilt:
        raise PolicyError("unreplaced placeholders after reassemble")
    return TrustedArticle(
        html=rebuilt,
        body_html=fragment.html,
        sha256=sha256_utf8(rebuilt),
        policy_version=HTML_POLICY_VERSION,
    )
