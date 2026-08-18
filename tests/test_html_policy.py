"""Deterministic html_policy unit tests. $0, no providers."""
import json

import pytest

from agent.html_policy import (
    HTML_POLICY_VERSION,
    PolicyError,
    TrustedFragment,
    assemble_trusted_article,
    build_citation_items,
    normalize_citation_url,
    render_citations_html,
    render_markdown_fragment,
    render_markdown_review_html,
    sanitize_fragment,
    serialize_json_ld,
    sha256_utf8,
)

EXPLOITS = {
    "script": '<script>alert(1)</script><p>ok</p>',
    "event_handler": '<p onclick="alert(1)">x</p>',
    "malformed_attr": '<p foo=`onmouseover=alert(1)`>x</p>',
    "javascript_url": '<a href="javascript:alert(1)">x</a>',
    "dangerous_data": '<a href="data:text/html,<script>alert(1)</script>">x</a>',
    "protocol_relative": '<a href="//evil.example/x">x</a>',
    "svg_active": '<svg><script>alert(1)</script></svg><p>ok</p>',
    "mathml": '<math><mi onclick="alert(1)">x</mi></math><p>ok</p>',
    "foreignObject": '<svg><foreignObject><script>alert(1)</script></foreignObject></svg>',
    "iframe": '<iframe src="https://evil.example"></iframe><p>ok</p>',
    "srcdoc": '<iframe srcdoc="<script>alert(1)</script>"></iframe>',
    "object_embed": '<object data="https://evil.example"></object><embed src="https://evil.example">',
    "form": '<form action="https://evil.example"><input name="x"></form><p>ok</p>',
    "meta_refresh": '<meta http-equiv="refresh" content="0;url=https://evil.example"><p>ok</p>',
    "base": '<base href="https://evil.example/"><p>ok</p>',
    "css_import": '<style>@import url("https://evil.example/x.css")</style><p>ok</p>',
    "css_url": '<div style="background:url(https://evil.example/x)">x</div>',
    "malformed_html": '<p><img src=x onerror=alert(1)></p>',
    "encoded_scheme": '<a href="java&#115;cript:alert(1)">x</a>',
}


def _article(**kwargs):
    defaults = dict(
        topic="Gradient Descent",
        slug="gradient-descent",
        meta_description="meta",
        series_label="Learning Log",
        breadcrumb_section="Learning Log",
        read_time="5",
        problem_framing=sanitize_fragment("<p>framing</p>"),
        technical_dive=sanitize_fragment("<p>dive</p>"),
        code_snippets=sanitize_fragment('<pre><code class="language-python">print(1)</code></pre>'),
        takeaways=sanitize_fragment("<ul><li>one</li></ul>"),
        citations_html="<li>Sources retrieved via Tavily web search.</li>",
    )
    defaults.update(kwargs)
    return assemble_trusted_article(**defaults)


@pytest.mark.parametrize("name,html", list(EXPLOITS.items()))
def test_exploit_fixtures_are_inert(name, html):
    cleaned = sanitize_fragment(html).html.lower()
    assert "<script" not in cleaned
    assert "onerror" not in cleaned
    assert "onclick" not in cleaned
    assert "javascript:" not in cleaned
    assert "<iframe" not in cleaned
    assert "<object" not in cleaned
    assert "<embed" not in cleaned
    assert "<form" not in cleaned
    assert "<meta" not in cleaned
    assert "<base" not in cleaned
    assert "<svg" not in cleaned
    assert "<math" not in cleaned
    assert "srcdoc" not in cleaned
    assert "url(" not in cleaned
    assert "@import" not in cleaned
    assert "<a " not in cleaned


def test_sanitizer_idempotent_on_safe_and_dirty_html():
    for html in list(EXPLOITS.values()) + ["<p>Hello <strong>there</strong></p>"]:
        once = sanitize_fragment(html).html
        twice = sanitize_fragment(once).html
        assert twice == once


def test_only_allowed_classes_survive():
    out = sanitize_fragment('<div class="callout callout-info evil">x</div>').html
    assert 'class="callout callout-info"' in out
    assert "evil" not in out


def test_th_scope_allowlist():
    ok = sanitize_fragment('<table><tr><th scope="row">h</th></tr></table>').html
    assert 'scope="row"' in ok
    bad = sanitize_fragment('<table><tr><th scope="bogus">h</th></tr></table>').html
    assert "scope=" not in bad


def test_ids_names_data_attrs_stripped():
    out = sanitize_fragment('<p id="x" name="y" data-x="1" tabindex="1">z</p>').html
    assert "id=" not in out
    assert "name=" not in out
    assert "data-" not in out
    assert "tabindex" not in out


def test_markdown_integration_strips_active_content():
    md = "# Title\n\n<script>alert(1)</script>\n\n[click](javascript:alert(1))\n\n![x](https://evil.example/x.png)\n"
    frag = render_markdown_fragment(md)
    html = frag.html.lower()
    assert "<script" not in html
    assert "<a " not in html
    assert "<img" not in html
    doc = render_markdown_review_html(md)
    assert "Content-Security-Policy" in doc
    assert "default-src 'none'" in doc


@pytest.mark.parametrize("url", [
    "javascript:alert(1)",
    "data:text/html,hi",
    "//evil.example/x",
    "http://example.com/x",
    "https://localhost/x",
    "https://foo.local/x",
    "https://127.0.0.1/x",
    "https://10.0.0.1/x",
    "https://192.168.1.1/x",
    "https://169.254.1.1/x",
    "https://[::1]/x",
    "https://user:pass@example.com/x",
    "https://example.com:8443/x",
    "https://example.com/x\\y",
    "https://example.com/%zz",
    "https://example.com/" + ("a" * 2100),
])
def test_unsafe_urls_rejected(url):
    assert normalize_citation_url(url) is None


def test_https_url_normalized():
    assert normalize_citation_url(" HTTPS://Example.COM/a/B?q=1#frag ") == "https://example.com/a/B?q=1"


def test_citation_authority_exact_match_only():
    web = [{"title": "Good", "url": "https://example.com/gd", "score": 0.9}]
    items = build_citation_items(
        [{"claim": "c", "status": "verified", "source_kind": "web",
          "source_ref": "https://example.com/gd", "source_url": "https://evil.example/not-canonical"}],
        web,
        [],
    )
    assert items[0]["url"] == "https://example.com/gd"
    hostile = build_citation_items(
        [{"claim": "c", "status": "verified", "source_kind": "web",
          "source_ref": "https://evil.example/x", "source_url": "https://evil.example/x"}],
        web,
        [],
    )
    assert hostile[0]["url"] is None
    assert hostile[0]["label"] == "Unresolved source"
    html = render_citations_html(hostile)
    assert "<a " not in html
    assert "Unresolved source" in html


def test_kb_citations_are_not_clickable():
    items = build_citation_items(
        [{"claim": "c", "status": "verified", "source_kind": "kb", "source_ref": "kb:gd.md"}],
        [{"url": "https://example.com/gd"}],
        [{"source": "gd.md"}],
    )
    html = render_citations_html(items)
    assert "<a " not in html
    assert "gd.md" in html


def test_hostile_title_and_claim_are_escaped_in_article():
    art = _article(topic='</script><script>alert(1)</script><img src=x onerror=alert(1)>')
    assert "<script>alert" not in art.html
    assert "\\u003c/script\\u003e" in art.html or "&lt;/script&gt;" in art.html
    assert art.policy_version == HTML_POLICY_VERSION
    assert sha256_utf8(art.html) == art.sha256


def test_json_ld_script_breakout_escaped():
    payload = {"headline": "</script><script>alert(1)</script>", "x": "a&b>c<d"}
    dumped = serialize_json_ld(payload)
    assert "<" not in dumped
    assert ">" not in dumped
    assert "&" not in dumped


def test_trusted_article_has_no_active_subresources():
    art = _article()
    html = art.html
    assert "shared.js" not in html
    assert "shared.css" not in html
    assert "fonts.googleapis.com" not in html
    assert "<script src" not in html.lower()
    assert "default-src 'none'" in html
    assert "script-src 'none'" in html


def test_fail_closed_on_empty_required_section():
    with pytest.raises(PolicyError):
        _article(problem_framing=TrustedFragment(html=""))
