"""Browser exploit tests with pinned Chromium. $0, no paid providers."""
import os
import socket
import threading
import time

import pytest
import uvicorn

os.environ.setdefault("API_BEARER_TOKEN", "test-token")
os.environ.setdefault("API_SYNC", "1")

from types import SimpleNamespace

from agent.html_policy import (
    ARTICLE_CSS,
    assemble_trusted_article,
    build_citation_items,
    normalize_citation_url,
    render_citations_html,
    render_markdown_fragment,
    render_publication_markdown,
    render_markdown_review_html,
    sanitize_fragment,
)
from agent.initial_html_equivalence import visible_key
from agent.html_policy import PolicyError
from tests.test_api_stream import FakeStreamGraph, _interrupt
from tests.test_html_policy import LEGACY_LOOPBACK, WHATWG_IPV4_EDGE_CASES, _article

pytest.importorskip("playwright")
from playwright.sync_api import sync_playwright

HOSTILE_MARKDOWN = (
    "# Draft\n\n<script>window.top.hacked=true</script>\n\n"
    "[click](javascript:alert(1))\n\n"
    "![x](https://evil.example/pixel.png)\n"
)
HOSTILE_MODEL_HTML = (
    '<script>window.top.hacked=true</script>'
    '<img src="https://evil.example/pixel.png" onerror="alert(1)">'
    '<p onclick="alert(1)">ok</p>'
    '<iframe src="https://evil.example/"></iframe>'
)


def _gate1_review():
    return {
        "type": "hitl_review",
        "draft_sha256": "a" * 64,
        "draft_review_html": render_markdown_review_html(HOSTILE_MARKDOWN),
        "html_policy_version": "p0-1-v2",
        "material_review": {
            "evidence_binding_sha256": "b" * 64,
            "material_claim_ids": ["C-001"],
            "claims": [{
                "claim_id": "C-001", "claim_text": "<script>alert(1)</script> material claim",
                "draft_contexts": [{"start": 0, "end": 8, "context": {"text": "Draft context"}}],
                "support_spans": [{"evidence_id": "WEB-001", "citation_identity": "WEB-001",
                                   "quote": "Bound source passage", "source_context": {"text": "Bound source passage in context"},
                                   "provenance": {"title": "Source", "url": "javascript:alert(1)"}}],
                "blockers": [],
            }],
        },
    }


def _gate2_review():
    article = assemble_trusted_article(
        topic="Gradient Descent",
        slug="gradient-descent",
        meta_description="meta",
        series_label="Learning Log",
        breadcrumb_section="Learning Log",
        read_time="5",
        problem_framing=sanitize_fragment("<p>framing</p>"),
        technical_dive=sanitize_fragment(
            HOSTILE_MODEL_HTML
            + "<h3>Preview heading</h3><p>Formula <code>E = mc^2</code></p>"
            + "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>"
            + "<tbody><tr><td>loss</td><td>0.1</td></tr></tbody></table>"
        ),
        code_snippets=sanitize_fragment("<pre><code>print(1)</code></pre>"),
        takeaways=sanitize_fragment("<ul><li>one</li></ul>"),
        citations_html="<li>Unresolved source</li>",
    )
    return {
        "type": "hitl_html_review",
        "html_output": article.html,
        "html_filename": "gradient-descent.html",
        "html_sha256": article.sha256,
        "html_policy_version": article.policy_version,
        "validation_warnings": [],
        "grounding_score": 0.8,
    }


class HostileReviewGraph(FakeStreamGraph):
    """Same two-gate stream as FakeStreamGraph; interrupt payloads are trusted-rendered."""

    def get_state(self, config):
        tid = config["configurable"]["thread_id"]
        phase = self.phase.get(tid)
        if phase == "draft_gate":
            return _interrupt(_gate1_review())
        if phase == "html_gate":
            return _interrupt(_gate2_review())
        return SimpleNamespace(tasks=(), values=self.values.get(tid, {}))


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    return port


@pytest.fixture
def live_server(monkeypatch):
    monkeypatch.setenv("HITL_REVIEWER_ID", "pilot-reviewer")
    monkeypatch.setenv("HITL_REVIEWER_TOKEN", "reviewer-browser-secret-0123456789abcdef")
    import tools.query_kb as kb
    monkeypatch.setattr(kb, "warmup", lambda: {"status": "skipped"})
    import api.server as srv
    srv.GRAPH = HostileReviewGraph()
    monkeypatch.setattr(srv, "material_review_bundle", lambda _state: _gate1_review()["material_review"])
    port = _free_port()
    config = uvicorn.Config(srv.app, host="127.0.0.1", port=port, log_level="error", lifespan="on")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.05)
    else:
        pytest.fail("uvicorn did not start")
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True


@pytest.fixture
def pw_page():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        pg = context.new_page()
        yield pg
        context.close()
        browser.close()


def test_code_equivalence_matches_chromium_display_contract(pw_page):
    code = 'if ready:\n    step()'
    block = f'<pre><code>{code}</code></pre>'
    inline = f'<code>{code}</code>'
    valid = sanitize_fragment(render_markdown_fragment(
        'A **gradient descent** learning note.\n\n1. Use `fit()` on training data.'
    ).html).html.replace('training data.', 'training data.<sup>1</sup>')
    pw_page.set_content(
        f'<style>{ARTICLE_CSS}</style><div class="sl-body">'
        f'<div id="block">{block}</div><div id="inline">{inline}</div>'
        f'<div id="break"><pre><code>if ready:<br>\n    step()</code></pre></div>'
        f'<div id="case"><pre><code class="sl-code-label">print("a")</code></pre></div>'
        '<div id="hidden" style="display:none">hidden claim</div>'
        f'<div id="valid">{valid}</div>'
        '</div>'
    )
    rendered = pw_page.evaluate("""() => Object.fromEntries(
        ['block', 'inline', 'break', 'case', 'hidden'].map(id => {
            const el = document.querySelector('#' + id);
            const code = el.querySelector('code') || el;
            const style = getComputedStyle(code);
            return [id, {textContent: code.textContent, innerText: code.innerText,
                         whiteSpace: style.whiteSpace, textTransform: style.textTransform,
                         display: getComputedStyle(el).display}];
        }))""")
    assert rendered["block"]["textContent"] == rendered["inline"]["textContent"]
    assert rendered["block"]["whiteSpace"] == "pre"
    assert rendered["inline"]["whiteSpace"] == "normal"
    assert "\n    step()" in rendered["block"]["innerText"]
    assert "\n    step()" not in rendered["inline"]["innerText"]
    assert rendered["break"]["innerText"].count("\n") > rendered["block"]["innerText"].count("\n")
    assert rendered["case"]["textTransform"] == "uppercase"
    assert rendered["hidden"]["display"] == "none"
    assert visible_key(block) != visible_key(inline)
    for unsafe in ('<pre><code>if ready:<br>    step()</code></pre>',
                   '<pre><code class="sl-code-label">print("a")</code></pre>',
                   '<div style="display:none">hidden claim</div>'):
        with pytest.raises(PolicyError):
            visible_key(unsafe)
    assert pw_page.locator("#valid strong").inner_text() == "gradient descent"
    assert pw_page.locator("#valid ol li code").inner_text() == "fit()"
    assert pw_page.locator("#valid sup").inner_text() == "1"


def test_publication_markdown_structure_in_chromium(pw_page):
    markdown = (
        "## Method\n\n3. Third\n4. Fourth\n   - Nested\n\n"
        "```python\nif ready:\n    step()\n```\n\n"
        "| Claim | Value |\n|---|---|\n| loss | 0.1 |\n\n"
        "Inline $x+y$\n\n$$\nx + y\n$$"
    )
    article = render_publication_markdown(markdown).html.replace(
        "Inline <span", "Inline <sup>1</sup> <span", 1)
    pw_page.set_content(f'<style>{ARTICLE_CSS}</style><div class="sl-body">{article}</div>')
    assert pw_page.locator(".sl-body h2").inner_text() == "Method"
    ordered = pw_page.locator(".sl-body ol")
    assert ordered.get_attribute("start") == "3"
    assert ordered.evaluate("el => el.start") == 3
    assert ordered.locator("li").first.evaluate(
        "el => getComputedStyle(el).display") == "list-item"
    assert ordered.locator("ul li").inner_text() == "Nested"
    code = pw_page.locator(".sl-body pre code")
    assert code.evaluate("el => getComputedStyle(el).whiteSpace") == "pre"
    assert "\n    step()" in code.inner_text()
    assert pw_page.locator(".sl-body table tr").count() == 2
    assert pw_page.locator(".sl-body tbody td").all_inner_texts() == ["loss", "0.1"]
    assert pw_page.locator(".sl-body .math-inline").inner_text() == "x+y"
    assert pw_page.locator(".sl-body .math-block").evaluate(
        "el => getComputedStyle(el).whiteSpace") == "pre-wrap"
    assert pw_page.locator(".sl-body sup").inner_text() == "1"


def test_numeric_subscript_changes_chromium_presentation_and_is_rejected(pw_page):
    plain = "<p>The example multiplier is 10.</p>"
    altered = "<p>The example multiplier is 1<sub>0</sub>.</p>"
    pw_page.set_content(f'<div id="plain">{plain}</div><div id="altered">{altered}</div>')
    assert pw_page.locator("#plain").inner_text() == pw_page.locator("#altered").inner_text()
    assert pw_page.locator("#altered sub").evaluate(
        "el => getComputedStyle(el).verticalAlign") == "sub"
    assert pw_page.locator("#altered sub").evaluate(
        "el => el.getBoundingClientRect().top") > pw_page.locator("#altered p").evaluate(
            "el => el.getBoundingClientRect().top")
    with pytest.raises(PolicyError):
        visible_key(altered)
    assert visible_key(plain)


def test_reviewer_ui_gate1_and_gate2_hostile_path(live_server, pw_page):
    seen = []
    popups = []
    pw_page.on("request", lambda req: seen.append(req.url))
    pw_page.on("popup", lambda p: popups.append(p.url))
    pw_page.goto(live_server + "/", wait_until="domcontentloaded")
    pw_page.fill("#token", "test-token")
    pw_page.fill("#reviewerToken", "reviewer-browser-secret-0123456789abcdef")
    pw_page.fill("#topic", "Gradient Descent")
    pw_page.click("#go")

    pw_page.wait_for_selector("#gate1:not(.hidden)", timeout=8000)
    g1 = pw_page.locator("#g1frame")
    assert g1.get_attribute("sandbox") == ""
    assert g1.get_attribute("referrerpolicy") == "no-referrer"
    g1_srcdoc = g1.get_attribute("srcdoc") or ""
    assert "<script>" not in g1_srcdoc.lower().replace(" ", "")
    assert "https://evil.example" not in g1_srcdoc
    assert pw_page.evaluate("() => window.hacked === true") is False

    pw_page.locator(".material-decision").select_option("SUPPORTED")
    pw_page.locator(".material-rationale").fill("The cited passage supports the exact claim.")
    pw_page.check("#g1noOmissions")
    pw_page.check("#g1materiality")
    pw_page.click('#gate1 button[data-act="approve"]')
    pw_page.wait_for_selector("#gate2:not(.hidden)", timeout=8000)
    g2 = pw_page.locator("#g2frame")
    assert g2.get_attribute("sandbox") == ""
    assert g2.get_attribute("referrerpolicy") == "no-referrer"
    g2_srcdoc = g2.get_attribute("srcdoc") or ""
    assert "<img" not in g2_srcdoc.lower()
    assert "<iframe" not in g2_srcdoc.lower().split("<iframe id")[0] if False else "<iframe src" not in g2_srcdoc.lower()
    assert "https://evil.example" not in g2_srcdoc
    assert pw_page.evaluate("() => window.hacked === true") is False
    assert live_server.rstrip("/") in pw_page.url
    assert "evil.example" not in pw_page.url
    assert popups == []
    assert all("test-token" not in u for u in seen)
    assert all("token=" not in u.lower() for u in seen)
    assert all("evil.example" not in u and not u.startswith("javascript:") for u in seen)
    assert pw_page.locator("#g2preview").count() == 0
    assert "allow-scripts" not in pw_page.content()
    assert "marked" not in pw_page.content()


def test_reviewer_ui_exposes_evidence_without_favorable_model_labels(live_server, pw_page):
    pw_page.goto(live_server + "/", wait_until="domcontentloaded")
    pw_page.fill("#token", "test-token")
    pw_page.fill("#reviewerToken", "reviewer-browser-secret-0123456789abcdef")
    pw_page.fill("#topic", "Gradient Descent")
    pw_page.click("#go")

    pw_page.wait_for_selector("#gate1:not(.hidden)", timeout=8000)
    review = pw_page.locator("#g1materialReview")
    assert "Material claims (1)" in review.inner_text()
    assert "Bound source passage" in review.inner_text()
    assert "Draft context" in review.inner_text()
    assert "javascript:alert" not in pw_page.locator("#g1materialReview a").all_text_contents()
    assert pw_page.locator("#g1materialReview a").count() == 0
    assert "REAL JUDGE SCORE" not in pw_page.locator("#g1meta").inner_text()
    assert "Verified:" not in pw_page.locator("#gate1").inner_text()


def test_material_evidence_indentation_is_visibly_distinct_and_readable(live_server, pw_page):
    pw_page.goto(live_server + "/", wait_until="domcontentloaded")
    first = "if ready:\n    return allow()\nreturn deny()"
    second = "if ready:\nreturn allow()\n    return deny()"
    assert first.replace(" ", "") == second.replace(" ", "")
    review = _gate1_review()
    review["material_review"]["claims"] = [
        {"claim_id": f"C-00{i}", "claim_text": "Indentation-sensitive Python",
         "draft_contexts": [{"start": 0, "end": len(code),
                             "context": {"text": code}}],
         "support_spans": [{"evidence_id": f"WEB-00{i}", "citation_identity": f"WEB-00{i}",
                            "quote": code, "source_context": {"text": code + "\n" + "x" * 500},
                            "provenance": {"title": "Python source"}}],
         "blockers": []}
        for i, code in enumerate((first, second), start=1)
    ]
    pw_page.evaluate("payload => onGate(payload)", review)
    quotes = pw_page.locator("#g1materialReview blockquote")
    assert quotes.count() == 2
    assert quotes.nth(0).screenshot() != quotes.nth(1).screenshot()
    for selector in (".draft-context", "blockquote", ".source-context"):
        for element in pw_page.locator(f"#g1materialReview {selector}").all():
            assert element.evaluate("el => getComputedStyle(el).whiteSpace") == "pre-wrap"
            assert element.evaluate("el => el.scrollWidth <= el.clientWidth + 1")


def test_material_source_links_follow_server_canonical_policy(live_server, pw_page):
    from agent.evidence_signoff import material_review_bundle
    from tests.signoff_fixtures import approved_material_state

    pw_page.goto(live_server + "/", wait_until="domcontentloaded")
    requests = []
    pw_page.on("request", lambda request: requests.append(request.url))
    pw_page.route("https://**", lambda route: route.abort())
    state = approved_material_state({"run_id": "canonical-link-probe",
                                     "draft_markdown": "Gradient descent minimizes a loss function.",
                                     "topic": "Gradient Descent", "slug": "gradient-descent"})
    for url in ["https://127.1/admin", *LEGACY_LOOPBACK, *WHATWG_IPV4_EDGE_CASES]:
        state["web_sources"][0]["url"] = url
        bundle = material_review_bundle(state)
        pw_page.evaluate("payload => onGate(payload)",
                         {**_gate1_review(), "material_review": bundle})
        assert pw_page.locator("#g1materialReview a").count() == 0, url
    state["web_sources"][0]["url"] = "https://example.com/gd"
    bundle = material_review_bundle(state)
    pw_page.evaluate("payload => onGate(payload)",
                     {**_gate1_review(), "material_review": bundle})
    link = pw_page.locator("#g1materialReview a")
    assert link.count() == 1
    assert link.get_attribute("href") == "https://example.com/gd"
    assert not any(url.startswith("https://") for url in requests)


def test_gate2_preview_keeps_trusted_article_styles_under_reviewer_csp(live_server, pw_page):
    pw_page.set_viewport_size({"width": 540, "height": 820})
    pw_page.goto(live_server + "/", wait_until="domcontentloaded")
    pw_page.fill("#token", "test-token")
    pw_page.fill("#reviewerToken", "reviewer-browser-secret-0123456789abcdef")
    pw_page.fill("#topic", "Gradient Descent")
    pw_page.click("#go")
    pw_page.wait_for_selector("#gate1:not(.hidden)", timeout=8000)
    pw_page.locator(".material-decision").select_option("SUPPORTED")
    pw_page.locator(".material-rationale").fill("The cited passage supports the exact claim.")
    pw_page.check("#g1noOmissions")
    pw_page.check("#g1materiality")
    pw_page.click('#gate1 button[data-act="approve"]')
    pw_page.wait_for_selector("#gate2:not(.hidden)", timeout=8000)

    frame = pw_page.frame_locator("#g2frame")
    assert frame.locator("h2").count() >= 1
    assert frame.locator("table").count() == 1
    assert frame.locator("code").count() >= 1
    assert frame.locator("footer svg").first.evaluate(
        "el => getComputedStyle(el).width"
    ) == "20px"
    assert frame.locator(".sl-body").evaluate(
        "el => getComputedStyle(el).paddingLeft"
    ) == "16px"


def test_article_iframe_no_script_no_attacker_request(pw_page):
    from html import escape
    dirty = HOSTILE_MODEL_HTML
    art = _article(technical_dive=sanitize_fragment(dirty))
    requests = []
    pw_page.on("request", lambda req: requests.append(req.url))
    wrapped = (
        "<!DOCTYPE html><html><body>"
        f'<iframe id="f" sandbox="" referrerpolicy="no-referrer" srcdoc="{escape(art.html, quote=True)}"></iframe>'
        "</body></html>"
    )
    pw_page.set_content(wrapped, wait_until="domcontentloaded")
    pw_page.wait_for_timeout(300)
    assert pw_page.evaluate("() => window.hacked === true") is False
    assert [u for u in requests if "evil" in u or u.startswith("javascript:")] == []
    assert pw_page.locator("#f").get_attribute("sandbox") == ""


def test_popup_and_top_navigation_blocked(pw_page):
    from html import escape
    html = sanitize_fragment('<p>ok</p><a href="https://evil.example/" target="_blank">x</a>').html
    assert "<a " not in html
    pw_page.set_content(
        f'<!DOCTYPE html><iframe sandbox="" srcdoc="{escape("<p>ok</p><script>window.top.location=\'https://evil.example\'</script>", quote=True)}"></iframe>',
        wait_until="domcontentloaded",
    )
    pw_page.wait_for_timeout(200)
    assert "evil.example" not in pw_page.url


def test_chromium_rejects_legacy_ipv4_citation_hrefs(pw_page):
    """Attack hosts Chromium maps to loopback must never become clickable citations."""
    import ipaddress

    blocked = []

    def _abort(route):
        blocked.append(route.request.url)
        route.abort()

    pw_page.route("https://**", _abort)
    pw_page.route("http://**", _abort)

    web = [
        {"title": "Good", "url": "https://example.com/gd", "score": 0.9},
    ]
    for url in [*LEGACY_LOOPBACK, *WHATWG_IPV4_EDGE_CASES]:
        host = pw_page.evaluate(
            """(u) => {
                try { return new URL(u).hostname; }
                catch (e) { return null; }
            }""",
            url,
        )
        if host:
            try:
                ip = ipaddress.ip_address(host)
            except ValueError:
                ip = None
            else:
                assert ip.is_loopback or ip.is_private or not ip.is_global
        assert normalize_citation_url(url) is None
        items = build_citation_items(
            [{
                "claim": "c",
                "status": "verified",
                "source_kind": "web",
                "source_ref": url,
                "source_url": "https://example.com/gd",
            }],
            web + [{"title": "Loop", "url": url, "score": 0.1}],
            [],
        )
        html = render_citations_html(items)
        assert items[0]["url"] is None
        assert "<a " not in html
        assert "Unresolved source" in html
        assert f'href="{url}"' not in html

    pw_page.set_content("<!DOCTYPE html><html><body>ok</body></html>", wait_until="domcontentloaded")
    pw_page.wait_for_timeout(100)
    assert blocked == []


@pytest.mark.parametrize('status', ['terminal_quality_exhausted', 'policy_blocked', 'execution_failed',
    'resource_exhausted', 'content_revision_limit', 'layout_revision_limit', 'rendering_failed',
    'terminal_rejected'])
def test_workflow_terminal_ui_cannot_offer_publication(live_server, pw_page, status):
    pw_page.goto(live_server + '/', wait_until='domcontentloaded')
    message = 'Quality gates were not met within the allowed attempts. Please try again.'
    pw_page.evaluate('(payload) => onDone(payload)', {'status': status, 'summary': {
        'terminal_status': status, 'terminal_message': message,
        'git_status': 'merged', 'html_filename': 'never-published.html'}})
    assert pw_page.locator('#doneBody p.bad').inner_text() == status + ': ' + message
    assert pw_page.locator('#publishBtns').is_hidden()
    assert pw_page.locator('#gate1').is_hidden()
    assert pw_page.locator('#gate2').is_hidden()


@pytest.mark.parametrize('legacy', [False, True])
def test_J_failed_archive_ui_is_never_success(live_server, pw_page, legacy, monkeypatch, base_state):
    pw_page.goto(live_server + '/', wait_until='domcontentloaded')
    import agent.nodes as nodes
    from unittest.mock import Mock
    monkeypatch.setenv('GIT_PUSH_ENABLED', 'false')
    monkeypatch.setattr(nodes, '_atomic_write_utf8', Mock(side_effect=OSError('disk write denied')))
    html = '<html><body>Approved article.</body></html>'
    sha = nodes.sha256_utf8(html)
    summary = nodes.git_node({**base_state, 'html_output': html, 'html_filename': 'article.html',
                              'html_sha256': sha, 'approved_html_sha256': sha})
    assert summary['git_status'] == 'failed'
    if legacy:
        summary.pop('terminal_status')
    pw_page.evaluate('(payload) => onDone(payload)', {'status': 'complete' if legacy else 'execution_failed',
                                                   'summary': summary})
    text = pw_page.locator('#doneBody').inner_text()
    assert 'execution_failed' in text and summary['terminal_message'] in text
    assert 'Published locally' not in text
    assert pw_page.locator('#doneBody p.ok').count() == 0
    assert pw_page.locator('#publishBtns').is_hidden()


def test_L_successful_local_publish_ui_is_preserved(live_server, pw_page):
    pw_page.goto(live_server + '/', wait_until='domcontentloaded')
    pw_page.evaluate('(payload) => onDone(payload)', {'status': 'complete', 'summary': {'git_status': 'merged'}})
    assert 'Merged locally' in pw_page.locator('#doneBody p.ok').inner_text()
    assert pw_page.locator('#publishBtns').is_visible()
