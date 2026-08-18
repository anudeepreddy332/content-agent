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
    assemble_trusted_article,
    build_citation_items,
    normalize_citation_url,
    render_citations_html,
    render_markdown_review_html,
    sanitize_fragment,
)
from tests.test_api_stream import FakeStreamGraph, _interrupt
from tests.test_html_policy import LEGACY_LOOPBACK, _article

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
        "draft_review_html": render_markdown_review_html(HOSTILE_MARKDOWN),
        "html_policy_version": "p0-1-v1",
        "grounding_score": 0.8,
        "reflection_score": 8,
        "reflection_notes": "ok",
        "review_claims": [{
            "claim": "<script>alert(1)</script> claim",
            "status": "verified",
            "confidence": 0.9,
            "source_kind": "web",
            "source_label": "Unresolved source",
            "source_url": None,
        }],
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
        technical_dive=sanitize_fragment(HOSTILE_MODEL_HTML),
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
    import tools.query_kb as kb
    monkeypatch.setattr(kb, "warmup", lambda: {"status": "skipped"})
    import api.server as srv
    srv.GRAPH = HostileReviewGraph()
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


def test_reviewer_ui_gate1_and_gate2_hostile_path(live_server, pw_page):
    seen = []
    popups = []
    pw_page.on("request", lambda req: seen.append(req.url))
    pw_page.on("popup", lambda p: popups.append(p.url))
    pw_page.goto(live_server + "/", wait_until="domcontentloaded")
    pw_page.fill("#token", "test-token")
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
    for url in LEGACY_LOOPBACK:
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
