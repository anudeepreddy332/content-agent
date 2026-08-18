"""Browser exploit tests with pinned Chromium. $0, no paid providers."""
import os
import socket
import threading
import time
from html import escape

import pytest
import uvicorn

os.environ.setdefault("API_BEARER_TOKEN", "test-token")
os.environ.setdefault("API_SYNC", "1")

from agent.html_policy import sanitize_fragment
from tests.test_html_policy import EXPLOITS, _article
from tests.test_api_stream import FakeStreamGraph

pytest.importorskip("playwright")
from playwright.sync_api import sync_playwright


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    return port


@pytest.fixture
def live_server(monkeypatch):
    import tools.query_kb as kb
    monkeypatch.setattr(kb, "warmup", lambda: {"status": "skipped"})
    import api.server as srv
    srv.GRAPH = FakeStreamGraph()
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


def test_reviewer_ui_empty_sandbox_and_no_token_in_url(live_server, pw_page):
    seen = []
    pw_page.on("request", lambda req: seen.append(req.url))
    pw_page.goto(live_server + "/", wait_until="domcontentloaded")
    pw_page.fill("#token", "test-token")
    pw_page.fill("#topic", "Gradient Descent")
    pw_page.click("#go")
    pw_page.wait_for_timeout(500)
    assert pw_page.input_value("#token") == ""
    assert all("test-token" not in u for u in seen)
    assert all("token=" not in u.lower() for u in seen)
    assert pw_page.locator("#g1frame").get_attribute("sandbox") == ""
    assert pw_page.locator("#g2frame").get_attribute("sandbox") == ""
    assert pw_page.locator("#g1frame").get_attribute("referrerpolicy") == "no-referrer"
    assert pw_page.locator("#g2preview").count() == 0
    html = pw_page.content()
    assert "allow-scripts" not in html
    assert "marked" not in html


def test_article_iframe_no_script_no_attacker_request(pw_page):
    dirty = EXPLOITS["script"] + EXPLOITS["iframe"] + EXPLOITS["malformed_html"] + EXPLOITS["javascript_url"]
    frag = sanitize_fragment(dirty)
    art = _article(technical_dive=frag)
    requests = []
    pw_page.on("request", lambda req: requests.append(req.url))
    wrapped = (
        "<!DOCTYPE html><html><body>"
        f'<iframe id="f" sandbox="" referrerpolicy="no-referrer" srcdoc="{escape(art.html, quote=True)}"></iframe>'
        "</body></html>"
    )
    pw_page.set_content(wrapped, wait_until="domcontentloaded")
    pw_page.wait_for_timeout(300)
    executed = pw_page.evaluate("() => window.hacked === true")
    assert executed is False
    attacker = [u for u in requests if "evil" in u or u.startswith("javascript:")]
    assert attacker == []
    assert pw_page.locator("#f").get_attribute("sandbox") == ""


def test_popup_and_top_navigation_blocked(pw_page):
    html = sanitize_fragment('<p>ok</p><a href="https://evil.example/" target="_blank">x</a>').html
    assert "<a " not in html
    pw_page.set_content(
        f'<!DOCTYPE html><iframe sandbox="" srcdoc="{escape("<p>ok</p><script>window.top.location=\'https://evil.example\'</script>", quote=True)}"></iframe>',
        wait_until="domcontentloaded",
    )
    pw_page.wait_for_timeout(200)
    assert "evil.example" not in pw_page.url
