"""Deterministic delimiter-policy qualification. $0, no providers.

Exercises the real sanitize_fragment → assemble/reassemble path.
"""
from __future__ import annotations

import json

import pytest

from agent.html_policy import (
    ARTICLE_CSS,
    HTML_POLICY_VERSION,
    PolicyError,
    reassemble_from_body,
    sanitize_fragment,
)
from main import _write_telemetry
from tests.conftest import fake_response
from tests.test_html_policy import _article


def _raise_assemble(html: str) -> PolicyError:
    with pytest.raises(PolicyError, match="unresolved template delimiters") as exc_info:
        _article(technical_dive=sanitize_fragment(html))
    return exc_info.value


def _safe_diag(exc: PolicyError) -> dict:
    diagnostic = exc.diagnostic
    assert diagnostic is not None
    assert "TOPIC" not in repr(diagnostic)
    assert "{{" not in repr(diagnostic)
    assert "}}" not in repr(diagnostic)
    return diagnostic


def test_html_policy_version_is_v2():
    assert HTML_POLICY_VERSION == "p0-1-v2"
    article = _article()
    assert article.policy_version == "p0-1-v2"
    assert sanitize_fragment("<p>ok</p>").policy_version == "p0-1-v2"


def test_ordinary_paragraph_placeholder_rejected():
    exc = _raise_assemble("<p>{{TOPIC}}</p>")
    diagnostic = _safe_diag(exc)
    assert diagnostic["rule_id"] == "template_delimiter_outside_code_v1"
    assert diagnostic["match_count"] == 1
    assert diagnostic["locations"][0]["element"] == "p"
    assert diagnostic["locations"][0]["token_class"] == "template_delimiter"


def test_standalone_code_template_accepted():
    article = _article(technical_dive=sanitize_fragment("<code>{{ user.name }}</code>"))
    assert "{{ user.name }}" in article.html


def test_pre_code_template_accepted():
    article = _article(technical_dive=sanitize_fragment(
        "<pre><code>{{ user.name }}</code></pre>"
    ))
    assert "{{ user.name }}" in article.html


def test_delimiter_immediately_before_code_rejected():
    exc = _raise_assemble("<p>{{TOPIC}}<code>x</code></p>")
    diagnostic = _safe_diag(exc)
    assert diagnostic["match_count"] == 1
    assert diagnostic["locations"][0]["element"] == "p"


def test_delimiter_immediately_after_code_rejected():
    exc = _raise_assemble("<p><code>x</code>{{TOPIC}}</p>")
    diagnostic = _safe_diag(exc)
    assert diagnostic["match_count"] == 1
    assert diagnostic["locations"][0]["element"] == "p"


def test_nested_markup_inside_pre_remains_literal():
    article = _article(technical_dive=sanitize_fragment(
        "<pre><code>before <strong>{{ user.name }}</strong> after</code></pre>"
    ))
    assert "{{ user.name }}" in article.html


def test_mixed_code_and_noncode_rejects_noncode_only():
    exc = _raise_assemble("<p>{{BAD}}</p><code>{{ user.name }}</code>")
    diagnostic = _safe_diag(exc)
    assert diagnostic["match_count"] == 1
    assert diagnostic["locations"][0]["element"] == "p"


def test_lone_open_delimiter_outside_code_rejected():
    exc = _raise_assemble("<p>keep {{ open</p>")
    diagnostic = _safe_diag(exc)
    assert diagnostic["match_count"] == 1


def test_lone_close_delimiter_outside_code_rejected():
    exc = _raise_assemble("<p>keep }} close</p>")
    diagnostic = _safe_diag(exc)
    assert diagnostic["match_count"] == 1


def test_malformed_misnested_html_sanitizer_then_scanner():
    repaired = sanitize_fragment(
        "<p>hello<pre><code>{{x}}</code></p><p>after {{TOPIC}}</p>"
    ).html
    assert repaired == (
        "<p>hello</p><pre><code>{{x}}</code><p></p><p>after {{TOPIC}}</p></pre>"
    )

    sibling = sanitize_fragment(
        "<pre><code>{{ user.name }}</code></pre><p>{{TOPIC}}</p>"
    ).html
    assert sibling == "<pre><code>{{ user.name }}</code></pre><p>{{TOPIC}}</p>"
    exc = _raise_assemble(sibling)
    diagnostic = _safe_diag(exc)
    assert diagnostic["match_count"] == 1
    assert diagnostic["locations"][0]["element"] == "p"


def test_encoded_delimiter_in_prose_does_not_bypass():
    cleaned = sanitize_fragment("<p>&#123;&#123;TOPIC&#125;&#125;</p>").html
    assert cleaned == "<p>{{TOPIC}}</p>"
    exc = _raise_assemble("<p>&#123;&#123;TOPIC&#125;&#125;</p>")
    _safe_diag(exc)


def test_encoded_delimiter_inside_code_remain_representable():
    cleaned = sanitize_fragment(
        "<code>&#123;&#123; user.name &#125;&#125;</code>"
    ).html
    assert cleaned == "<code>{{ user.name }}</code>"
    article = _article(technical_dive=sanitize_fragment(cleaned))
    assert "{{ user.name }}" in article.html


def test_delimiter_in_attributes_stripped_or_rejected():
    assert sanitize_fragment('<p class="{{TOPIC}}">x</p>').html == "<p>x</p>"
    assert sanitize_fragment('<p id="{{TOPIC}}">x</p>').html == "<p>x</p>"
    assert sanitize_fragment('<p data-x="{{TOPIC}}">x</p>').html == "<p>x</p>"
    mixed = sanitize_fragment(
        '<code class="language-python {{oops}}">print(1)</code>'
    ).html
    assert mixed == '<code class="language-python">print(1)</code>'
    article = _article(technical_dive=sanitize_fragment(mixed))
    assert article.policy_version == HTML_POLICY_VERSION


def test_article_css_braces_do_not_trigger_delimiter_rule():
    assert "}}" in ARTICLE_CSS
    article = _article()
    assert ARTICLE_CSS in article.html


def test_single_brace_math_and_state_remain_representable():
    article = _article(technical_dive=sanitize_fragment(
        '<p>c_e: S → {True, False}; state = {&quot;messages&quot;: []}</p>'
    ))
    assert "{True, False}" in article.html


def test_multiple_invalid_occurrences_match_count():
    exc = _raise_assemble("<p>{{A}}</p><p>{{B}}</p>")
    diagnostic = _safe_diag(exc)
    assert diagnostic["match_count"] == 2
    assert len(diagnostic["locations"]) == 2
    assert [loc["element"] for loc in diagnostic["locations"]] == ["p", "p"]


def test_diagnostic_scope_body_document_revised():
    body_exc = _raise_assemble("<p>{{TOPIC}}</p>")
    body_diag = _safe_diag(body_exc)
    assert body_diag["scope"] == "article_body"

    with pytest.raises(PolicyError, match="unresolved template delimiters") as doc_info:
        _article(meta_description="{{META}}")
    doc_diag = _safe_diag(doc_info.value)
    assert doc_diag["scope"] == "document"

    with pytest.raises(PolicyError, match="unresolved template delimiters") as rev_info:
        reassemble_from_body(
            topic="Gradient Descent",
            slug="gradient-descent",
            meta_description="meta",
            series_label="Learning Log",
            breadcrumb_section="Learning Log",
            read_time="5",
            body_html="<p>{{TOPIC}}</p>",
            citations_html="<li>Sources retrieved via Tavily web search.</li>",
        )
    rev_diag = _safe_diag(rev_info.value)
    assert rev_diag["scope"] == "revised_body"


def test_html_revise_scanner_failure_persists_safe_diagnostic(base_state, monkeypatch, tmp_path):
    import agent.nodes as nodes

    original = _article()
    original_body = "<p>{{TOPIC}}</p>"
    revised = "<p><strong>{{TOPIC}}</strong></p>"
    base_state.update(
        html_output=original.html,
        article_body_html=original_body,
        html_sha256=original.sha256,
        html_feedback="wrap in strong",
        run_id="revise-policy-run",
    )
    monkeypatch.setattr(nodes, "_get_client", lambda: object())
    monkeypatch.setattr(nodes, "_llm_call", lambda c, **kw: fake_response(revised))

    out = nodes.html_revise_node(base_state)

    assert out["html_output"] == original.html
    assert out["article_body_html"] == original_body
    assert out["html_sha256"] == original.sha256
    assert any("DISCARDED" in entry and "reassemble" in entry for entry in out["error_log"])
    assert out["policy_diagnostics"]
    diagnostic = out["policy_diagnostics"][0]
    assert diagnostic["stage"] == "html_revise"
    assert diagnostic["scope"] == "revised_body"
    assert diagnostic["rule_id"] == "template_delimiter_outside_code_v1"
    assert diagnostic["locations"][0]["element"] == "strong"
    assert "TOPIC" not in repr(out["policy_diagnostics"])
    assert "<p>{{TOPIC}}</p>" not in repr(out["policy_diagnostics"])
    assert "<p><strong>{{TOPIC}}</strong></p>" not in repr(out["policy_diagnostics"])
    assert "<p><strong>{{TOPIC}}</strong></p>" not in "".join(out["error_log"])

    monkeypatch.chdir(tmp_path)
    state = {**base_state, **out}
    path = _write_telemetry(state)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["policy_diagnostics"] == out["policy_diagnostics"]
    dumped = path.read_text(encoding="utf-8")
    assert "<p><strong>{{TOPIC}}</strong></p>" not in dumped
    assert "TOPIC" not in json.dumps(record["policy_diagnostics"])
