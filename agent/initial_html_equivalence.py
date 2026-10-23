"""Deterministic initial-render content binding for the Gate-1-approved draft."""

from __future__ import annotations

import json
from html.parser import HTMLParser

from agent.html_policy import (ALLOWED_CLASSES, ALLOWED_TAGS, HTML_POLICY_VERSION, PolicyError, reassemble_from_body,
                               render_markdown_fragment, sanitize_fragment, sha256_utf8)

CONTRACT_VERSION = "initial-html-content-v1"
SECTIONS = ("problem_framing", "technical_dive", "code_snippets", "takeaways")
_VOID = frozenset({"br", "hr"})
_BLOCK = frozenset({"p", "div", "h1", "h2", "h3", "h4", "li", "ul", "ol", "blockquote", "table", "tr", "td", "th"})
_CODE_LANGUAGES = frozenset(css for css in ALLOWED_CLASSES if css.startswith("language-"))
_CASE_CHANGING_CLASSES = frozenset({"callout-label", "sl-code-label"})


class _Visible(HTMLParser):
    """Keep ordered prose and exact code; reject malformed or ambiguous markup."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.classes: list[frozenset[str]] = []
        self.events: list[tuple[str, str]] = []
        self.text: list[str] = []
        self.code: list[str] = []
        self.pre_has_code = False
        self.citation: list[str] = []

    def _flush_text(self) -> None:
        if self.text:
            import re
            import unicodedata

            value = re.sub(r"\s+", " ", unicodedata.normalize("NFC", "".join(self.text))).strip()
            if value:
                self.events.append(("text", value))
            self.text.clear()

    @staticmethod
    def _check_tag(tag: str, attrs: list) -> None:
        if tag not in ALLOWED_TAGS:
            raise PolicyError("untrusted article element")
        if tag == "pre" and attrs:
            raise PolicyError("block code wrapper has unproven attributes")
        if tag == "code" and attrs != [] and (
            len(attrs) != 1 or attrs[0][0] != "class" or attrs[0][1] not in _CODE_LANGUAGES
        ):
            raise PolicyError("code wrapper has unproven attributes")
        seen: set[str] = set()
        for name, value in attrs:
            if name in seen or not isinstance(value, str):
                raise PolicyError("ambiguous article attribute")
            seen.add(name)
            if tag == "hr" and name == "class" and value == "sl-hr":
                continue
            if tag in {"div", "span", "pre", "code"} and name == "class":
                classes = value.split()
                if classes and all(css_class in ALLOWED_CLASSES for css_class in classes):
                    continue
            if tag == "th" and name == "scope" and value in {"row", "col"}:
                continue
            raise PolicyError("untrusted article attribute")

    def handle_starttag(self, tag: str, attrs: list) -> None:
        self._check_tag(tag, attrs)
        if "sup" in self.stack:
            raise PolicyError("nested citation markup")
        if tag in ("pre", "code") and any(
            classes & _CASE_CHANGING_CLASSES for classes in self.classes
        ):
            raise PolicyError("presentation class changes displayed code")
        if any(t in ("pre", "code") for t in self.stack):
            # Only the inert <pre><code> wrapper is unambiguous. In particular,
            # <br> changes displayed line structure without contributing data.
            if tag != "code" or self.stack[-1] != "pre":
                raise PolicyError("markup inside code changes displayed content")
            if self.pre_has_code:
                raise PolicyError("multiple code wrappers in one block")
            self.pre_has_code = True
        if tag in _VOID:
            if tag == "br":
                self.text.append(" ")
            return
        if tag == "sup":
            self._flush_text()
            self.citation.clear()
        elif tag in ("pre", "code") and not any(t in ("pre", "code") for t in self.stack):
            self._flush_text()
            self.code.clear()
            if tag == "pre":
                self.pre_has_code = False
        elif tag in _BLOCK and not any(t in ("pre", "code") for t in self.stack):
            self.text.append(" ")
        self.stack.append(tag)
        self.classes.append(frozenset(next((value.split() for name, value in attrs if name == "class"), ())))

    def handle_startendtag(self, tag: str, attrs: list) -> None:
        self._check_tag(tag, attrs)
        if any(t in ("pre", "code") for t in self.stack):
            raise PolicyError("markup inside code changes displayed content")
        if tag not in _VOID:
            raise PolicyError("ambiguous self-closing article markup")

    def handle_endtag(self, tag: str) -> None:
        if not self.stack or self.stack[-1] != tag:
            raise PolicyError("malformed article markup")
        self.stack.pop()
        self.classes.pop()
        if tag == "sup":
            import re

            value = "".join(self.citation)
            if not re.fullmatch(r"[1-9][0-9]*(?:,[1-9][0-9]*)*", value):
                raise PolicyError("invalid citation marker")
            self.events.append(("citation", value))
            self.citation.clear()
        elif tag == "pre":
            if not self.pre_has_code:
                raise PolicyError("block code requires a code wrapper")
            self.events.append(("block_code", "".join(self.code)))
            self.code.clear()
            self.pre_has_code = False
        elif tag == "code" and "pre" not in self.stack:
            self.events.append(("inline_code", "".join(self.code)))
            self.code.clear()
        elif tag in _BLOCK and not any(t in ("pre", "code") for t in self.stack):
            self.text.append(" ")

    def handle_data(self, data: str) -> None:
        if "sup" in self.stack:
            self.citation.append(data)
        elif "pre" in self.stack and "code" not in self.stack:
            raise PolicyError("text outside the block code wrapper")
        elif any(t in ("pre", "code") for t in self.stack):
            self.code.append(data)
        else:
            self.text.append(data)

    def handle_comment(self, data: str) -> None:
        raise PolicyError("comments are not article content")

    def unknown_decl(self, data: str) -> None:
        raise PolicyError("declarations are not article content")

    def handle_decl(self, data: str) -> None:
        raise PolicyError("declarations are not article content")

    def finish(self) -> tuple[tuple[str, str], ...]:
        self.close()
        if self.stack or self.code:
            raise PolicyError("unclosed article markup")
        self._flush_text()
        merged: list[tuple[str, str]] = []
        for kind, value in self.events:
            if kind == "text" and merged and merged[-1][0] == "text":
                merged[-1] = ("text", f"{merged[-1][1]} {value}")
            else:
                merged.append((kind, value))
        return tuple(merged)


def visible_key(fragment: str) -> tuple[tuple[str, str], ...]:
    parser = _Visible()
    parser.feed(fragment)
    return parser.finish()


def _digest(value: object) -> str:
    return sha256_utf8(json.dumps(value, ensure_ascii=True, separators=(",", ":")))


def _source_fragment(section: str, raw: str) -> str:
    if section == "code_snippets" and "```" not in raw:
        import html

        return f"<pre><code>{html.escape(raw)}</code></pre>"
    return sanitize_fragment(render_markdown_fragment(raw).html).html


def source_key(section: str, raw: str) -> tuple[tuple[str, str], ...]:
    return visible_key(_source_fragment(section, raw))


def source_identity(state: dict) -> str:
    from agent.nodes import _assemble_markdown

    sections = state.get("draft_sections")
    draft = state.get("draft_markdown")
    topic = state.get("topic")
    if (not isinstance(topic, str) or not isinstance(sections, dict)
            or any(not isinstance(sections.get(key), str) or not sections[key].strip()
                   for key in SECTIONS)
            or not isinstance(draft, str) or _assemble_markdown(topic, sections) != draft):
        raise PolicyError("render source differs from Gate 1-approved draft")
    return _digest([(key, source_key(key, sections[key])) for key in SECTIONS])


def assert_section_equivalent(section: str, raw: str, rendered: str) -> None:
    if section not in SECTIONS or source_key(section, raw) != visible_key(rendered):
        raise PolicyError(f"{section} visible content differs from approved draft")


def receipt_for_article(state: dict, body_html: str) -> dict:
    source = source_identity(state)
    return {
        "contract_version": CONTRACT_VERSION,
        "run_id": state.get("run_id"),
        "draft_sha256": sha256_utf8(state["draft_markdown"]),
        "source_key_sha256": source,
        "body_key_sha256": _digest(visible_key(body_html)),
        "html_policy_version": HTML_POLICY_VERSION,
    }


def receipt_valid(state: dict) -> bool:
    receipt = state.get("initial_html_equivalence")
    html = state.get("html_output")
    body = state.get("article_body_html")
    if not isinstance(receipt, dict) or not isinstance(html, str) or not isinstance(body, str):
        return False
    try:
        source = source_identity(state)
        body_key = _digest(visible_key(body))
    except (PolicyError, KeyError, TypeError, ValueError):
        return False
    if receipt != {
        "contract_version": CONTRACT_VERSION,
        "run_id": state.get("run_id"),
        "draft_sha256": sha256_utf8(state["draft_markdown"]),
        "source_key_sha256": source,
        "body_key_sha256": body_key,
        "html_policy_version": HTML_POLICY_VERSION,
    }:
        return False
    if state.get("html_policy_version") != HTML_POLICY_VERSION or state.get("html_sha256") != sha256_utf8(html):
        return False
    marker = '<div class="sl-body">\n'
    end = '\n<div class="sl-sources">'
    if html.count(marker) != 1 or html.count(end) != 1 or html.split(marker, 1)[1].split(end, 1)[0] != body:
        return False
    try:
        from agent.nodes import _article_shell_fields, _citations_html_for_state

        expected = reassemble_from_body(
            **_article_shell_fields(state), body_html=body,
            citations_html=_citations_html_for_state(state),
        ).html
    except (PolicyError, KeyError, TypeError, ValueError):
        return False
    shell_start = '<div class="sl-body">'
    shell_end = '<div class="sl-sources">'
    return (html.split(shell_start, 1)[0] == expected.split(shell_start, 1)[0]
            and html.split(shell_end, 1)[1] == expected.split(shell_end, 1)[1])
