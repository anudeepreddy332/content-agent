"""Exact UTF-8 source span helpers for CSWP structural parsing."""

from __future__ import annotations

import bisect
import re

from agent.cswp.errors import StructuralError


class Source:
    def __init__(self, raw):
        self.raw = raw
        self.text = raw.decode("utf-8", errors="strict")
        self.lines = re.findall(r"[^\r\n]*(?:\r\n|\r|\n|$)", self.text)
        if self.lines and self.lines[-1] == "":
            self.lines.pop()
        self.starts = [0]
        for line in self.lines:
            self.starts.append(self.starts[-1] + len(line))
        self.bytes = [0]
        for char in self.text:
            self.bytes.append(self.bytes[-1] + len(char.encode("utf-8")))

    def span(self, start, end, role="content"):
        if not 0 <= start < end <= len(self.text):
            raise StructuralError(f"invalid source span: {start}:{end}")
        return {
            "source_char_start": start,
            "source_char_end": end,
            "source_byte_start": self.bytes[start],
            "source_byte_end": self.bytes[end],
            "start_line": bisect.bisect_right(self.starts, start),
            "end_line": bisect.bisect_right(self.starts, end - 1),
            "role": role,
        }

    def line_span(self, start, end, role="content"):
        return self.span(self.starts[start], self.starts[end], role)

    def extract(self, span):
        return self.text[span["source_char_start"] : span["source_char_end"]]
