"""Turn streamed LLM text into speakable sentence chunks."""

from __future__ import annotations

import re
from typing import AsyncIterator

_SENTENCE_END = re.compile(r"([.!?…]+[\"')\]]*)(\s+|$)")
_CLAUSE_END = re.compile(r"([,;:—])(\s+)")
_MARKDOWN = re.compile(r"[*_`#>|~]+")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0000FE0F\U0001F1E6-\U0001F1FF]")
_SPEAKER_TAG = re.compile(r"^\s*\[?\*{0,2}kiki\*{0,2}\]?\s*:\s*", re.IGNORECASE)


_SOURCES = re.compile(r"\n\s*(sources?|references?)\s*:.*", re.IGNORECASE | re.DOTALL)
_MD_LINK = re.compile(r"\[([^\]]*)\]\((https?://[^)]*)\)")
_URL = re.compile(r"\(?https?://\S+\)?")


def strip_sources(text: str) -> str:
    """Drop a trailing 'Sources:' list, markdown links (keep their text) and bare URLs."""
    text = _SOURCES.sub("", text)
    text = _MD_LINK.sub(r"\1", text)
    return _URL.sub("", text).strip()


def clean(text: str) -> str:
    """Strip things a TTS shouldn't read out: markdown, emoji, a leading '[Kiki]:'."""
    text = strip_sources(text)
    text = _SPEAKER_TAG.sub("", text)
    text = _MARKDOWN.sub("", text)
    text = _EMOJI.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


async def sentences(
    deltas: AsyncIterator[str], min_chars: int = 12, first_clause_chars: int = 30
) -> AsyncIterator[str]:
    """Yield cleaned sentences as soon as each one is complete.

    Very short sentences ("Oh!") are merged with the next so TTS doesn't produce
    choppy audio; the first chunk is still emitted early to keep latency down.
    """
    buf = ""
    first = True
    async for d in deltas:
        buf += d
        while True:
            m = _SENTENCE_END.search(buf)
            if not m and first:
                # get the first audio out sooner: a long-enough opening clause will do
                c = _CLAUSE_END.search(buf, first_clause_chars)
                if c:
                    m = c
            if not m:
                break
            cut = m.end(1)
            candidate = clean(buf[:cut])
            if len(candidate) < min_chars and not first:
                # too short: try to extend to the next sentence end
                nxt = _SENTENCE_END.search(buf, m.end())
                if not nxt:
                    break
                cut = nxt.end(1)
                candidate = clean(buf[:cut])
            buf = buf[cut:].lstrip()
            if candidate:
                first = False
                yield candidate
    tail = clean(buf)
    if tail:
        yield tail
