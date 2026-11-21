"""Minimal async client for a llama.cpp server (OpenAI-compatible chat API)."""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx


class LLM:
    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self._http = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def stream(
        self, messages: list[dict], max_tokens: int = 200, temperature: float = 0.7, slot: int | None = None
    ) -> AsyncIterator[str]:
        """Yield text deltas as they're generated. `slot` pins a llama.cpp slot so
        its KV cache (warmed by `prewarm`) is reused."""
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature,
                "stream": True, "cache_prompt": True}
        if slot is not None:
            body["id_slot"] = slot
        async with self._http.stream("POST", "/v1/chat/completions", json=body) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                delta = json.loads(data)["choices"][0].get("delta", {}).get("content")
                if delta:
                    yield delta

    async def prewarm(self, messages: list[dict], slot: int | None = None) -> None:
        """Process the prompt into the slot's KV cache without really generating,
        so the next real request only pays for the new tail."""
        body = {"messages": messages, "max_tokens": 1, "cache_prompt": True}
        if slot is not None:
            body["id_slot"] = slot
        try:
            (await self._http.post("/v1/chat/completions", json=body)).raise_for_status()
        except httpx.HTTPError as e:
            print(f"[llm] prewarm failed: {e!r}", flush=True)

    async def json(self, messages: list[dict], schema: dict, max_tokens: int = 200) -> dict:
        """One completion constrained to a JSON schema (grammar-based in llama.cpp)."""
        body = {
            "messages": messages, "max_tokens": max_tokens, "temperature": 0.0, "cache_prompt": True,
            "response_format": {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}},
        }
        r = await self._http.post("/v1/chat/completions", json=body)
        r.raise_for_status()
        return json.loads(r.json()["choices"][0]["message"]["content"])

    async def healthy(self) -> bool:
        try:
            return (await self._http.get("/health")).status_code == 200
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        await self._http.aclose()
