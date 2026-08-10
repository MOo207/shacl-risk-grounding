"""Anthropic Claude API client for thesis SLM-classification experiments.

Anthropic Claude API client used by the LLM experiments. The SLM
classification scripts can swap backends with --backend anthropic.

Key design choice — prompt caching: the system message and the few-shot
examples are byte-identical across all calls in a single evaluation run.
We mark the system block with cache_control so calls 2..N read the
cached prefix at ~0.1x the input price (vs 1.25x cache-write on the
first call). For N=180 with ~3000 prefix tokens this saves ~87% on
input cost.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import anthropic


class AnthropicClient:
    """Drop-in replacement for ZAIClient that uses the Anthropic Messages API.

    The constructor signature matches ZAIClient for symmetry. Sampling
    parameters from the ZAI side (temperature) are accepted but ignored
    on Opus 4.7 (which removes them); other Claude models accept them.

    Args:
        model: Claude model ID. Defaults to claude-opus-4-7 (most capable).
        temperature: Accepted for API compatibility; only forwarded for
            non-Opus-4.7 models.
        max_tokens: Output cap. 64 is plenty for single-class output.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-7",
        temperature: float = 0,
        max_tokens: int = 64,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    def _supports_sampling_params(self) -> bool:
        """Opus 4.7 removes temperature/top_p/top_k. Other models accept them."""
        return not self.model.startswith("claude-opus-4-7")

    def generate(self, system_msg: str, user_msg: str, retries: int = 5) -> Optional[str]:
        """Call Claude with the constant system prompt cached for reuse.

        Splits system content so the system block (which contains the
        few-shot examples) is marked cache_control:ephemeral. Calls 2..N
        in the same run hit the cache at ~0.1x cost.

        Returns the response text, or None on persistent failure.
        """
        # cache_control on a system block caches the system content.
        # The few-shot examples are appended into the user message in
        # the existing run_slm_*.py scripts, so the entire user message
        # is also constant-prefix + variable-suffix — but to keep the
        # interface compatible, we cache only the system block here and
        # let the prompt builder put few-shot in the user content.
        # Caller can split user_msg into [cached_prefix, variable_suffix]
        # via the .generate_split() method below for maximum savings.
        system_blocks = [
            {
                "type": "text",
                "text": system_msg,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        messages = [{"role": "user", "content": user_msg}]

        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_blocks,
            "messages": messages,
        }
        if self._supports_sampling_params():
            kwargs["temperature"] = self.temperature

        for attempt in range(retries):
            try:
                resp = self.client.messages.create(**kwargs)
                # Response.content is a list of content blocks.
                for block in resp.content:
                    if block.type == "text":
                        return block.text
                return ""  # no text block (shouldn't happen for classification)
            except anthropic.RateLimitError as e:
                wait = 30 * (attempt + 1)
                print(f"  [retry {attempt+1}/{retries-1}] RateLimit: sleeping {wait}s...")
                time.sleep(wait)
            except anthropic.APIStatusError as e:
                if e.status_code >= 500:
                    if attempt < retries - 1:
                        wait = 15 * (attempt + 1)
                        print(f"  [retry {attempt+1}/{retries-1}] {e.status_code}: sleeping {wait}s...")
                        time.sleep(wait)
                    else:
                        print(f"API error after {retries} retries: {e.message}")
                        return None
                else:
                    # 4xx errors (except rate limit) are not retryable
                    print(f"API error (non-retryable): {e.status_code} {e.message}")
                    return None
            except anthropic.APIConnectionError as e:
                if attempt < retries - 1:
                    wait = 15 * (attempt + 1)
                    print(f"  [retry {attempt+1}/{retries-1}] Connection: sleeping {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"Connection error after {retries} retries: {e}")
                    return None
        return None

    def generate_split(
        self,
        system_msg: str,
        cached_prefix: str,
        variable_suffix: str,
        retries: int = 5,
    ) -> Optional[str]:
        """Like generate(), but splits the user message into cached + variable parts.

        Useful for SLM classification where few-shot examples live in the
        user message: pass the few-shot block as cached_prefix and the
        per-flow query as variable_suffix. Caches both the system block
        AND the user-message prefix.

        Args:
            system_msg: System prompt text (cached).
            cached_prefix: Stable user-message prefix, e.g. few-shot examples.
            variable_suffix: Per-call variable content, e.g. the flow description.

        Returns:
            Response text, or None on persistent failure.
        """
        system_blocks = [
            {
                "type": "text",
                "text": system_msg,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        # Two content blocks in the user message; cache the prefix.
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": cached_prefix,
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": variable_suffix},
                ],
            }
        ]

        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_blocks,
            "messages": messages,
        }
        if self._supports_sampling_params():
            kwargs["temperature"] = self.temperature

        for attempt in range(retries):
            try:
                resp = self.client.messages.create(**kwargs)
                # Optional: log cache hit on the first response for debugging.
                if attempt == 0 and hasattr(resp, "usage"):
                    cw = getattr(resp.usage, "cache_creation_input_tokens", 0) or 0
                    cr = getattr(resp.usage, "cache_read_input_tokens", 0) or 0
                    if cw or cr:
                        # Single line, only logged when first attempt has cache activity.
                        pass  # caller can inspect resp if needed
                for block in resp.content:
                    if block.type == "text":
                        return block.text
                return ""
            except anthropic.RateLimitError:
                wait = 30 * (attempt + 1)
                print(f"  [retry {attempt+1}/{retries-1}] RateLimit: sleeping {wait}s...")
                time.sleep(wait)
            except anthropic.APIStatusError as e:
                if e.status_code >= 500:
                    if attempt < retries - 1:
                        wait = 15 * (attempt + 1)
                        print(f"  [retry {attempt+1}/{retries-1}] {e.status_code}: sleeping {wait}s...")
                        time.sleep(wait)
                    else:
                        print(f"API error after {retries} retries: {e.message}")
                        return None
                else:
                    print(f"API error (non-retryable): {e.status_code} {e.message}")
                    return None
            except anthropic.APIConnectionError as e:
                if attempt < retries - 1:
                    wait = 15 * (attempt + 1)
                    time.sleep(wait)
                else:
                    print(f"Connection error after {retries} retries: {e}")
                    return None
        return None
