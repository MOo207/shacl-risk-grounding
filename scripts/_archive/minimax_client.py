"""MiniMax API client — OpenAI-compatible REST interface.

Reads the API key from env var MINIMAX_API_KEY.

Usage:
    export MINIMAX_API_KEY="your-key"
    client = MiniMaxClient(model="MiniMax-Text-01")
    text = client.generate(system_msg, user_msg)

MiniMax API is OpenAI-compatible:
    POST https://api.minimaxi.chat/v1/chat/completions
    Authorization: Bearer <key>
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

import requests


class MiniMaxClient:
    """Drop-in for ZAIClient / ClaudeCLIClient using the MiniMax REST API."""

    DEFAULT_BASE_URL = "https://api.minimaxi.chat/v1"
    DEFAULT_MODEL    = "MiniMax-Text-01"

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0,
        max_tokens: int = 2000,
        timeout_sec: int = 120,
        base_url: Optional[str] = None,
    ) -> None:
        self.model       = model or self.DEFAULT_MODEL
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.timeout_sec = timeout_sec
        self.base_url    = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.api_key     = os.environ.get("MINIMAX_API_KEY", "")
        if not self.api_key:
            raise EnvironmentError(
                "MINIMAX_API_KEY not set. "
                "Run: $env:MINIMAX_API_KEY='your-key'  (PowerShell) "
                "or:  export MINIMAX_API_KEY='your-key'  (bash)"
            )

    def generate(
        self,
        system_msg: str,
        user_msg: str,
        retries: int = 5,
    ) -> Optional[str]:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ]
        payload = {
            "model":       self.model,
            "messages":    messages,
            "temperature": self.temperature,
            "max_tokens":  self.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }
        url = f"{self.base_url}/chat/completions"

        for attempt in range(retries):
            try:
                resp = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_sec,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except requests.HTTPError as e:
                status = getattr(e.response, "status_code", None)
                body   = getattr(e.response, "text", "")[:200]
                if status == 429:
                    wait = 30 * (attempt + 1)
                    print(f"  [retry {attempt+1}/{retries-1}] rate-limit 429 — sleeping {wait}s...")
                    time.sleep(wait)
                elif attempt < retries - 1:
                    wait = 15 * (attempt + 1)
                    print(f"  [retry {attempt+1}/{retries-1}] HTTP {status}: {body} — sleeping {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"MiniMax API error after {retries} retries: HTTP {status}: {body}")
                    return None
            except Exception as e:
                if attempt < retries - 1:
                    wait = 15 * (attempt + 1)
                    print(f"  [retry {attempt+1}/{retries-1}] {type(e).__name__}: sleeping {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"MiniMax API error after {retries} retries: {e}")
                    return None
        return None
