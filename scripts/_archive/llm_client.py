"""ZAI API client for thesis experiments. OpenAI-compatible API."""
import json
import os
import time
from typing import Optional
import requests

class ZAIClient:
    def __init__(self, model="glm-4.7", temperature=0, max_tokens=2048):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = "https://api.z.ai/api/coding/paas/v4"
        self.api_key = self._load_api_key()

    def _load_api_key(self):
        # 1. Environment variable
        key = os.environ.get("ZAI_API_KEY", "")
        if key:
            return key
        # 2. OpenClaw agent auth-profiles
        auth_paths = [
            os.path.expanduser("~/.openclaw/agents/navi/agent/auth-profiles.json"),
            os.path.expanduser("~/.openclaw/agents/main/agent/auth-profiles.json"),
        ]
        for path in auth_paths:
            try:
                d = json.load(open(path))
                for k, v in d.get("profiles", {}).items():
                    if "zai" in k:
                        return v.get("key", "")
            except:
                continue
        return ""

    def generate(self, system_msg, user_msg, retries=5):
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        for attempt in range(retries):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json={"model": self.model, "messages": messages, "temperature": self.temperature, "max_tokens": self.max_tokens},
                    timeout=120
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                if attempt < retries - 1:
                    wait = 15 * (attempt + 1)  # 15s, 30s, 45s, 60s — longer waits for DNS recovery
                    print(f"  [retry {attempt+1}/{retries-1}] {type(e).__name__}: sleeping {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"API error after {retries} retries: {e}")
                    return None
