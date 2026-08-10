"""Claude Code CLI client for thesis SLM-classification experiments.

Claude CLI client for the LLM experiments. Each call invokes the
`claude` CLI as a subprocess in headless mode (-p / --print) with the
existing user OAuth credentials from the keychain. No API key needed.

Trade-offs vs the Anthropic API client:
  + Uses existing user auth — no API key setup required.
  + Cost is billed against the user's Claude subscription, not API credits.
  - Slower: ~14 seconds per call due to CLI startup + new session per call.
  - No prompt caching (each --no-session-persistence call is fresh).
  - Subject to subscription rate limits, not API rate limits.

For 180 samples this is ~40 minutes wall clock.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import Optional

_D_TEMP = r"D:\temp_claude"
os.makedirs(_D_TEMP, exist_ok=True)


class ClaudeCLIClient:
    """Claude client that shells out to the `claude` CLI.

    Args:
        model: Optional model alias for --model (e.g. "opus", "sonnet",
            "haiku"). If None, the CLI's default is used.
        temperature: Accepted for API compatibility, ignored (CLI does not
            expose sampling parameters).
        max_tokens: Accepted for API compatibility, ignored (CLI determines
            output length per the model defaults).
        timeout_sec: Per-call subprocess timeout in seconds.
        cli_path: Path to the `claude` executable; auto-detected if None.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0,
        max_tokens: int = 64,
        timeout_sec: int = 180,
        cli_path: Optional[str] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_sec = timeout_sec
        self.cli_path = cli_path or shutil.which("claude") or "claude"

    def generate(
        self,
        system_msg: str,
        user_msg: str,
        retries: int = 3,
    ) -> Optional[str]:
        """Invoke `claude -p` with system_msg + user_msg via stdin.

        Each call is fully isolated (no session persistence) so per-flow
        evaluations are independent.

        Returns the response text (stripped), or None on persistent failure.
        """
        cmd = [
            self.cli_path,
            "-p",
            "--no-session-persistence",
            "--output-format", "text",
            "--system-prompt", system_msg,
        ]
        if self.model:
            cmd.extend(["--model", self.model])

        # Redirect TEMP/TMP to D: so the CLI can write even when C: is full.
        _env = os.environ.copy()
        _env["TEMP"] = _D_TEMP
        _env["TMP"] = _D_TEMP

        for attempt in range(retries):
            proc = None
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    encoding="utf-8",
                    errors="replace",
                    env=_env,
                )
                try:
                    stdout, stderr = proc.communicate(
                        input=user_msg, timeout=self.timeout_sec
                    )
                except subprocess.TimeoutExpired:
                    # Force-kill entire process tree on Windows, then on POSIX.
                    try:
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                            capture_output=True,
                        )
                    except Exception:
                        proc.kill()
                    proc.communicate()  # drain pipes
                    if attempt < retries - 1:
                        print(f"  [retry {attempt+1}/{retries-1}] timeout after {self.timeout_sec}s")
                        time.sleep(5)
                        continue
                    print(f"CLI timed out after {retries} retries ({self.timeout_sec}s each)")
                    return None

                if proc.returncode != 0:
                    err = (stderr or "").strip()
                    if attempt < retries - 1:
                        wait = 10 * (attempt + 1)
                        print(f"  [retry {attempt+1}/{retries-1}] CLI exit {proc.returncode}: "
                              f"{err[:120]} — sleeping {wait}s...")
                        time.sleep(wait)
                        continue
                    print(f"CLI failed after {retries} retries: exit {proc.returncode}: {err[:300]}")
                    return None
                response = (stdout or "").strip()
                if not response:
                    if attempt < retries - 1:
                        time.sleep(5)
                        continue
                    return ""
                return response
            except FileNotFoundError:
                print(f"`claude` CLI not found at {self.cli_path!r}. "
                      f"Install via npm install -g @anthropic-ai/claude-code, or pass --cli-path.")
                return None
            except Exception as e:  # pragma: no cover - belt and braces
                if proc is not None:
                    try:
                        proc.kill()
                        proc.communicate()
                    except Exception:
                        pass
                if attempt < retries - 1:
                    time.sleep(10)
                else:
                    print(f"CLI invocation error after {retries} retries: {e!r}")
                    return None
        return None
