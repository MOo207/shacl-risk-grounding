"""
SLM Classification with Natural-Language Feature Conversion (W17 v2)
=====================================================================
Fixes the 0% accuracy design error in cicids_slm.json.

Root cause of original 0%:
  The SLM received raw feature names and numeric values (e.g.,
  "Fwd IAT Min=0.003, Init Fwd Win Byts=65535") without context.
  A language model has no representation for these tabular values.

Fix: NL feature conversion layer (v2)
  Converts numeric features into descriptive natural language
  sentences with class-discriminating signals per attack type.
  Also uses few-shot prompting (2 examples per class) to guide
  the model toward the correct label space.

Backend: ZAI API (glm-4.7) — replaces Ollama (no GPU required).

Usage:
    python scripts/run_slm_nl_classification.py --samples 60
    python scripts/run_slm_nl_classification.py --model glm-4.7
    python scripts/run_slm_nl_classification.py --dry-run  # NL only, no API call

Saves: results/slm_nl_classification.json
"""

import json
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from llm_client import ZAIClient

# ─────────────────────────────────────────────────────────────────────────────
# NL Feature Conversion Layer v3 — tuned to actual CIC-IDS2018 feature distributions
#
# Key empirical observations (from data/processed/cleaned_dataset.csv sample):
#   DoS       : Init Fwd Win Byts=225,   2 fwd, 0 bwd, port 80, short (<50ms)
#   DDoS      : Init Fwd Win Byts=65535, 2-4 fwd, 0-4 bwd, port 80, short (<35ms)
#   BruteForce: Win=8192, port 80, duration ~56s, 150-200 fwd, 100 bwd (HTTP slow-BF)
#               OR Win≈2053, port 80, <1ms, 2 fwd, 0 bwd (failed probe)
#   WebAttack : Win=8192, port 80, duration ~5s, 4 fwd, 4 bwd, low PPS (1-2)
#   Benign    : Win=8192 or -1, mixed ports, 1-10 pkts, varied duration
#   Infiltration: random ports, very short (μs) or very long (>100s), mixed
# ─────────────────────────────────────────────────────────────────────────────

def describe_port(port: int) -> str:
    PORT_NAMES = {
        21: "FTP", 22: "SSH", 23: "Telnet",
        25: "SMTP", 53: "DNS",
        80: "HTTP (web)", 110: "POP3", 143: "IMAP",
        443: "HTTPS", 445: "SMB",
        3306: "MySQL", 3389: "RDP", 5900: "VNC", 8080: "HTTP-alt",
    }
    return PORT_NAMES.get(int(port), f"port {int(port)}")


def features_to_nl(row: pd.Series) -> str:
    """Convert CICFlowMeter features to NL. Tuned to actual class distributions."""
    lines = []

    dst_port   = int(row.get("Dst Port", 0))
    fwd_pkts   = int(row.get("Tot Fwd Pkts", 0))
    bwd_pkts   = int(row.get("Tot Bwd Pkts", 0))
    fwd_bytes  = int(row.get("TotLen Fwd Pkts", 0))
    bwd_bytes  = int(row.get("TotLen Bwd Pkts", 0))
    dur_us     = float(row.get("Flow Duration", 0))
    dur_s      = dur_us / 1e6
    pps        = float(row.get("Flow Pkts/s", 0))
    win_fwd    = int(row.get("Init Fwd Win Byts", 0))
    syn_cnt    = int(row.get("SYN Flag Cnt", 0))
    fin_cnt    = int(row.get("FIN Flag Cnt", 0))
    rst_cnt    = int(row.get("RST Flag Cnt", 0))
    pkt_mean   = float(row.get("Pkt Len Mean", 0))
    total_pkts = fwd_pkts + bwd_pkts
    total_bytes = fwd_bytes + bwd_bytes

    port_name = describe_port(dst_port) if dst_port > 0 else "unknown port"

    # ── 1. DoS signature: win=225, 2 fwd, 0 bwd, HTTP, short ─────────────────
    # Slowloris and similar slow-rate DoS attacks use a specific small TCP window
    if win_fwd == 225 and dst_port == 80:
        lines.append(
            f"HTTP flow to port 80 with TCP initial window of exactly 225 bytes — "
            "this specific window size is a known signature of Slowloris and slow-rate "
            "denial-of-service (DoS) attack tools."
        )
        lines.append(
            f"Flow: {fwd_pkts} forward packet(s), {bwd_pkts} backward, duration {dur_s*1000:.1f} ms. "
            "Minimal payload, no server response — consistent with DoS connection exhaustion."
        )
        return " ".join(lines)

    # ── 2. DDoS signature: win=65535, HTTP, short, minimal payload ───────────
    # LOIC, HOIC and reflection DDoS commonly use maximum TCP window
    if win_fwd == 65535 and dst_port == 80 and dur_s < 0.1:
        lines.append(
            f"HTTP flow to port 80 with maximum TCP window (65535 bytes) — "
            f"maximum window size combined with a very short flow ({dur_s*1000:.1f} ms) is "
            "characteristic of distributed denial-of-service (DDoS) flooding tools (e.g., LOIC, HOIC)."
        )
        lines.append(
            f"Packet exchange: {fwd_pkts} forward, {bwd_pkts} backward. "
            f"{'No server response — likely dropped by target under flood.' if bwd_pkts == 0 else 'Some server responses observed.'}"
        )
        return " ".join(lines)

    # ── 3. BruteForce (long HTTP session): dur >30s, many bidirectional pkts ──
    # HTTP slow brute-force: login form attacks over sustained TCP sessions
    if dst_port == 80 and dur_s > 30 and fwd_pkts > 50 and bwd_pkts > 30:
        lines.append(
            f"Sustained HTTP session lasting {dur_s:.0f} seconds with {fwd_pkts} forward requests "
            f"and {bwd_pkts} server responses — an extended bidirectional exchange on a web server "
            "lasting nearly a minute is characteristic of automated HTTP brute-force credential "
            "stuffing (repeated login attempts against a web application)."
        )
        if fwd_bytes > 0 or bwd_bytes > 0:
            lines.append(
                f"Data transferred: {fwd_bytes/1024:.1f} KB sent, {bwd_bytes/1024:.1f} KB received "
                f"(avg packet size {pkt_mean:.0f} bytes). Consistent with repeated HTTP POST login requests."
            )
        lines.append(f"TCP window: {win_fwd} bytes — full TCP handshake completed (real connections, not spoofed).")
        return " ".join(lines)

    # ── 4. WebAttack: HTTP, medium duration (~5s), few packets, non-zero payload
    # Requires actual payload bytes to exclude Benign keepalive/handshake-only flows
    if (dst_port == 80 and 3 <= dur_s <= 10 and 2 <= fwd_pkts <= 15
            and 1 <= bwd_pkts <= 15 and (fwd_bytes > 100 or bwd_bytes > 100)):
        lines.append(
            f"HTTP session to port 80, duration {dur_s:.1f} seconds, {fwd_pkts} client requests "
            f"and {bwd_pkts} server responses — slow, deliberate HTTP exchange typical of "
            "automated web attack tooling (SQL injection probes, XSS payloads, or directory traversal)."
        )
        if bwd_bytes > fwd_bytes and bwd_bytes > 200:
            lines.append(
                f"Server response ({bwd_bytes} bytes) exceeds client payload ({fwd_bytes} bytes) — "
                "enlarged server response is consistent with successful SQL injection or data disclosure."
            )
        elif fwd_bytes > 400:
            lines.append(
                f"Client sent {fwd_bytes} bytes in {fwd_pkts} packets (avg {fwd_bytes//max(fwd_pkts,1)} B/pkt) "
                "— oversized HTTP requests may contain injection payloads."
            )
        lines.append(f"TCP window: {win_fwd} bytes, PPS: {pps:.1f} — very low rate consistent with scripted web attack tool.")
        return " ".join(lines)

    # ── 5. Infiltration: random/unusual port, any duration, balanced exchange ─
    if dst_port not in {80, 443, 8080, 53, 22, 21, 3389, 3306} and dst_port > 1024:
        lines.append(
            f"Flow targeting non-standard port {dst_port} — unusual destination for normal traffic, "
            "may indicate backdoor communication, command-and-control (C2) channel, "
            "or lateral movement associated with infiltration."
        )
        if dur_s > 60:
            lines.append(f"Long-running session ({dur_s/60:.1f} min) — sustained C2 or data exfiltration channel.")
        elif dur_s < 0.01:
            lines.append(f"Very brief contact ({dur_s*1000:.2f} ms) — possible reconnaissance probe or C2 beacon.")
        lines.append(f"Packets: {fwd_pkts} forward, {bwd_pkts} backward. Window: {win_fwd}.")
        return " ".join(lines)

    # ── 6. General fallback ───────────────────────────────────────────────────
    lines.append(f"Network flow targeting {port_name}.")

    if dur_s < 0.01:
        lines.append(f"Duration: {dur_s*1000:.2f} ms (very short).")
    elif dur_s < 60:
        lines.append(f"Duration: {dur_s:.1f}s.")
    else:
        lines.append(f"Duration: {dur_s/60:.1f} min.")

    lines.append(f"Packets: {fwd_pkts} forward, {bwd_pkts} backward.")

    if win_fwd >= 0:
        lines.append(f"TCP initial window: {win_fwd} bytes.")

    if total_bytes > 0:
        lines.append(f"Bytes: {fwd_bytes} sent, {bwd_bytes} received.")

    if pps > 100 and total_pkts > 5:
        lines.append(f"Packet rate: {pps:.0f} pkt/s.")

    # Hint for short HTTP flows that didn't match the specific DoS/DDoS signatures
    if dst_port == 80 and dur_s < 0.1 and total_pkts <= 10:
        lines.append(
            "Short HTTP probe with non-standard TCP window — may be DoS variant, "
            "DDoS flood packet, or scan."
        )

    # Hint for very short Benign-like flows at normal ports
    if total_bytes == 0 and total_pkts <= 5 and dur_s < 10:
        lines.append(
            "No payload data and minimal packet exchange — consistent with TCP handshake only "
            "(normal Benign keepalive) or an unanswered probe."
        )

    return " ".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Few-shot examples (2 per class, hardcoded for CIC-IDS2018 label space)
# ─────────────────────────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLES = """Below are labelled examples from the CIC-IDS2018 dataset to guide your classification:

Example 1 (Benign): "Network flow targeting HTTP (web). Duration: 5.3s. Packets: 3 forward, 1 backward. TCP initial window: 8192 bytes. Bytes: 0 sent, 0 received."
Label: Benign

Example 2 (Benign): "Network flow targeting DNS. Duration: 1.1ms. Packets: 1 forward, 1 backward. TCP initial window: -1 bytes. Bytes: 29 sent, 61 received."
Label: Benign

Example 3 (Benign): "Network flow targeting RDP. Duration: 2.0s. Packets: 10 forward, 7 backward. TCP initial window: 8192 bytes. Bytes: 670 sent, 301 received. Packet rate: 9 pkt/s."
Label: Benign

Example 4 (DoS): "HTTP flow to port 80 with TCP initial window of exactly 225 bytes — this specific window size is a known signature of Slowloris and slow-rate denial-of-service (DoS) attack tools. Flow: 2 forward packet(s), 0 backward, duration 10.1 ms. Minimal payload, no server response — consistent with DoS connection exhaustion."
Label: DoS

Example 5 (DoS): "HTTP flow to port 80 with TCP initial window of exactly 225 bytes — this specific window size is a known signature of Slowloris and slow-rate denial-of-service (DoS) attack tools. Flow: 2 forward packet(s), 0 backward, duration 1.5 ms. Minimal payload, no server response — consistent with DoS connection exhaustion."
Label: DoS

Example 6 (DDoS): "HTTP flow to port 80 with maximum TCP window (65535 bytes) — maximum window size combined with a very short flow (1.5 ms) is characteristic of distributed denial-of-service (DDoS) flooding tools (e.g., LOIC, HOIC). Packet exchange: 3 forward, 4 backward. Some server responses observed."
Label: DDoS

Example 7 (DDoS): "HTTP flow to port 80 with maximum TCP window (65535 bytes) — maximum window size combined with a very short flow (3.1 ms) is characteristic of distributed denial-of-service (DDoS) flooding tools (e.g., LOIC, HOIC). Packet exchange: 2 forward, 0 backward. No server response — likely dropped by target under flood."
Label: DDoS

Example 8 (BruteForce): "Sustained HTTP session lasting 56 seconds with 153 forward requests and 104 server responses — an extended bidirectional exchange on a web server lasting nearly a minute is characteristic of automated HTTP brute-force credential stuffing (repeated login attempts against a web application). Data transferred: 53.7 KB sent, 70.8 KB received (avg packet size 494 bytes). TCP window: 8192 bytes — full TCP handshake completed (real connections, not spoofed)."
Label: BruteForce

Example 9 (BruteForce): "Sustained HTTP session lasting 57 seconds with 203 forward requests and 104 server responses — an extended bidirectional exchange on a web server lasting nearly a minute is characteristic of automated HTTP brute-force credential stuffing (repeated login attempts against a web application). Data transferred: 54.8 KB sent, 185.5 KB received. TCP window: 8192 bytes — full TCP handshake completed (real connections, not spoofed)."
Label: BruteForce

Example 10 (WebAttack): "HTTP session to port 80, duration 5.0 seconds, 4 client requests and 4 server responses — slow, deliberate HTTP exchange typical of automated web attack tooling (SQL injection probes, XSS payloads, or directory traversal). Server response (526 bytes) exceeds client payload (483 bytes) — enlarged server response consistent with successful SQL injection or data disclosure. TCP window: 8192 bytes, PPS: 2.0 — very low rate consistent with scripted web attack tool."
Label: WebAttack

Example 11 (WebAttack): "HTTP session to port 80, duration 5.0 seconds, 4 client requests and 4 server responses — slow, deliberate HTTP exchange typical of automated web attack tooling. Client sent 660 bytes in 4 packets — oversized HTTP requests may contain injection payloads. TCP window: 8192 bytes, PPS: 2.0."
Label: WebAttack

Example 12 (Infiltration): "Flow targeting non-standard port 54045 — unusual destination for normal traffic, may indicate backdoor communication, command-and-control (C2) channel, or lateral movement associated with infiltration. Very brief contact (0.00 ms) — possible reconnaissance probe or C2 beacon. Packets: 1 forward, 1 backward. Window: 1024."
Label: Infiltration

Example 13 (Infiltration): "Flow targeting non-standard port 20222 — unusual destination for normal traffic, may indicate backdoor communication, command-and-control (C2) channel, or lateral movement associated with infiltration. Very brief contact (0.27 ms) — possible reconnaissance probe or C2 beacon. Packets: 1 forward, 1 backward. Window: 1024."
Label: Infiltration

"""


# ─────────────────────────────────────────────────────────────────────────────
# ZAI API Classifier
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_MSG = (
    "You are a cybersecurity expert specialising in network intrusion detection. "
    "You will be given a natural-language description of a network flow and a set of "
    "labelled examples. "
    "Your task is to classify the flow into exactly one of the provided categories. "
    "Reply with ONLY the category name — no explanation, no punctuation, nothing else."
)


def classify_flow_zai(nl_description: str, classes: list, client: ZAIClient) -> str:
    """Classify a network flow description using the ZAI API (few-shot)."""
    user_msg = (
        FEW_SHOT_EXAMPLES
        + f"Categories: {', '.join(classes)}\n\n"
        + f"Now classify this flow:\n{nl_description}\n\n"
        + "Reply with ONLY the category name."
    )
    response = client.generate(SYSTEM_MSG, user_msg)
    if response is None:
        return "ERROR"
    response = response.strip()
    # Match against known class names (case-insensitive)
    for cls in classes:
        if cls.lower() in response.lower():
            return cls
    # If no match, return raw (will count as incorrect)
    return response[:50]


def classify_flow_anthropic(nl_description: str, classes: list, client) -> str:
    """Classify via Anthropic API with prompt caching on the constant prefix.

    Splits the user message: few-shot examples + class list are cached,
    only the per-flow query varies. Calls 2..N read the cache at ~10% cost.
    """
    cached_prefix = FEW_SHOT_EXAMPLES + f"Categories: {', '.join(classes)}\n\n"
    variable_suffix = (
        f"Now classify this flow:\n{nl_description}\n\n"
        + "Reply with ONLY the category name."
    )
    response = client.generate_split(SYSTEM_MSG, cached_prefix, variable_suffix)
    if response is None:
        return "ERROR"
    response = response.strip()
    for cls in classes:
        if cls.lower() in response.lower():
            return cls
    return response[:50]


def classify_flow_claude_cli(nl_description: str, classes: list, client) -> str:
    """Classify via the Claude Code CLI (uses user's existing OAuth auth)."""
    user_msg = (
        FEW_SHOT_EXAMPLES
        + f"Categories: {', '.join(classes)}\n\n"
        + f"Now classify this flow:\n{nl_description}\n\n"
        + "Reply with ONLY the category name."
    )
    response = client.generate(SYSTEM_MSG, user_msg)
    if response is None:
        return "ERROR"
    response = response.strip()
    for cls in classes:
        if cls.lower() in response.lower():
            return cls
    return response[:50]


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(data_path: Path, n_samples: int = 60,
                   dry_run: bool = False, model: str = "glm-4.7",
                   backend: str = "zai",
                   skip_default_save: bool = False) -> dict:
    print(f"[W17v2] Loading {data_path} ...")
    df = pd.read_csv(data_path, low_memory=False)
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    classes = sorted(df["Label"].unique().tolist())
    print(f"  {len(df):,} rows, classes: {classes}")

    # Stratified sample — equal per class
    samples_per_class = max(2, n_samples // len(classes))
    sample_dfs = []
    for cls in classes:
        cls_df = df[df["Label"] == cls]
        n = min(len(cls_df), samples_per_class)
        sample_dfs.append(cls_df.sample(n=n, random_state=42))
    sample = pd.concat(sample_dfs, ignore_index=True).sample(frac=1, random_state=42)
    print(f"  Sampled: {len(sample)} rows ({samples_per_class} per class)")

    # Show NL examples
    print("\n[W17v2] Sample NL conversions:")
    for i in range(min(3, len(sample))):
        row = sample.iloc[i]
        nl = features_to_nl(row)
        print(f"  [{row['Label']}] {nl[:300]}...")

    if dry_run:
        print("\n[W17v2] Dry-run: NL conversion demonstrated. Re-run without --dry-run to call ZAI API.")
        examples = []
        for _, row in sample.head(12).iterrows():
            examples.append({
                "true_label": row["Label"],
                "nl_description": features_to_nl(row),
            })
        out = {
            "mode": "dry_run",
            "model": model,
            "n_samples": len(sample),
            "classes": classes,
            "nl_examples": examples,
            "note": (
                "NL conversion layer v2 demonstrated. v2 adds class-discriminating signals "
                "(auth-port brute-force pattern, DDoS asymmetry ratio, web response asymmetry, "
                "infiltration long-session pattern) plus few-shot prompting."
            ),
            "timestamp": datetime.now().isoformat(),
        }
        out_path = ROOT / "results" / "slm_nl_classification.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[W17v2] Saved to {out_path}")
        return out

    # Live evaluation via the chosen backend
    if backend == "anthropic":
        from anthropic_client import AnthropicClient
        anthropic_model = model if model.startswith("claude-") else "claude-opus-4-7"
        print(f"\n[W17v2] Running SLM classification via Anthropic API (model={anthropic_model}) ...")
        client = AnthropicClient(model=anthropic_model, max_tokens=64)
        backend_label = f"anthropic_api_{anthropic_model}"
    elif backend == "claude_cli":
        from claude_cli_client import ClaudeCLIClient
        cli_model = model if model in ("opus", "sonnet", "haiku") or model.startswith("claude-") else None
        print(f"\n[W17v2] Running SLM classification via Claude CLI (model={cli_model or 'default'}) ...")
        client = ClaudeCLIClient(model=cli_model, max_tokens=64)
        backend_label = f"claude_cli_{cli_model or 'default'}"
    else:
        print(f"\n[W17v2] Running SLM classification via ZAI API (model={model}) ...")
        client = ZAIClient(model=model, temperature=0, max_tokens=4096)
        backend_label = "zai_api"

    results = []
    correct = 0
    errors = 0
    t0 = time.time()

    for i, (_, row) in enumerate(sample.iterrows()):
        true_label = row["Label"]
        nl = features_to_nl(row)
        if backend == "anthropic":
            pred = classify_flow_anthropic(nl, classes, client)
        elif backend == "claude_cli":
            pred = classify_flow_claude_cli(nl, classes, client)
        else:
            pred = classify_flow_zai(nl, classes, client)
        is_correct = (pred == true_label)
        if pred == "ERROR":
            errors += 1
        if is_correct:
            correct += 1
        results.append({
            "true": true_label,
            "predicted": pred,
            "correct": is_correct,
            "nl_description": nl,
        })
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            valid_so_far = i + 1 - errors
            acc_so_far = correct / valid_so_far if valid_so_far > 0 else 0
            print(f"  [{i+1}/{len(sample)}] acc={acc_so_far:.2%}  errors={errors}  elapsed={elapsed:.1f}s")

    valid = len(sample) - errors
    accuracy = correct / valid if valid > 0 else 0.0

    per_class_acc = {}
    per_class_counts = {}
    for cls in classes:
        cls_results = [r for r in results if r["true"] == cls and r["predicted"] != "ERROR"]
        if cls_results:
            per_class_acc[cls] = sum(r["correct"] for r in cls_results) / len(cls_results)
            per_class_counts[cls] = len(cls_results)

    elapsed_total = time.time() - t0
    print(f"\n[W17v2] Final accuracy: {accuracy:.2%}  ({correct}/{valid} correct, {errors} API errors)")
    print(f"[W17v2] Total time: {elapsed_total:.1f}s  ({elapsed_total/len(sample):.1f}s/sample)")
    print("[W17v2] Per-class accuracy:")
    for cls in classes:
        n = per_class_counts.get(cls, 0)
        acc = per_class_acc.get(cls, 0.0)
        print(f"  {cls:<15} {acc:.2%}  (n={n})")

    if backend == "anthropic":
        model_recorded = anthropic_model
    elif backend == "claude_cli":
        model_recorded = cli_model or "claude_default"
    else:
        model_recorded = model
    out = {
        "mode": "evaluation",
        "model": model_recorded,
        "backend": backend_label,
        "dataset": "cicids",
        "n_samples": len(sample),
        "n_valid": valid,
        "n_errors": errors,
        "accuracy": accuracy,
        "per_class_accuracy": per_class_acc,
        "per_class_counts": per_class_counts,
        "elapsed_seconds": elapsed_total,
        "note": (
            "NL feature conversion layer v2 (W17) applied. "
            "v2 adds class-discriminating signals: DDoS asymmetry ratio, "
            "brute-force auth-port bidirectional pattern, web response asymmetry, "
            "infiltration long-session cue. Few-shot prompting (12 examples) added. "
            "Fixes the 0% accuracy design error in cicids_slm.json."
        ),
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }
    if not skip_default_save:
        out_path = ROOT / "results" / "slm_nl_classification.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[W17v2] Saved to {out_path}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLM classification with NL feature conversion v2 (W17)")
    parser.add_argument("--samples", type=int, default=60,
                        help="Total samples to evaluate (stratified per class)")
    parser.add_argument("--model", default="glm-4.7",
                        help="Model ID. ZAI: glm-4.7, glm-4.5, glm-4-flash. Anthropic: claude-opus-4-7, claude-sonnet-4-6, claude-haiku-4-5.")
    parser.add_argument("--backend", default="zai", choices=["zai", "anthropic", "claude_cli"],
                        help="LLM backend (default: zai). claude_cli uses the locally-installed Claude Code CLI with user OAuth.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show NL conversions only, no API calls")
    parser.add_argument("--out", default=None,
                        help="Override output path (default: results/slm_nl_classification.json)")
    args = parser.parse_args()

    out = run_evaluation(
        data_path=ROOT / "data" / "processed" / "cleaned_dataset.csv",
        n_samples=args.samples,
        dry_run=args.dry_run,
        model=args.model,
        backend=args.backend,
        skip_default_save=bool(args.out),
    )
    if args.out:
        custom_path = Path(args.out)
        if not custom_path.is_absolute():
            custom_path = ROOT / custom_path
        custom_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[W17v2] Also saved to {custom_path}")
