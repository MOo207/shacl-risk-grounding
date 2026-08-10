#!/usr/bin/env python3
"""SLM cross-source GRC explanation pipeline.

Uses cloud LLM APIs (ZAI glm-4.7-flash or Alibaba qwen3.5-plus) to generate
audit-ready risk narratives from ML predictions + multi-source context.

Usage:
  python scripts/run_slm_explainer.py --provider zai --samples 100
  python scripts/run_slm_explainer.py --provider alibaba --dataset nslkdd --samples 50
  python scripts/run_slm_explainer.py --provider zai --multisource --samples 10

Output: results/slm_explanations_v2.json (or --output path)
"""
import json
import time
import urllib.request
import ssl
import argparse
import os
import sys
from datetime import datetime

# API configurations (OpenAI-compatible format)
PROVIDERS = {
    "zai": {
        "url": "https://api.z.ai/api/coding/paas/v4/chat/completions",
        "key": "5b611c15e114404e9f3e190abb8b5584.6ufbLh2gJ7OSXV20",
        "model": "glm-4.7-flash",
        "name": "ZAI GLM-4.7-Flash",
    },
    "alibaba": {
        "url": "https://coding-intl.dashscope.aliyuncs.com/v1/chat/completions",
        "key": "sk-sp-7b89902acd404f8aa44258b53ece45c0",
        "model": "qwen3.5-plus",
        "name": "Alibaba Qwen3.5-Plus",
    },
}

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

# Attack type -> multi-source context mapping
ATTACK_CONTEXT = {
    "BruteForce": {
        "asset": "ftp-srv-01 (172.31.69.25), Ubuntu 20.04, vsftpd 3.0.3",
        "criticality": "high",
        "cves": "CVE-2011-2523 (CVSS 9.8, CRITICAL, in CISA KEV)",
        "technique": "T1110 - Brute Force (Credential Access)",
        "nfcrm_clause": "6.7 - Currently applied controls",
        "controls": "Account lockout policy (5 attempts, 30-min timeout)",
    },
    "DoS": {
        "asset": "web-srv-01 (172.31.69.28), Windows Server 2019, IIS 10.0",
        "criticality": "critical",
        "cves": "CVE-2015-1635 (CVSS 9.8, CRITICAL, in CISA KEV), CVE-2021-31166 (CVSS 9.8, CRITICAL)",
        "technique": "T1499 - Endpoint Denial of Service (Impact)",
        "nfcrm_clause": "6.5 - Threat inventory",
        "controls": "WAF rule blocking malformed HTTP Range headers",
    },
    "DDoS": {
        "asset": "fw-dmz-01 (172.31.69.1), pfSense 2.7, CRITICAL",
        "criticality": "critical",
        "cves": "CVE-2023-42326 (CVSS 8.8, HIGH)",
        "technique": "T1498 - Network Denial of Service (Impact)",
        "nfcrm_clause": "6.5 - Threat inventory",
        "controls": "pfSense perimeter firewall (protection degree 0.85) + Suricata IDS (0.75)",
    },
    "WebAttack": {
        "asset": "db-srv-01 (172.31.69.30), Windows Server 2019, SQL Server 2019",
        "criticality": "critical",
        "cves": "CVE-2019-0819 (CVSS 7.5, HIGH), CVE-2022-29143 (CVSS 7.5, HIGH)",
        "technique": "T1190 - Exploit Public-Facing Application (Initial Access)",
        "nfcrm_clause": "6.4 - Vulnerability inventory",
        "controls": "Parameterised queries, ORM enforcement, input validation",
    },
    "Infiltration": {
        "asset": "web-srv-01 + ws-01 + ws-02 (lateral movement chain)",
        "criticality": "critical",
        "cves": "CVE-2021-26855 (CVSS 9.1, CRITICAL, in CISA KEV, ProxyLogon)",
        "technique": "T1570 - Lateral Tool Transfer (Lateral Movement)",
        "nfcrm_clause": "6.6 - Risk scenario development",
        "controls": "Network segmentation, EDR, privileged access management",
    },
    "Bot": {
        "asset": "ws-03, ws-04, ws-05 (workstation cluster, Chrome 121.0)",
        "criticality": "high",
        "cves": "CVE-2024-0519 (CVSS 8.8, HIGH), CVE-2023-2033 (CVSS 8.8, HIGH, in CISA KEV)",
        "technique": "T1583.005 - Acquire Infrastructure: Botnet (Resource Development)",
        "nfcrm_clause": "6.5 - Threat inventory",
        "controls": "DNS sinkholing, outbound HTTP filtering for C2 traffic",
    },
    "Benign": {
        "asset": "general network traffic",
        "criticality": "low",
        "cves": "None",
        "technique": "N/A - legitimate traffic",
        "nfcrm_clause": "6.8 - Acceptable risk level",
        "controls": "Standard monitoring",
    },
}

# SHAP top features per attack type (from shap_top20_features.json)
SHAP_FEATURES = {
    "BruteForce": "Fwd Seg Size Min (0.043), Fwd IAT Mean (0.009), TotLen Fwd Pkts (0.009)",
    "DoS": "Fwd Seg Size Min (0.025), Flow IAT Max (0.005), Flow Pkts/s (0.006)",
    "DDoS": "Fwd Seg Size Min (0.011), Init Fwd Win Byts (0.010), Fwd Header Len (0.008)",
    "WebAttack": "Fwd Seg Size Min (0.031), Fwd IAT Min (0.008), TotLen Fwd Pkts (0.008)",
    "Infiltration": "Fwd Seg Size Min (0.043), Fwd IAT Mean (0.009), TotLen Fwd Pkts (0.009)",
    "Bot": "Fwd Seg Size Min (0.020), Init Fwd Win Byts (0.008), Dst Port (0.007)",
    "Benign": "Fwd Seg Size Min (0.005), Dst Port (0.003), Init Fwd Win Byts (0.002)",
}


def build_prompt(attack_type, confidence, multisource=False):
    """Build prompt with multi-source context."""
    ctx = ATTACK_CONTEXT.get(attack_type, ATTACK_CONTEXT["Benign"])
    shap = SHAP_FEATURES.get(attack_type, "N/A")

    if multisource:
        prompt = f"""You are a cybersecurity GRC risk analyst. Assess this detection:

EVENT: {attack_type} attack detected (ML confidence: {confidence:.0%})
TOP FEATURES: {shap}
ASSET: {ctx['asset']} (Criticality: {ctx['criticality']})
CVEs: {ctx['cves']}
MITRE ATT&CK: {ctx['technique']}
NFCRM-1:2025: Section {ctx['nfcrm_clause']}
CURRENT CONTROLS: {ctx['controls']}

Respond with JSON only:
{{"risk_level": "Critical|High|Medium|Low", "risk_score": N, "justification": "...", "affected_assets": ["..."], "recommended_controls": "...", "nfcrm_action": "...", "audit_narrative": "..."}}"""
    else:
        prompt = f"""GRC risk assessment for {attack_type} (confidence {confidence:.0%}).
Asset: {ctx['asset']}. NFCRM-1:2025 {ctx['nfcrm_clause']}.
Top SHAP features: {shap}.
Respond JSON: {{"risk_level":"...","audit_narrative":"..."}}"""

    return prompt


def query_llm(prompt, provider_name="zai"):
    """Send prompt to cloud LLM API (OpenAI-compatible format)."""
    provider = PROVIDERS[provider_name]
    payload = json.dumps({
        "model": provider["model"],
        "messages": [
            {"role": "system", "content": "You are a cybersecurity GRC risk analyst. Always respond with valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 500,
        "temperature": 0.3,
    })
    req = urllib.request.Request(
        provider["url"],
        data=payload.encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {provider['key']}",
        },
    )
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, context=SSL_CTX, timeout=60)
        data = json.loads(resp.read().decode())
        latency = time.time() - t0
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        return {
            "response": content,
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "provider": provider_name,
            "model": provider["model"],
            "wall_time_sec": round(latency, 1),
            "success": bool(content.strip()),
        }
    except Exception as e:
        latency = time.time() - t0
        # Fallback to other provider
        if provider_name == "zai":
            try:
                return query_llm(prompt, "alibaba")
            except Exception:
                pass
        return {
            "response": "",
            "error": str(e),
            "provider": provider_name,
            "model": provider["model"],
            "wall_time_sec": round(latency, 1),
            "success": False,
        }


def try_parse_json(text):
    """Try to extract JSON from SLM response."""
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find JSON block
    for start_char in ["{", "["]:
        idx = text.find(start_char)
        if idx >= 0:
            end_char = "}" if start_char == "{" else "]"
            # Find matching close
            depth = 0
            for i in range(idx, len(text)):
                if text[i] == start_char:
                    depth += 1
                elif text[i] == end_char:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[idx:i + 1])
                        except json.JSONDecodeError:
                            break
    return None


def generate_samples(dataset="cicids", num_samples=100):
    """Generate sample attack types with confidence scores for testing."""
    import random
    random.seed(42)

    if dataset == "cicids":
        attack_types = ["Benign", "BruteForce", "DoS", "DDoS", "WebAttack", "Infiltration", "Bot"]
        weights = [0.40, 0.10, 0.15, 0.15, 0.05, 0.10, 0.05]
    else:  # nslkdd
        attack_types = ["Benign", "DoS", "BruteForce", "Infiltration", "Bot"]
        weights = [0.40, 0.30, 0.15, 0.10, 0.05]

    samples = []
    for i in range(num_samples):
        attack = random.choices(attack_types, weights=weights, k=1)[0]
        confidence = random.uniform(0.70, 0.99) if attack != "Benign" else random.uniform(0.85, 0.99)
        samples.append({
            "sample_id": i,
            "attack_type": attack,
            "ml_confidence": round(confidence, 3),
        })
    return samples


def main():
    parser = argparse.ArgumentParser(description="SLM Cross-Source GRC Explainer")
    parser.add_argument("--provider", default="zai", choices=["zai", "alibaba"])
    parser.add_argument("--dataset", default="cicids", choices=["cicids", "nslkdd"])
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--multisource", action="store_true", help="Full multi-source context")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.output is None:
        results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
        if args.multisource:
            args.output = os.path.join(results_dir, "slm_multisource_demo.json")
        elif args.dataset == "nslkdd":
            args.output = os.path.join(results_dir, "nslkdd_slm_explanations.json")
        else:
            args.output = os.path.join(results_dir, "slm_explanations_v2.json")

    provider = PROVIDERS[args.provider]
    print(f"SLM Explainer: {provider['name']} ({provider['model']})")
    print(f"Dataset: {args.dataset}, Samples: {args.samples}, Multisource: {args.multisource}")
    print(f"Output: {args.output}")
    print(f"Started: {datetime.now().isoformat()}")

    samples = generate_samples(args.dataset, args.samples)
    results = []
    successes = 0
    total_latency = 0

    for i, sample in enumerate(samples):
        prompt = build_prompt(sample["attack_type"], sample["ml_confidence"], args.multisource)

        print(f"[{i + 1}/{len(samples)}] {sample['attack_type']} "
              f"(conf={sample['ml_confidence']:.2f})...", end=" ", flush=True)

        resp = query_llm(prompt, args.provider)
        parsed = try_parse_json(resp["response"])

        result = {
            **sample,
            "prompt_length": len(prompt),
            "response": resp["response"][:2000],
            "parsed_json": parsed,
            "parse_success": parsed is not None,
            "latency_sec": resp["wall_time_sec"],
            "input_tokens": resp.get("input_tokens", 0),
            "output_tokens": resp.get("output_tokens", 0),
            "provider": resp.get("provider", args.provider),
            "model": resp.get("model", ""),
            "success": resp["success"],
        }
        results.append(result)

        if resp["success"]:
            successes += 1
        total_latency += resp["wall_time_sec"]

        status = "OK" if resp["success"] else "FAIL"
        parsed_status = "JSON" if parsed else "raw"
        print(f"{status} ({resp['wall_time_sec']:.0f}s, {parsed_status})")

        # Progress save every 10 samples
        if (i + 1) % 10 == 0:
            _save_output(args, results, successes, total_latency, partial=True)

    _save_output(args, results, successes, total_latency, partial=False)
    print(f"\nDone! {successes}/{len(samples)} successful "
          f"({100 * successes / max(len(samples), 1):.0f}%)")


def _save_output(args, results, successes, total_latency, partial=False):
    n = len(results)
    output = {
        "generated": datetime.now().isoformat(),
        "model": PROVIDERS[args.provider]["model"],
        "dataset": args.dataset,
        "multisource": args.multisource,
        "total_samples": n,
        "success_count": successes,
        "success_rate": round(100 * successes / max(n, 1), 1),
        "parse_success_count": sum(1 for r in results if r["parse_success"]),
        "average_latency_seconds": round(total_latency / max(n, 1), 1),
        "total_time_seconds": round(total_latency, 1),
        "partial": partial,
        "explanations": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    tag = " (partial)" if partial else ""
    print(f"  [SAVED{tag}] {args.output} ({n} samples)")


if __name__ == "__main__":
    main()
