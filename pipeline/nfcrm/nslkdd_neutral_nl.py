"""Neutral (non-leaking) NL-description generator for NSL-KDD connection records.

Context: the original generator (scripts/_archive/run_slm_nslkdd_classification.py
:: features_to_nl_nslkdd()) inserts class-revealing tokens directly into the
prose it hands to the LLM -- named-attack substrings ("httptunnel", "mailbomb",
"warezmaster", "snmpguess"/"snmpgetattack", "rootkit installation", the
literal 'back' denial-of-service phrase) and literal class-name / class-category
phrases ("(Probe)", "(U2R)", "(R2L)", "(DoS)", "denial-of-service",
"network reconnaissance", "user-to-root", "remote-to-local"). An audit
(results/nslkdd_ablation/leakage_tagging_clean_subset.json) found 30/50 of the
existing safety-evaluation test cases leak the answer this way, and the
flagship safety finding's significance collapses once leaky cases are excluded.

This module is a REPLACEMENT generator for a clean rerun. It converts the same
NSL-KDD tabular fields the original generator reads (protocol, service, flag,
byte counts, connection/error-rate statistics, login/privilege indicators)
into neutral, objective prose that reports OBSERVED VALUES ONLY -- no
interpretive attack-class language, no named-attack substrings, no hedge
words that imply a verdict (e.g. "flood", "scan", "brute-force",
"exfiltration", "privilege escalation", "reconnaissance", "denial-of-service").
The LLM must infer the attack class from the numbers itself, exactly as a
human analyst would from a raw connection-log line.

Validate with `scripts/build_nslkdd_unified_rerun.py`'s leakage checker, which
reuses the same regex patterns as scripts/tag_nslkdd_leakage_and_clean_subset.py.
"""
from __future__ import annotations

from typing import Any


def _get(row: Any, key: str, default):
    """Works for both pandas.Series and plain dict rows."""
    if hasattr(row, "get"):
        v = row.get(key, default)
    else:
        v = row[key] if key in row else default
    return default if v is None else v


def features_to_nl_neutral(row: Any) -> str:
    """Convert an NSL-KDD 41-feature row to a neutral, non-leaking NL description.

    Reports the same underlying signal families as the original generator
    (protocol/service/flag, byte counts, connection counts, error/diversity
    rates, login & privilege indicators) but purely as observed numbers --
    no attack-class name, category tag, or named-attack substring anywhere
    in the output text.
    """
    protocol     = str(_get(row, "protocol_type", "tcp")).lower()
    service      = str(_get(row, "service", "other")).lower()
    flag         = str(_get(row, "flag", "SF")).upper()
    duration     = int(_get(row, "duration", 0))
    src_bytes    = int(_get(row, "src_bytes", 0))
    dst_bytes    = int(_get(row, "dst_bytes", 0))
    land         = int(_get(row, "land", 0))
    wrong_frag   = int(_get(row, "wrong_fragment", 0))
    hot          = int(_get(row, "hot", 0))
    num_failed   = int(_get(row, "num_failed_logins", 0))
    logged_in    = int(_get(row, "logged_in", 0))
    num_comp     = int(_get(row, "num_compromised", 0))
    root_shell   = int(_get(row, "root_shell", 0))
    su_attempted = int(_get(row, "su_attempted", 0))
    num_root     = int(_get(row, "num_root", 0))
    num_shells   = int(_get(row, "num_shells", 0))
    num_access   = int(_get(row, "num_access_files", 0))
    is_guest     = int(_get(row, "is_guest_login", 0))
    count        = int(_get(row, "count", 0))
    srv_count    = int(_get(row, "srv_count", 0))
    serror_rate  = float(_get(row, "serror_rate", 0.0))
    srv_serror   = float(_get(row, "srv_serror_rate", 0.0))
    rerror_rate  = float(_get(row, "rerror_rate", 0.0))
    srv_rerror   = float(_get(row, "srv_rerror_rate", 0.0))
    same_srv     = float(_get(row, "same_srv_rate", 1.0))
    diff_srv     = float(_get(row, "diff_srv_rate", 0.0))
    dst_h_count  = int(_get(row, "dst_host_count", 0))
    dst_h_srv    = int(_get(row, "dst_host_srv_count", 0))
    dst_h_same   = float(_get(row, "dst_host_same_srv_rate", 1.0))
    dst_h_diff   = float(_get(row, "dst_host_diff_srv_rate", 0.0))
    dst_h_serr   = float(_get(row, "dst_host_serror_rate", 0.0))
    dst_h_rerr   = float(_get(row, "dst_host_rerror_rate", 0.0))

    parts: list[str] = []

    parts.append(
        f"Connection record: protocol={protocol.upper()}, service='{service}', "
        f"flag={flag}, duration={duration}s."
    )
    parts.append(
        f"Data volume: sent={src_bytes}B (client->server), received={dst_bytes}B (server->client)."
    )
    parts.append(
        f"Same-host activity in the observation window: {count} connection(s) to the same "
        f"destination host, {srv_count} to the same service."
    )
    parts.append(
        f"Service diversity: same-service rate={same_srv:.0%}, different-service rate={diff_srv:.0%}."
    )
    parts.append(
        f"Error rates on this connection: SYN-error rate={serror_rate:.0%} "
        f"(service-level={srv_serror:.0%}), REJ-error rate={rerror_rate:.0%} "
        f"(service-level={srv_rerror:.0%})."
    )
    parts.append(
        f"Destination-host 2-second window: {dst_h_count} total connection(s), "
        f"{dst_h_srv} to the same service; same-service rate={dst_h_same:.0%}, "
        f"different-service rate={dst_h_diff:.0%}; SYN-error rate={dst_h_serr:.0%}, "
        f"REJ-error rate={dst_h_rerr:.0%}."
    )

    session_bits = []
    session_bits.append(f"authenticated session={'yes' if logged_in == 1 else 'no'}")
    if is_guest:
        session_bits.append("guest login=yes")
    if num_failed > 0:
        session_bits.append(f"failed login attempts={num_failed}")
    if hot > 0:
        session_bits.append(f"hot indicator count={hot}")
    if root_shell:
        session_bits.append("root_shell flag=set")
    if su_attempted:
        session_bits.append("su_attempted flag=set")
    if num_root > 0:
        session_bits.append(f"num_root={num_root}")
    if num_shells > 0:
        session_bits.append(f"num_shells={num_shells}")
    if num_access > 0:
        session_bits.append(f"num_access_files={num_access}")
    if num_comp > 0:
        session_bits.append(f"system_state_flag_count={num_comp}")
    if wrong_frag > 0:
        session_bits.append(f"wrong_fragment_count={wrong_frag}")
    if land:
        session_bits.append("land flag=set (source host equals destination host)")
    parts.append("Session/host indicators: " + "; ".join(session_bits) + ".")

    return " ".join(parts)


__all__ = ["features_to_nl_neutral"]
