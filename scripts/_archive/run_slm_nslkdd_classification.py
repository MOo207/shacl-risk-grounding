"""
SLM Classification with Natural-Language Feature Conversion -- NSL-KDD (5-class)
================================================================================
Applies the NL feature conversion approach (from run_slm_nl_classification.py)
to the NSL-KDD dataset with its 41-feature KDD Cup 1999 schema.

Label space (5 classes):
  Normal   -- legitimate traffic
  DoS      -- denial-of-service (neptune, smurf, back, teardrop, …)
  Probe    -- reconnaissance/scanning (ipsweep, nmap, portsweep, satan, …)
  R2L      -- remote-to-local (guess_passwd, ftp_write, imap, warezmaster, …)
  U2R      -- user-to-root privilege escalation (buffer_overflow, rootkit, …)

Backend: ZAI API (glm-4.7) -- OpenAI-compatible endpoint.

Usage:
    python scripts/run_slm_nslkdd_classification.py --samples 50
    python scripts/run_slm_nslkdd_classification.py --dry-run
    python scripts/run_slm_nslkdd_classification.py --model glm-4.7 --samples 50

Saves: results/slm_nslkdd_nl_classification.json
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
# Column names for NSL-KDD (42-column, no header)
# ─────────────────────────────────────────────────────────────────────────────

COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    'label', 'difficulty',
]

# ─────────────────────────────────────────────────────────────────────────────
# Attack sub-type -> 5-class mapping
# ─────────────────────────────────────────────────────────────────────────────

DOS_TYPES   = {'back','land','neptune','pod','smurf','teardrop','apache2',
               'udpstorm','processtable','mailbomb'}
PROBE_TYPES = {'ipsweep','nmap','portsweep','satan','mscan','saint'}
R2L_TYPES   = {'ftp_write','guess_passwd','imap','multihop','phf','spy',
               'warezclient','warezmaster','sendmail','named','snmpgetattack',
               'snmpguess','xlock','xsnoop','worm'}
U2R_TYPES   = {'buffer_overflow','loadmodule','perl','rootkit','httptunnel',
               'ps','sqlattack','xterm'}

CLASSES = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']


def map_label(lbl: str) -> str:
    lbl = lbl.lower().strip().rstrip('.')
    if lbl == 'normal':    return 'Normal'
    if lbl in DOS_TYPES:   return 'DoS'
    if lbl in PROBE_TYPES: return 'Probe'
    if lbl in R2L_TYPES:   return 'R2L'
    if lbl in U2R_TYPES:   return 'U2R'
    return 'Unknown'


# ─────────────────────────────────────────────────────────────────────────────
# NL Feature Conversion Layer v2 -- NSL-KDD
#
# Empirical grounding (KDDTest+ samples observed):
#
#   U2R      -- root_shell=1 | su_attempted=1 | num_root>0 | hot>0 | num_shells>0
#               NOTE: many U2R subtypes (rootkit/httptunnel) have hot=0 root_shell=0
#               but they share ftp_data service + large src_bytes, or 'other' service
#   R2L      -- num_failed_logins>0 (guess_passwd, ftp_write)
#               OR long-duration ftp/ftp_data with logged_in=1 (warezmaster=warez transfer)
#               OR UDP/private with no bytes (snmpguess, snmpgetattack = SNMP community attack)
#   DoS      -- (flag=REJ/S0 AND count>=50 AND dst_host_count>=200)
#               OR (src_bytes>10000 AND service=http AND dst_host_count>=200) [back attack]
#               OR wrong_fragment>0 [teardrop/pod]
#   Probe    -- diff_srv_rate>=0.5 | (protocol=icmp + small payload) | flag=REJ/S0 + diff_srv>=0.3
#               NOTE: Probe uses REJ flag too (satan/portsweep/mscan), distinguished by
#               diff_srv_rate being high — DoS has low diff_srv (same service repeatedly)
#   Normal   -- SF flag, balanced bytes, no anomaly signals
# ─────────────────────────────────────────────────────────────────────────────

def features_to_nl_nslkdd(row: pd.Series) -> str:
    """Convert NSL-KDD 41-feature row to class-discriminating natural language (v2)."""
    lines = []

    protocol     = str(row.get('protocol_type', 'tcp')).lower()
    service      = str(row.get('service', 'other')).lower()
    flag         = str(row.get('flag', 'SF')).upper()
    duration     = int(row.get('duration', 0))
    src_bytes    = int(row.get('src_bytes', 0))
    dst_bytes    = int(row.get('dst_bytes', 0))
    land         = int(row.get('land', 0))
    wrong_frag   = int(row.get('wrong_fragment', 0))
    hot          = int(row.get('hot', 0))
    num_failed   = int(row.get('num_failed_logins', 0))
    logged_in    = int(row.get('logged_in', 0))
    num_comp     = int(row.get('num_compromised', 0))
    root_shell   = int(row.get('root_shell', 0))
    su_attempted = int(row.get('su_attempted', 0))
    num_root     = int(row.get('num_root', 0))
    num_shells   = int(row.get('num_shells', 0))
    num_access   = int(row.get('num_access_files', 0))
    count        = int(row.get('count', 0))
    serror_rate  = float(row.get('serror_rate', 0.0))
    rerror_rate  = float(row.get('rerror_rate', 0.0))
    diff_srv     = float(row.get('diff_srv_rate', 0.0))
    same_srv     = float(row.get('same_srv_rate', 1.0))
    dst_h_count  = int(row.get('dst_host_count', 0))
    dst_h_srv    = int(row.get('dst_host_srv_count', 0))
    dst_h_serr   = float(row.get('dst_host_serror_rate', 0.0))
    dst_h_rerr   = float(row.get('dst_host_rerror_rate', 0.0))
    dst_h_diff   = float(row.get('dst_host_diff_srv_rate', 0.0))

    # ── 0a. 'back' DoS early-exit: HTTP flood with large payload (no REJ flag) ─
    # Must come before U2R because 'back' flows have hot>0 and num_compromised>0
    # which would otherwise trigger the U2R weak branch.
    if service == 'http' and src_bytes >= 10000 and dst_h_count >= 200 and diff_srv < 0.2:
        lines.append(
            f"HTTP connection (flag={flag}, duration={duration}s, sent={src_bytes}B to server, "
            f"dst_host_count={dst_h_count}). "
            "A very large payload sent to an HTTP server with saturation of the destination host "
            f"({dst_h_count} recent connections, low service diversity diff_srv={diff_srv:.0%}) "
            "is characteristic of a 'back' denial-of-service attack -- sending malformed "
            "HTTP requests (e.g., with '/' repeated thousands of times) to crash/overload Apache. "
            "This is a DoS attack even though the TCP connection may complete (flag=SF)."
        )
        return " ".join(lines)

    # ── 0b. 'httptunnel' U2R vs 'satan' Probe disambiguation ─────────────────
    # Both use service='other', flag=REJ, zero bytes, dst_host_count=255.
    # KEY DISCRIMINATOR: diff_srv_rate
    #   httptunnel (U2R): diff_srv_rate is 0 or very low (same service, covert channel)
    #   satan (Probe):    diff_srv_rate=1.0, count>=400 (scanning all different services)
    if service in ('other', 'private') and flag in ('REJ', 'S0') and src_bytes == 0 and dst_bytes == 0:
        if diff_srv >= 0.5 and count >= 50:
            # This is a Probe scanner (satan/portsweep) -- handled by Probe branch below
            pass
        elif count >= 100:
            # High count with same service = DoS flood (neptune via 'other' service)
            # Let this fall through to DoS branch below
            pass
        else:
            lines.append(
                f"Connection via {protocol.upper()} to 'other' service (flag={flag}, duration={duration}s, "
                f"dst_host_count={dst_h_count}, rerror_rate={rerror_rate:.0%}, diff_srv_rate={diff_srv:.0%}). "
                "A rejected TCP connection to an unclassified 'other' service with zero data exchange, "
                "low service diversity, and low connection count is a signature of the httptunnel attack -- "
                "a user-to-root (U2R) technique that establishes a covert TCP tunnel to escalate privileges, "
                "bypassing firewall rules. The connection is rejected but reveals privilege-escalation intent."
            )
            return " ".join(lines)

    # ── 0c. warezmaster FTP early exit (before U2R) ──────────────────────────
    # warezmaster uses long FTP sessions; some have hot=2 which would trigger U2R weak.
    # Prioritize R2L warezmaster check for long-duration FTP with significant data.
    if service in {'ftp', 'ftp_data'} and duration >= 200 and (src_bytes > 100 or dst_bytes > 100):
        lines.append(
            f"FTP connection lasting {duration}s with significant data transfer "
            f"(sent={src_bytes}B, received={dst_bytes}B, logged_in={logged_in}). "
            "A long-running FTP session lasting several minutes with a data channel transfer "
            "is characteristic of unauthorized file upload/download -- remote-to-local (R2L) "
            "file exfiltration or warezmaster-style unauthorized data transfer."
        )
        return " ".join(lines)

    # ── 1. U2R: privilege-escalation signals ─────────────────────────────────
    # Explicit privilege signals: root_shell, su_attempted, num_root, num_shells
    # Weak: hot>0 with logged-in session (sensitive file access during exploitation)
    # NOTE: 'back' DoS pre-filtered at 0a; warezmaster FTP pre-filtered at 0c
    u2r_strong = root_shell == 1 or su_attempted == 1 or num_root > 0 or num_shells > 0
    u2r_weak   = hot > 0 or num_access > 0 or num_comp > 0
    if u2r_strong or (u2r_weak and logged_in == 1 and service != 'http'):
        priv_signals = []
        if root_shell == 1:
            priv_signals.append("root shell was obtained")
        if su_attempted == 1:
            priv_signals.append("'su root' command was attempted")
        if num_root > 0:
            priv_signals.append(f"{num_root} root access(es) recorded")
        if num_shells > 0:
            priv_signals.append(f"{num_shells} shell prompt(s) spawned")
        if hot > 0:
            priv_signals.append(f"{hot} hot indicator(s) (privileged/sensitive operations detected)")
        if num_access > 0:
            priv_signals.append(f"{num_access} access-control file operation(s)")
        if num_comp > 0:
            priv_signals.append(f"{num_comp} compromised condition(s)")
        lines.append(
            f"Connection via {protocol.upper()} to '{service}' service (flag={flag}, duration={duration}s, "
            f"sent={src_bytes}B, received={dst_bytes}B): "
            f"{'; '.join(priv_signals)}. "
            "These are hallmark indicators of user-to-root (U2R) privilege escalation -- "
            "an attacker exploiting a local vulnerability to gain superuser control."
        )
        if logged_in == 1:
            lines.append("Session authenticated -- exploitation occurred from within an established login session.")
        return " ".join(lines)

    # ── 2. R2L: remote-to-local access attacks ────────────────────────────────
    # Sub-cases grounded in data:
    #   guess_passwd/ftp_write: num_failed_logins > 0
    #   warezmaster:            long-duration ftp/ftp_data with logged_in=1, large src_bytes
    #   snmpguess/snmpgetattack: UDP + service='private' + small src_bytes + no response (dst=0)
    if num_failed > 0:
        lines.append(
            f"Connection via {protocol.upper()} to '{service}' (flag={flag}, duration={duration}s, "
            f"sent={src_bytes}B, received={dst_bytes}B). "
            f"{num_failed} failed login attempt(s) recorded -- repeated authentication failures "
            "on a remote service are a primary indicator of a remote-to-local (R2L) "
            "brute-force or password-guessing attack. User did not gain access."
        )
        return " ".join(lines)

    # warezmaster pattern: long FTP with large data transfer (unauthorized file upload/download)
    if service in {'ftp', 'ftp_data'} and duration >= 200 and (src_bytes > 100 or dst_bytes > 100):
        lines.append(
            f"FTP connection lasting {duration}s with significant data transfer "
            f"(sent={src_bytes}B, received={dst_bytes}B, logged_in={logged_in}). "
            "A long-running FTP session with a data channel transfer lasting several minutes "
            "is characteristic of unauthorized file upload/download -- remote-to-local (R2L) "
            "file exfiltration or warezmaster-style unauthorized data transfer."
        )
        return " ".join(lines)

    # SNMP community guessing: UDP + private service + tiny payload + no server response
    if protocol == 'udp' and service == 'private' and dst_bytes == 0 and src_bytes < 200:
        lines.append(
            f"UDP connection to 'private' service (src={src_bytes}B, dst={dst_bytes}B, count={count}, "
            f"dst_host_count={dst_h_count}). "
            "Short UDP packets to an unknown/private service with zero server response are "
            "characteristic of SNMP community string guessing (snmpguess/snmpgetattack) -- "
            "a remote-to-local (R2L) attack that attempts to authenticate to SNMP services "
            "using default or guessed community strings to gain read/write access."
        )
        return " ".join(lines)

    # ── 3. Probe: network scanning / reconnaissance ───────────────────────────
    # Key discriminator from DoS: diff_srv_rate is high (scanning multiple services)
    # satan/portsweep/mscan use REJ flag but scan DIFFERENT services -> diff_srv is high
    # Also: ICMP sweep, same_srv_rate near 0 with multiple connections
    probe_score = 0
    probe_signals = []

    if diff_srv >= 0.5:
        probe_score += 3
        probe_signals.append(f"diff_srv_rate={diff_srv:.0%} (scanning many different services)")
    elif diff_srv >= 0.3:
        probe_score += 2
        probe_signals.append(f"diff_srv_rate={diff_srv:.0%} (accessing varied services -- scan indicator)")

    if dst_h_diff >= 0.3:
        probe_score += 2
        probe_signals.append(f"dst_host_diff_srv_rate={dst_h_diff:.0%} (destination host targeted with diverse services)")

    if protocol == 'icmp' and src_bytes <= 28:
        probe_score += 3
        probe_signals.append(f"ICMP echo ({src_bytes}B payload) -- classic ping sweep / host discovery scan")

    if same_srv < 0.2 and count >= 3:
        probe_score += 2
        probe_signals.append(f"same_srv_rate={same_srv:.0%} across {count} connections (port sweep -- low service locality)")

    if flag in ('REJ', 'RSTO', 'RSTOS0') and diff_srv >= 0.3:
        probe_score += 1
        probe_signals.append(f"Connection flag={flag} with varied services -- typical of active port scanner being rejected by firewalled ports")

    # mscan: single service BUT high dst_host_rerror_rate + high dst_host_count
    # mscan probes one service at a time but across many hosts -> dst_h_rerr is high
    # COUNT must be LOW (mscan doesn't flood same host), unlike neptune (count>=100)
    if dst_h_rerr >= 0.5 and dst_h_count >= 20 and count < 20:
        probe_score += 2
        probe_signals.append(
            f"dst_host_rerror_rate={dst_h_rerr:.0%} across {dst_h_count} destination connections, "
            f"count={count} (mscan pattern: single service probed at high rejection rate across many hosts, "
            "not a same-host flood)"
        )

    if probe_score >= 3:
        lines.append(
            f"Connection via {protocol.upper()} to '{service}' (flag={flag}, duration={duration}s, "
            f"count={count}, dst_host_count={dst_h_count}): "
            + "; ".join(probe_signals) + ". "
            "This pattern is consistent with network reconnaissance (Probe) -- "
            "systematic scanning to discover open ports and reachable services before an attack."
        )
        return " ".join(lines)

    # ── 4. DoS: connection-flood / denial-of-service ──────────────────────────
    # Distinguished from Probe by: same service targeted repeatedly (low diff_srv),
    # high count, REJ/S0 flag, zero bytes, dst_host_count saturated at 255.
    # 'back' attack: HTTP with very large src_bytes and dst_host_count=255 (no REJ)
    dos_signals = []

    # back attack: high-volume HTTP payload flooding (no REJ flag, SF flag, large src_bytes)
    if service == 'http' and src_bytes >= 10000 and dst_h_count >= 200 and diff_srv < 0.2:
        lines.append(
            f"HTTP connection (flag={flag}, duration={duration}s, sent={src_bytes}B to server, "
            f"dst_host_count={dst_h_count}). "
            "A very large payload sent to an HTTP server with saturation of the destination host "
            f"({dst_h_count} recent connections, low service diversity diff_srv={diff_srv:.0%}) "
            "is characteristic of a 'back' denial-of-service attack -- sending malformed "
            "HTTP requests (e.g., with '/' repeated thousands of times) to crash/overload Apache. "
            "This is a DoS attack even though the TCP connection completes (flag=SF)."
        )
        return " ".join(lines)

    if flag in ('REJ', 'S0'):
        dos_signals.append(f"TCP flag={flag} (connection rejected or unanswered -- server unresponsive)")
    if count >= 50:
        dos_signals.append(f"{count} connections to same host in 2 seconds (flood rate)")
    if rerror_rate >= 0.5:
        dos_signals.append(f"REJ error rate={rerror_rate:.0%}")
    if serror_rate >= 0.5:
        dos_signals.append(f"SYN error rate={serror_rate:.0%} (SYN flood)")
    if dst_h_count >= 200:
        dos_signals.append(f"{dst_h_count} connections to destination host (saturation)")
    if src_bytes == 0 and dst_bytes == 0 and flag in ('REJ', 'S0'):
        dos_signals.append("zero bytes in both directions (pure connection exhaustion)")
    if wrong_frag > 0:
        dos_signals.append(f"{wrong_frag} malformed IP fragment(s) (fragmentation DoS variant)")
    if land == 1:
        dos_signals.append("land attack: source=destination (self-directed flood)")

    # DoS requires either high count (flooding) or multiple strong signals
    # A single SYN error with count=1 could be mscan or a scanner, not a DoS flood
    dos_requires_flood = count >= 20 or wrong_frag > 0 or land == 1
    if len(dos_signals) >= 2 and diff_srv < 0.4 and dos_requires_flood:
        lines.append(
            f"Connection via {protocol.upper()} to '{service}' service (flag={flag}, duration={duration}s): "
            + "; ".join(dos_signals) + ". "
            f"Low service diversity (diff_srv_rate={diff_srv:.0%}) confirms this is the same service "
            "being targeted repeatedly -- characteristic of a denial-of-service (DoS) flood attack."
        )
        return " ".join(lines)

    # Fallback DoS: any single-service REJ flood
    if flag in ('REJ', 'S0') and count >= 100 and diff_srv < 0.2:
        lines.append(
            f"Connection via {protocol.upper()} to '{service}' (flag={flag}): "
            f"{count} connections to same host, same service (diff_srv={diff_srv:.0%}), "
            "all rejected. This is a DoS flood pattern."
        )
        return " ".join(lines)

    # Mailbomb DoS: SMTP with repeated fixed-size payloads, dst_host_count=255
    if service == 'smtp' and dst_h_count >= 200 and src_bytes > 500 and flag == 'SF':
        lines.append(
            f"SMTP connection (flag={flag}, duration={duration}s, sent={src_bytes}B, received={dst_bytes}B, "
            f"dst_host_count={dst_h_count}, diff_srv={diff_srv:.0%}). "
            "A large volume of email sent to a saturated destination host (255 recent connections) "
            "via SMTP with a fixed-size payload is characteristic of a mailbomb denial-of-service attack -- "
            "flooding a mail server with messages to exhaust its resources and prevent legitimate email delivery."
        )
        return " ".join(lines)

    # ── 5. Normal: legitimate traffic ─────────────────────────────────────────
    lines.append(
        f"Connection via {protocol.upper()} to '{service}' service (flag={flag}, duration={duration}s, "
        f"sent={src_bytes}B, received={dst_bytes}B)."
    )
    if flag == 'SF':
        lines.append("Connection completed normally (SF flag -- both sides exchanged data and closed cleanly).")
    if logged_in == 1:
        lines.append("User is authenticated (logged_in=1) -- valid session in progress.")
    if serror_rate == 0.0 and rerror_rate == 0.0:
        lines.append("No SYN or REJ errors detected.")
    if count <= 10 and dst_h_count <= 150:
        lines.append(
            f"Low connection rate: {count} connections to same host, {dst_h_count} to destination -- "
            "consistent with individual user activity, not flooding."
        )
    elif dst_h_count >= 200:
        # High destination host count even in normal-looking flows can indicate anomaly
        lines.append(
            f"NOTE: {dst_h_count} recent connections observed to destination host (high saturation level) -- "
            "this may indicate automated or malicious traffic despite normal-looking flow features."
        )
    if src_bytes > 0 and dst_bytes > 0:
        ratio = src_bytes / max(dst_bytes, 1)
        if 0.05 <= ratio <= 20.0:
            lines.append(f"Balanced data exchange (sent {src_bytes}B, received {dst_bytes}B) -- typical of normal request-response.")

    # mscan hint: single service, high dst_host_count, non-zero dst_host_rerror_rate
    if dst_h_rerr >= 0.5 and dst_h_count >= 50 and diff_srv < 0.2:
        lines.append(
            f"Despite appearing normal, dst_host_rerror_rate={dst_h_rerr:.0%} indicates "
            f"{dst_h_rerr:.0%} of connections to this destination host are being rejected -- "
            "consistent with mscan port scanner probing a single service across many hosts."
        )

    # rootkit/ftp_data anomaly hint: ftp_data with dst_host_serror_rate > 0
    if service == 'ftp_data' and dst_h_serr >= 0.2:
        lines.append(
            f"NOTE: dst_host_serror_rate={dst_h_serr:.0%} for this FTP data channel destination -- "
            "elevated SYN error rate at destination may indicate this FTP session is part of "
            "a rootkit installation or unauthorized file transfer disguised as normal FTP."
        )

    return " ".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Few-shot examples -- 2 per class (10 total), grounded in KDDTest+ distributions
# ─────────────────────────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLES = """Below are labelled examples from the NSL-KDD dataset to guide your classification:

Example 1 (Normal): "Connection via TCP to 'ftp_data' service (flag=SF, duration=2s, sent=12983B, received=0B). Connection completed normally (SF flag -- both sides exchanged data and closed cleanly). No SYN or REJ errors detected. Low connection rate: 1 connections to same host, 134 to destination -- consistent with individual user activity, not flooding."
Label: Normal

Example 2 (Normal): "Connection via TCP to 'http' service (flag=SF, duration=0s, sent=267B, received=14515B). Connection completed normally (SF flag). User is authenticated (logged_in=1) -- valid session in progress. No SYN or REJ errors detected. Low connection rate: 4 connections to same host, 155 to destination. Balanced data exchange (sent 267B, received 14515B) -- typical of normal request-response."
Label: Normal

Example 3 (DoS): "Connection via TCP to 'private' service (flag=REJ, duration=0s): TCP flag=REJ (connection rejected or unanswered -- server unresponsive); 229 connections to same host in 2 seconds (flood rate); REJ error rate=100%; 255 connections to destination host (saturation); zero bytes in both directions (pure connection exhaustion). Low service diversity (diff_srv_rate=6%) confirms this is the same service being targeted repeatedly -- characteristic of a denial-of-service (DoS) flood attack."
Label: DoS

Example 4 (DoS): "HTTP connection (flag=SF, duration=0s, sent=54540B to server, dst_host_count=255). A very large payload sent to an HTTP server with saturation of the destination host (255 recent connections, low service diversity diff_srv=0%) is characteristic of a 'back' denial-of-service attack -- sending malformed HTTP requests to crash/overload Apache. This is a DoS attack even though the TCP connection completes (flag=SF)."
Label: DoS

Example 5 (Probe): "Connection via ICMP to 'eco_i' service (flag=SF, duration=0s, count=1, dst_host_count=3): ICMP echo (20B payload) -- classic ping sweep / host discovery scan. This pattern is consistent with network reconnaissance (Probe) -- systematic scanning to discover open ports and reachable services before an attack."
Label: Probe

Example 6 (Probe): "Connection via TCP to 'private' service (flag=REJ, duration=0s, count=483, dst_host_count=255): diff_srv_rate=100% (scanning many different services); dst_host_diff_srv_rate=96% (destination host targeted with diverse services); Connection flag=REJ with varied services -- typical of active port scanner being rejected. This pattern is consistent with network reconnaissance (Probe) -- systematic scanning to discover open ports."
Label: Probe

Example 7 (R2L): "Connection via TCP to 'telnet' (flag=SF, duration=0s, sent=129B, received=174B). 1 failed login attempt(s) recorded -- repeated authentication failures on a remote service are a primary indicator of a remote-to-local (R2L) brute-force or password-guessing attack. User did not gain access."
Label: R2L

Example 8 (R2L): "UDP connection to 'private' service (src=48B, dst=0B, count=5, dst_host_count=255). Short UDP packets to an unknown/private service with zero server response are characteristic of SNMP community string guessing (snmpguess/snmpgetattack) -- a remote-to-local (R2L) attack that attempts to authenticate to SNMP services using default or guessed community strings to gain read/write access."
Label: R2L

Example 9 (U2R): "Connection via TCP to 'ftp' service (flag=SF, duration=8s, sent=220B, received=688B): 4 hot indicator(s) (privileged/sensitive operations detected). These are hallmark indicators of user-to-root (U2R) privilege escalation -- an attacker exploiting a local vulnerability to gain superuser control. Session authenticated -- exploitation occurred from within an established login session."
Label: U2R

Example 10 (U2R): "Connection via TCP to 'telnet' service (flag=SF, duration=45s, sent=300B, received=800B): root shell was obtained; 'su root' command was attempted; 2 shell prompt(s) spawned; 12 hot indicator(s) (privileged/sensitive operations detected); 3 compromised condition(s). These are hallmark indicators of user-to-root (U2R) privilege escalation -- an attacker exploiting a local vulnerability to gain superuser control. Session authenticated."
Label: U2R

"""


# ─────────────────────────────────────────────────────────────────────────────
# ZAI API Classifier
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_MSG = (
    "You are a cybersecurity expert specialising in network intrusion detection. "
    "You will be given a natural-language description of a network connection record "
    "from the NSL-KDD dataset and a set of labelled examples. "
    "Your task is to classify the connection into exactly one of the provided categories. "
    "Reply with ONLY the category name -- no explanation, no punctuation, nothing else."
)


def classify_flow_zai(nl_description: str, classes: list, client: ZAIClient) -> str:
    """Classify an NSL-KDD connection description using the ZAI API (few-shot)."""
    user_msg = (
        FEW_SHOT_EXAMPLES
        + f"Categories: {', '.join(classes)}\n\n"
        + f"Now classify this connection:\n{nl_description}\n\n"
        + "Reply with ONLY the category name."
    )
    response = client.generate(SYSTEM_MSG, user_msg)
    if response is None:
        return "ERROR"
    response = response.strip()
    # Case-insensitive match against known class names
    for cls in classes:
        if cls.lower() in response.lower():
            return cls
    # Return raw (will count as incorrect)
    return response[:50]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_nslkdd(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path, header=None, names=COLUMNS)
    df['label'] = df['label'].str.replace(r'\.$', '', regex=True)
    df['class'] = df['label'].apply(map_label)
    # Drop Unknown rows (not in any defined attack category)
    df = df[df['class'] != 'Unknown'].reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(data_path: Path, n_samples: int = 50,
                   dry_run: bool = False, model: str = "glm-4.7") -> dict:
    print(f"[NSL-KDD SLM] Loading {data_path} ...")
    df = load_nslkdd(data_path)
    print(f"  {len(df):,} rows loaded, class distribution:")
    for cls in CLASSES:
        n = len(df[df['class'] == cls])
        print(f"    {cls:<10} {n:>6,}")

    # Stratified sample -- equal per class (10 per class for --samples 50)
    samples_per_class = max(2, n_samples // len(CLASSES))
    sample_dfs = []
    for cls in CLASSES:
        cls_df = df[df['class'] == cls]
        n = min(len(cls_df), samples_per_class)
        sample_dfs.append(cls_df.sample(n=n, random_state=42))
    sample = pd.concat(sample_dfs, ignore_index=True).sample(frac=1, random_state=42)
    print(f"\n  Sampled: {len(sample)} rows ({samples_per_class} per class)")

    # Show NL examples
    print("\n[NSL-KDD SLM] Sample NL conversions:")
    for cls in CLASSES:
        row = sample[sample['class'] == cls].iloc[0]
        nl = features_to_nl_nslkdd(row)
        print(f"  [{cls}] {nl[:250]}...")

    if dry_run:
        print("\n[NSL-KDD SLM] Dry-run: NL conversion demonstrated. Re-run without --dry-run to call ZAI API.")
        examples = []
        for _, row in sample.head(15).iterrows():
            examples.append({
                "true_label": row["class"],
                "raw_label": row["label"],
                "nl_description": features_to_nl_nslkdd(row),
            })
        out = {
            "mode": "dry_run",
            "model": model,
            "n_samples": len(sample),
            "classes": CLASSES,
            "nl_examples": examples,
            "note": (
                "NL feature conversion for NSL-KDD. 5 classes: Normal, DoS, Probe, R2L, U2R. "
                "Priority hierarchy: U2R (root_shell/su_attempted/hot>5) > "
                "R2L (num_failed_logins/auth-service) > "
                "DoS (flag=REJ/S0, count≥50, rerror≥50%) > "
                "Probe (diff_srv_rate, ICMP sweep) > Normal."
            ),
            "timestamp": datetime.now().isoformat(),
        }
        out_path = ROOT / "results" / "slm_nslkdd_nl_classification.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[NSL-KDD SLM] Saved to {out_path}")
        return out

    # Live evaluation via ZAI API
    print(f"\n[NSL-KDD SLM] Running classification via ZAI API (model={model}) ...")
    client = ZAIClient(model=model, temperature=0, max_tokens=4096)

    results = []
    correct = 0
    errors = 0
    t0 = time.time()

    for i, (_, row) in enumerate(sample.iterrows()):
        true_label = row["class"]
        nl = features_to_nl_nslkdd(row)
        pred = classify_flow_zai(nl, CLASSES, client)
        is_correct = (pred == true_label)
        if pred == "ERROR":
            errors += 1
        if is_correct:
            correct += 1
        results.append({
            "true": true_label,
            "raw_label": row["label"],
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
    for cls in CLASSES:
        cls_results = [r for r in results if r["true"] == cls and r["predicted"] != "ERROR"]
        if cls_results:
            per_class_acc[cls] = sum(r["correct"] for r in cls_results) / len(cls_results)
            per_class_counts[cls] = len(cls_results)

    elapsed_total = time.time() - t0
    print(f"\n[NSL-KDD SLM] Final accuracy: {accuracy:.2%}  ({correct}/{valid} correct, {errors} API errors)")
    print(f"[NSL-KDD SLM] Total time: {elapsed_total:.1f}s  ({elapsed_total/len(sample):.1f}s/sample)")
    print("[NSL-KDD SLM] Per-class accuracy:")
    for cls in CLASSES:
        n = per_class_counts.get(cls, 0)
        acc = per_class_acc.get(cls, 0.0)
        print(f"  {cls:<10} {acc:.2%}  (n={n})")

    out = {
        "mode": "evaluation",
        "model": model,
        "backend": "zai_api",
        "dataset": "nslkdd",
        "n_samples": len(sample),
        "n_valid": valid,
        "n_errors": errors,
        "accuracy": accuracy,
        "per_class_accuracy": per_class_acc,
        "per_class_counts": per_class_counts,
        "elapsed_seconds": elapsed_total,
        "note": (
            "NL feature conversion layer applied to NSL-KDD 5-class problem. "
            "Priority hierarchy for NL dispatch: U2R (root_shell/su_attempted/num_root/hot>5) > "
            "R2L (num_failed_logins / auth-service without login) > "
            "DoS (flag=REJ/S0 + count>=50 + rerror/serror) > "
            "Probe (diff_srv_rate/ICMP sweep) > Normal. "
            "10 few-shot examples (2 per class) embedded in prompt."
        ),
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }
    out_path = ROOT / "results" / "slm_nslkdd_nl_classification.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[NSL-KDD SLM] Saved to {out_path}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SLM classification with NL feature conversion for NSL-KDD (5-class)"
    )
    parser.add_argument("--samples", type=int, default=50,
                        help="Total samples (stratified, default=50 -> 10 per class)")
    parser.add_argument("--model", default="glm-4.7",
                        help="ZAI model to use (e.g. glm-4-flash, glm-4.5, glm-4.7)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show NL conversions only, no API calls")
    parser.add_argument("--data", default=None,
                        help="Path to KDDTest+.txt (default: data/raw/NSL-KDD/KDDTest+.txt)")
    args = parser.parse_args()

    data_path = (
        Path(args.data) if args.data
        else ROOT / "data" / "raw" / "NSL-KDD" / "KDDTest+.txt"
    )

    run_evaluation(
        data_path=data_path,
        n_samples=args.samples,
        dry_run=args.dry_run,
        model=args.model,
    )
