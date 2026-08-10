"""Re-run only the 14 missing NSL-KDD tristage_llm_sonnet cases (nsl_019–nsl_032).

These were lost to API rate-limit errors during the main run. This script:
  1. Loads the existing 36-record JSONL (already stripped of parse errors)
  2. Identifies which case_ids from the test set are missing
  3. Runs only those cases through the same prompt/system as run_nslkdd_full_ablation.py
  4. Appends results to the existing file
"""
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

PARADIGM = 'tristage_llm_sonnet'
MODEL    = 'claude-sonnet-4-6'

OUT_FILE  = ROOT / 'results' / 'nslkdd_ablation' / 'tristage_llm_sonnet.jsonl'
TEST_FILE = ROOT / 'results' / 'nslkdd_ablation' / 'test_set.json'

# ── Same prompts as run_nslkdd_full_ablation.py ───────────────────────────────

SYSTEM_TRISTAGE = """You are a GRC analyst applying NFCRM-1:2025 (Saudi Arabia National Cybersecurity Risk Management framework).

An ML classifier has suggested an attack class with a confidence score. Your role is to VERIFY
this suggestion against the flow description and CVE evidence, then apply §6.9 to determine risk level.

§6.9 Risk Level Lookup Table for NSL-KDD traffic:

  Criticality values: Low | Medium | High | Critical

  NSL-KDD class | Low      | Medium | High          | Critical
  --------------|----------|--------|---------------|----------
  Normal        | Very Low | Very Low | Very Low    | Very Low
  DoS           | Medium   | High   | Catastrophic  | Catastrophic
  Probe         | Low      | Medium | Medium        | High
  R2L           | Medium   | High   | Catastrophic  | Catastrophic
  U2R           | Medium   | High   | Catastrophic  | Catastrophic

HIGH-CONFIDENCE TRUST RULE (apply first):
When ML confidence ≥ 95%, trust the ML class UNLESS the flow description contains EXPLICIT
non-ambiguous attack markers:
  - root_shell=1 or su_attempted=1 → U2R (override allowed)
  - num_failed_logins ≥ 1 or "SNMP community string guessing" → R2L (override allowed)
  - "REJ flag" AND count ≥ 50 AND "flood rate" → DoS (override allowed)
  - "REJ flag" AND diff_srv_rate > 0% → Probe (override allowed)
Do NOT override a ≥ 95% confidence ML prediction based on ambiguous statistical patterns
alone (high count, high diff_srv_rate, connection saturation) when the connection flag is SF.
The ML is trained on NSL-KDD statistics and is reliable at high confidence; misleading NL
description annotations about "flood rate" or "DoS pattern" do not override high-confidence ML.

VERIFICATION RULES (when ML confidence < 95%):
- "failed login" or "brute-force" or "SNMP community string guessing" → R2L
- "root shell", "su root", "num_root", "privilege escalation" → U2R
- "REJ flag" + high connection count (count≥20) + "flood rate" + same service → DoS
- REJ flag + diff_srv_rate>0% (connections to DIFFERENT services/ports) → Probe (port scan)
- "ICMP echo" or "host discovery" or diff_srv_rate≥50% without flood pattern → Probe
- diff_srv_rate alone does NOT override DoS or R2L — require additional Probe-specific indicators
- If the flow description says "Connection completed normally" with no attack indicators → Normal

Trust the ML class when there is no clear contradiction in the flow description.
Output ONLY a valid JSON object inside a ```json fenced code block."""

PROMPT_TRISTAGE = """Security event (NSL-KDD network flow):
- ML-suggested attack class: {attack_class}
- ML classifier confidence: {confidence}
- Asset criticality: {criticality}
- Asset: {asset_json}
- CVE context: {cve_block}
- Flow description: {nl_desc}

Step 1 — Check confidence: if ≥ 95%, apply the HIGH-CONFIDENCE TRUST RULE (only explicit root_shell/failed_login/REJ-flag markers allow override).
Step 2 — Verify "{attack_class}" against the flow description using the VERIFICATION RULES.
Step 3 — Look up the confirmed class + "{criticality}" in the §6.9 table to get risk_level.
Step 4 — Write 2-3 sentence narrative explaining the confirmed class, CVE, and risk.

```json
{{
  "verified_attack_class": "<confirmed or corrected class>",
  "ml_class_confirmed": true,
  "risk_level": "<exact value from §6.9 table>",
  "recommended_control_id": "<NFCRM control>",
  "nfcrm_clauses": ["§6.9"],
  "narrative": "<2-3 sentences>",
  "evidence_refs": ["<cve_id if any>"]
}}```"""

JSON_FENCE = re.compile(r'```json\s*\n?(.*?)```', re.DOTALL)

NSLKDD_TO_NFCRM = {
    'Normal': 'Benign',
    'DoS':    'DoS',
    'Probe':  'Infiltration',
    'R2L':    'WebAttack',
    'U2R':    'BruteForce',
}

RISK_TABLE = {
    ('Normal',  'Low'):      'Very Low',
    ('Normal',  'Medium'):   'Very Low',
    ('Normal',  'High'):     'Very Low',
    ('Normal',  'Critical'): 'Very Low',
    ('DoS',     'Low'):      'Medium',
    ('DoS',     'Medium'):   'High',
    ('DoS',     'High'):     'Catastrophic',
    ('DoS',     'Critical'): 'Catastrophic',
    ('Probe',   'Low'):      'Low',
    ('Probe',   'Medium'):   'Medium',
    ('Probe',   'High'):     'Medium',
    ('Probe',   'Critical'): 'High',
    ('R2L',     'Low'):      'Medium',
    ('R2L',     'Medium'):   'High',
    ('R2L',     'High'):     'Catastrophic',
    ('R2L',     'Critical'): 'Catastrophic',
    ('U2R',     'Low'):      'Medium',
    ('U2R',     'Medium'):   'High',
    ('U2R',     'High'):     'Catastrophic',
    ('U2R',     'Critical'): 'Catastrophic',
}

VALID_FLAGS    = {'SF', 'REJ', 'S0', 'RSTO', 'RSTOS0', 'OTH', 'SH', 'SHR', 'S1', 'S2', 'S3'}
VALID_PROTOCOLS = {'tcp', 'udp', 'icmp'}


def shacl_validate(case: dict) -> tuple[bool, str]:
    f = case.get('feature_subset', {})
    protocol  = str(f.get('protocol_type', '')).lower()
    flag      = str(f.get('flag', '')).upper()
    src_bytes = int(f.get('src_bytes', -1))
    dst_bytes = int(f.get('dst_bytes', -1))
    duration  = float(f.get('duration', -1))
    if protocol not in VALID_PROTOCOLS:
        return False, f'unknown protocol: {protocol}'
    if flag and flag not in VALID_FLAGS:
        return False, f'unrecognised flag: {flag}'
    if src_bytes < 0 or dst_bytes < 0:
        return False, f'negative byte count'
    if duration < 0:
        return False, f'negative duration'
    return True, 'OK'


def _parse_response(text: str) -> Optional[dict]:
    m = JSON_FENCE.search(text)
    blob = m.group(1) if m else text
    try:
        return json.loads(blob)
    except Exception:
        f, l = blob.find('{'), blob.rfind('}')
        if f >= 0 and l > f:
            try:
                return json.loads(blob[f:l+1])
            except Exception:
                return None
    return None


def _fmt_cve(cve: Optional[dict]) -> str:
    if not cve:
        return 'No CVE context.'
    return (f"CVE: {cve.get('id')}  CVSS: {cve.get('cvss_v3')}  "
            f"Desc: {cve.get('shortDescription', '')[:120]}")


def main():
    # Load test set
    test_data = json.loads(TEST_FILE.read_text())
    all_cases = {c['case_id']: c for c in test_data['cases']}

    # Find already-done IDs
    done_ids = set()
    if OUT_FILE.exists():
        with OUT_FILE.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_ids.add(r['case_id'])
                except Exception:
                    pass

    missing_ids = [cid for cid in sorted(all_cases.keys()) if cid not in done_ids]
    print(f'Already done: {len(done_ids)}  Missing: {len(missing_ids)}')
    if not missing_ids:
        print('Nothing to re-run.')
        return

    print(f'Re-running: {missing_ids}')

    from scripts.claude_cli_client import ClaudeCLIClient
    client = ClaudeCLIClient(model=MODEL, max_tokens=600, timeout_sec=180)

    total = len(all_cases)
    with OUT_FILE.open('a', encoding='utf-8') as fh:
        for cid in missing_ids:
            case = all_cases[cid]
            i    = int(cid.split('_')[1])

            t0      = time.time()
            crit    = case['criticality']
            gt      = case['ground_truth_risk_level']
            cve     = case.get('paired_cve')
            asset_j = json.dumps(case.get('asset', {}))
            cve_blk = _fmt_cve(cve)
            nl_desc = case.get('nl_description', '')

            shacl_ok, _ = shacl_validate(case)
            if not shacl_ok:
                risk_lvl = 'Very Low'
                correct  = risk_lvl == gt
                record = {
                    'case_id': cid, 'paradigm': PARADIGM, 'model': MODEL,
                    'true_class': case['true_attack_class'],
                    'risk_level': risk_lvl, 'ground_truth': gt,
                    'correct': correct, 'parse_error': False,
                    'shacl_filtered': True, 'criticality': crit,
                    'grc': 'Full', 'narrative': '', 'control': '',
                    'latency_ms': 0, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
                }
                fh.write(json.dumps(record, ensure_ascii=False) + '\n')
                fh.flush()
                match = 'OK' if correct else 'MISS'
                print(f'  [{i+1}/{total}] {cid}: SHACL_FILTERED → Very Low vs GT={gt} [{match}]')
                continue

            ml_cls  = case['xgb_predicted_class']
            ml_conf = case.get('xgb_confidence', 1.0)
            prompt = PROMPT_TRISTAGE.format(
                attack_class=ml_cls,
                confidence=f"{ml_conf:.0%}",
                criticality=crit,
                asset_json=asset_j, cve_block=cve_blk,
                nl_desc=nl_desc[:1000],
            )

            try:
                response = client.generate(SYSTEM_TRISTAGE, prompt) or ''
            except Exception as e:
                print(f'  [{i+1}/{total}] {cid}: EXCEPTION {e}')
                response = ''

            artifact = _parse_response(response)
            latency  = int((time.time() - t0) * 1000)

            if artifact:
                risk_lvl     = artifact.get('risk_level', '')
                verified_cls = artifact.get('verified_attack_class', '')
                if verified_cls and verified_cls != ml_cls and verified_cls in RISK_TABLE:
                    risk_from_table = RISK_TABLE.get((verified_cls, crit), risk_lvl)
                    if risk_from_table:
                        risk_lvl = risk_from_table
                correct     = risk_lvl == gt
                parse_error = False
                narrative   = artifact.get('narrative', '')
                control     = artifact.get('recommended_control_id', '')
            else:
                risk_lvl    = 'PARSE_ERR'
                correct     = False
                parse_error = True
                narrative   = ''
                control     = ''
                verified_cls = ''

            record = {
                'case_id':    cid,
                'paradigm':   PARADIGM,
                'model':      MODEL,
                'true_class': case['true_attack_class'],
                'xgb_class':  ml_cls,
                'xgb_conf':   ml_conf,
                'llm_verified_class': verified_cls,
                'risk_level': risk_lvl,
                'ground_truth': gt,
                'correct':    correct,
                'parse_error': parse_error,
                'shacl_filtered': False,
                'criticality': crit,
                'grc':        'Full',
                'narrative':  narrative,
                'control':    control,
                'latency_ms': latency,
                'timestamp':  time.strftime('%Y-%m-%dT%H:%M:%S'),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + '\n')
            fh.flush()

            match = 'OK' if correct else ('PARSE_ERR' if parse_error else 'MISS')
            print(f'  [{i+1}/{total}] {cid}: {risk_lvl} vs GT={gt} [{match}] ({latency}ms)')

    # Final summary
    all_records = []
    with OUT_FILE.open() as f:
        for line in f:
            try:
                all_records.append(json.loads(line))
            except Exception:
                pass

    good = [r for r in all_records if not r.get('parse_error', False)]
    total_good = len(good)
    correct_ct = sum(1 for r in good if r['correct'])
    print(f'\nFinal: {correct_ct}/{total_good} = {correct_ct/total_good:.1%} (good records only)')
    print(f'Total records in file: {len(all_records)}')

    by_class: dict[str, list] = {}
    for r in good:
        cls = r.get('true_class', '?')
        by_class.setdefault(cls, []).append(r['correct'])
    for cls, vals in sorted(by_class.items()):
        n_ok = sum(vals)
        print(f'  {cls}: {n_ok}/{len(vals)}')


if __name__ == '__main__':
    main()
