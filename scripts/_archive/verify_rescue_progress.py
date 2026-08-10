#!/usr/bin/env python3
"""
INTERSYMBOLIC-GRC Thesis Rescue V2 — Progress Verification Script

This script verifies that Phases -1, 0, 1, and 3 have been completed successfully.
Run this to confirm the rescue progress before proceeding to Phase 4-7 (thesis rewrite).

Usage:
    python3 scripts/verify_rescue_progress.py

Expected output: All checks pass with ✅
"""

import json
import os
import sys
import subprocess
from pathlib import Path

# Dynamic repo root - works on any machine
REPO_ROOT = Path(__file__).parent.parent

def check_file_exists(path: str, description: str) -> bool:
    """Check if a file exists."""
    exists = os.path.isfile(path)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {path}")
    return exists

def check_dir_exists(path: str, description: str) -> bool:
    """Check if a directory exists."""
    exists = os.path.isdir(path)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {path}")
    return exists

def check_file_not_exists(path: str, description: str) -> bool:
    """Check that a file does NOT exist (was cleaned up)."""
    exists = os.path.exists(path)
    status = "✅" if not exists else "❌"
    print(f"{status} {description} (should not exist): {path}")
    return not exists

def check_json_label(path: str, key: str, expected: str, description: str) -> bool:
    """Check a JSON file has the expected label."""
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        actual = data.get(key, "")
        matches = actual == expected
        status = "✅" if matches else "❌"
        print(f"{status} {description}: {key}='{actual}' (expected: '{expected}')")
        return matches
    except Exception as e:
        print(f"❌ {description}: Error reading {path} - {e}")
        return False

def check_nfcrm_references() -> bool:
    """Check that NFCRM references use correct §5.x/§6.x format."""
    try:
        # Check for old format (3-digit numbers)
        result = subprocess.run(
            ['grep', '-rn', 'NFCRM-1:2025-[0-9][0-9][0-9]', '--include=*.py', '--include=*.ttl', '.'],
            capture_output=True, text=True, cwd=REPO_ROOT
        )
        old_format_found = len(result.stdout.strip()) > 0
        
        # Check for old format (section 3.x, 4.x, 5.x)
        result2 = subprocess.run(
            ['grep', '-rn', 'NFCRM-1:2025-[3-5]\\.', '--include=*.py', '--include=*.ttl', '.'],
            capture_output=True, text=True, cwd=REPO_ROOT
        )
        old_section_found = len(result2.stdout.strip()) > 0
        
        status = "✅" if not (old_format_found or old_section_found) else "❌"
        print(f"{status} NFCRM references use correct §5.x/§6.x format (no old format found)")
        return not (old_format_found or old_section_found)
    except Exception as e:
        print(f"❌ NFCRM reference check failed: {e}")
        return False

def count_results_jsons() -> bool:
    """Count results JSON files."""
    results_dir = REPO_ROOT / 'results'
    json_files = list(results_dir.glob('*.json'))
    count = len(json_files)
    status = "✅" if count >= 19 else "❌"
    print(f"{status} Results JSON files: {count} (expected: 19+)")
    return count >= 19

def count_figures() -> bool:
    """Count thesis figures."""
    figures_dir = REPO_ROOT / 'thesis' / 'figures'
    png_files = list(figures_dir.glob('*.png'))
    count = len(png_files)
    status = "✅" if count >= 7 else "❌"
    print(f"{status} Thesis figures: {count} (expected: 7+)")
    return count >= 7

def check_external_data() -> bool:
    """Check external data files exist."""
    required_files = [
        'NFCRM-1-2025.pdf',
        'nfcrm_clause_mapping.json',
        'cisa_kev.json',
        'enterprise-attack.json',
        'cmdb_assets.json',
        'nvd_enrichment.json'
    ]
    all_exist = True
    for filename in required_files:
        path = REPO_ROOT / 'data' / 'external' / filename
        exists = os.path.isfile(path)
        status = "✅" if exists else "❌"
        print(f"{status} External data: {filename}")
        if not exists:
            all_exist = False
    return all_exist

def main():
    print("=" * 70)
    print("INTERSYMBOLIC-GRC Thesis Rescue V2 — Progress Verification")
    print("=" * 70)
    print()
    
    results = []
    
    print("--- Phase -1: Filesystem Cleanup ---")
    results.append(check_dir_exists(REPO_ROOT / 'data' / 'external', 'External data dir'))
    results.append(check_dir_exists(REPO_ROOT / 'thesis' / 'figures', 'Thesis figures dir'))
    results.append(check_file_not_exists(REPO_ROOT / 'data' / 'raw' / 'UNSW-NB15', 'UNSW-NB15 cleanup'))
    results.append(check_file_not_exists(REPO_ROOT / 'slms', 'SLMs dir cleanup'))
    results.append(check_file_not_exists(REPO_ROOT / 'exports', 'Empty exports dir'))
    results.append(check_file_not_exists(REPO_ROOT / 'risk_states', 'Empty risk_states dir'))
    results.append(check_file_not_exists(REPO_ROOT / 'test_results', 'Empty test_results dir'))
    print()
    
    print("--- Phase 0: NFCRM Reconciliation ---")
    results.append(check_json_label(
        REPO_ROOT / 'results' / 'rf_baseline.json',
        'dataset', 'NSL-KDD', 'rf_baseline.json label'
    ))
    results.append(check_nfcrm_references())
    results.append(check_file_exists(
        REPO_ROOT / 'data' / 'external' / 'nfcrm_clause_mapping.json',
        'NFCRM clause mapping'
    ))
    print()
    
    print("--- Phase 1: Multi-Source Data ---")
    results.append(check_external_data())
    results.append(count_results_jsons())
    print()
    
    print("--- Phase 3: Figures ---")
    results.append(count_figures())
    print()
    
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    percentage = (passed / total) * 100 if total > 0 else 0
    
    if all(results):
        print(f"✅ ALL CHECKS PASSED ({passed}/{total})")
        print("Phases -1, 0, 1, and 3 are COMPLETE. Ready for Phase 4-7 (thesis rewrite).")
        return 0
    else:
        print(f"⚠️  SOME CHECKS FAILED ({passed}/{total} passed, {percentage:.1f}%)")
        print("Review failed checks above before proceeding.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
