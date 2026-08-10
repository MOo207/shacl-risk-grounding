#!/usr/bin/env python3
"""
CSE-CIC-IDS2018 Minimal Dataset Downloader
============================================
Downloads the MINIMUM set of daily CSV files needed to run all 3 thesis approaches:

  Approach 1 — SHACL + Sigma rule-based engine
  Approach 2 — ML baselines (Random Forest, XGBoost, GNN)
  Approach 3 — local SLM fine-tuning (deprecated; framework uses the Claude API)

Why NOT 452 GB?
  452 GB = raw PCAP network captures (NOT needed — CICFlowMeter already extracted features)
  CSV files = ~4-6 GB total; this script downloads only 6 of the 10 days = ~1.9 GB

Why these 6 days?
  Every day has both Benign + attack traffic mixed together.
  These 6 cover ALL 7 canonical attack classes:

  Day      | File size | Attack types present
  ---------|-----------|------------------------------------------
  thu15    | ~358 MB   | FTP-BruteForce, SSH-Bruteforce
  fri16    | ~336 MB   | DoS-Hulk, DoS-SlowHTTPTest, DoS-GoldenEye, DoS-Slowloris
  wed21    | ~102 MB   | Web Attack-BruteForce, Web Attack-XSS, SQL Injection (87 samples)
  thu22    | ~115 MB   | Infiltration
  fri23    | ~365 MB   | Bot (Ares botnet)
  thu01    | ~600 MB   | DDOS-LOIC-HTTP, DDOS-HOIC
  ---------|-----------|------------------------------------------
  TOTAL    | ~1.9 GB   | All 7 attack types + Benign in every file

Requirements:
    pip install boto3 tqdm

Usage:
    python scripts/download_dataset.py          # download 6 files (~1.9 GB)
    python scripts/download_dataset.py --all    # download all 10 files (~4-6 GB)
    python scripts/download_dataset.py --list   # show file info and exit
"""

import argparse
import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw" / "CSE-CIC-IDS2018"

S3_BUCKET = "cse-cic-ids2018"
S3_PREFIX = "Processed Traffic Data for ML Algorithms/"
S3_REGION = "us-east-1"      # confirmed working region

# (key, filename, size_mb, attacks, in_minimal_set)
# Note: actual filenames use CICFlowMeter (NOT CaptureFlowID)
# Note: "Thuesday" is the real spelling in the bucket for Tuesday 20 Feb (4 GB - excluded from minimal)
ALL_FILES = [
    ("wed14",  "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",   358, "Benign only (no attacks this day)",                        False),
    ("thu15",  "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",    376, "FTP-BruteForce, SSH-Bruteforce",                           True),
    ("fri16",  "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",      334, "DoS-Hulk, DoS-SlowHTTPTest, DoS-GoldenEye, DoS-Slowloris", True),
    ("tue20",  "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv",   4055, "DDoS (massive file — 4 GB, optional)",                     False),
    ("wed21",  "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",   329, "Web Attack-BruteForce, Web Attack-XSS, SQL Injection",     True),
    ("thu22",  "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",    383, "Infiltration",                                             True),
    ("fri23",  "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",      383, "Bot (Ares botnet)",                                        True),
    ("wed28",  "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",   209, "Infiltration (continued)",                                 False),
    ("thu01",  "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",    108, "DDOS-LOIC-HTTP, DDOS-HOIC",                                True),
    ("fri02",  "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",      352, "DDOS-LOIC-UDP, DDOS-LOIC-HTTP",                            False),
]

MINIMAL = [(k, f, s, a) for k, f, s, a, m in ALL_FILES if m]
FULL    = [(k, f, s, a) for k, f, s, a, _ in ALL_FILES]


def check_deps():
    missing = []
    for pkg in ("boto3", "tqdm"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[ERROR] Missing packages: {', '.join(missing)}")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)


def list_files():
    print("\nCSE-CIC-IDS2018 CSV files  (452 GB = raw PCAP, NOT downloaded here)")
    print("-" * 70)
    for key, fname, size_mb, attacks, minimal in ALL_FILES:
        tag = "[MINIMAL]" if minimal else "         "
        print(f"  {tag} ~{size_mb:>4} MB  {key}  {attacks}")
    m_mb = sum(s for _, _, s, _ in MINIMAL)
    a_mb = sum(s for _, _, s, _ in FULL)
    print(f"\n  Minimal set (6 files, all 7 attack types): ~{m_mb} MB")
    print(f"  Full set    (9 files):                      ~{a_mb} MB")


def download_files(files: list, out_dir: Path):
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    from tqdm import tqdm

    out_dir.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client(
        "s3",
        config=Config(signature_version=UNSIGNED),
        region_name=S3_REGION,
    )

    n = len(files)
    for i, (day, fname, size_mb, attacks) in enumerate(files, 1):
        dest   = out_dir / fname
        s3_key = S3_PREFIX + fname

        print(f"\n[{i}/{n}] {day.upper()} — {attacks}")

        # Check if already fully downloaded
        if dest.exists():
            try:
                head     = s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
                expected = head["ContentLength"]
                if dest.stat().st_size == expected:
                    print(f"  [SKIP] {fname}  ({expected/1e6:.0f} MB already present)")
                    continue
            except Exception:
                pass

        # Get real size
        try:
            head  = s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
            total = head["ContentLength"]
        except Exception:
            total = size_mb * 1_048_576

        print(f"  {fname}")
        print(f"  ~{total/1e6:.0f} MB  from s3://{S3_BUCKET}/{s3_key}")

        with tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024,
                  desc=f"  {fname[:40]}", leave=True) as bar:
            s3.download_file(
                Bucket=S3_BUCKET,
                Key=s3_key,
                Filename=str(dest),
                Callback=lambda n: bar.update(n),
            )
        print(f"  [OK] -> {dest}")

    downloaded_mb = sum(f.stat().st_size for f in out_dir.glob("*.csv")) / 1e6
    print(f"\n{'='*60}")
    print(f"Done. {len(list(out_dir.glob('*.csv')))} file(s), {downloaded_mb:.0f} MB in {out_dir}")
    print("Next steps:")
    print("  python scripts/validate_dataset.py")
    print("  python scripts/prepare_slm_dataset.py")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Download CIC-IDS2018 minimal CSV set for SHACL + ML + SLM (~1.9 GB)"
    )
    parser.add_argument("--all",  action="store_true", help="Download all 9 files (~4-6 GB)")
    parser.add_argument("--list", action="store_true", help="List available files and exit")
    parser.add_argument("--out",  type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    if args.list:
        list_files()
        return

    check_deps()

    files = FULL if args.all else MINIMAL
    total_mb = sum(s for _, _, s, _ in files)

    print("=" * 60)
    print("CSE-CIC-IDS2018 Downloader  (CSV files only, no PCAP)")
    print(f"  Mode   : {'Full (9 files)' if args.all else 'Minimal (6 files, all 7 attack types)'}")
    print(f"  Size   : ~{total_mb} MB")
    print(f"  Output : {args.out}")
    print(f"  Source : s3://{S3_BUCKET}/{S3_PREFIX}")
    print(f"  Auth   : None (AWS Open Data, public bucket)")
    print("=" * 60)

    download_files(files, args.out)


if __name__ == "__main__":
    main()
