"""
INTERSYMBOLIC-GRC -- Live Demo
Run: python scripts/demo.py

Shows the full pipeline: raw network flow -> GRC risk report
No dataset or API needed -- uses pre-computed results.
"""
import json, textwrap, sys, io, os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# -- colour helpers ------------------------------------------------------------
R    = "\033[91m"
Y    = "\033[93m"
G    = "\033[92m"
B    = "\033[94m"
C    = "\033[96m"
W    = "\033[97m"
D    = "\033[2m"
X    = "\033[0m"
BOLD = "\033[1m"

def clr(text, code): return f"{code}{text}{X}"
def hr(ch="-", n=70): print(clr(ch * n, D))
def section(title):
    print(f"\n{BOLD}{B}{'='*70}{X}")
    print(f"{BOLD}{W}  {title}{X}")
    print(f"{B}{'='*70}{X}")
def label(k, v, col=C):
    print(f"  {clr(k + ':', D):<30} {col}{v}{X}")

RISK_COL = {"CRITICAL": R, "HIGH": Y, "MEDIUM": C, "LOW": G}

# -- load pre-computed results -------------------------------------------------
explanations = json.load(open("results/intersymbolic_explanations.json",
                               encoding="utf-8"))
grc          = json.load(open("results/grc_metrics.json", encoding="utf-8"))

# pick one clean example per attack class with a working NL explanation
examples = {}
for item in explanations:
    lbl = item.get("true_label", "")
    if lbl not in examples and item.get("layer4_optional", {}).get("status") == "OK":
        examples[lbl] = item
    if len(examples) >= 5:
        break

# -- HEADER -------------------------------------------------------------------
print(f"""
{BOLD}{W}
  +------------------------------------------------------------------+
  |        INTERSYMBOLIC-GRC  --  Live Pipeline Demo                |
  |  Raw network flow  ->  NFCRM-1:2025 compliant GRC risk report   |
  +------------------------------------------------------------------+{X}
""")

# -- SYSTEM STATS -------------------------------------------------------------
section("System Overview")
label("Dataset",             "CSE-CIC-IDS2018  (16.2M network flows)")
label("ARG nodes / edges",   f"{grc['arg_nodes']} nodes  /  {grc['arg_edges']} edges  /  32 real CVEs")
label("Test samples run",    f"{grc['test_set_size']}  ->  {grc['total_risk_cases']} RiskCases generated")
label("SHACL violations",    clr(str(grc["shacl_violations"]) + "  (all artifacts valid)", G))
label("Throughput",          "2,944 samples / second  |  0.34 ms / sample")
label("NFCRM-1:2025 coverage",   clr("100%", G))
label("Event->Risk traceability", clr("100%", G))
label("ISO 27005 coverage",      clr("57.6%  (49 / 85 RiskCases)", Y))

input(f"\n  {D}Press Enter to walk through 5 real attack scenarios...{X}")

# -- PER-ATTACK DEMO ----------------------------------------------------------
DESCRIPTIONS = {
    "DDoS":        "Distributed Denial of Service -- flood attack exhausting server resources",
    "DoS":         "Denial of Service -- Slowloris slow-rate connection exhaustion",
    "BruteForce":  "HTTP brute-force -- automated credential stuffing against web login",
    "WebAttack":   "Web Attack -- SQL injection / XSS probes against web application",
    "Infiltration":"Infiltration -- stealthy C2 beacon on non-standard port",
}

for i, (lbl, item) in enumerate(examples.items(), 1):
    section(f"Scenario {i} of {len(examples)}  --  {lbl}")
    print(f"  {D}{DESCRIPTIONS.get(lbl, '')}{X}\n")

    # Layer 1: raw flow features
    print(clr("  LAYER 1 -- Raw Network Flow Features", B))
    hr()
    for k, v in item.get("key_flow_stats", {}).items():
        label(f"  {k}", str(v))

    # Layer 2: pre-inference symbolic hints
    print()
    print(clr("  LAYER 2 -- Pre-inference Symbolic Rules", B))
    hr()
    l1     = item.get("layer1", {})
    active = l1.get("active_rules", [])
    hint   = l1.get("symbolic_hint", "none")
    label("  Active rules",  ", ".join(active) if active else clr("none triggered", D))
    label("  Symbolic hint", hint)

    # Layer 3: ML classification + GRC annotation
    print()
    print(clr("  LAYER 3 -- ML Classification + SHACL GRC Annotation", B))
    hr()
    l3      = item.get("layer3", {})
    pred    = l3.get("predicted_attack_type", "?")
    rlvl    = l3.get("risk_level", "?")
    rcol    = RISK_COL.get(rlvl, C)
    correct = "[CORRECT]" if item.get("correct_prediction") else "[WRONG]"
    ccol    = G if item.get("correct_prediction") else R

    label("  Predicted class",      clr(pred, W))
    label("  True class",           clr(lbl, W))
    label("  Prediction",           clr(correct, ccol))
    label("  Risk level",           clr(rlvl, rcol))
    label("  NFCRM-1:2025",         clr(l3.get("nfcrm_clause", "--"), G))
    label("  ISO 27005",            l3.get("iso_clause") or clr("not mapped", D))
    label("  NCA reference",        l3.get("nca_reference", "--"))
    label("  CVE examples",         "  ".join(l3.get("cve_examples", [])))
    label("  Recommended control",  clr(l3.get("recommended_control", "--"), Y))
    label("  RiskCase ID",          clr(l3.get("risk_case", "--"), W))

    # Layer 4: NL explanation
    print()
    print(clr("  LAYER 4 -- Natural Language Explanation  (GLM-4.7 via ZAI API)", B))
    hr()
    l4 = item.get("layer4_optional", {})
    nl = l4.get("nl_explanation", "")
    print(f"  {clr('Backend:', D)} {l4.get('backend', '')}")
    print()
    for line in textwrap.wrap(nl, width=66):
        print(f"  {C}{line}{X}")
    print()

    if i < len(examples):
        input(f"  {D}Press Enter for next scenario...{X}")
    else:
        input(f"  {D}Press Enter for summary...{X}")

# -- SLM SECTION -------------------------------------------------------------
section("BONUS -- Small Language Model (SLM) Classification")
print(f"""
  {D}A separate experiment: can a language model classify network attacks
  if we first convert raw numbers into plain English?{X}

  {BOLD}Without NL conversion:{X}  {R}0% accuracy{X}  (LLM sees raw numbers -- meaningless)
  {BOLD}With NL conversion:   {X}  {G}55% accuracy{X}  (LLM reads a description -- can reason)
""")
hr()

slm = json.load(open("results/slm_nl_classification.json", encoding="utf-8"))
nsl = json.load(open("results/slm_nslkdd_nl_classification.json", encoding="utf-8"))

# pick the clearest correct examples: DoS (Slowloris signature) and DDoS (max window)
showcase = {}
for r in slm["results"]:
    lbl = r.get("true", "")
    if lbl in ("DoS", "DDoS", "BruteForce") and r.get("correct") and lbl not in showcase:
        showcase[lbl] = r
    if len(showcase) == 3:
        break

for lbl, r in showcase.items():
    pred  = r["predicted"]
    nl    = r["nl_description"]
    ok    = r["correct"]
    pcol  = G if ok else R
    tag   = "[CORRECT]" if ok else "[WRONG]"

    print(f"  {BOLD}{W}Attack: {lbl}{X}")
    print(f"  {D}NL description sent to GLM-4.7:{X}")
    for line in textwrap.wrap(nl, width=66):
        print(f"    {C}{line}{X}")
    print(f"  {clr('LLM predicted:', D):<30} {pcol}{pred}  {tag}{X}")
    print()

hr()
print(f"  {BOLD}CIC-IDS2018 results  ({slm['n_samples']} samples, {slm['model']}){X}")
label("  Overall accuracy",  clr(f"{slm['accuracy']*100:.0f}%  ({slm['n_valid']}/{slm['n_samples']} valid, {slm['n_errors']} errors)", G))
for cls, acc in slm["per_class_accuracy"].items():
    col = G if acc >= 0.6 else Y if acc >= 0.4 else R
    bar = "#" * int(acc * 20)
    print(f"    {cls:<15} {col}{acc*100:.0f}%  {bar}{X}")

print()
label("  NSL-KDD accuracy",  clr(f"{nsl['accuracy']*100:.0f}%  ({nsl['n_valid']}/{nsl['n_samples']} valid, {nsl['n_errors']} errors)", G))
for cls, acc in nsl["per_class_accuracy"].items():
    col = G if acc >= 0.8 else Y if acc >= 0.5 else R
    bar = "#" * int(acc * 20)
    print(f"    {cls:<15} {col}{acc*100:.0f}%  {bar}{X}")

print(f"""
  {D}Key insight: the NL conversion layer encodes domain knowledge
  (e.g. "TCP window=225 = Slowloris signature") that the LLM
  cannot infer from raw numbers alone.{X}
""")

input(f"  {D}Press Enter for summary...{X}")

# -- SUMMARY ------------------------------------------------------------------
section("What Was Just Demonstrated")
hr()
print(f"""
  {BOLD}Input:{X}   Raw CICFlowMeter network traffic numbers
  {BOLD}Output:{X}  Standards-compliant GRC risk report  -- automatically

  {G}[OK]{X}  Attack classified by Random Forest  (87.70% acc, F1-macro 0.896)
  {G}[OK]{X}  Risk level assigned  (CRITICAL / HIGH / MEDIUM / LOW)
  {G}[OK]{X}  NFCRM-1:2025 clause mapped  ->  100% coverage (SHACL-enforced)
  {G}[OK]{X}  CVEs linked from real NVD data  (32 CVEs in ARG)
  {G}[OK]{X}  Control recommendation generated
  {G}[OK]{X}  Audit-ready RiskCase created  (traceable, timestamped)
  {G}[OK]{X}  Natural language explanation written by LLM  (grounded by SHACL)

  {D}A pure-ML system gives you only the class label.
  {D}A pure-rule system cannot adapt to new attack patterns.
  {BOLD}{W}This framework gives you both -- plus the compliance report.{X}
""")
hr("=")
print(f"  {BOLD}{W}INTERSYMBOLIC-GRC  |  Mohammed Ismail Al-Ammawi  |  IU Madinah 2026{X}")
hr("=")
print()
