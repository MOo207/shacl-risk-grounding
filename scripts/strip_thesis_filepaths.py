"""One-shot: strip internal file-path references (results/, scripts/, pipeline/,
data/, evidence/) from thesis body prose and table captions, keeping only the
Reproducibility Package section and the implementation appendix where listing
code artefacts is appropriate.

Read-through-then-write; prints a per-line before/after report for review.
"""
import re
import sys

TEX = "thesis/INTERSYMBOLIC-GRC_Thesis.tex"

# Line ranges (1-based, inclusive) where file/module references are LEGITIMATE
# and must be preserved verbatim.
KEEP_RANGES = [
    (1789, 1874),   # \section{Reproducibility Package}
    (4040, 4070),   # NFCRM compliance appendix: implementation module listing
]
KEEP_LINES = {1944, 3454}  # repo/prompt-template reproducibility notes

PATH_RE = re.compile(r"\\texttt\{(?:results|scripts|pipeline|data|evidence)/")


def in_keep(idx1):
    if idx1 in KEEP_LINES:
        return True
    return any(lo <= idx1 <= hi for lo, hi in KEEP_RANGES)


# A single \texttt{...} unit (path content has no literal '}')
TT = r"\\texttt\{[^}]*\}"

# Pass 1: parenthetical groups led by a provenance keyword.
P_KEYED_PAREN = re.compile(
    r"\s*\((?:source|sources|see|script|builder|driver|output)\s*:[^()]*"
    + TT + r"[^()]*\)",
    re.IGNORECASE,
)
# Pass 2: parenthetical groups that START with a file-path texttt (optionally
# followed by more prose/paths), e.g. (\texttt{results/x.json}) or
# (\texttt{pipeline/nfcrm/asset_risk.py}, driver \texttt{scripts/y.py}).
P_PATH_PAREN = re.compile(
    r"\s*\(\\texttt\{(?:results|scripts|pipeline|data|evidence)/[^}]*\}[^()]*\)"
)
# Pass 3a: trailing "Source: ... ." clause preceded by a period (captions).
P_SRC_TRAIL = re.compile(
    r"\.\s+Sources?:\s*" + TT + r"(?:[,;]?\s*(?:and\s+)?" + TT + r")*\s*\."
)
# Pass 3b: "Sources: ... ." not preceded by a period (after ':' etc.).
P_SRC_BARE = re.compile(
    r"\s*Sources?:\s*" + TT + r"(?:[,;]?\s*(?:and\s+)?" + TT + r")*\s*\."
)
# Pass 4: "Full results: ... ." / "results: ... ." trailing clause.
P_FULLRES = re.compile(
    r"\.\s+(?:Full results|results):\s*" + TT
    + r"(?:[,;]?\s*(?:and\s+)?" + TT + r")*\s*\.",
    re.IGNORECASE,
)
# Pass 5: "; \texttt{path})" inside a parenthetical -> drop the path, keep ')'.
P_SEMI_PATH = re.compile(r";\s*" + TT + r"\)")


def clean(line):
    line = P_KEYED_PAREN.sub("", line)
    line = P_PATH_PAREN.sub("", line)
    line = P_SRC_TRAIL.sub(".", line)
    line = P_SRC_BARE.sub("", line)
    line = P_FULLRES.sub(".", line)
    line = P_SEMI_PATH.sub(")", line)
    # tidy (preserve leading indentation; only collapse spaces after content)
    line = re.sub(r"(?<=\S) {2,}", " ", line)
    line = re.sub(r" +([.,;:])", r"\1", line)
    line = line.replace("( ", "(").replace(" )", ")")
    return line


# Exact rewrites for lines the regex passes cannot clean without breaking grammar.
# (old, new) operate on the post-regex full text; each old is unique.
MANUAL_PAIRS = [
    (r"(the RF configuration is recorded in \texttt{results/cleaned\_rf\_baseline.json} and reused by the stratified ablation run \texttt{results/ablation\_study\_v2.json}; XGBoost from \texttt{results/xgb\_baseline.json}):",
     r"(the RF configuration is reused by the stratified ablation run; XGBoost is configured separately):"),
    (r"justifies excluding override rules from the final pipeline. Full calibration data is in \texttt{results/threshold\_calibration.json} (confidence threshold: 0.7).",
     r"justifies excluding override rules from the final pipeline. Calibration used a confidence threshold of 0.7."),
    (r" Results are stored in \texttt{results/kfold\_cv.json}.", r""),
    (r"measured from \texttt{data/processed/cleaned\_dataset.csv} ($n=14{,}146$ rows, verified).}",
     r"measured from the cleaned dataset ($n=14{,}146$ rows, verified).}"),
    (r"criticality--class matrix. All nine JSONL result files are in \texttt{results/nslkdd\_ablation/}; the unified runner is \texttt{scripts/run\_nslkdd\_full\_ablation.py}.",
     r"criticality--class matrix."),
    (r"Section~\ref{sec:data-sources}) logged in \texttt{results/multisource\_arg.json}: CIC-IDS2018 (16.2M",
     r"Section~\ref{sec:data-sources}): CIC-IDS2018 (16.2M"),
    (r"(sub-project B in Appendix~\ref{app:nfcrm-compliance}). The implementation is in \texttt{pipeline/nfcrm/} and is invoked via \texttt{scripts/compute\_inherent\_risk.py}.",
     r"(sub-project B in Appendix~\ref{app:nfcrm-compliance}). It is implemented in the framework's NFCRM module."),
    (r"computed by \texttt{pipeline/nfcrm} per NFCRM-1:2025 ", r"computed by the NFCRM module per NFCRM-1:2025 "),
    (r" N/A for Benign and empty predictions. Full per-flow output is in \texttt{results/inherent\_risk\_register.json}.",
     r" N/A for Benign and empty predictions."),
    (r"Four NFCRM-1:2025 control rules in \texttt{pipeline/post\_inference/grc\_annotation\_rules.py} now carry",
     r"Four NFCRM-1:2025 control rules in the post-inference annotation engine now carry"),
    (r"Pure-rule baseline does not generate automated GRC artifacts. Full results in \texttt{results/grc\_metrics.json}.}",
     r"Pure-rule baseline does not generate automated GRC artifacts.}"),
    (r"are shown in Table~\ref{tab:in-inference-ssh} and stored in \texttt{results/in\_inference\_ssh.json}.",
     r"are shown in Table~\ref{tab:in-inference-ssh}."),
    (r"rule-only accuracy, source: \texttt{results/rule\_baseline.json})", r"rule-only accuracy)"),
    (r"across three additional dimensions. Sources: \texttt{scripts/run\_cic\_risk\_ablation.py}, \texttt{scripts/analyze\_cic\_risk\_ablation.py}, \texttt{scripts/analyze\_cic\_risk\_symgt.py}; results: \texttt{results/cic\_risk\_ablation/summary.json}, \texttt{results/cic\_risk\_ablation/summary\_symgt.json}.",
     r"across three additional dimensions."),
    (r" Sources: \texttt{scripts/run\_cic\_risk\_ablation.py}, \texttt{scripts/analyze\_cic\_risk\_multirun.py}; results: \texttt{results/cic\_risk\_ablation/runs/}, \texttt{results/cic\_risk\_ablation/multirun\_summary.json}.",
     r""),
    (r" This analysis is stored in \texttt{results/confidence\_gated\_override.json} and \texttt{results/shap\_top20\_features.json}.",
     r""),
    (r"The \texttt{correct} and \texttt{accuracy} fields in \texttt{results/slm\_multisource\_demo.json} are \texttt{null} by design.",
     r"The \texttt{correct} and \texttt{accuracy} fields in the multisource LLM demo output are \texttt{null} by design."),
    (r"Generated narratives from \texttt{results/slm\_multisource\_demo.json} cite CVEs",
     r"Generated narratives from the multisource LLM demo cite CVEs"),
    (r"\textbf{Representative NL descriptions} (from \texttt{results/slm\_nl\_classification.json}):",
     r"\textbf{Representative NL descriptions}:"),
    (r"\textbf{E2E Pipeline Benchmark}: From \texttt{results/e2e\_pipeline\_benchmark.json}:",
     r"\textbf{E2E Pipeline Benchmark}:"),
    (r"; \texttt{random\_state=42}; source: \texttt{results/layer4\_evaluation.json}, script: \texttt{scripts/run\_layer4\_evaluation.py}).",
     r"; \texttt{random\_state=42})."),
    (r"using the 3-run variance data extracted from \texttt{results/llm\_risk\_generation.json} and \texttt{results/llm\_control\_recommendation.json}.",
     r"using the 3-run variance data from both LLM generation experiments."),
    (r" in the system prompt; source: \texttt{results/unified\_ablation/tristage\_llm\_sonnet.jsonl}. This limitation",
     r" in the system prompt. This limitation"),
    (r"\ref{tab:unified-ablation}; \texttt{results/ablation\_study\_xgb.json}; \texttt{results/unified\_ablation/*.jsonl}; Section~\ref{sec:five-paradigm} (Exp~2/3).",
     r"\ref{tab:unified-ablation}; Section~\ref{sec:five-paradigm} (Exp~2/3)."),
    (r"$\to$ \texttt{pipeline/nfcrm/} sub-projects B--G (clause-by-clause deliverables).}",
     r"$\to$ NFCRM module sub-projects B--G (clause-by-clause deliverables).}"),
    (r"Refinement: sub-projects B--G in \texttt{pipeline/nfcrm/} now implement clause-by-clause deliverable functions, validated by 57 unit tests; \texttt{scripts/run\_nfcrm\_full\_cycle.py} demonstrates the full Identification $\to$ Assessment $\to$ Treatment $\to$ Monitoring cycle.",
     r"Refinement: sub-projects B--G in the NFCRM module now implement clause-by-clause deliverable functions, validated by 57 unit tests, with a full-cycle demonstration exercising the Identification $\to$ Assessment $\to$ Treatment $\to$ Monitoring cycle."),
    (r"confirming the headline result is reproducible.\footnote{Results saved in \texttt{results/t5\_multirun\_results.json}.}",
     r"confirming the headline result is reproducible."),
    (r"(Section~\ref{sec:confidence-gated}; source: \texttt{results/confidence\_gated\_override.json}):",
     r"(Section~\ref{sec:confidence-gated}):"),
    (r"57 unit tests validate the implementation; \texttt{scripts/run\_nfcrm\_full\_cycle.py} demonstrates the full Identification $\rightarrow$ Assessment $\rightarrow$ Treatment $\rightarrow$ Monitoring cycle end-to-end.",
     r"57 unit tests validate the implementation, with a full-cycle demonstration exercising the Identification $\rightarrow$ Assessment $\rightarrow$ Treatment $\rightarrow$ Monitoring cycle end-to-end."),
    (r"algorithmically supported by the \texttt{pipeline/nfcrm/} module", r"algorithmically supported by the NFCRM module"),
    (r"the algorithm or schema the clause requires is present in \texttt{pipeline/nfcrm/}, with unit tests validating it.",
     r"the algorithm or schema the clause requires is present in the NFCRM module, with unit tests validating it."),
]


def main():
    with open(TEX, encoding="utf-8") as f:
        lines = f.readlines()

    out = []
    changed = 0
    still_has = []
    for i, line in enumerate(lines, 1):
        if in_keep(i) or not PATH_RE.search(line):
            out.append(line)
            continue
        new = clean(line)
        if new != line:
            changed += 1
            print(f"--- line {i} ---")
            print("OLD:", line.rstrip()[:300])
            print("NEW:", new.rstrip()[:300])
        out.append(new)

    text = "".join(out)

    # apply exact manual rewrites
    missing = []
    for old, new in MANUAL_PAIRS:
        if old in text:
            text = text.replace(old, new)
        else:
            missing.append(old[:70])

    # final tidy on affected text only is unsafe globally; just collapse runs
    # of 2+ spaces that follow content (preserves indentation).
    text = re.sub(r"(?<=\S)  +", " ", text)

    with open(TEX, "w", encoding="utf-8") as f:
        f.write(text)

    # report any stripped-zone paths still present outside keep zones
    leftover = []
    for i, line in enumerate(text.splitlines(), 1):
        if in_keep(i):
            continue
        if PATH_RE.search(line):
            leftover.append(i)

    print(f"\n=== changed {changed} lines via regex ===")
    print(f"=== manual pairs NOT found (check): {missing} ===")
    print(f"=== stripped-zone paths still present outside keep zones: {leftover} ===")


if __name__ == "__main__":
    main()
