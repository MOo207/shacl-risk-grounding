# Rule+LLM runner — haiku + sonnet, 3 SHACL + 1 no-SHACL each, N=3 per class (30 cases)
$ErrorActionPreference = "Continue"
$logfile = "results/llm_experiment/raw/_rule_llm_runner.log"
"$(Get-Date -Format o) START" | Out-File -Encoding utf8 $logfile

$N_PER_CLASS = 3   # 3 × 5 classes × 2 subsets = 30 cases total

$jobs = @(
    # --- Sonnet (3 SHACL + 1 no-SHACL) ---
    @{model="claude-sonnet-4-6"; run=1; shacl=$true;  out="results/llm_experiment/raw/run_1_arm_rule_llm_sonnet.jsonl"},
    @{model="claude-sonnet-4-6"; run=2; shacl=$true;  out="results/llm_experiment/raw/run_2_arm_rule_llm_sonnet.jsonl"},
    @{model="claude-sonnet-4-6"; run=3; shacl=$true;  out="results/llm_experiment/raw/run_3_arm_rule_llm_sonnet.jsonl"},
    @{model="claude-sonnet-4-6"; run=1; shacl=$false; out="results/llm_experiment/raw/run_1_arm_rule_llm_sonnet_no_shacl.jsonl"},

    # --- Haiku (3 SHACL + 1 no-SHACL) ---
    @{model="claude-haiku-4-5-20251001"; run=1; shacl=$true;  out="results/llm_experiment/raw/run_1_arm_rule_llm_haiku.jsonl"},
    @{model="claude-haiku-4-5-20251001"; run=2; shacl=$true;  out="results/llm_experiment/raw/run_2_arm_rule_llm_haiku.jsonl"},
    @{model="claude-haiku-4-5-20251001"; run=3; shacl=$true;  out="results/llm_experiment/raw/run_3_arm_rule_llm_haiku.jsonl"},
    @{model="claude-haiku-4-5-20251001"; run=1; shacl=$false; out="results/llm_experiment/raw/run_1_arm_rule_llm_haiku_no_shacl.jsonl"}
)

foreach ($j in $jobs) {
    $cmdArgs = @("-m", "scripts.run_llm_artifact_arm_rule_llm",
                 "--test-set", "results/llm_experiment/test_set.json",
                 "--out", $j.out,
                 "--model", $j.model,
                 "--n-per-class", $N_PER_CLASS)
    if (-not $j.shacl) { $cmdArgs += "--no-shacl" }

    $label = "$($j.model) run=$($j.run) shacl=$($j.shacl)"
    "$(Get-Date -Format o) BEGIN $label -> $($j.out)" | Add-Content -Encoding utf8 $logfile
    Write-Host "`n=== $label ===" -ForegroundColor Cyan
    & python @cmdArgs 2>&1 | Tee-Object -Append -FilePath $logfile
    "$(Get-Date -Format o) END   $label exit=$LASTEXITCODE" | Add-Content -Encoding utf8 $logfile
}

"$(Get-Date -Format o) ALL DONE" | Add-Content -Encoding utf8 $logfile
Write-Host "`nAll jobs finished. See $logfile" -ForegroundColor Green
