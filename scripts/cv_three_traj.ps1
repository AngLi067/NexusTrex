#Requires -Version 5.1
<#
  Batch: three closed-loop TXT logs (Teacher / NoScanner / WithScanner) + plots.

  Usage (from NexusTrex root, or pass -Root):
    1) Set the three checkpoint paths below
    2) Run:  .\scripts\cv_three_traj.ps1

  Outputs:
    cv_out\b.txt   -- nexustrex-cv-b-v0  Teacher/PPO
    cv_out\d.txt   -- nexustrex-cv-d-v0  distilled NoScanner
    cv_out\s.txt   -- nexustrex-cv-s-v0  distilled WithScanner
    cv_out\three_curve.png        -- three trajectories
    cv_out\pair_b_d.png etc.       -- pairwise figures

  Print commands only (no sim): add -DryRun
#>

param(
    [string] $Root = (Split-Path $PSScriptRoot -Parent),
    [string] $OutDir = "cv_out",
    [int] $Steps = 2000,
    [int] $NumEnvs = 16,
    [switch] $DryRun
)

# ========= Set your checkpoint paths here =========
$TeacherCkpt = "C:\Users\ROG\Desktop\FYP\IsaacLab\logs\rsl_rl\nexus-trex-basic-v0\YOUR_RUN\model_XXXX.pt"
$StudentDCkpt = "C:\Users\ROG\Desktop\FYP\IsaacLab\logs\rsl_rl\nexus-trex-distillation\YOUR_RUN\model_XXXX.pt"
$StudentSCkpt = "C:\Users\ROG\Desktop\FYP\IsaacLab\logs\rsl_rl\nexus-trex-distillation-with-scanner\YOUR_RUN\model_XXXX.pt"
# ================================================

Set-Location $Root
$out = Join-Path $Root $OutDir
New-Item -ItemType Directory -Force -Path $out | Out-Null

$runFlag = if ($DryRun) { @() } else { @("--run") }

Write-Host "=== 1) Generate three TXT logs (each launch starts Isaac; slow) ===" -ForegroundColor Cyan
python scripts\cv_run.py `
    --teacher $TeacherCkpt `
    --d $StudentDCkpt `
    --s $StudentSCkpt `
    --out $OutDir `
    --steps $Steps `
    --num_envs $NumEnvs `
    --circle `
    @runFlag

if ($DryRun) {
    Write-Host "`n[DryRun] No simulation run. Remove -DryRun to execute." -ForegroundColor Yellow
    exit 0
}

Write-Host "`n=== 2) Three-way figure (order Teacher / NoScanner / WithScanner) ===" -ForegroundColor Cyan
python scripts\cv_plot.py $OutDir --name three_curve --order "b,d,s" --dpi 200 `
    --labels "Teacher" "NoScanner" "WithScanner" `
    --title "Teacher vs NoScanner vs WithScanner`nNexusTrex · closed-loop traj · cv-b / cv-d / cv-s"

Write-Host "`n=== 3) Pairwise comparison figures ===" -ForegroundColor Cyan
python scripts\cv_pair.py (Join-Path $out "b.txt") (Join-Path $out "d.txt") `
    --n1 "Teacher" --n2 "NoScanner" -o (Join-Path $out "pair_b_d.png")

python scripts\cv_pair.py (Join-Path $out "d.txt") (Join-Path $out "s.txt") `
    --n1 "NoScanner" --n2 "WithScanner" -o (Join-Path $out "pair_d_s.png")

python scripts\cv_pair.py (Join-Path $out "b.txt") (Join-Path $out "s.txt") `
    --n1 "Teacher" --n2 "WithScanner" -o (Join-Path $out "pair_b_s.png")

Write-Host "`nDone. Output directory: $out" -ForegroundColor Green
Write-Host "  three_curve.png  pair_b_d.png  pair_d_s.png  pair_b_s.png"
