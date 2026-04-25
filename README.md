# NexusTrex Quick Run Guide




Run all commands from repo root: `NexusTrex/`.

---

## 0) Environment self-check and export (`env_isaaclab`)

Minimal reproducible env file (recommended):

- `scripts/env_isaaclab_minimal.yml`

Create environment from minimal file:

```bash
conda env create -f scripts/env_isaaclab_minimal.yml
conda activate env_isaaclab
```

Local self-check result on this machine:

- Active conda env: `env_isaaclab`
- Python executable: `.conda\envs\env_isaaclab\python.exe`
- Python prefix: `.conda\envs\env_isaaclab`

Quick sanity check:

```bash
conda run -n env_isaaclab python -c "import isaaclab, isaaclab_tasks, isaaclab_rl; print('isaaclab ok')"
conda run -n env_isaaclab python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

Full snapshots (for exact backup/debug):

- `scripts/env_isaaclab_export.yml`
- `scripts/env_isaaclab_conda_list.txt`
- `scripts/env_isaaclab_pip_freeze.txt`

---

## 1) Main program: training

### 1.1 Train teacher (PPO)

```bash
python scripts/rsl_rl/train.py --task nexustrex-basic-v0  
```

### 1.2 Train student (NoScanner distillation)

```bash
python scripts/rsl_rl/train.py --task nexustrex-distillation-v0 --agent rsl_rl_distillation_cfg_entry_point
```

### 1.3 Train student (WithScanner distillation)

```bash
python scripts/rsl_rl/train.py --task nexustrex-distillation-with-scanner-v0 --agent rsl_rl_distillation_cfg_entry_point
```

> Distillation runner loads teacher from the configured `load_experiment` in
> `tasks/agents/rsl_rl_distillation_cfg.py` or `...with_scanner_cfg.py`.

---

## 2) Main program: evaluation / play

### 2.1 Basic play

```bash
python scripts/rsl_rl/play.py --task nexustrex-basic-v0 --num_envs 32
python scripts/rsl_rl/play.py --task nexustrex-distillation-v0 --agent rsl_rl_distillation_cfg_entry_point --num_envs 32
```

### 2.2 Play with LQR residual

```bash
python scripts/rsl_rl/play.py \
  --task nexustrex-distillation-v0 \
  --agent rsl_rl_distillation_cfg_entry_point \
  --checkpoint "ckpt/model.pt" \
  --num_envs 16 \
  --use_lqr \
  --lqr_gain 0.05 \
  --lqr_smooth 0.88
```

Useful LQR options in `play.py`:

- `--lqr_flip_sign` (flip residual direction)
- `--lqr_adaptive` (enable adaptive mode)
- `--lqr_adaptive_ki 0.08`
- `--lqr_adaptive_schedule 0.25`
- `--lqr_int_leak 0.998`

### 2.3 Log tracking TXT (for plotting)

```bash
python scripts/rsl_rl/play.py \
  --task nexustrex-cv-d-v0 \
  --agent rsl_rl_distillation_cfg_entry_point \
  --checkpoint "ckpt/model.pt" \
  --num_envs 16 \
  --log_tracking --log_steps 1000 --log_out cv_out/d.txt \
  --circle
```

Closed-loop tracking options:

- `--traj eight|circle|lissa`
- `--circle_r 1.5`
- `--circle_omega 0.10`
- `--circle_kp 0.5`
- `--circle_kp_yaw 1.2`
- `--traj_kp_lat 0.2`
- `--traj_smooth 0.12`
- `--traj_no_ff`

---

## 3) Batch benchmark logs (recommended)

Generate Teacher / NoScanner / WithScanner / NoScanner+LQR logs in one command:

```bash
python scripts/cv_run.py \
  --teacher "/ckpt/teacher.pt" \
  --d "ckpt/student_ns.pt" \
  --s "ckpt/student_ws.pt" \
  --lqr_d --lqr_adaptive \
  --circle --traj eight --traj_smooth 0.12 \
  --run --out cv_out
```

Outputs:

- `cv_out/b.txt`
- `cv_out/d.txt`
- `cv_out/s.txt`
- `cv_out/d_lqr.txt` (if `--lqr_d`)

---

## 4) Plot scripts

### 4.1 Pair comparison

```bash
python scripts/cv_pair.py cv_out/b.txt cv_out/d.txt --n1 Teacher --n2 NoScanner -o cv_out/pair_b_d.png
```

### 4.2 Multi-log plot in one folder

```bash
python scripts/cv_plot.py cv_out --order b,d,s,d_lqr --labels Teacher NoScanner WithScanner NoScanner+LQR --name all_compare
```

### 4.3 Four-condition paper figures

```bash
python scripts/cv_overview.py cv_out/b.txt cv_out/d.txt cv_out/s.txt cv_out/d_lqr.txt \
  --out-xy report/figures/traj_fourway_xy.png \
  --out-rmse report/figures/traj_fourway_rmse.png
```

---

## 5) Useful task IDs

- `nexustrex-basic-v0`
- `nexustrex-basic-flat-v0`
- `nexustrex-distillation-v0`
- `nexustrex-distillation-with-scanner-v0`
- `nexustrex-cv-b-v0`
- `nexustrex-cv-d-v0`
- `nexustrex-cv-s-v0`

Registered in: `tasks/__init__.py`.

---

## 6) Common issue

If a plot script says log file not found:

1. Check path from repo root.
2. Confirm `.txt` logs were generated (`cv_run.py --run` or `play.py --log_tracking`).
3. Use absolute checkpoint / log paths first.

If LQR performance is unstable:

1. Try `--lqr_flip_sign`.
2. Reduce `--lqr_gain` (example: `0.03`).
3. Increase `--lqr_smooth` (example: `0.92`).
4. Then test adaptive mode (`--lqr_adaptive`, `--lqr_adaptive_ki`, `--lqr_adaptive_schedule`).

