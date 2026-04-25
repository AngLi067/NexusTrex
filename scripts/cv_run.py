"""
cv_run.py -- controlled runs: play short benchmark tasks (nexustrex-cv-*-v0), short TXT names.

By default prints commands only; pass --run to execute (each launch starts Isaac; slow).

Output layout (editable):
  NexusTrex/cv_out/b.txt, d.txt, s.txt, d_lqr.txt ...

Usage:
  python scripts/cv_run.py --teacher ... --d ... --s ... --circle --run
  python scripts/cv_run.py --teacher ... --d ... --s ... --circle --run --traj eight --traj_smooth 0.15
  python scripts/cv_run.py --teacher ... --d ...  # print-only
  python scripts/cv_plot.py cv_out --order b,d,s --labels Teacher NoScanner WithScanner --name three_curve
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description="Batch controlled play (short task id + --log_out)")
    p.add_argument("--teacher", type=str, default="", help="Teacher/PPO checkpoint (cv-b)")
    p.add_argument("--d", type=str, default="", help="NoScanner student ckpt (cv-d)")
    p.add_argument("--s", type=str, default="", help="WithScanner student ckpt (cv-s)")
    p.add_argument("--out", type=str, default="cv_out", help="Output folder (relative to repo root)")
    p.add_argument("--steps", type=int, default=1000, help="--log_steps")
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--run", action="store_true", help="Execute; omit to print commands only")
    p.add_argument("--lqr_d", action="store_true", help="Also run cv-d with LQR -> d_lqr.txt")
    p.add_argument("--lqr_adaptive", action="store_true")
    p.add_argument(
        "--circle",
        action="store_true",
        help="Pass --circle to play (closed-loop tracking; default eight + feedforward/low-pass)",
    )
    p.add_argument("--traj", type=str, default=None, help="Forwarded to play: eight / circle / lissa")
    p.add_argument("--traj_smooth", type=float, default=None, help="Forwarded to play: --traj_smooth")
    args = p.parse_args()

    out_dir = (ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    play_py = ROOT / "scripts" / "rsl_rl" / "play.py"
    rows: list[tuple[str, str, str, str, list[str]]] = []

    if args.teacher:
        rows.append(
            (
                "b",
                "nexustrex-cv-b-v0",
                "rsl_rl_cfg_entry_point",
                args.teacher,
                [],
            )
        )
    if args.d:
        rows.append(
            (
                "d",
                "nexustrex-cv-d-v0",
                "rsl_rl_distillation_cfg_entry_point",
                args.d,
                [],
            )
        )
        if args.lqr_d:
            extra = ["--use_lqr"]
            if args.lqr_adaptive:
                extra += ["--lqr_adaptive"]
            rows.append(
                (
                    "d_lqr",
                    "nexustrex-cv-d-v0",
                    "rsl_rl_distillation_cfg_entry_point",
                    args.d,
                    extra,
                )
            )
    if args.s:
        rows.append(
            (
                "s",
                "nexustrex-cv-s-v0",
                "rsl_rl_distillation_cfg_entry_point",
                args.s,
                [],
            )
        )

    if not rows:
        print(
            "Provide at least one of --teacher / --d / --s. "
            "Terrain and command defaults: tasks/nexustrex_bench_task.py (BENCH_*)."
        )
        sys.exit(1)

    for name, task, agent, ckpt, extra in rows:
        log_out = out_dir / f"{name}.txt"
        cmd = [
            sys.executable,
            str(play_py),
            "--task",
            task,
            "--agent",
            agent,
            "--checkpoint",
            ckpt,
            "--num_envs",
            str(args.num_envs),
            "--log_tracking",
            "--log_steps",
            str(args.steps),
            "--log_out",
            str(log_out),
        ] + extra
        if args.circle:
            cmd.append("--circle")
            if args.traj:
                cmd += ["--traj", args.traj]
            if args.traj_smooth is not None:
                cmd += ["--traj_smooth", str(args.traj_smooth)]
        line = " ".join(f'"{c}"' if (" " in c or "\\" in c) else c for c in cmd)
        print(line)
        if args.run:
            subprocess.check_call(cmd, cwd=str(ROOT))

    print(f"\n[INFO] TXT directory: {out_dir}")
    print(
        f'       Three-way plot: python scripts/cv_plot.py "{out_dir}" --name three '
        '--title "Teacher / NoScanner / WithScanner"'
    )
    print(
        f'       Pair plot: python scripts/cv_pair.py "{out_dir / "b.txt"}" '
        f'"{out_dir / "d.txt"}" --n1 Teacher --n2 NoScanner'
    )


if __name__ == "__main__":
    main()
