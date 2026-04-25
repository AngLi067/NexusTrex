"""
Build two figures from four closed-loop TXT logs:
  1) Top-down XY overlay (reference + four conditions)
  2) Position RMSE bar chart (recomputed from logs)

Defaults read cv_circle/b.txt, d.txt, s.txt, d_lqr.txt under repo root;
writes report/figures/traj_fourway_xy.png and traj_fourway_rmse.png.

  python scripts/cv_overview.py
  python scripts/cv_overview.py --out-xy report/figures/traj_fourway_xy.png \\
      --out-rmse report/figures/traj_fourway_rmse.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

BASE_RC = {
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "lines.linewidth": 1.35,
}

COLORS = ["#1565c0", "#c62828", "#2e7d32", "#6a1b9a"]
DEFAULT_FILES = ["cv_circle/b.txt", "cv_circle/d.txt", "cv_circle/s.txt", "cv_circle/d_lqr.txt"]
DEFAULT_LABELS = ["Teacher", "NoScanner", "WithScanner", "NoScanner+LQR"]
DEF_XY = ROOT / "report" / "figures" / "traj_fourway_xy.png"
DEF_RMSE = ROOT / "report" / "figures" / "traj_fourway_rmse.png"


def load_log(path: Path) -> dict:
    data = np.genfromtxt(str(path), delimiter=",", names=True, dtype=float)
    if data.size == 0:
        raise ValueError(f"empty or invalid: {path}")
    return {name: data[name] for name in data.dtype.names}


def pos_rmse_mm(d: dict) -> float:
    pe = np.sqrt((d["pos_x"] - d["ref_x"]) ** 2 + (d["pos_y"] - d["ref_y"]) ** 2)
    return float(np.sqrt(np.mean(pe**2)) * 1000.0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Four-condition benchmark: top-down trajectory + RMSE bar chart."
    )
    ap.add_argument(
        "txts",
        nargs="*",
        default=None,
        help="Four TXT paths (order: Teacher, NoScanner, WithScanner, NoScanner+LQR)",
    )
    ap.add_argument("--labels", nargs=4, default=None, metavar=("L1", "L2", "L3", "L4"))
    ap.add_argument("--out-xy", type=str, default=str(DEF_XY), help="Output top-view PNG")
    ap.add_argument("--out-rmse", type=str, default=str(DEF_RMSE), help="Output RMSE bar PNG")
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    rels = args.txts if args.txts else DEFAULT_FILES
    if len(rels) != 4:
        ap.error("Need exactly four TXT paths (or rely on defaults).")
    labels = list(args.labels) if args.labels else DEFAULT_LABELS

    paths: list[Path] = []
    for r in rels:
        p = Path(r)
        if not p.is_file():
            p = ROOT / r
        if not p.is_file():
            raise SystemExit(f"Log not found: {r}")
        paths.append(p.resolve())

    datasets = [load_log(p) for p in paths]
    for d in datasets:
        for k in ("pos_x", "pos_y", "ref_x", "ref_y"):
            if k not in d:
                raise SystemExit(f"Log missing column: {k}")

    d0 = datasets[0]
    rmses: list[float] = []

    plt.rcParams.update({**BASE_RC, "figure.figsize": (5.9, 5.2)})
    fig1, ax_xy = plt.subplots()
    ax_xy.plot(
        d0["ref_x"],
        d0["ref_y"],
        "k--",
        lw=1.25,
        alpha=0.75,
        label="Reference (figure-eight)",
    )
    for d, lb, c in zip(datasets, labels, COLORS):
        ax_xy.plot(d["pos_x"], d["pos_y"], color=c, label=lb)
        ax_xy.plot(d["pos_x"][0], d["pos_y"][0], "o", color=c, ms=5, zorder=5)
        ax_xy.plot(d["pos_x"][-1], d["pos_y"][-1], "s", color=c, ms=5, zorder=5)
        rmses.append(pos_rmse_mm(d))

    ax_xy.set_xlabel("World $x$ (m)")
    ax_xy.set_ylabel("World $y$ (m)")
    ax_xy.set_title("Closed-loop benchmark: top view")
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.grid(True, alpha=0.35)
    ax_xy.legend(loc="best", framealpha=0.92)
    out_xy = Path(args.out_xy)
    if not out_xy.is_absolute():
        out_xy = (ROOT / out_xy).resolve()
    out_xy.parent.mkdir(parents=True, exist_ok=True)
    fig1.savefig(out_xy, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig1)

    plt.rcParams.update({**BASE_RC, "figure.figsize": (6.2, 4.0)})
    fig2, ax_bar = plt.subplots()
    xpos = np.arange(len(labels))
    ax_bar.bar(xpos, rmses, color=COLORS, edgecolor="0.2", linewidth=0.6)
    ax_bar.set_xticks(xpos)
    ax_bar.set_xticklabels(labels, rotation=22, ha="right")
    ax_bar.set_ylabel("Position RMSE (mm)")
    ax_bar.set_title("Position RMSE recomputed from logs")
    ax_bar.grid(True, axis="y", alpha=0.35)
    out_rmse = Path(args.out_rmse)
    if not out_rmse.is_absolute():
        out_rmse = (ROOT / out_rmse).resolve()
    out_rmse.parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(out_rmse, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig2)

    print(f"[OK] {out_xy}")
    print(f"[OK] {out_rmse}")
    for lb, v in zip(labels, rmses):
        print(f"     {lb}: Pos RMSE = {v:.1f} mm")


if __name__ == "__main__":
    main()
