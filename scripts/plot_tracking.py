"""
plot_tracking.py -- visualize TXT logs from play.py --log_tracking.

Layouts:
  --layout paper (default): multi-panel figure (3D path, velocity, errors, top view, attitude, LQR)
  --layout simple: legacy 4x2 grid

Usage:
  python scripts/plot_tracking.py path/to/track_xxx.txt
  python scripts/plot_tracking.py a.txt b.txt --labels "A" "B" --save --out figure.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- register 3d projection

plt.rcParams.update({
    "figure.figsize": (18, 12),
    "axes.grid": True,
    "grid.alpha": 0.35,
    "lines.linewidth": 1.4,
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
})

# Paper-style palette (multi-curve)
COLORS = [
    "#2e7d32",  # green
    "#c62828",  # red-brown
    "#1565c0",  # blue
    "#f9a825",  # yellow/gold (reference / analytic accent)
    "#6a1b9a",
    "#00838f",
    "#4e342e",
    "#ad1457",
]


def load_log(path: str) -> dict:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(
            f"Log file not found:\n  {p}\n"
            "Pass the full path to a TXT produced by play.py --log_tracking.\n"
            "Typical: logs/rsl_rl/<experiment>/<run>/tracking_logs/track_*.txt"
        )
    data = np.genfromtxt(str(p), delimiter=",", names=True, dtype=float)
    if data.size == 0:
        raise ValueError(f"Empty or unparseable file: {p}")
    return {name: data[name] for name in data.dtype.names}


def compute_metrics(d) -> dict:
    vx_err = d["cmd_vx"] - d["act_vx"]
    wz_err = d["cmd_wz"] - d["act_wz"]
    m = {
        "vx_rmse": float(np.sqrt(np.mean(vx_err**2))),
        "wz_rmse": float(np.sqrt(np.mean(wz_err**2))),
        "pitch_rmse_deg": float(np.degrees(np.sqrt(np.mean(d["pitch_rad"]**2)))),
        "roll_rmse_deg": float(np.degrees(np.sqrt(np.mean(d["roll_rad"]**2)))),
        "orient_rmse_rad": float(np.sqrt(np.mean(d["pitch_rad"]**2 + d["roll_rad"]**2))),
        "mean_height": float(np.mean(d["pos_z"])),
        "steps": len(d["time_s"]),
        "duration_s": float(d["time_s"][-1] - d["time_s"][0]),
    }
    if "ref_x" in d:
        pe = np.sqrt((d["pos_x"] - d["ref_x"]) ** 2 + (d["pos_y"] - d["ref_y"]) ** 2)
        m["pos_rmse_mm"] = float(np.sqrt(np.mean(pe**2)) * 1000.0)
        m["pos_max_mm"] = float(np.max(pe) * 1000.0)
    return m


# ---- legacy simple layout -------------------------------------------------

def plot_velocity_tracking(ax_vx, ax_wz, d, label, color):
    t = d["time_s"]
    ax_vx.plot(t, d["cmd_vx"], "--", color=color, alpha=0.5, label=f"{label} cmd vx")
    ax_vx.plot(t, d["act_vx"], color=color, label=f"{label} act vx")
    ax_vx.set_ylabel("Vx (m/s)")
    ax_vx.set_title("Linear Velocity Tracking (X)")

    ax_wz.plot(t, d["cmd_wz"], "--", color=color, alpha=0.5, label=f"{label} cmd ωz")
    ax_wz.plot(t, d["act_wz"], color=color, label=f"{label} act ωz")
    ax_wz.set_ylabel("ωz (rad/s)")
    ax_wz.set_title("Angular Velocity Tracking (Yaw)")


def plot_orientation(ax_pitch, ax_roll, d, label, color):
    t = d["time_s"]
    pitch_deg = np.degrees(d["pitch_rad"])
    roll_deg = np.degrees(d["roll_rad"])
    ax_pitch.plot(t, pitch_deg, color=color, label=label)
    ax_pitch.axhline(0, color="k", ls=":", lw=0.6)
    ax_pitch.set_ylabel("Pitch (°)")
    ax_pitch.set_title("Body Pitch")

    ax_roll.plot(t, roll_deg, color=color, label=label)
    ax_roll.axhline(0, color="k", ls=":", lw=0.6)
    ax_roll.set_ylabel("Roll (°)")
    ax_roll.set_title("Body Roll")


def plot_actions(ax, d, label, color):
    t = d["time_s"]
    ax.plot(t, d["action_0"], color=color, alpha=0.7, label=f"{label} a0 DLM")
    ax.plot(t, d["action_1"], "--", color=color, alpha=0.7, label=f"{label} a1 DLP")
    ax.set_ylabel("Action (rad)")
    ax.set_title("Joint Actions (Left Leg)")


def plot_lqr_residual(ax, d, label, color):
    t = d["time_s"]
    ax.plot(t, d["lqr_left"], color=color, label=f"{label} LQR L")
    ax.plot(t, d["lqr_right"], "--", color=color, alpha=0.7, label=f"{label} LQR R")
    ax.set_ylabel("Residual (rad)")
    ax.set_title("LQR Residual")


def plot_trajectory_2d(ax, d, label, color):
    ax.plot(d["pos_x"], d["pos_y"], color=color, label=label)
    ax.plot(d["pos_x"][0], d["pos_y"][0], "o", color=color, ms=8)
    ax.plot(d["pos_x"][-1], d["pos_y"][-1], "s", color=color, ms=8)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("2D Trajectory (Top View)")
    ax.set_aspect("equal", adjustable="datalim")


def plot_height(ax, d, label, color):
    t = d["time_s"]
    ax.plot(t, d["pos_z"], color=color, label=label)
    ax.set_ylabel("Height Z (m)")
    ax.set_title("Body Height")


def run_simple_layout(files, datasets, labels, dpi, out_path: Path | None, suptitle: str | None = None):
    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    _st = suptitle if suptitle is not None else default_paper_suptitle(labels) + " (simple)"
    fig.suptitle(_st, fontsize=14, fontweight="bold")

    for i, (d, lb) in enumerate(zip(datasets, labels)):
        c = COLORS[i % len(COLORS)]
        plot_velocity_tracking(axes[0, 0], axes[0, 1], d, lb, c)
        plot_orientation(axes[1, 0], axes[1, 1], d, lb, c)
        plot_actions(axes[2, 0], d, lb, c)
        plot_lqr_residual(axes[2, 1], d, lb, c)
        plot_trajectory_2d(axes[3, 0], d, lb, c)
        plot_height(axes[3, 1], d, lb, c)

    for row in axes:
        for ax in row:
            ax.legend(fontsize=7, loc="best")
    axes[3, 0].set_xlabel("X (m)")
    axes[3, 1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if out_path:
        fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
        print(f"\n[INFO] Figure saved -> {out_path}")
    else:
        plt.show()
    plt.close(fig)


# ---- paper-style layout (multi-panel trajectory figure) -------------------------

def default_paper_suptitle(labels: list[str]) -> str:
    """Auto suptitle for two-way or multi-way comparisons."""
    if len(labels) == 2:
        return f"{labels[0]}  vs  {labels[1]}\nNexusTrex · velocity / attitude / trajectory"
    if len(labels) > 2:
        return "  |  ".join(labels) + "\nNexusTrex · velocity / attitude / trajectory"
    return f"{labels[0]}\nNexusTrex · play log"


def run_paper_layout(
    files,
    datasets,
    labels,
    dpi,
    out_path: Path | None,
    suptitle: str | None = None,
):
    """Multi-panel layout: 3D path, velocity, errors, top view, attitude, LQR."""
    fig = plt.figure(figsize=(18, 13.5))
    _st = suptitle if suptitle is not None else default_paper_suptitle(labels)
    fig.suptitle(_st, fontsize=13, fontweight="bold", y=0.995)
    # 4 rows: column 0 has (a) rows 0-1, (d) row 2, (g) row 3; avoids GridSpec overlap
    gs = GridSpec(
        4,
        3,
        figure=fig,
        height_ratios=[1.0, 1.0, 1.0, 1.0],
        hspace=0.42,
        wspace=0.38,
        left=0.06,
        right=0.98,
        top=0.92,
        bottom=0.07,
    )

    has_ref = "ref_x" in datasets[0]

    # (a) 3D trajectory — mm
    ax3d = fig.add_subplot(gs[0:2, 0], projection="3d")
    if has_ref:
        d0 = datasets[0]
        rx = d0["ref_x"] * 1000.0
        ry = d0["ref_y"] * 1000.0
        rz = np.full_like(rx, np.mean(d0["pos_z"]) * 1000.0)
        ax3d.plot(rx, ry, rz, "k--", lw=1.0, alpha=0.55, label="Reference")
    for i, (d, lb) in enumerate(zip(datasets, labels)):
        c = COLORS[i % len(COLORS)]
        xm = d["pos_x"] * 1000.0
        ym = d["pos_y"] * 1000.0
        zm = d["pos_z"] * 1000.0
        ax3d.plot(xm, ym, zm, color=c, label=lb, lw=1.6)
        ax3d.scatter(xm[0], ym[0], zm[0], color=c, s=36, marker="o", edgecolors="k", linewidths=0.5, zorder=5)
        ax3d.scatter(xm[-1], ym[-1], zm[-1], color=c, s=36, marker="s", edgecolors="k", linewidths=0.5, zorder=5)
    ax3d.set_xlabel("X (mm)")
    ax3d.set_ylabel("Y (mm)")
    ax3d.set_zlabel("Z (mm)")
    ax3d.set_title("(a) 3D trajectory")
    ax3d.legend(loc="upper left", fontsize=7, bbox_to_anchor=(0.0, 1.02))
    try:
        ax3d.view_init(elev=22, azim=-55)
    except Exception:
        pass
    try:
        ax3d.set_box_aspect((1.0, 1.0, 0.55))
    except Exception:
        pass

    # Plot reference command once from first log (visual only; runs may differ slightly)
    d0 = datasets[0]
    t0 = d0["time_s"]

    # (b) Linear velocity: left — cmd ref + actual; right — |error|
    ax_vx = fig.add_subplot(gs[0, 1])
    ax_vx_r = ax_vx.twinx()
    ax_vx.plot(t0, d0["cmd_vx"], "k--", lw=1.2, alpha=0.85, label="Cmd $v_x$ (ref.)")
    for i, (d, lb) in enumerate(zip(datasets, labels)):
        c = COLORS[i % len(COLORS)]
        ax_vx.plot(d["time_s"], d["act_vx"], color=c, lw=1.2, label=f"{lb} $v_x$")
        ev = np.abs(d["cmd_vx"] - d["act_vx"])
        ax_vx_r.plot(d["time_s"], ev, color=c, lw=1.0, ls=":", alpha=0.9, label=f"{lb} $|e_{{v_x}}|$")
    ax_vx.set_xlabel("Time (s)")
    ax_vx.set_ylabel("$v_x$ (m/s)", color="0.2")
    ax_vx_r.set_ylabel("$|v_x^{cmd}-v_x^{act}|$ (m/s)", color="0.35")
    ax_vx.set_title("(b) $v_x$: command, actual, $|e_{v_x}|$ (twin axis)")
    ax_vx.tick_params(axis="y", labelcolor="0.2")
    ax_vx_r.tick_params(axis="y", labelcolor="0.45")
    h1, l1 = ax_vx.get_legend_handles_labels()
    h2, l2 = ax_vx_r.get_legend_handles_labels()
    ax_vx.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=6, ncol=2)

    # (c) Angular velocity z: same pattern
    ax_wz = fig.add_subplot(gs[0, 2])
    ax_wz_r = ax_wz.twinx()
    ax_wz.plot(t0, d0["cmd_wz"], "k--", lw=1.2, alpha=0.85, label=r"Cmd $\omega_z$ (ref.)")
    for i, (d, lb) in enumerate(zip(datasets, labels)):
        c = COLORS[i % len(COLORS)]
        ax_wz.plot(d["time_s"], d["act_wz"], color=c, lw=1.2, label=rf"{lb} $\omega_z$")
        ew = np.abs(d["cmd_wz"] - d["act_wz"])
        ax_wz_r.plot(d["time_s"], ew, color=c, lw=1.0, ls=":", alpha=0.9, label=rf"{lb} $|e_{{\omega_z}}|$")
    ax_wz.set_xlabel("Time (s)")
    ax_wz.set_ylabel(r"$\omega_z$ (rad/s)", color="0.2")
    ax_wz_r.set_ylabel(r"$|\omega_z^{cmd}-\omega_z^{act}|$ (rad/s)", color="0.35")
    ax_wz.set_title(r"(c) $\omega_z$: command, actual, $|e_{\omega_z}|$ (twin axis)")
    ax_wz.tick_params(axis="y", labelcolor="0.2")
    ax_wz_r.tick_params(axis="y", labelcolor="0.45")
    h1, l1 = ax_wz.get_legend_handles_labels()
    h2, l2 = ax_wz_r.get_legend_handles_labels()
    ax_wz.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=6, ncol=2)

    # (d) Proxy stress metric: combined velocity error (when no Jacobian condition number)
    ax_cond = fig.add_subplot(gs[2, 0])
    if has_ref:
        for i, (d, lb) in enumerate(zip(datasets, labels)):
            c = COLORS[i % len(COLORS)]
            pe = np.sqrt((d["pos_x"] - d["ref_x"]) ** 2 + (d["pos_y"] - d["ref_y"]) ** 2)
            ax_cond.plot(d["time_s"], pe * 1000.0, color=c, lw=1.3, label=lb)
        ax_cond.set_xlabel("Time (s)")
        ax_cond.set_ylabel("Position error (mm)")
        ax_cond.set_title("(d) Closed-loop position tracking error")
    else:
        for i, (d, lb) in enumerate(zip(datasets, labels)):
            c = COLORS[i % len(COLORS)]
            vx_e = np.abs(d["cmd_vx"] - d["act_vx"])
            wz_e = np.abs(d["cmd_wz"] - d["act_wz"])
            vy_e = np.abs(d["cmd_vy"] - d["act_vy"])
            stress = np.sqrt(vx_e ** 2 + vy_e ** 2 + (0.15 * wz_e) ** 2)
            ax_cond.plot(d["time_s"], stress, color=c, lw=1.3, label=lb)
        ax_cond.set_xlabel("Time (s)")
        ax_cond.set_ylabel(r"$\Vert e_v \Vert_{\mathrm{proxy}}$ (m/s)")
        ax_cond.set_title("(d) Combined velocity error (tracking stress)")
    ax_cond.legend(loc="upper right", fontsize=7)

    # (e) Linear velocity error magnitude
    ax_ev = fig.add_subplot(gs[1, 1])
    for i, (d, lb) in enumerate(zip(datasets, labels)):
        c = COLORS[i % len(COLORS)]
        ev = np.abs(d["cmd_vx"] - d["act_vx"])
        ax_ev.plot(d["time_s"], ev, color=c, lw=1.3, label=lb)
    ax_ev.set_xlabel("Time (s)")
    ax_ev.set_ylabel(r"$|e_{v_x}|$ (m/s)")
    ax_ev.set_title("(e) Scalar $|e_{v_x}|$ (no command overlay)")
    ax_ev.legend(loc="upper right", fontsize=7)

    # (f) Angular velocity error
    ax_ew = fig.add_subplot(gs[1, 2])
    for i, (d, lb) in enumerate(zip(datasets, labels)):
        c = COLORS[i % len(COLORS)]
        ew = np.abs(d["cmd_wz"] - d["act_wz"])
        ax_ew.plot(d["time_s"], ew, color=c, lw=1.3, label=lb)
    ax_ew.set_xlabel("Time (s)")
    ax_ew.set_ylabel(r"$|e_{\omega_z}|$ (rad/s)")
    ax_ew.set_title(r"(f) Scalar $|e_{\omega_z}|$ (no command overlay)")
    ax_ew.legend(loc="upper right", fontsize=7)

    # (g) Top view XY (bottom of column 0)
    ax_xy = fig.add_subplot(gs[3, 0])
    if has_ref:
        d0r = datasets[0]
        ax_xy.plot(d0r["ref_x"], d0r["ref_y"], "k--", lw=1.2, alpha=0.6, label="Reference")
    for i, (d, lb) in enumerate(zip(datasets, labels)):
        c = COLORS[i % len(COLORS)]
        ax_xy.plot(d["pos_x"], d["pos_y"], color=c, lw=1.4, label=lb)
        ax_xy.plot(d["pos_x"][0], d["pos_y"][0], "o", color=c, ms=7)
        ax_xy.plot(d["pos_x"][-1], d["pos_y"][-1], "s", color=c, ms=7)
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title("(g) Planar trajectory (top view)")
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.legend(loc="best", fontsize=7)

    # (h) Attitude error -- spans rows 2-3
    ax_att = fig.add_subplot(gs[2:4, 1])
    for i, (d, lb) in enumerate(zip(datasets, labels)):
        c = COLORS[i % len(COLORS)]
        att = np.sqrt(d["pitch_rad"] ** 2 + d["roll_rad"] ** 2)
        ax_att.plot(d["time_s"], np.degrees(att), color=c, lw=1.3, label=lb)
    ax_att.set_xlabel("Time (s)")
    ax_att.set_ylabel(r"$\sqrt{\phi^2+\psi^2}$ (deg)")
    ax_att.set_title("(h) Attitude deviation (norm)")
    ax_att.legend(loc="upper right", fontsize=7)

    # (i) LQR residual -- same height as (h)
    ax_lqr = fig.add_subplot(gs[2:4, 2])
    for i, (d, lb) in enumerate(zip(datasets, labels)):
        c = COLORS[i % len(COLORS)]
        lq = np.sqrt(d["lqr_left"] ** 2 + d["lqr_right"] ** 2)
        ax_lqr.plot(d["time_s"], lq, color=c, lw=1.3, label=f"{lb} $\\|u_{{LQR}}\\|$")
    ax_lqr.set_xlabel("Time (s)")
    ax_lqr.set_ylabel("LQR residual norm (rad)")
    ax_lqr.set_title("(i) LQR residual magnitude")
    ax_lqr.legend(loc="upper right", fontsize=7)

    if out_path:
        fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
        print(f"\n[INFO] Figure saved -> {out_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot play.py --log_tracking TXT logs")
    parser.add_argument("files", nargs="+", help="One or more TXT log paths")
    parser.add_argument("--labels", nargs="*", default=None, help="Legend labels (same order as files)")
    parser.add_argument("--layout", choices=["paper", "simple"], default="paper", help="paper or simple 4x2")
    parser.add_argument("--save", action="store_true", help="Save PNG (no GUI)")
    parser.add_argument("--out", type=str, default=None, help="Output PNG path; default next to first TXT")
    parser.add_argument("--dpi", type=int, default=200, help="PNG DPI")
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Figure suptitle; default auto from labels",
    )
    args = parser.parse_args()

    files = args.files
    labels = args.labels or [Path(f).stem for f in files]
    if len(labels) < len(files):
        labels += [Path(f).stem for f in files[len(labels):]]

    datasets = []
    for f in files:
        d = load_log(f)
        datasets.append(d)
        m = compute_metrics(d)
        print(f"\n=== {f} ===")
        print(f"  Steps       : {m['steps']}")
        print(f"  Duration    : {m['duration_s']:.2f} s")
        print(f"  Vx RMSE     : {m['vx_rmse']:.4f} m/s")
        print(f"  wz RMSE     : {m['wz_rmse']:.4f} rad/s")
        print(f"  Pitch RMSE  : {m['pitch_rmse_deg']:.2f} deg")
        print(f"  Roll RMSE   : {m['roll_rmse_deg']:.2f} deg")
        print(f"  Orient RMSE : {m['orient_rmse_rad']:.5f} rad")
        print(f"  Mean Height : {m['mean_height']:.4f} m")
        if "pos_rmse_mm" in m:
            print(f"  Pos RMSE    : {m['pos_rmse_mm']:.1f} mm")
            print(f"  Pos Max Err : {m['pos_max_mm']:.1f} mm")

    n = len(datasets)

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    elif args.save:
        out_path = Path(files[0]).expanduser().resolve().with_suffix(".png")
    else:
        out_path = None

    save_or_show_path = out_path if (args.save or args.out) else None
    if args.layout == "paper":
        run_paper_layout(files, datasets, labels, args.dpi, save_or_show_path, suptitle=args.title)
    else:
        run_simple_layout(files, datasets, labels, args.dpi, save_or_show_path, suptitle=args.title)

    if n > 1:
        print("\n" + "=" * 70)
        print(f"{'Metric':<20}", end="")
        for lb in labels:
            print(f"{lb:>20}", end="")
        print()
        print("-" * 70)
        all_m = [compute_metrics(d) for d in datasets]
        table_keys = ["vx_rmse", "wz_rmse", "pitch_rmse_deg", "roll_rmse_deg", "orient_rmse_rad", "mean_height"]
        if "pos_rmse_mm" in all_m[0]:
            table_keys += ["pos_rmse_mm", "pos_max_mm"]
        for key in table_keys:
            print(f"{key:<20}", end="")
            for m in all_m:
                print(f"{m[key]:>20.4f}", end="")
            print()
        print("=" * 70)


if __name__ == "__main__":
    main()
