"""
cv_plot.py -- combine all *.txt in cv_out (or a folder) into one comparison figure.

Usage:
  python scripts/cv_plot.py
  python scripts/cv_plot.py cv_out
  python scripts/cv_plot.py cv_out --name cmp
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder", nargs="?", default="cv_out", help="Directory containing .txt logs")
    p.add_argument("--name", type=str, default="all", help="Output PNG basename (default all.png)")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--title", type=str, default=None, help="Suptitle passed to plot_tracking.py --title")
    p.add_argument(
        "--order",
        type=str,
        default=None,
        help='Comma-separated stems to plot in order, e.g. "b,d,s" (default: all *.txt sorted)',
    )
    p.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Legend labels; same order as files. E.g. --labels Teacher NoScanner WithScanner",
    )
    args = p.parse_args()

    folder = Path(args.folder)
    if not folder.is_absolute():
        folder = (ROOT / folder).resolve()
    if args.order:
        stems = [s.strip() for s in args.order.split(",") if s.strip()]
        files = []
        for st in stems:
            pth = folder / f"{st}.txt"
            if not pth.is_file():
                print(f"[ERROR] missing file: {pth}")
                sys.exit(1)
            files.append(pth)
    else:
        files = sorted(folder.glob("*.txt"))
    if not files:
        print(f"[ERROR] no TXT in: {folder}")
        sys.exit(1)

    if args.labels:
        labels = list(args.labels)
        if len(labels) < len(files):
            labels += [f.stem for f in files[len(labels) :]]
    else:
        labels = [f.stem for f in files]
    out_png = folder / f"{args.name}.png"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "plot_tracking.py"),
        *[str(f) for f in files],
        "--labels",
        *labels,
        "--layout",
        "paper",
        "--save",
        "--out",
        str(out_png),
        "--dpi",
        str(args.dpi),
    ]
    if args.title:
        cmd += ["--title", args.title]
    print(" ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))
    print(f"[INFO] -> {out_png}")


if __name__ == "__main__":
    main()
