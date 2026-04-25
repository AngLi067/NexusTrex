"""
Pairwise comparison of two tracking TXT files into one figure (no simulation).

  python scripts/cv_pair.py a.txt b.txt
  python scripts/cv_pair.py a.txt b.txt --n1 A --n2 B -o cmp.png

Without -o, PNG is written next to the first TXT as stem1_vs_stem2.png
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _resolve_txt(arg: str, which: str) -> Path:
    """Resolve path: CWD first, then NexusTrex repo root."""
    s = arg.strip().strip('"').strip("'")
    p = Path(s).expanduser()
    tried: list[Path] = []
    cands: list[Path] = []
    if p.is_absolute():
        cands.append(p)
    else:
        cands.append(Path.cwd() / p)
        cands.append(ROOT / p)
    for c in cands:
        try:
            r = c.resolve()
        except OSError:
            continue
        tried.append(r)
        if r.is_file():
            return r
    lines = "\n  ".join(str(t) for t in tried) if tried else "(none)"
    msg = (
        f"Cannot find {which} TXT:\n  input: {arg}\n"
        f"tried:\n  {lines}\n\n"
        f"CWD: {Path.cwd()}\n"
        f"NexusTrex root: {ROOT}\n\n"
        "Generate logs first, e.g. from repo root:\n"
        "  python scripts/cv_run.py --teacher <ckpt> --d <ckpt> --circle --run\n"
        "or play.py with --log_tracking --log_out cv_out/b.txt\n"
        "You may also pass an absolute TXT path."
    )
    print(msg, file=sys.stderr)
    sys.exit(1)


def _resolve_out(arg: str | None, pa: Path, pb: Path) -> Path:
    if not arg:
        return pa.parent / f"{pa.stem}_vs_{pb.stem}.png"
    s = arg.strip().strip('"').strip("'")
    o = Path(s).expanduser()
    if o.is_absolute():
        return o.resolve()
    # Relative paths are resolved under repo root (matches cv_out convention)
    return (ROOT / o).resolve()


def main():
    p = argparse.ArgumentParser(description="Pairwise tracking TXT comparison plot")
    p.add_argument("a", type=str, help="First .txt")
    p.add_argument("b", type=str, help="Second .txt")
    p.add_argument("--n1", type=str, default=None, help="Legend label 1 (default: first stem)")
    p.add_argument("--n2", type=str, default=None, help="Legend label 2")
    p.add_argument("-o", "--out", type=str, default=None, help="Output PNG; default stem1_vs_stem2.png")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--title", type=str, default=None, help="Figure suptitle; default auto from labels")
    args = p.parse_args()

    pa = _resolve_txt(args.a, "first")
    pb = _resolve_txt(args.b, "second")
    l1 = args.n1 or pa.stem
    l2 = args.n2 or pb.stem
    png = _resolve_out(args.out, pa, pb)
    png.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "plot_tracking.py"),
        str(pa),
        str(pb),
        "--labels",
        l1,
        l2,
        "--layout",
        "paper",
        "--save",
        "--out",
        str(png),
        "--dpi",
        str(args.dpi),
    ]
    if args.title:
        cmd += ["--title", args.title]
    subprocess.check_call(cmd, cwd=str(ROOT))
    print(f"[OK] {png}")


if __name__ == "__main__":
    main()
