"""Command-line interface for selly-fft.

Usage
-----
    selly scan PROBE FILE... [options]

Fuzzy-scan files for a probe and report every span above the threshold,
with positions in the original file text.  Exit code is grep-like:
0 when at least one match was found, 1 when none, 2 on error.
"""

from __future__ import annotations

import argparse
import glob as globmod
import sys
from pathlib import Path

import numpy as np

from selly_fft.text import TextAssociativeMemory


def _parse_threshold(raw: str):
    if raw == "auto":
        return "auto"
    try:
        v = float(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid threshold {raw!r}: use a float in [0, 1] or 'auto'"
        )
    if not 0.0 <= v <= 1.0:
        raise argparse.ArgumentTypeError(f"threshold {v} out of [0, 1]")
    return v


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="selly",
        description="FFT-accelerated fuzzy subsequence search (sharp, one-hot text path).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    scan = sub.add_parser("scan", help="fuzzy-scan files for a probe")
    scan.add_argument("probe", help="text to search for")
    scan.add_argument("files", nargs="+", help="files or glob patterns to scan")
    scan.add_argument(
        "--threshold",
        default="0.5",
        type=_parse_threshold,
        help="score floor in [0, 1], or 'auto' for significance-gated "
        "reporting (default: 0.5)",
    )
    scan.add_argument("--case-sensitive", action="store_true")
    scan.add_argument(
        "--float32",
        action="store_true",
        help="encode in float32 (half memory; use for large corpora)",
    )
    scan.add_argument(
        "--uint8",
        action="store_true",
        help="encode in uint8 (1 byte/char/channel — 8x smaller than float64)",
    )
    scan.add_argument(
        "--build-alphabet",
        action="store_true",
        help="derive the alphabet from the corpus (supports arbitrary Unicode)",
    )
    scan.add_argument(
        "--context",
        type=int,
        default=0,
        metavar="N",
        help="show N characters of surrounding context",
    )
    scan.add_argument(
        "--max-per-file",
        type=int,
        default=20,
        help="cap reported matches per file (default: 20)",
    )
    return parser


def _expand_files(patterns) -> list[str]:
    out: list[str] = []
    for pat in patterns:
        hits = sorted(globmod.glob(pat, recursive=True))
        out.extend(hits if hits else [pat])
    # de-dupe, preserve order
    seen: set[str] = set()
    uniq = []
    for f in out:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq


def _scan(args) -> int:
    mem = TextAssociativeMemory(
        case_sensitive=args.case_sensitive,
        dtype=np.uint8 if args.uint8 else (np.float32 if args.float32 else np.float64),
    )
    paths = _expand_files(args.files)
    texts: dict[str, str] = {}
    for p in paths:
        try:
            texts[p] = Path(p).read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"selly: {p}: {e}", file=sys.stderr)
            return 2

    if args.build_alphabet:
        mem.build_alphabet(args.probe, *texts.values())

    found = 0
    for p, text in texts.items():
        try:
            spans = mem.find_spans(args.probe, text, threshold=args.threshold)
        except ValueError as e:
            print(
                f"selly: {e}. Hint: --build-alphabet derives the alphabet "
                f"from your corpus (needed for non-ASCII text).",
                file=sys.stderr,
            )
            return 2
        for sp in spans[: args.max_per_file]:
            found += 1
            if args.context:
                lo = max(0, sp.orig_start - args.context)
                hi = min(len(text), sp.orig_end + args.context)
                excerpt = text[lo:hi].replace("\n", "⏎")
                print(f"{p}:{sp.orig_start}: {sp.score:.3f} …{excerpt}…")
            else:
                line = text.count("\n", 0, sp.orig_start) + 1
                print(f"{p}:{line}:{sp.orig_start}: {sp.score:.3f} {sp.text!r}")
        if len(spans) > args.max_per_file:
            print(f"{p}: … {len(spans) - args.max_per_file} more (raise --max-per-file)")
    return 0 if found else 1


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "scan":
        return _scan(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
