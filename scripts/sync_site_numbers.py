#!/usr/bin/env python
"""Write the numbers of ``site/figures/numbers.json`` into the website's captions, in place.

``scripts/make_site_figures.py`` writes the numbers its figures show to ``numbers.json``. The page marks
every caption number that comes from there as

    <span data-num="figures.noise_floor.n_eff[0]" data-fmt="int">125</span>

This script looks up each path, formats the value and replaces the span's text, so the captions are
regenerated with the figures and never hand-edited. Formats:

    int      integer                      f1 .. f4   fixed decimals
    pct      x 100, integer               s1 .. s4   signed (true minus or plus), fixed decimals

Rounding is half-up on the value as written in the JSON (0.0495 -> 0.050).

    python scripts/sync_site_numbers.py            # update site/index.html
    python scripts/sync_site_numbers.py --check    # change nothing; exit 1 if a number is stale

``--page`` and ``--numbers`` point at other files. The Pages workflow runs ``--check``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPAN = re.compile(r'(<span\b[^>]*\bdata-num="([^"]+)"[^>]*\bdata-fmt="([^"]+)"[^>]*>)([^<]*)(</span>)')
STEP = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)|\[(\d+)\]")
MINUS = "−"


def lookup(data: dict, path: str) -> float:
    """Resolve a dotted path with [i] list indices, for example ``a.b[0][1]``."""
    node = data
    for part in path.split("."):
        steps = STEP.findall(part)
        if not steps or "".join(n or f"[{i}]" for n, i in steps) != part:
            raise KeyError(f"malformed path '{path}' (at '{part}')")
        for name, index in steps:
            try:
                node = node[name] if name else node[int(index)]
            except (KeyError, IndexError, TypeError) as exc:
                raise KeyError(f"numbers.json has no '{path}' (failed at '{name or index}')") from exc
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        raise TypeError(f"'{path}' is not a number: {node!r}")
    return node


def fmt(value: float, kind: str) -> str:
    """Format ``value`` as ``kind`` (see the module docstring)."""
    d = Decimal(repr(value))
    if kind == "int":
        return f"{d.quantize(Decimal(1), ROUND_HALF_UP):f}"
    if kind == "pct":
        return f"{(d * 100).quantize(Decimal(1), ROUND_HALF_UP):f}"
    m = re.fullmatch(r"([fs])([1-4])", kind)
    if not m:
        raise ValueError(f"unknown format '{kind}'")
    q = d.quantize(Decimal(1).scaleb(-int(m.group(2))), ROUND_HALF_UP)
    text = f"{abs(q):f}"
    if m.group(1) == "f":
        return MINUS + text if q < 0 else text
    return (MINUS if q < 0 else "+") + text


def sync(html: str, data: dict) -> tuple[str, list[str], int]:
    """Return the page with every marked number rewritten, the list of changes and the number count."""
    changes: list[str] = []
    count = 0

    def repl(m: re.Match) -> str:
        nonlocal count
        count += 1
        new = fmt(lookup(data, m.group(2)), m.group(3))
        if new != m.group(4):
            changes.append(f"{m.group(2)}: {m.group(4)!r} -> {new!r}")
        return m.group(1) + new + m.group(5)

    return SPAN.sub(repl, html), changes, count


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--page", type=Path, default=ROOT / "site" / "index.html")
    ap.add_argument("--numbers", type=Path, default=ROOT / "site" / "figures" / "numbers.json")
    ap.add_argument("--check", action="store_true", help="change nothing; exit 1 if any number is stale")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        data = json.loads(args.numbers.read_text(encoding="utf-8"))
        html = args.page.read_text(encoding="utf-8")
        out, changes, count = sync(html, data)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for line in changes:
        print(("stale  " if args.check else "update ") + line)
    print(f"{count} numbers in {args.page.name}, {len(changes)} {'stale' if args.check else 'changed'}")
    if count == 0:
        print("error: the page marks no numbers (data-num); is it the right file?", file=sys.stderr)
        return 2
    if args.check:
        return 1 if changes else 0
    if changes:
        args.page.write_text(out, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
