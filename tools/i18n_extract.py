#!/usr/bin/env python
"""Audit or update the GUI translation tables.

python tools/i18n_extract.py                # coverage of every shipped language
python tools/i18n_extract.py --missing ja   # list the untranslated strings of one language
python tools/i18n_extract.py --update ja    # add missing keys (empty values) to translations/ja.json
python tools/i18n_extract.py --dump         # print every source string (one per line)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from al_dvc.gui.i18n import SUPPORTED_LANGUAGES  # noqa: E402
from al_dvc.gui.i18n_tools import audit_table, extract_sources, update_table  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--missing", metavar="CODE", help="print the missing / empty strings of this language")
    ap.add_argument("--update", metavar="CODE", help="add missing keys to this language's table")
    ap.add_argument("--prune", action="store_true", help="with --update: drop keys no longer in the code")
    ap.add_argument("--dump", action="store_true", help="print every source string")
    args = ap.parse_args(argv)
    sources = extract_sources()
    if args.dump:
        for text in sorted(sources):
            print(text)
        return 0
    if args.update:
        audit = update_table(args.update, sources, prune=args.prune)
        print(audit.summary())
        return 0
    if args.missing:
        audit = audit_table(args.missing, sources)
        for text in audit.missing + audit.empty:
            print(text)
        print(audit.summary(), file=sys.stderr)
        return 0
    worst = 0
    for code in SUPPORTED_LANGUAGES:
        if code == "en":
            continue
        audit = audit_table(code, sources)
        print(audit.summary())
        worst = max(worst, len(audit.missing) + len(audit.empty))
    return 1 if worst else 0


if __name__ == "__main__":
    raise SystemExit(main())
