"""clean_data_html.py - one-pass cleanup of the shipped knowledge base.

PubMed efetch returns titles/abstracts with typographic markup: <i> around species
names, <sub>/<sup> for sub/superscripts (which leak into entity names like
"Ca<sup>2+</sup>" or "EspG<sub>1</sub>"), the odd <b>/<u>/<sc>, and rare MathML.
That markup is formatting, not content, so we *unwrap* it (drop the tag, keep the
inner text) and decode HTML entities, across every string in data/*.json - titles,
abstracts, entity names, relationship endpoints, synonym keys/values. A few prose
fragments the extractor grabbed as entity names are hand-fixed too.

Whitelist-based on purpose: a generic <[^>]+> strip would corrupt legitimate text
like "p<0.05" or "IC50 > 2"; here only known formatting tags are removed.

Re-run this after regenerating the KB (a fresh pipeline run re-introduces PubMed
markup). Idempotent. Run:  python clean_data_html.py
"""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# Tags PubMed emits as pure formatting (unwrap: drop tag, keep inner text).
_FORMAT_TAGS = (
    "i", "b", "u", "em", "strong", "sup", "sub", "sc", "p",
    # MathML (rare); unwrapping keeps the numbers/operators as plain text.
    "math", "semantics", "annotation", "mrow", "mo", "mn", "mi", "ms",
    "mtext", "msup", "msub", "msubsup", "mfrac", "msqrt", "mspace",
)
_TAG_RE = re.compile(r"</?(?:%s)\b[^>]*>" % "|".join(_FORMAT_TAGS), re.IGNORECASE)
_WS_RE = re.compile("[ \t ]{2,}")  # space / tab / non-breaking-space runs

# Prose fragments the extractor grabbed as entity names -> clean canonical form.
# Renames (not deletions): each appears in a real, scored relationship we keep.
MANUAL_FIXES = {
    "DBLα1 not var3": "DBLα1",
    "single nucleotide variant within the pmv gene": "pmv variant",
    "auxiliary proteins of the ESX-1 system": "ESX-1 auxiliary proteins",
}


def sanitize_text(s):
    """Unwrap formatting tags, decode HTML entities, apply manual fixes."""
    if not isinstance(s, str):
        return s
    if "<" in s or "&" in s:
        s = _TAG_RE.sub("", s)
        s = html.unescape(s)
        s = _WS_RE.sub(" ", s).strip()
    return MANUAL_FIXES.get(s, s)


def clean(obj):
    """Recursively sanitize every string in a JSON-like structure (keys too)."""
    if isinstance(obj, str):
        return sanitize_text(obj)
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    if isinstance(obj, dict):
        return {sanitize_text(k): clean(v) for k, v in obj.items()}
    return obj


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    changed = 0
    for f in sorted(DATA.glob("*.json")):
        before = f.read_text(encoding="utf-8")
        after = json.dumps(clean(json.loads(before)), ensure_ascii=False, indent=2)
        if before.endswith("\n"):
            after += "\n"
        if after != before:
            f.write_text(after, encoding="utf-8")
            changed += 1
            print(f"[cleaned]   {f.name}")
        else:
            print(f"[unchanged] {f.name}")
    print(f"\n[DONE] {changed} file(s) cleaned")


if __name__ == "__main__":
    main()
