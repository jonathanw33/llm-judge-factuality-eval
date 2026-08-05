"""Reference-free entity coverage: stock-ticker recall and precision.

Factuality (see metrics/factuality.py) penalises claims the source does not
support, but says nothing about claims the summary left out. This metric
supplies the missing axis. It treats the set of tickers discussed in the source
as ground truth and asks how many of them survive into the summary.

It is fully deterministic -- no model, no reference summary -- which makes it
the most auditable of the three axes used in the paper.

Extraction is regex plus a whitelist of listed Indonesia Stock Exchange codes,
plus a stop-list for codes that collide with ordinary Indonesian or English
words. That last filter matters more than it sounds: "main saham" means "to
trade stocks", not the ticker MAIN (Malindo Feedmill), and a chat about a
holiday in Bali is not a discussion of $BALI. Ambiguous codes are therefore
only counted when explicitly written with a leading "$".
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

DEFAULT_WHITELIST = Path(__file__).resolve().parent.parent / "data" / "idx_tickers.csv"

# Codes that are also common words. Counted only with an explicit "$" prefix.
# Extend freely; every entry here was added after a real false positive.
AMBIGUOUS_WORD_TICKERS = {
    # Indonesian words and trading-chat slang
    "MAIN", "MARI", "PADI", "AMIN", "RAJA", "KOPI", "IKAN", "KAYU",
    "SOTO", "HOKI", "TOTO", "CUAN", "AMAN", "ENAK", "PRAY", "BOSS",
    "SOUL", "HERO", "PALM", "BUKA", "JAYA", "TIRA", "AGRO", "SINI",
    # Indonesian place names that are also tickers
    "BALI",
    # English words and slang that collide with tickers
    "STAR", "CHIP", "GOLD", "KING", "NATO", "BULL", "BEAR", "FUEL",
    "FAST", "FUND", "PURE", "COAL", "INDO", "WIFI", "LIFE", "TRUS",
    "NICE", "WOOD", "PORT", "GOOD", "BABY", "FISH", "FOOD", "RICH",
    "HOPE", "CARE", "CITY", "SAFE", "FIRE", "GOLF", "LAND", "LINK",
    "LION", "MARK", "POOL", "SAGE", "SOFA",
    # Deliberately NOT listed, because they are genuinely discussed and rarely
    # ambiguous in context: BSDE (Bumi Serpong Damai), BUMI (Bumi Resources).
}

_DOLLAR = re.compile(r"\$([A-Z]{2,5})")
_BARE = re.compile(r"\b([A-Z]{2,5})\b")


def load_whitelist(path: Path | str = DEFAULT_WHITELIST) -> Set[str]:
    """Load listed IDX codes. One column, header `ticker`."""
    with open(path, newline="", encoding="utf-8") as fh:
        return {row["ticker"].strip().upper() for row in csv.DictReader(fh) if row.get("ticker")}


def extract_tickers(text: str, whitelist: Set[str]) -> Set[str]:
    """Return the set of whitelisted tickers mentioned in `text`."""
    if not text:
        return set()
    upper = text.upper()
    dollar = set(_DOLLAR.findall(upper))
    candidates = dollar | set(_BARE.findall(upper))

    found = set()
    for ticker in candidates:
        if ticker not in whitelist:
            continue
        # An ambiguous code only counts when written as "$TICKER".
        if ticker in AMBIGUOUS_WORD_TICKERS and ticker not in dollar:
            continue
        found.add(ticker)
    return found


def compute_coverage(source: str, summary: str, whitelist: Optional[Set[str]] = None) -> Dict:
    """Ticker recall and precision for one (source, summary) pair.

    Recall and precision are None when the corresponding side mentions no
    ticker at all, so that callers can distinguish "scored zero" from
    "not applicable". The paper excludes None cells rather than treating
    them as zero.
    """
    if whitelist is None:
        whitelist = load_whitelist()

    src = extract_tickers(source, whitelist)
    summ = extract_tickers(summary, whitelist)
    overlap = src & summ

    return {
        "recall": (len(overlap) / len(src)) if src else None,
        "precision": (len(overlap) / len(summ)) if summ else None,
        "source_count": len(src),
        "summary_count": len(summ),
        "intersection_count": len(overlap),
        "missed": sorted(src - summ),
        "fabricated": sorted(summ - src),
    }


def f1(recall: Optional[float], precision: Optional[float]) -> Optional[float]:
    """Harmonic mean, or None when either side is undefined or both are zero."""
    if recall is None or precision is None or (recall + precision) == 0:
        return None
    return 2 * recall * precision / (recall + precision)
