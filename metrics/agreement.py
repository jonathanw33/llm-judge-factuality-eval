"""Inter-judge agreement for the multi-judge LLM scorecard panel.

A panel mean is only meaningful if the panel agrees. Reporting one without a
chance-corrected agreement coefficient hides the case where the mean is an
average over judges who disagree with each other. In the accompanying paper
that turned out to be exactly the situation: mean alpha = 0.401 across 41
fully-scored sessions.

Thresholds are Krippendorff's own:

    alpha >= 0.800  conclusions can be relied on
    alpha >= 0.667  tentative conclusions only
    alpha <  0.667  not sufficient for either

Source of the thresholds:
    Krippendorff, K. (2004). Content Analysis: An Introduction to Its
    Methodology, 2nd ed. Sage. (Chapter 11.)

Note the citation carefully. The thresholds are often attributed to
Krippendorff's 2011 "Computing Krippendorff's Alpha-Reliability" memo, which
derives the coefficient but does not state these cut-offs. Do not use the
Landis and Koch (1977) descriptors ("fair", "moderate") here either -- those
were defined for Cohen's kappa and make a weak alpha sound acceptable.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def compute_judge_alpha(scorecards: List[Dict], level: str = "interval") -> Optional[float]:
    """Krippendorff's alpha across one session's judge scorecards.

    Args:
        scorecards: [{"judge": <name>, "scores": {"V1": 8.4, "V2": 7.1, ...}}, ...]
        level: measurement level passed through to the krippendorff package.
               Scores are continuous 1-10, so "interval" is the right choice.

    Returns:
        alpha, or None when there is too little data to compute it (fewer than
        two judges, or fewer than two versions). None rather than 0.0, so a
        caller can tell "no agreement" apart from "no data" -- collapsing the
        two would drag the reported mean alpha down for the wrong reason.

    Judges that skipped a version are represented as NaN and handled natively
    by the coefficient; there is no need to drop or impute them.
    """
    import krippendorff

    if not scorecards or len(scorecards) < 2:
        return None

    versions = sorted({v for sc in scorecards for v in sc.get("scores", {})})
    if len(versions) < 2:
        return None

    matrix = np.array(
        [[sc.get("scores", {}).get(v, np.nan) for v in versions] for sc in scorecards],
        dtype=float,
    )
    if np.isnan(matrix).all():
        return None

    try:
        alpha = krippendorff.alpha(reliability_data=matrix, level_of_measurement=level)
    except Exception:
        return None
    return float(alpha) if alpha is not None else None


def alpha_tier(alpha: Optional[float]) -> str:
    """Krippendorff's own bands. Deliberately not the Landis-Koch wording."""
    if alpha is None:
        return "unknown"
    if alpha >= 0.800:
        return "reliable"
    if alpha >= 0.667:
        return "tentative-only"
    return "insufficient"


def self_enhancement_delta(
    scores: List[Dict], judge_family: Dict[str, str], version_family: Dict[str, str]
) -> Dict[str, float]:
    """Mean points a judge adds when grading its own model family.

    For each (session, version) cell, compare the score given by judges of the
    same family as the generating model against the mean score given by all
    other judges for that same cell. Averaging those differences per judge
    gives the self-enhancement estimate reported in the paper.

    Args:
        scores: [{"session": s, "version": v, "judge": j, "score": x}, ...]
        judge_family: judge name -> model family
        version_family: version id -> model family that generated it
    """
    from collections import defaultdict

    by_cell = defaultdict(list)
    for r in scores:
        by_cell[(r["session"], r["version"])].append(r)

    deltas = defaultdict(list)
    for (_, version), rows in by_cell.items():
        fam = version_family.get(version)
        own = [r for r in rows if judge_family.get(r["judge"]) == fam]
        other = [r for r in rows if judge_family.get(r["judge"]) != fam]
        if not own or not other:
            continue
        other_mean = sum(r["score"] for r in other) / len(other)
        for r in own:
            deltas[r["judge"]].append(r["score"] - other_mean)

    return {j: sum(d) / len(d) for j, d in deltas.items() if d}
