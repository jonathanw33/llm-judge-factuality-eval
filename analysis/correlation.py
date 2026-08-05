"""Reproduce the paper's central result from the shipped aggregate data.

    python analysis/correlation.py

Per-cell data (596 individual observations) is not shipped -- see data/README.md
-- so this operates on the 12 per-version means in
results/judge_vs_factuality_by_version.csv. That is enough to reproduce the
rank dissociation, which is the part a reader most wants to check.

The cell-level Pearson correlation reported in the paper is
r = -0.075, p = 0.067, n = 596.
"""

from __future__ import annotations

import csv
import pathlib

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"

PAPER_R = -0.075
PAPER_P = 0.067
PAPER_N = 596


def load():
    with open(RESULTS / "judge_vs_factuality_by_version.csv", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def spearman(a, b):
    """Rank correlation without scipy, so the script has no hard dependency."""
    n = len(a)
    mean_a, mean_b = sum(a) / n, sum(b) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    va = sum((x - mean_a) ** 2 for x in a) ** 0.5
    vb = sum((y - mean_b) ** 2 for y in b) ** 0.5
    return cov / (va * vb) if va and vb else float("nan")


def main() -> None:
    rows = load()
    jr = [int(r["judge_rank"]) for r in rows]
    fr = [int(r["factuality_rank"]) for r in rows]

    print(f"Cell-level result reported in the paper: "
          f"r = {PAPER_R:+.3f}, p = {PAPER_P:.3f}, n = {PAPER_N}")
    print(f"Version-level rank correlation from shipped means: "
          f"rho = {spearman(jr, fr):+.3f} over {len(rows)} configurations\n")

    print("How far each configuration moves between the two metrics:\n")
    print(f"  {'cfg':>4}  {'model':<15} {'strategy':<7} "
          f"{'judge':>6} {'fact':>6}  {'rank move':>10}")
    for r in sorted(rows, key=lambda x: int(x["judge_rank"])):
        move = int(r["factuality_rank"]) - int(r["judge_rank"])
        print(f"  {r['version']:>4}  {r['model']:<15} {r['strategy']:<7} "
              f"{r['judge_rank']:>6} {r['factuality_rank']:>6}  {move:>+10}")

    top = min(rows, key=lambda r: int(r["judge_rank"]))
    best_fact = min(rows, key=lambda r: int(r["factuality_rank"]))
    print(f"\nThe configuration judges rank #1 ({top['version']}) is "
          f"#{top['factuality_rank']} on factuality.")
    print(f"The most factual configuration ({best_fact['version']}) is "
          f"#{best_fact['judge_rank']} by judge score.")

    simple_top6 = all(
        r["strategy"] == "Simple" for r in rows if int(r["judge_rank"]) <= 6
    )
    print(f"\nAll six top judge ranks held by Simple: {simple_top6}")


if __name__ == "__main__":
    main()
