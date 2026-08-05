# Evaluation harness: LLM-as-a-Judge vs. objective factuality

Metrics, prompts and aggregate results for a study of automatic summarisation
of Indonesian financial group conversation.

Six frontier models were crossed with two prompting strategies, giving 12
summaries per session across 84 sessions. Every summary was scored three ways:
by a six-model LLM judge panel, by a reference-free NLI factuality metric, and
by deterministic stock-ticker entity coverage.

## What the evaluation found

**Judge scores carry almost no information about factuality.** Across 596
paired observations the correlation is r = -0.075 (p = 0.067). The
configuration the panel ranks first is only 7th most factual; the most factual
configuration is ranked 5th by the panel; the two the panel ranks last are 3rd
and 5th most factual.

**The panel is not reliable enough to stand alone.** Krippendorff's alpha
averages 0.401 across 41 fully-scored sessions, below the 0.667 threshold for
even tentative conclusions. Judges span 1.41 points of calibration, and
self-enhancement bias reaches +1.51 points.

**Raw conversation beats a classical NLP pipeline.** Passing the transcript
straight to the model outperforms sentiment/NER/topic preprocessing by 6.8
points of entity recall and 0.028 of factuality (paired t over 412 pairs,
t = 4.75, p < 0.0001), consistently across model families.

The practical consequence: choosing a production system on judge scores alone
selects against groundedness.

## Layout

```
metrics/
  factuality.py         SummaC-ZS scorer over a multilingual NLI model
  entity_coverage.py    ticker recall/precision, deterministic
  agreement.py          Krippendorff's alpha, self-enhancement delta
prompts/
  judge_rubric.md            the five criteria and the scorecard format
  summarization_strategies.md  what distinguishes the two strategies
results/                aggregate figures behind every number in the paper
analysis/
  correlation.py        reproduces the rank dissociation from results/
data/                   ticker whitelist; see data/README.md for exclusions
```

## Install

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

`transformers` and `torch` are only needed for `metrics/factuality.py`. The
entity and agreement metrics, and `analysis/correlation.py`, run without them.

## Use

```python
from metrics.factuality import scorer
from metrics.entity_coverage import compute_coverage, load_whitelist
from metrics.agreement import compute_judge_alpha

scorer.score(source, summary)                          # -> [0, 1] or None
compute_coverage(source, summary, load_whitelist())    # -> recall/precision/...
compute_judge_alpha(scorecards)                        # -> alpha or None
```

```bash
python analysis/correlation.py
```

## Three things worth knowing before reusing this

**The factuality score is relative, not calibrated.** Use it to rank systems
against one another. A score of 0.55 does not mean a summary is 55% true.

**Sample sizes are not interchangeable.** 943 summaries carry a factuality
score, 902 carry an entity score, and 596 cells carry both. Any judge-versus-
factuality statement belongs to the 596, and mixing them silently changes what
is being claimed.

**Judges and summarisers are the same six models.** That is what makes
self-enhancement measurable, but it also means judge and generation errors are
not independent. Say so if you copy the design.

## Data

The conversation corpus and the generated summaries are not published. Members
were promised their messages would not be reproduced with identifiers, and
that constrains what can ship. `data/README.md` explains what is excluded and
how to run the metrics on your own corpus.

## Citation

See `CITATION.md`.

## Licence

MIT, `LICENSE`.
