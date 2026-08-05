# Data

## What is here

`idx_tickers.csv` — 914 listed Indonesia Stock Exchange codes, used as the
whitelist by `metrics/entity_coverage.py`. Public reference data.

## What is not here, and why

The evaluation corpus is 84 hourly sessions of private conversation from a
retail-investing Telegram community, together with the 943 generated summaries
scored against it. **None of it is published, and it will not be.**

Members were told their messages fed an automated summariser being studied for
a thesis, and were promised that nothing would be quoted with a username or
other identifier. Publishing the transcripts, or the summaries derived from
them, would break that promise. Group and channel identifiers are excluded for
the same reason.

What is published instead is `results/`: aggregates at model and configuration
level. Those are sufficient to check every number in the paper and to reproduce
the rank dissociation, and they carry no message content.

## Running the metrics on your own data

Nothing here is specific to the source community. Each metric takes plain
strings:

```python
from metrics.factuality import scorer
from metrics.entity_coverage import compute_coverage, load_whitelist

scorer.score(source_text, summary_text)                       # SummaC-ZS, [0, 1]
compute_coverage(source_text, summary_text, load_whitelist()) # recall/precision
```

For a different market, replace `idx_tickers.csv` with that exchange's codes
and revise `AMBIGUOUS_WORD_TICKERS` in `metrics/entity_coverage.py` — the
collisions are language-specific, and getting them wrong inflates recall on
both sides of any comparison.
