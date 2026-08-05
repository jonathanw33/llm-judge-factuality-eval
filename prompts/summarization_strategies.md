# The two prompting strategies

Six models crossed with two strategies gives the 12 configurations (`V1`..`V12`)
evaluated in the paper. Both strategies use identical decoding parameters and
identical output-format instructions. **The only difference is what reaches the
model.**

| Versions | Strategy | Model input |
|---|---|---|
| `V1`-`V6` | NLP pipeline | a structured digest produced by preprocessing |
| `V7`-`V12` | Simple | the raw conversation transcript |

Version-to-model mapping is in `results/per_version_entity.csv`.

## NLP pipeline (V1-V6)

The session is preprocessed before generation, and the model never sees the
raw messages. Three stages feed the digest:

1. **Sentiment** — per-message classification, aggregated per ticker.
2. **Entity recognition** — ticker extraction (the same regex plus whitelist
   plus ambiguity filter used by `metrics/entity_coverage.py`).
3. **Topic clustering** — BERTopic over the session's messages.

The model receives the aggregate: ranked tickers with sentiment, cluster
labels, activity counts, and a set of representative quoted messages.

## Simple (V7-V12)

The raw transcript is passed through with an instruction to summarise. No
preprocessing, no extraction, no clustering.

## Result

Simple wins on both objective axes: entity recall `0.555` vs `0.487`
(+6.8 points) and factuality `0.514` vs `0.483` (paired t over 412 pairs,
t = 4.75, p < 0.0001). The five top-ranked configurations by recall are all
Simple.

Precision is high everywhere (>0.88), so the pipeline is not injecting errors
— it is discarding information. Sentiment labels, extracted entities and topic
clusters are a lossy compression, and what is dropped upstream cannot be
recovered downstream.

## A note on the prompt text

The prompt templates are not reproduced verbatim here. In the deployed system
they name two individuals — the community's owner and a co-leader — because
the summaries weight those two voices differently from general chat. Publishing
them as written would identify the community, which the study committed not to
do.

Where an authority is named, the templates use:

```
<AUTHORITY_1>   community owner
<AUTHORITY_2>   co-leader
```

Substituting any two names reproduces the behaviour. Nothing else in either
template is community-specific: both are otherwise generic Indonesian
trading-discussion summarisation instructions with a fixed section layout
(activity, most-discussed tickers, trending topics, key-member perspectives,
cross-group insight, recommendation).
