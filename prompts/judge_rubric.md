# Judge rubric

Each judge receives the full source conversation for one session plus all 12
candidate summaries, in a single prompt, and returns one scorecard.

The same six models act as both summarisers and judges. That overlap is
deliberate: it is what makes self-enhancement bias directly measurable, by
comparing a judge's score for summaries from its own family against what the
other judges gave the same summaries.

## Criteria

Every summary is scored 1-10 on five criteria. The reported total is their
mean.

| Key | Criterion | Question put to the judge |
|---|---|---|
| `Acc` | Factual accuracy | Are price levels and attributions (who said what) correct? |
| `Comp` | Completeness | Are all important tickers and insights covered? |
| `Ctx` | Context understanding | Does it capture sarcasm, disagreement, and reply context? |
| `Hall` | Absence of hallucination | Does it invent prices or facts? |
| `Read` | Readability | Is it clear and well structured? |

The judge also nominates a single best version and gives a one-line reason.

`Hall` is the criterion judges disagree on most (between-judge sigma = 1.63)
and `Read` the least (sigma = 0.76). Judges converge on style and diverge on
truth, which is the same pattern the correlation result shows from the other
direction.

## Prompt shape

The prompt is written in Indonesian, matching the language of the data. Its
structure:

1. Role: evaluator of Indonesian trading-discussion summaries.
2. Task: compare N summaries against the original conversation, score each on
   the five criteria above.
3. The five criteria, stated as questions.
4. The source conversation, labelled as ground truth.
5. The candidate summaries, separated by rule lines and labelled `V1`..`V12`.
6. A rigid plain-text output format (a ranked list, one line per version:
   `Vn | Acc:X Comp:X Ctx:X Hall:X Read:X | Total:X.X | note`), then
   `BEST:` and `WHY:` lines.

Plain text rather than JSON: across these six models it parsed more reliably,
and a malformed scorecard costs an entire session of judging.

## Reproducing

The parser is tolerant of minor format drift but requires the `Vn | ... |
Total:` line shape. Judges occasionally invent a version that was not offered
(a "V13"); those rows are dropped rather than repaired, and the session still
counts, since the coefficient handles missing cells natively.
