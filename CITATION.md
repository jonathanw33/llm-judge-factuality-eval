# Citation

The paper describing this evaluation is **not yet published**. It is under
submission to the 12th IEEE International Conference on Data and Software
Engineering (ICoDSE 2026). Please do not cite it as published work.

Until acceptance, cite the repository:

```bibtex
@software{wiguna2026evalharness,
  author  = {Jonathan Wiguna},
  title   = {Evaluation harness: {LLM-as-a-Judge} vs. objective factuality
             in {Indonesian} financial conversation summarisation},
  year    = {2026},
  url     = {https://github.com/jonathanw33/llm-judge-factuality-eval}
}
```

Replace this block with the conference entry once the paper is accepted.

## Underlying work

The harness comes from an undergraduate final project in the Information
Systems and Technology programme, School of Electrical Engineering and
Informatics, Institut Teknologi Bandung, supervised by Ir. Windy Gambetta,
M.B.A.

## Methods this builds on

The factuality metric follows the zero-shot SummaC variant:

> Laban, P., Schnabel, T., Bennett, P. N., and Hearst, M. A. (2022).
> SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in
> Summarization. *TACL* 10:163-177.

The entailment model is `joeddav/xlm-roberta-large-xnli`, which is
XLM-RoBERTa large (Conneau et al., 2020) fine-tuned on XNLI (Conneau et al.,
2018).

Agreement thresholds are from Krippendorff, K. (2004), *Content Analysis: An
Introduction to Its Methodology*, 2nd ed., Sage — not from the 2011 alpha
memo, to which they are frequently misattributed.
