"""Reference-free factuality scoring, following the zero-shot SummaC variant.

Reference:
    Laban, Schnabel, Bennett and Hearst (2022). SummaC: Re-Visiting NLI-based
    Models for Inconsistency Detection in Summarization. TACL 10:163-177.

Every source sentence is treated as a premise and every summary sentence as a
hypothesis. Each summary sentence is credited by the single source sentence
that best entails it, and the summary's score is the mean over its sentences:

    Fact(S, D) = (1 / |S|) * sum_{s in S} max_{d in D} P_entail(s | d)

The score is bounded in [0, 1] and needs no reference summary, which is what
makes it usable on live conversation. It is a *relative* measure: use it to
rank systems against one another, not as a calibrated probability that a
summary is true.

The entailment model is `joeddav/xlm-roberta-large-xnli`, chosen because it
covers Indonesian with no in-domain training. Any multilingual NLI checkpoint
with an `entailment` label can be substituted via `model_id`.
"""

from __future__ import annotations

import re
from typing import List, Optional

import torch
import torch.nn.functional as F

MODEL_ID = "joeddav/xlm-roberta-large-xnli"

MAX_PREMISE_SENTENCES = 120   # cap source sentences; longer sessions are truncated
MAX_HYPOTHESIS_SENTENCES = 40  # cap summary sentences
MIN_SENTENCE_CHARS = 15        # below this a "sentence" is a fragment, not a claim

_MARKDOWN = re.compile(r"[*_`#>]+")
_VERSION_TAG = re.compile(r"^\s*\[[^\]]*\]\s*")
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]+",
    flags=re.UNICODE,
)


def clean(text: str) -> str:
    """Strip markdown, a leading version tag, and emoji.

    Emoji matter here: left in place they derail sentence tokenisation, which
    silently changes how many hypotheses a summary contributes.
    """
    text = _VERSION_TAG.sub("", text or "")
    text = _EMOJI.sub(" ", text)
    text = _MARKDOWN.sub("", text)
    return text.strip()


def split_sentences(text: str, cap: int) -> List[str]:
    """Sentence-split, drop fragments, and cap the count.

    Splits on newlines first because generated summaries are formatted with
    line breaks that nltk would otherwise swallow into one long sentence.
    """
    from nltk.tokenize import sent_tokenize

    if not text:
        return []
    out: List[str] = []
    for block in clean(text).split("\n"):
        block = block.strip()
        if not block:
            continue
        try:
            out.extend(sent_tokenize(block))
        except LookupError:
            out.append(block)
    kept = [s.strip() for s in out if len(s.strip()) >= MIN_SENTENCE_CHARS]
    return kept[:cap]


class NLIFactualityScorer:
    """Lazily-loaded SummaC-ZS scorer."""

    def __init__(self, model_id: str = MODEL_ID, device: Optional[str] = None):
        self.model_id = model_id
        self._device = device
        self._tokenizer = None
        self._model = None
        self._entail_idx = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        self._model.to(self._device).eval()

        # Label order differs between checkpoints; find `entailment` by name
        # rather than assuming an index.
        labels = {v.lower(): k for k, v in self._model.config.id2label.items()}
        self._entail_idx = labels.get("entailment", len(labels) - 1)

    def _entailment_probs(self, premises: List[str], hypotheses: List[str], batch: int = 32) -> List[float]:
        self._ensure_loaded()
        probs: List[float] = []
        for i in range(0, len(premises), batch):
            enc = self._tokenizer(
                premises[i:i + batch], hypotheses[i:i + batch],
                return_tensors="pt", truncation=True, padding=True, max_length=512,
            ).to(self._device)
            with torch.no_grad():
                logits = self._model(**enc).logits
            probs.extend(F.softmax(logits, dim=-1)[:, self._entail_idx].tolist())
        return probs

    def score(self, source: str, summary: str) -> Optional[float]:
        """Return Fact(S, D) in [0, 1], or None if either side has no scorable sentence."""
        source_sents = split_sentences(source, MAX_PREMISE_SENTENCES)
        summary_sents = split_sentences(summary, MAX_HYPOTHESIS_SENTENCES)
        if not source_sents or not summary_sents:
            return None

        premises: List[str] = []
        hypotheses: List[str] = []
        for hyp in summary_sents:
            premises.extend(source_sents)
            hypotheses.extend([hyp] * len(source_sents))

        flat = self._entailment_probs(premises, hypotheses)

        n_src = len(source_sents)
        per_hypothesis_max = [
            max(flat[i * n_src:(i + 1) * n_src]) for i in range(len(summary_sents))
        ]
        return sum(per_hypothesis_max) / len(per_hypothesis_max)


scorer = NLIFactualityScorer()
