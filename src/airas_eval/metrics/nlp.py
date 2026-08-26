"""NLP metrics.

Pure in-house: exact match and token-level F1 (SQuAD-style normalization).
Wrapped (optional ``nlp`` extra): BLEU/chrF/TER via ``sacrebleu`` and
ROUGE via Google's ``rouge-score`` — the community-accepted scorers.
Model-dependent scores (perplexity, BERTScore) are out of scope for the core.
"""

import re
import string
from collections.abc import Sequence

import numpy as np


def _normalize_answer(text: str) -> list[str]:
    """SQuAD normalization: lowercase, strip punctuation/articles, split."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return text.split()


def exact_match(predicted: Sequence[str], reference: Sequence[str]) -> float:
    if len(predicted) != len(reference):
        raise ValueError(f"length mismatch: {len(predicted)} vs {len(reference)}")
    if not predicted:
        raise ValueError("cannot compute a metric on zero examples")
    matches = [
        _normalize_answer(p) == _normalize_answer(r)
        for p, r in zip(predicted, reference, strict=False)
    ]
    return float(np.mean(matches))


def token_f1(predicted: Sequence[str], reference: Sequence[str]) -> float:
    """Mean SQuAD-style token F1 over examples."""
    if len(predicted) != len(reference):
        raise ValueError(f"length mismatch: {len(predicted)} vs {len(reference)}")
    if not predicted:
        raise ValueError("cannot compute a metric on zero examples")
    scores = []
    for pred, ref in zip(predicted, reference, strict=False):
        p_tokens = _normalize_answer(pred)
        r_tokens = _normalize_answer(ref)
        if not p_tokens or not r_tokens:
            scores.append(float(p_tokens == r_tokens))
            continue
        common: dict[str, int] = {}
        for tok in p_tokens:
            common[tok] = common.get(tok, 0)
        overlap = 0
        r_counts: dict[str, int] = {}
        for tok in r_tokens:
            r_counts[tok] = r_counts.get(tok, 0) + 1
        for tok in p_tokens:
            if r_counts.get(tok, 0) > 0:
                overlap += 1
                r_counts[tok] -= 1
        if overlap == 0:
            scores.append(0.0)
            continue
        precision = overlap / len(p_tokens)
        recall = overlap / len(r_tokens)
        scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(scores))


def bleu(predicted: Sequence[str], references: Sequence[Sequence[str]]) -> float:
    """Corpus BLEU via sacrebleu (the canonical scorer). Requires the nlp extra.

    ``references[i]`` is the list of acceptable references for ``predicted[i]``.
    """
    import sacrebleu  # noqa: PLC0415 - optional dependency

    if len(predicted) != len(references):
        raise ValueError(f"length mismatch: {len(predicted)} vs {len(references)}")
    # sacrebleu expects references transposed: one stream per reference index.
    n_refs = max(len(r) for r in references)
    streams = [
        [refs[i] if i < len(refs) else refs[0] for refs in references]
        for i in range(n_refs)
    ]
    return float(sacrebleu.corpus_bleu(list(predicted), streams).score)


def chrf(predicted: Sequence[str], references: Sequence[Sequence[str]]) -> float:
    """Corpus chrF via sacrebleu. Requires the nlp extra."""
    import sacrebleu  # noqa: PLC0415 - optional dependency

    if len(predicted) != len(references):
        raise ValueError(f"length mismatch: {len(predicted)} vs {len(references)}")
    n_refs = max(len(r) for r in references)
    streams = [
        [refs[i] if i < len(refs) else refs[0] for refs in references]
        for i in range(n_refs)
    ]
    return float(sacrebleu.corpus_chrf(list(predicted), streams).score)


def rouge_l(predicted: Sequence[str], reference: Sequence[str]) -> float:
    """Mean ROUGE-L F-measure via Google's rouge-score. Requires the nlp extra."""
    from rouge_score import rouge_scorer  # noqa: PLC0415 - optional dependency

    if len(predicted) != len(reference):
        raise ValueError(f"length mismatch: {len(predicted)} vs {len(reference)}")
    if not predicted:
        raise ValueError("cannot compute a metric on zero examples")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(r, p)["rougeL"].fmeasure
        for p, r in zip(predicted, reference, strict=False)
    ]
    return float(np.mean(scores))
