import pytest

from airas_eval.metrics import nlp


def test_exact_match_normalization():
    assert nlp.exact_match(["The Cat!"], ["cat"]) == pytest.approx(1.0)
    assert nlp.exact_match(["a dog"], ["cat"]) == pytest.approx(0.0)
    assert nlp.exact_match(["cat", "dog"], ["cat", "bird"]) == pytest.approx(0.5)


def test_token_f1_hand_computed():
    # pred tokens {new, york, city}, ref tokens {york, city}:
    # precision 2/3, recall 1 -> F1 = 0.8
    assert nlp.token_f1(["New York City"], ["York City"]) == pytest.approx(0.8)


def test_token_f1_no_overlap_is_zero():
    assert nlp.token_f1(["alpha"], ["omega"]) == pytest.approx(0.0)


def test_bleu_wrapper_if_available():
    sacrebleu = pytest.importorskip("sacrebleu")
    ours = nlp.bleu(["the cat sat on the mat"], [["the cat sat on the mat"]])
    theirs = float(
        sacrebleu.corpus_bleu(
            ["the cat sat on the mat"], [["the cat sat on the mat"]]
        ).score
    )
    assert ours == pytest.approx(theirs)


def test_rouge_wrapper_if_available():
    pytest.importorskip("rouge_score")
    assert nlp.rouge_l(["the cat sat"], ["the cat sat"]) == pytest.approx(1.0)
