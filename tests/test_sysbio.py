import math

import pytest

from airas_eval import evaluate, validate_inputs
from airas_eval.metrics import regression
from airas_eval.tasks.sysbio import _metric_sets


def _rx(reactants, products, modifiers=()):
    return {
        "reactants": list(reactants),
        "products": list(products),
        "modifiers": list(modifiers),
    }


# 2 instances. Instance 0: 3 removed reactions, agent adds 2 of them (one with a
# wrong modifier) plus 1 spurious. Instance 1: nothing added.
INPUTS = {
    "predicted_reactions": [
        [_rx("A", "B"), _rx("B", "C", "E"), _rx("A", "C")],
        [],
    ],
    "reference_reactions": [
        [_rx("A", "B"), _rx("B", "C", "F"), _rx("C", "D")],
        [_rx("X", "Y")],
    ],
    "predicted_trajectories": [
        {"A": [1.0, 0.5, 0.0], "B": [0.0, 0.5, 1.0]},
        {"X": [2.0, 2.0]},
    ],
    "reference_trajectories": [
        {"A": [1.0, 0.5, 0.0], "B": [0.0, 1.0, 1.0]},
        {"X": [2.0, 1.0], "Y": [0.0, 1.0]},
    ],
}


def test_reaction_network_inference_matches_hand_computation():
    report = evaluate("reaction_network_inference", INPUTS)
    m = report.metrics
    # instance 0 (w/o modifiers): pred {AB, BC, AC}, ref {AB, BC, CD} -> 2/3, 2/3, 2/3
    # instance 1: nothing predicted -> 0 by the benchmark convention
    assert m["reaction_precision"] == pytest.approx((2 / 3 + 0) / 2)
    assert m["reaction_recall"] == pytest.approx((2 / 3 + 0) / 2)
    assert m["reaction_f1"] == pytest.approx((2 / 3 + 0) / 2)
    # with modifiers: only AB matches -> 1/3, 1/3, 1/3
    assert m["reaction_precision_with_modifiers"] == pytest.approx(1 / 6)
    assert m["reaction_recall_with_modifiers"] == pytest.approx(1 / 6)
    assert m["reaction_f1_with_modifiers"] == pytest.approx(1 / 6)
    # instance 0: one point differs (0.5 vs 1.0 -> 0.5/1.5), 6 points -> 1/18
    # instance 1: species Y missing from the prediction -> 1
    assert m["trajectory_smape"] == pytest.approx((1 / 18 + 1.0) / 2)
    assert report.inputs_summary["n_instances"] == 2
    fits = _metric_sets.per_instance_trajectory_fit(
        INPUTS["predicted_trajectories"], INPUTS["reference_trajectories"]
    )
    assert fits == pytest.approx([1 - 1 / 18, 0.0])
    assert not any(report.skipped.values())


def test_reaction_match_ignores_order_and_stoichiometry():
    inputs = {
        "predicted_reactions": [[_rx(["B", "A", "A"], ["C"])]],
        "reference_reactions": [[_rx(["A", "B"], ["C"])]],
        "predicted_trajectories": [{"A": [0.0]}],
        "reference_trajectories": [{"A": [0.0]}],
    }
    m = evaluate("reaction_network_inference", inputs).metrics
    assert m["reaction_f1"] == 1.0
    assert m["trajectory_smape"] == 0.0  # 0/0 counts as no error


def test_length_mismatch_scores_maximum_error():
    inputs = {
        "predicted_reactions": [[]],
        "reference_reactions": [[_rx("A", "B")]],
        "predicted_trajectories": [{"A": [1.0, 2.0]}],
        "reference_trajectories": [{"A": [1.0, 2.0, 3.0]}],
    }
    report = evaluate("reaction_network_inference", inputs)
    assert report.metrics["trajectory_smape"] == 1.0


def test_inputs_must_line_up():
    bad = dict(INPUTS, reference_trajectories=INPUTS["reference_trajectories"][:1])
    with pytest.raises(ValueError):
        validate_inputs("reaction_network_inference", bad)
    with pytest.raises(ValueError):
        validate_inputs(
            "reaction_network_inference",
            dict(INPUTS, predicted_trajectories=[{"A": [math.nan]}, {"X": [1.0]}]),
        )


def test_smape_bounded():
    assert regression.smape_bounded([[1.0, 0.0]], [[3.0, 0.0]]) == pytest.approx(0.25)
    assert regression.smape_bounded([1.0], [1.0]) == 0.0
    with pytest.raises(ValueError):
        regression.smape_bounded([[1.0, 2.0]], [[1.0]])
