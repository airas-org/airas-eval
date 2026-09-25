import json
from pathlib import Path

import pytest

from airas_eval import evaluate, validate_inputs
from airas_eval.tasks.scigym._inputs import MANIFEST
from airas_eval.tasks.scigym._metric_sets import PUBLISHED, official_scores

FIXTURE = Path(__file__).parent / "fixtures" / "scigym" / "BIOMD0000000027"


def _instance(submitted=None):
    return {
        "id": "BIOMD0000000027",
        "reference_sbml": (FIXTURE / "truth.xml").read_text(),
        "incomplete_sbml": (FIXTURE / "partial.xml").read_text(),
        "reference_sedml": (FIXTURE / "truth.sedml").read_text(),
        "submitted_sbml": submitted,
    }


def _scigym_installed() -> bool:
    try:
        official_scores([])
    except ImportError:
        return False
    return True


def test_manifest_pins_the_small_split():
    assert len(MANIFEST) == 137
    validate_inputs("scigym_small", {"instances": [_instance()]})


def test_inputs_must_be_the_official_files():
    tampered = _instance()
    tampered["reference_sbml"] += "\n"
    with pytest.raises(ValueError, match="differs from the official release"):
        validate_inputs("scigym_small", {"instances": [tampered]})
    with pytest.raises(ValueError, match="not in SciGym-small"):
        validate_inputs("scigym_small", {"instances": [dict(_instance(), id="BIOMD9")]})
    with pytest.raises(ValueError, match="unique"):
        validate_inputs("scigym_small", {"instances": [_instance(), _instance()]})


def test_published_table_rows():
    assert len(PUBLISHED) == 6
    assert PUBLISHED["Gemini-2.5-Pro"]["trajectory_smape"] == 0.3212


def test_without_scigym_every_metric_is_skipped():
    if _scigym_installed():
        pytest.skip("scigym is installed")
    report = evaluate("scigym_small", {"instances": [_instance()]})
    assert not report.metrics
    assert set(report.skipped["missing_dependency"]) >= {
        "trajectory_smape",
        "reaction_f1",
    }
    assert report.inputs_summary["n_instances"] == 1


@pytest.mark.skipif(not _scigym_installed(), reason="needs scigym at the pinned commit")
def test_matches_the_official_controller_output():
    # evaluation.json was written by SciGym's Controller for this exact submission
    official = json.loads((FIXTURE / "evaluation.json").read_text())
    submitted = (FIXTURE / "final_model.xml").read_text()
    report = evaluate("scigym_small", {"instances": [_instance(submitted)]})
    assert not report.skipped["missing_dependency"], report.skipped
    m = report.metrics
    assert m["trajectory_smape"] == pytest.approx(official["observe_smape"])
    for ours, theirs in [
        ("reaction_precision", "rp_precision"),
        ("reaction_recall", "rp_recall"),
        ("reaction_f1", "rp_f1"),
        ("reaction_precision_with_modifiers", "rpm_precision"),
        ("reaction_f1_with_modifiers", "rpm_f1"),
    ]:
        assert m[ours] == pytest.approx(official[theirs])
    assert m["trajectory_smape_vs_best_published"] == pytest.approx(
        official["observe_smape"] - 0.3212
    )
    # NTS is not in the Controller's evaluation.json; check it against the
    # Evaluator's own functions on the same models
    from scigym.data import SBML
    from scigym.eval.utils import (
        evaluate_species_interaction_f1,
        evaluate_typed_species_interaction_f1,
    )

    true, pred = SBML(_instance()["reference_sbml"]), SBML(submitted)
    edges = evaluate_species_interaction_f1(true.model, pred.model)
    typed = evaluate_typed_species_interaction_f1(true.model, pred.model)
    assert m["topology_f1"] == pytest.approx(edges["species_edges_undirected_f1"])
    assert m["topology_f1_reactant_product"] == pytest.approx(
        typed["reactant_product_f1"]
    )
    for key in ("topology_f1_reactant_modifier", "topology_f1_modifier_product"):
        assert key in m or key in report.skipped["undefined_on_data"]
    assert report.inputs_summary["n_valid_submissions"] == 1
    assert report.provenance["versions"]["scigym"] != "not installed"


@pytest.mark.skipif(not _scigym_installed(), reason="needs scigym at the pinned commit")
def test_missing_or_invalid_submission_scores_the_incomplete_model():
    none = evaluate("scigym_small", {"instances": [_instance(None)]}).metrics
    invalid = evaluate("scigym_small", {"instances": [_instance("<not sbml>")]}).metrics
    assert none == invalid
    assert none["reaction_f1"] == 0.0
    assert 0 < none["trajectory_smape"] < 1
