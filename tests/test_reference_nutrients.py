import numpy as np
import pytest
from backend.contracts import Interpretation, Intervention, RunSettings
from backend.nutrients import reference_nutrients
from backend.simulation import simulate
from backend.equation_engine import catalog_from_result


def candy(**kwargs):
    return Interpretation(title="Candy ingestion", interventions=[Intervention(
        kind="nutrition", label="One candy just eaten", entity="candy", route="oral", **kwargs)])


def test_unknown_candy_reaches_existing_transport_without_inventing_actual_mass():
    original = candy()
    result = simulate(original, RunSettings(), {})
    assert original.interventions[0].quantity is None
    assert result["interpretation"]["interventions"][0]["quantity"] is None
    assert result["plan"]["interpretation"]["interventions"][0]["quantity"] is None
    assert result["plan"]["modeled_interpretation"]["interventions"][0]["quantity"] == 1
    assert result["reference_scenarios"][0]["entity"] == "sucrose"
    assert result["nodes"][0]["intervention"]["quantity"] is None
    assert result["nodes"][0]["reference_scenario"]["basis"] == "reference"
    assert catalog_from_result(result)["reference_scenarios"]
    assert result["event_coverage"][0]["native"]
    for key in ("enzyme:sucrase", "flux:sglt1", "flux:glut5", "flux:glut2"):
        assert max(result["traces"][key]) > 0
    assert result["traces"]["meal:delivered:glucose"][-1] > 0
    assert result["traces"]["meal:delivered:fructose"][-1] > 0
    assert np.ptp(result["traces"]["glucose"]) > .1
    assert np.ptp(result["traces"]["insulin"]) > .01
    assert all(b["passed"] for b in result["validation"]["balances"])


@pytest.mark.parametrize("species", ["sucrose", "glucose", "fructose", "lactose", "starch"])
def test_reference_policy_reuses_any_known_missing_mass_nutrient(species):
    event = candy().interventions[0].model_copy(update={"entity": species})
    plan = Interpretation(title="Nutrient", interventions=[event])
    modeled, refs = reference_nutrients(plan, True)
    assert modeled.interventions[0].quantity == 1
    assert refs[0]["entity"] == species
    assert plan.interventions[0].quantity is None


@pytest.mark.parametrize("label", ["sugar-free candy", "무설탕 사탕", "제로 사탕", "Xylitol candy"])
def test_sugar_free_food_never_acquires_sucrose_from_a_reference(label):
    plan = candy()
    plan.interventions[0].label = label
    plan.interventions[0].entity = "sucrose"  # Even if parsing suggests the wrong sugar.
    assert reference_nutrients(plan, True)[1] == []


def test_explicit_doses_unsupported_foods_and_no_assumptions_are_preserved():
    plan = candy(quantity=12, unit="g")
    plan.interventions[0].entity = "sucrose"
    assert reference_nutrients(plan, True)[1] == []
    assert reference_nutrients(candy(), False)[1] == []
    plan.interventions[0] = Intervention(kind="nutrition", label="Bread", entity="bread", route="oral")
    assert reference_nutrients(plan, True)[1] == []


def test_reference_keeps_delayed_timing_and_does_not_multiply_piece_count():
    plan = candy(quantity=3, unit="pieces", start_min=30)
    modeled, refs = reference_nutrients(plan, True)
    assert modeled.interventions[0].start_min == 30
    assert modeled.interventions[0].quantity == 1
    assert refs[0]["quantity"] == 1
