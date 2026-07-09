import pytest
from pydantic import ValidationError

from autoedit.models import EditOp, EditPlan


def test_valid_edit_plan_from_fixture(edit_plan_data):
    plan = EditPlan(**edit_plan_data)
    assert len(plan.ops) == 4
    assert plan.ops[0].tool == "cutter"
    assert plan.ops[0].params["keep"] == ["s1", "s2"]
    assert plan.confidence == pytest.approx(0.88)


def test_round_trips_through_json(edit_plan_data):
    plan = EditPlan(**edit_plan_data)
    reloaded = EditPlan.model_validate_json(plan.model_dump_json())
    assert reloaded == plan


def test_edit_op_allows_arbitrary_params():
    op = EditOp(tool="emoji", params={"glyph": "🔥", "at": 14.2})
    assert op.params["glyph"] == "🔥"


def test_edit_op_params_default_to_empty_dict():
    op = EditOp(tool="cutter")
    assert op.params == {}


def test_rejects_empty_ops_list(edit_plan_data):
    edit_plan_data["ops"] = []
    with pytest.raises(ValidationError):
        EditPlan(**edit_plan_data)


def test_rejects_empty_tool_name():
    with pytest.raises(ValidationError):
        EditOp(tool="", params={})


def test_rejects_confidence_above_one(edit_plan_data):
    edit_plan_data["confidence"] = 1.5
    with pytest.raises(ValidationError):
        EditPlan(**edit_plan_data)


def test_rejects_confidence_below_zero(edit_plan_data):
    edit_plan_data["confidence"] = -0.1
    with pytest.raises(ValidationError):
        EditPlan(**edit_plan_data)


def test_rejects_missing_confidence(edit_plan_data):
    del edit_plan_data["confidence"]
    with pytest.raises(ValidationError):
        EditPlan(**edit_plan_data)


def test_rejects_op_with_extra_field(edit_plan_data):
    edit_plan_data["ops"][0]["unexpected"] = True
    with pytest.raises(ValidationError):
        EditPlan(**edit_plan_data)


def test_rejects_extra_field_on_plan(edit_plan_data):
    edit_plan_data["extra"] = "nope"
    with pytest.raises(ValidationError):
        EditPlan(**edit_plan_data)
