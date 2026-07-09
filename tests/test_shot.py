import pytest
from pydantic import ValidationError

from autoedit.models import Shot


def test_valid_shot_from_fixture(shot_data):
    shot = Shot(**shot_data)
    assert shot.id == "s1"
    assert shot.in_ == 0.0
    assert shot.out_ == 2.5
    assert shot.scale == "close"


def test_accepts_field_names_or_aliases(shot_data):
    by_alias = Shot(**shot_data)
    by_field_name = Shot(
        id=shot_data["id"],
        source=shot_data["source"],
        in_=shot_data["in"],
        out_=shot_data["out"],
        dur=shot_data["dur"],
        motion=shot_data["motion"],
        brightness=shot_data["brightness"],
        sharpness=shot_data["sharpness"],
        faces=shot_data["faces"],
        scale=shot_data["scale"],
    )
    assert by_alias == by_field_name


def test_round_trips_through_json(shot_data):
    shot = Shot(**shot_data)
    reloaded = Shot.model_validate_json(shot.model_dump_json(by_alias=True))
    assert reloaded == shot


def test_rejects_out_before_in(shot_data):
    shot_data["in"] = 5.0
    shot_data["out"] = 2.0
    with pytest.raises(ValidationError):
        Shot(**shot_data)


def test_rejects_out_equal_to_in(shot_data):
    shot_data["in"] = 2.0
    shot_data["out"] = 2.0
    shot_data["dur"] = 0.0
    with pytest.raises(ValidationError):
        Shot(**shot_data)


def test_rejects_dur_mismatched_with_span(shot_data):
    shot_data["dur"] = 99.0
    with pytest.raises(ValidationError):
        Shot(**shot_data)


def test_rejects_negative_in(shot_data):
    shot_data["in"] = -1.0
    with pytest.raises(ValidationError):
        Shot(**shot_data)


def test_rejects_negative_faces(shot_data):
    shot_data["faces"] = -1
    with pytest.raises(ValidationError):
        Shot(**shot_data)


def test_rejects_brightness_out_of_range(shot_data):
    shot_data["brightness"] = 1.5
    with pytest.raises(ValidationError):
        Shot(**shot_data)


def test_rejects_negative_sharpness(shot_data):
    shot_data["sharpness"] = -0.1
    with pytest.raises(ValidationError):
        Shot(**shot_data)


def test_rejects_invalid_scale(shot_data):
    shot_data["scale"] = "medium"
    with pytest.raises(ValidationError):
        Shot(**shot_data)


def test_rejects_extra_field(shot_data):
    shot_data["extra"] = "nope"
    with pytest.raises(ValidationError):
        Shot(**shot_data)
