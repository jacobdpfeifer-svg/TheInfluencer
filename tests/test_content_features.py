import pytest
from pydantic import ValidationError

from autoedit.models import ContentFeatures


def test_valid_content_features_from_fixture(content_features_data):
    features = ContentFeatures(**content_features_data)
    assert len(features.shots) == 2
    assert features.motion == "med"
    assert features.is_vertical is True
    assert features.music_bpm == pytest.approx(128.0)


def test_music_bpm_optional(content_features_data):
    content_features_data["music_bpm"] = None
    features = ContentFeatures(**content_features_data)
    assert features.music_bpm is None


def test_beat_times_defaults_to_empty_list(content_features_data):
    features = ContentFeatures(**content_features_data)
    assert features.beat_times == []


def test_beat_times_accepts_a_list_of_floats(content_features_data):
    content_features_data["beat_times"] = [0.5, 1.0, 1.5]
    features = ContentFeatures(**content_features_data)
    assert features.beat_times == [0.5, 1.0, 1.5]


def test_round_trips_through_json(content_features_data):
    features = ContentFeatures(**content_features_data)
    reloaded = ContentFeatures.model_validate_json(features.model_dump_json())
    assert reloaded == features


def test_rejects_empty_shots_list(content_features_data):
    content_features_data["shots"] = []
    with pytest.raises(ValidationError):
        ContentFeatures(**content_features_data)


def test_rejects_invalid_shot_in_list(content_features_data):
    content_features_data["shots"][0]["scale"] = "medium"
    with pytest.raises(ValidationError):
        ContentFeatures(**content_features_data)


def test_rejects_invalid_motion_bucket(content_features_data):
    content_features_data["motion"] = "extreme"
    with pytest.raises(ValidationError):
        ContentFeatures(**content_features_data)


def test_rejects_non_positive_aspect(content_features_data):
    content_features_data["aspect"] = 0
    with pytest.raises(ValidationError):
        ContentFeatures(**content_features_data)


def test_rejects_non_positive_music_bpm(content_features_data):
    content_features_data["music_bpm"] = 0
    with pytest.raises(ValidationError):
        ContentFeatures(**content_features_data)


def test_rejects_non_bool_is_vertical(content_features_data):
    content_features_data["is_vertical"] = "sideways"
    with pytest.raises(ValidationError):
        ContentFeatures(**content_features_data)


def test_rejects_missing_required_field(content_features_data):
    del content_features_data["has_face"]
    with pytest.raises(ValidationError):
        ContentFeatures(**content_features_data)


def test_rejects_extra_field(content_features_data):
    content_features_data["extra"] = "nope"
    with pytest.raises(ValidationError):
        ContentFeatures(**content_features_data)
