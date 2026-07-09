import pytest
from pydantic import ValidationError

from autoedit.models import CaptionStyleFreq, StyleProfile


def test_valid_style_profile_from_fixture(style_profile_data):
    profile = StyleProfile(**style_profile_data)
    assert profile.sample_count == 12
    assert profile.cut_on_beat is True
    assert isinstance(profile.caption_style_freq, CaptionStyleFreq)
    assert profile.caption_style_freq.karaoke == pytest.approx(0.7)


def test_round_trips_through_json(style_profile_data):
    profile = StyleProfile(**style_profile_data)
    reloaded = StyleProfile.model_validate_json(profile.model_dump_json())
    assert reloaded == profile


def test_rejects_caption_freq_not_summing_to_one(style_profile_data):
    style_profile_data["caption_style_freq"] = {"karaoke": 0.9, "static": 0.9}
    with pytest.raises(ValidationError):
        StyleProfile(**style_profile_data)


def test_rejects_caption_freq_out_of_range(style_profile_data):
    style_profile_data["caption_style_freq"] = {"karaoke": 1.5, "static": -0.5}
    with pytest.raises(ValidationError):
        StyleProfile(**style_profile_data)


def test_rejects_sample_count_zero(style_profile_data):
    style_profile_data["sample_count"] = 0
    with pytest.raises(ValidationError):
        StyleProfile(**style_profile_data)


def test_rejects_negative_sample_count(style_profile_data):
    style_profile_data["sample_count"] = -3
    with pytest.raises(ValidationError):
        StyleProfile(**style_profile_data)


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("aspect", 0),
        ("aspect", -1.0),
        ("shot_len_median", 0),
        ("shot_len_spread", -0.1),
        ("caption_density", -0.1),
        ("text_amount", -1.0),
        ("effect_freq", -0.1),
    ],
)
def test_rejects_bad_numeric_fields(style_profile_data, field, bad_value):
    style_profile_data[field] = bad_value
    with pytest.raises(ValidationError):
        StyleProfile(**style_profile_data)


def test_rejects_non_bool_cut_on_beat(style_profile_data):
    # Note: pydantic v2 coerces strings like "yes"/"true" to bool, so use a
    # value with no sensible bool coercion.
    style_profile_data["cut_on_beat"] = "sometimes"
    with pytest.raises(ValidationError):
        StyleProfile(**style_profile_data)


def test_rejects_extra_field(style_profile_data):
    style_profile_data["extra"] = "nope"
    with pytest.raises(ValidationError):
        StyleProfile(**style_profile_data)
