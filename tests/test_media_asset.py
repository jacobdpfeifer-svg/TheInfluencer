import pytest
from pydantic import ValidationError

from autoedit.models import MediaAsset


def test_valid_media_asset_from_fixture(media_asset_data):
    asset = MediaAsset(**media_asset_data)
    assert asset.path == "raw/clip_001.mp4"
    assert asset.type == "video"
    assert asset.width == 1080
    assert asset.height == 1920


def test_round_trips_through_json(media_asset_data):
    asset = MediaAsset(**media_asset_data)
    reloaded = MediaAsset.model_validate_json(asset.model_dump_json())
    assert reloaded == asset


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("duration", 0),
        ("duration", -1.0),
        ("width", 0),
        ("height", -100),
        ("fps", 0),
        ("audio_channels", -1),
        ("path", ""),
        ("codec", ""),
    ],
)
def test_rejects_bad_numeric_fields(media_asset_data, field, bad_value):
    media_asset_data[field] = bad_value
    with pytest.raises(ValidationError):
        MediaAsset(**media_asset_data)


def test_rejects_unknown_media_type(media_asset_data):
    media_asset_data["type"] = "hologram"
    with pytest.raises(ValidationError):
        MediaAsset(**media_asset_data)


def test_rejects_missing_required_field(media_asset_data):
    del media_asset_data["fps"]
    with pytest.raises(ValidationError):
        MediaAsset(**media_asset_data)


def test_rejects_extra_field(media_asset_data):
    media_asset_data["unexpected"] = "surprise"
    with pytest.raises(ValidationError):
        MediaAsset(**media_asset_data)
