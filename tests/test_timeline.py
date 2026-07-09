import pytest
from pydantic import ValidationError

from autoedit.models import Timeline, TimelineItem, Track


def test_valid_timeline_from_fixture(timeline_data):
    timeline = Timeline(**timeline_data)
    assert len(timeline.tracks) == 2
    assert timeline.tracks[0].kind == "video"
    assert timeline.tracks[0].items[0].payload["shot"] == "s1"


def test_round_trips_through_json(timeline_data):
    timeline = Timeline(**timeline_data)
    reloaded = Timeline.model_validate_json(timeline.model_dump_json())
    assert reloaded == timeline


def test_empty_timeline_is_valid():
    timeline = Timeline()
    assert timeline.tracks == []


def test_track_with_no_items_is_valid():
    track = Track(name="v1", kind="video")
    assert track.items == []


def test_timeline_item_payload_is_freeform():
    item = TimelineItem(id="x", start=0, end=1, payload={"anything": [1, 2, {"nested": True}]})
    assert item.payload["anything"][2]["nested"] is True


def test_rejects_item_end_before_start():
    with pytest.raises(ValidationError):
        TimelineItem(id="x", start=5.0, end=2.0, payload={})


def test_rejects_item_end_equal_to_start():
    with pytest.raises(ValidationError):
        TimelineItem(id="x", start=2.0, end=2.0, payload={})


def test_rejects_negative_start():
    with pytest.raises(ValidationError):
        TimelineItem(id="x", start=-1.0, end=2.0, payload={})


def test_rejects_invalid_track_kind():
    with pytest.raises(ValidationError):
        Track(name="v1", kind="holograms")


def test_rejects_nested_invalid_item_in_track(timeline_data):
    timeline_data["tracks"][0]["items"][0]["end"] = 0.0
    timeline_data["tracks"][0]["items"][0]["start"] = 0.0
    with pytest.raises(ValidationError):
        Timeline(**timeline_data)


def test_rejects_empty_track_name():
    with pytest.raises(ValidationError):
        Track(name="", kind="video")


def test_rejects_extra_field_on_timeline(timeline_data):
    timeline_data["extra"] = "nope"
    with pytest.raises(ValidationError):
        Timeline(**timeline_data)
