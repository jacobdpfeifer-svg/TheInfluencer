"""Tests for `autoedit.style.aggregate.aggregate`.

Pure fixture-JSON tests (per AGENTS.md's testing rule) — `aggregate` never
touches media itself, it only folds already-computed extractor features
(see `autoedit.style.aggregate.VideoFeatures`) across many reference videos.
"""

from __future__ import annotations

import pytest

from autoedit.models.style_profile import StyleProfile
from autoedit.style.aggregate import VideoFeatures, aggregate


@pytest.fixture
def three_videos(
    video_features_fast_cuts_data, video_features_slow_static_data, video_features_mixed_data
) -> list[VideoFeatures]:
    return [
        VideoFeatures(**video_features_fast_cuts_data),
        VideoFeatures(**video_features_slow_static_data),
        VideoFeatures(**video_features_mixed_data),
    ]


def test_aggregates_three_videos_into_a_valid_style_profile(three_videos):
    profile = aggregate(three_videos)

    assert isinstance(profile, StyleProfile)
    assert profile.sample_count == 3


def test_aspect_is_the_median_across_videos(three_videos):
    # fast_cuts=0.5625, slow_static=0.5, mixed=0.6 -> median is the middle value.
    profile = aggregate(three_videos)
    assert profile.aspect == pytest.approx(0.5625)


def test_shot_length_median_and_spread_pool_all_shots(three_videos):
    # Pooled shot lengths: [0.5,0.5,0.5,0.5, 3,3,3, 1,3,2] -> median 1.5.
    profile = aggregate(three_videos)
    assert profile.shot_len_median == pytest.approx(1.5)
    assert profile.shot_len_spread > 0.0


def test_cut_on_beat_majority_vote_ignores_videos_without_beats(three_videos):
    # Only fast_cuts has beat data at all, and every one of its cuts lands
    # exactly on a beat -> the (only) vote is True.
    profile = aggregate(three_videos)
    assert profile.cut_on_beat is True


def test_caption_style_freq_counts_shots_across_all_videos(three_videos):
    # karaoke: 4 shots (fast_cuts). static: 3 (slow_static) + 2 (mixed) = 5.
    # The mixed video's one uncaptioned ("none") shot is excluded entirely.
    profile = aggregate(three_videos)
    assert profile.caption_style_freq.karaoke == pytest.approx(4 / 9)
    assert profile.caption_style_freq.static == pytest.approx(5 / 9)


def test_caption_density_and_text_amount_are_averaged_across_videos(three_videos):
    profile = aggregate(three_videos)
    # per-video: fast_cuts=8/2=4.0, slow_static=3/9=0.333, mixed=2/6=0.333
    assert profile.caption_density == pytest.approx((4.0 + 3 / 9 + 2 / 6) / 3)
    # per-video text_amount (fraction of duration covered by on-screen text):
    # fast_cuts=1.0, slow_static=8.4/9, mixed=3.9/6
    assert profile.text_amount == pytest.approx((1.0 + 8.4 / 9 + 3.9 / 6) / 3)


def test_effect_freq_defaults_to_zero_with_no_effects_extractor(three_videos):
    profile = aggregate(three_videos)
    assert profile.effect_freq == 0.0


def test_rejects_empty_video_list():
    with pytest.raises(ValueError):
        aggregate([])


def test_warns_when_aggregating_fewer_than_recommended_samples(video_features_fast_cuts_data):
    video = VideoFeatures(**video_features_fast_cuts_data)
    with pytest.warns(UserWarning, match="single reference"):
        profile = aggregate([video])
    assert profile.sample_count == 1


def test_no_warning_with_at_least_three_videos(three_videos, recwarn):
    aggregate(three_videos)
    assert len(recwarn) == 0


def test_neutral_caption_style_freq_when_no_video_has_captions(video_features_fast_cuts_data):
    video = VideoFeatures(**video_features_fast_cuts_data)
    for shot in video.text.shots:
        shot.style = "none"
        shot.events = []

    profile = aggregate([video, video, video])
    assert profile.caption_style_freq.karaoke == pytest.approx(0.5)
    assert profile.caption_style_freq.static == pytest.approx(0.5)


def test_cut_on_beat_is_false_when_no_video_has_beat_data(
    video_features_slow_static_data, video_features_mixed_data
):
    videos = [
        VideoFeatures(**video_features_slow_static_data),
        VideoFeatures(**video_features_mixed_data),
    ]
    with pytest.warns(UserWarning, match="single reference"):
        profile = aggregate(videos)
    assert profile.cut_on_beat is False


def test_shot_len_spread_is_zero_for_a_single_uniform_shot_length(video_features_fast_cuts_data):
    # fast_cuts' shot_lengths are all 0.5 -> zero spread regardless of method.
    video = VideoFeatures(**video_features_fast_cuts_data)
    with pytest.warns(UserWarning, match="single reference"):
        profile = aggregate([video])
    assert profile.shot_len_spread == pytest.approx(0.0)
