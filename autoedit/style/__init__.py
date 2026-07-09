"""Phase A — aggregating per-video extractor features into a `StyleProfile`."""

from autoedit.style.aggregate import VideoFeatures, aggregate
from autoedit.style.extract_video_features import extract_video_features

__all__ = [
    "VideoFeatures",
    "aggregate",
    "extract_video_features",
]
