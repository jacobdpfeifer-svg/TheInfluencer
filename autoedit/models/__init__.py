"""Pydantic v2 data contracts shared by every stage of the pipeline.

See AGENTS.md "Data schemas" for the canonical description of each model.
"""

from autoedit.models.content_features import ContentFeatures, MotionBucket
from autoedit.models.media_asset import MediaAsset, MediaType
from autoedit.models.plan import EditOp, EditPlan
from autoedit.models.shot import Shot, ShotScale
from autoedit.models.style_profile import CaptionStyleFreq, StyleProfile
from autoedit.models.template import SlotRole, Template, TemplateMusic, TemplateSlot, TextSlot
from autoedit.models.timeline import Timeline, Track, TimelineItem, TrackKind

__all__ = [
    "MediaAsset",
    "MediaType",
    "Shot",
    "ShotScale",
    "StyleProfile",
    "CaptionStyleFreq",
    "ContentFeatures",
    "MotionBucket",
    "EditOp",
    "EditPlan",
    "Timeline",
    "Track",
    "TimelineItem",
    "TrackKind",
    "Template",
    "TemplateSlot",
    "TemplateMusic",
    "TextSlot",
    "SlotRole",
]
