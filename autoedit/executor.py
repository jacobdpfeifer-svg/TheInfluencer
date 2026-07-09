"""executor — dispatch an EditPlan's ops through the tool manifest, then render once.

Per `.cursor/rules/subsystems.mdc`: "The executor walks an EditPlan,
dispatches each op to its subsystem by name (the tool manifest is the map),
then calls the renderer exactly once at the end." This module is the seam
between the director's plan and the renderer's pixels: it never mutates a
Timeline's semantics itself (that's each subsystem's job) and never
composites (that's the renderer's job) — it only sequences the two,
plus seeds the very first Timeline from `ContentFeatures`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from autoedit.models.content_features import ContentFeatures
from autoedit.models.plan import EditPlan
from autoedit.models.timeline import Timeline, TimelineItem, Track
from autoedit.renderer import render
from autoedit.subsystems import TOOL_MANIFEST

_VIDEO_TRACK_KIND = "video"
_VIDEO_TRACK_NAME = "v1"

ToolManifest = dict[str, Callable[[Timeline, dict], Timeline]]


class ExecutorError(Exception):
    """Raised when an EditPlan op names a tool absent from the manifest."""


def build_initial_timeline(features: ContentFeatures) -> Timeline:
    """Seed a Timeline's video track with every shot in `features`, at its natural (uncut) position.

    Each item's payload carries `shot`/`source`/`in`/`out` — everything
    `cutter`, `effects`, and `renderer.build_render_plan` need downstream
    (see `.cursor/rules/subsystems.mdc` and `fixtures/timeline.json`).
    """
    items = [
        TimelineItem(
            id=f"clip-{shot.id}",
            start=shot.in_,
            end=shot.out_,
            payload={"shot": shot.id, "source": shot.source, "in": shot.in_, "out": shot.out_},
        )
        for shot in features.shots
    ]
    return Timeline(tracks=[Track(name=_VIDEO_TRACK_NAME, kind=_VIDEO_TRACK_KIND, items=items)])


def execute(plan: EditPlan, timeline: Timeline, *, manifest: ToolManifest = TOOL_MANIFEST) -> Timeline:
    """Dispatch every op in `plan`, in order, through `manifest`. Returns the final Timeline.

    Raises:
        ExecutorError: an op names a tool not present in `manifest`. A plan
            that already passed `director.validate_plan` can't hit this,
            but the executor doesn't assume every caller validated first.
    """
    for op in plan.ops:
        if op.tool not in manifest:
            raise ExecutorError(f"executor: unknown tool {op.tool!r}; not in the tool manifest")
        timeline = manifest[op.tool](timeline, op.params)
    return timeline


def run(
    features: ContentFeatures,
    plan: EditPlan,
    output_path: str | Path,
    *,
    manifest: ToolManifest = TOOL_MANIFEST,
) -> Path:
    """Seed a Timeline from `features`, execute `plan` against it, render once to `output_path`.

    This is the one call that goes all the way from features + a plan to an
    actual mp4 — the renderer only ever runs here, exactly once.
    """
    timeline = build_initial_timeline(features)
    timeline = execute(plan, timeline, manifest=manifest)
    return render(timeline, output_path)
