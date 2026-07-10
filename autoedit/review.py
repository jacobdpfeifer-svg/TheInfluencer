"""review — plan-level sanity checks + a simple results gallery.

Per AGENTS.md build order step 8 ("Plan-level review + publish gallery"):
before ever handing an `EditPlan` to the executor, sanity-check it against
the `StyleProfile` it was supposedly built from. This is **text-only** — it
reads `EditPlan`/`StyleProfile` JSON (plus, optionally, `ContentFeatures`
for real shot durations), never a `Timeline`, a video, or a rendered frame.

An `EditPlan`'s ops don't carry absolute timing by themselves (that lives on
the `Timeline`, seeded from `ContentFeatures.shots` — see `executor.py`), so
`check_plan` accepts an *optional* `features` argument: pass it (e.g. the
same `ContentFeatures` the plan was built from) for an accurate check using
real per-shot durations; without it, `check_plan` still works, falling back
to a generic per-shot duration estimate (documented below) for a rougher
— but still meaningful and non-circular — check.

Checks: caption density, text amount, implied cut pacing (all relative to
the learned `StyleProfile`), and total duration against a fixed, style-
independent short-form budget (`StyleProfile` has no total-duration field
of its own to check against).
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from autoedit.models.content_features import ContentFeatures
from autoedit.models.plan import EditPlan
from autoedit.models.style_profile import StyleProfile
from autoedit.subsystems.text_adder import TextParams

# Used only as a duration estimate when the caller can't supply `features`
# — a generic per-shot assumption, deliberately NOT derived from the
# StyleProfile being checked against (that would make the pacing check
# circular and always pass).
_FALLBACK_SHOT_LEN_ESTIMATE_SEC = 2.0

# How far a plan's caption density / text amount may drift from the
# profile's own value (as a fraction of that value) before being flagged.
_RELATIVE_TOLERANCE = 0.5
# When the profile's own value is ~0 (e.g. a style that never captions),
# any measured amount above this absolute epsilon is still worth flagging.
_ZERO_TARGET_EPSILON = 0.02
# How many multiples of `style.shot_len_spread` the plan's implied average
# shot length may differ from `style.shot_len_median` before being flagged.
_SHOT_LEN_SPREAD_SLACK = 1.5

# Absolute short-form duration budget, independent of the learned style
# (StyleProfile has no total-duration field to check against — see
# `.cursor/rules/director.mdc`'s "15-60s" guidance, echoed in
# `director/prompts.py`'s SYSTEM_PROMPT). Deliberately a little more
# permissive than that guidance on both ends, so this only ever catches a
# plan that's genuinely degenerate (near-empty) or runaway (way over any
# platform's "short-form" definition), not a merely-atypical one.
_MIN_PLAN_DURATION_SEC = 3.0
_MAX_PLAN_DURATION_SEC = 90.0


class ReviewResult(BaseModel):
    """The verdict `check_plan` returns: whether the plan passed, and why not."""

    model_config = ConfigDict(extra="forbid")

    passed: bool = Field(description="True iff no check was flagged.")
    warnings: list[str] = Field(default_factory=list, description="Human-readable descriptions of each flagged check.")
    metrics: dict[str, float | int | bool] = Field(
        default_factory=dict, description="The raw numbers behind the checks, for debugging/the gallery."
    )


def check_plan(
    plan: EditPlan, style: StyleProfile, *, features: Optional[ContentFeatures] = None
) -> ReviewResult:
    """Sanity-check `plan` against the learned `style`: caption density, text amount, cut pacing.

    Pass `features` (the `ContentFeatures` the plan was built from) for an
    accurate check using real shot durations; without it, durations are
    estimated (see module docstring), which still supports a meaningful
    caption density / text amount check but a rougher pacing check.
    """
    keep = _cutter_keep(plan)
    shot_count = len(keep)
    duration, duration_is_estimated = _resolve_duration(keep, features)

    text_ops_durations = [TextParams.model_validate(op.params).duration for op in plan.ops if op.tool == "text"]
    caption_count = len(text_ops_durations)
    text_covered = sum(text_ops_durations)

    metrics: dict[str, float | int | bool] = {
        "shot_count": shot_count,
        "cut_count": max(shot_count - 1, 0),
        "duration": duration,
        "duration_is_estimated": duration_is_estimated,
        "caption_count": caption_count,
        "text_covered": text_covered,
    }

    warnings: list[str] = []
    if not (_MIN_PLAN_DURATION_SEC <= duration <= _MAX_PLAN_DURATION_SEC):
        estimated_note = " (estimated durations)" if duration_is_estimated else ""
        warnings.append(
            f"total duration {duration:.2f}s is outside the short-form budget "
            f"[{_MIN_PLAN_DURATION_SEC:.0f}s, {_MAX_PLAN_DURATION_SEC:.0f}s]{estimated_note}"
        )

    if duration > 0:
        caption_density = caption_count / duration
        text_amount = text_covered / duration
        metrics["caption_density"] = caption_density
        metrics["text_amount"] = text_amount

        if _exceeds_tolerance(caption_density, style.caption_density):
            warnings.append(
                f"caption density {caption_density:.2f}/s is outside the learned range "
                f"(profile median {style.caption_density:.2f}/s ± {_RELATIVE_TOLERANCE:.0%})"
            )
        if _exceeds_tolerance(text_amount, style.text_amount):
            warnings.append(
                f"text amount {text_amount:.2f} is outside the learned range "
                f"(profile median {style.text_amount:.2f} ± {_RELATIVE_TOLERANCE:.0%})"
            )

        if shot_count > 0:
            implied_shot_len = duration / shot_count
            metrics["implied_shot_len"] = implied_shot_len
            low = max(style.shot_len_median - _SHOT_LEN_SPREAD_SLACK * style.shot_len_spread, 0.0)
            high = style.shot_len_median + _SHOT_LEN_SPREAD_SLACK * style.shot_len_spread
            if not (low <= implied_shot_len <= high):
                estimated_note = " (estimated durations)" if duration_is_estimated else ""
                warnings.append(
                    f"implied average shot length {implied_shot_len:.2f}s is outside the learned "
                    f"range [{low:.2f}s, {high:.2f}s]{estimated_note}"
                )

    return ReviewResult(passed=not warnings, warnings=warnings, metrics=metrics)


def _cutter_keep(plan: EditPlan) -> list[str]:
    cutter_op = next((op for op in plan.ops if op.tool == "cutter"), None)
    if cutter_op is None:
        return []
    return list(cutter_op.params.get("keep", []))


def _resolve_duration(keep: list[str], features: Optional[ContentFeatures]) -> tuple[float, bool]:
    """Return `(duration, was_estimated)` for the shots in `keep`."""
    if features is not None:
        durations_by_id = {shot.id: shot.dur for shot in features.shots}
        total = sum(durations_by_id[shot_id] for shot_id in keep if shot_id in durations_by_id)
        if total > 0:
            return total, False
    return len(keep) * _FALLBACK_SHOT_LEN_ESTIMATE_SEC, True


def _exceeds_tolerance(actual: float, target: float) -> bool:
    if target <= 0:
        return actual > _ZERO_TARGET_EPSILON
    lower = target * (1 - _RELATIVE_TOLERANCE)
    upper = target * (1 + _RELATIVE_TOLERANCE)
    return not (lower <= actual <= upper)


# --- publish gallery ---------------------------------------------------------


class GalleryEntry(BaseModel):
    """One rendered result, ready to show in the gallery."""

    model_config = ConfigDict(extra="forbid")

    video_path: str = Field(description="Path to the source raw footage.")
    output_path: str = Field(description="Path to the rendered mp4.")
    confidence: float = Field(ge=0, le=1, description="The EditPlan's own confidence.")
    review: ReviewResult = Field(description="This plan's check_plan() result.")


_GALLERY_HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>autoedit gallery</title>
<style>
  body {{ font-family: -apple-system, sans-serif; background: #111; color: #eee; padding: 2rem; }}
  .card {{ border: 1px solid #333; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; max-width: 420px; }}
  .pass {{ color: #4caf50; font-weight: bold; }}
  .warn {{ color: #ff9800; font-weight: bold; }}
  video {{ max-width: 100%; border-radius: 4px; background: #000; }}
</style>
</head>
<body>
<h1>autoedit gallery</h1>
{cards}
</body>
</html>
"""

_EMPTY_GALLERY_BODY = "<p>No videos yet — run <code>autoedit make</code> to add one.</p>"


def build_gallery(entries: list[GalleryEntry]) -> str:
    """Render a simple, self-contained HTML page listing every entry's video + review status."""
    cards = "\n".join(_render_gallery_card(entry) for entry in entries) if entries else _EMPTY_GALLERY_BODY
    return _GALLERY_HTML_TEMPLATE.format(cards=cards)


def _render_gallery_card(entry: GalleryEntry) -> str:
    status_class = "pass" if entry.review.passed else "warn"
    status_text = "PASS" if entry.review.passed else f"WARN ({len(entry.review.warnings)})"
    warnings_html = "".join(f"<li>{html.escape(warning)}</li>" for warning in entry.review.warnings)
    warnings_block = f"<ul>{warnings_html}</ul>" if entry.review.warnings else ""
    return (
        '<div class="card">\n'
        f"  <h2>{html.escape(Path(entry.output_path).name)}</h2>\n"
        f'  <video controls src="{html.escape(entry.output_path)}"></video>\n'
        f"  <p>Source: {html.escape(entry.video_path)}</p>\n"
        f'  <p>Confidence: {entry.confidence:.2f} &mdash; <span class="{status_class}">{status_text}</span></p>\n'
        f"  {warnings_block}\n"
        "</div>"
    )


_MANIFEST_FILENAME = "gallery.json"
_HTML_FILENAME = "gallery.html"


def append_gallery_entry(entry: GalleryEntry, gallery_dir: str | Path) -> Path:
    """Append `entry` to the gallery at `gallery_dir`, rewriting its JSON manifest and HTML page.

    Returns the path to the (re)written `gallery.html`.
    """
    directory = Path(gallery_dir)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / _MANIFEST_FILENAME

    entries = _load_gallery_manifest(manifest_path)
    entries.append(entry)

    manifest_path.write_text(json.dumps([e.model_dump() for e in entries], indent=2))
    html_path = directory / _HTML_FILENAME
    html_path.write_text(build_gallery(entries))
    return html_path


def _load_gallery_manifest(manifest_path: Path) -> list[GalleryEntry]:
    if not manifest_path.exists():
        return []
    data = json.loads(manifest_path.read_text())
    return [GalleryEntry.model_validate(item) for item in data]
