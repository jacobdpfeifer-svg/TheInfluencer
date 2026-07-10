# autoedit — project spec

Autonomous short-form video editor. The user pastes reference videos they like;
the program learns their style into a preferences file. The user then pastes their
own raw footage; the program reduces it to machine-readable features, an LLM
**director** turns features + preferences + a tool menu into an edit **plan**, and
deterministic subsystems execute that plan into a finished vertical video.

**The AI only directs. It never edits pixels.** Everything the AI does is: read
JSON, hold the user's preference context, emit a validated `EditPlan`. If the AI
is disabled, a heuristic planner produces a valid plan and the program still works
end to end.

---

## Architecture (two phases, one set of extractors)

**Phase A — Learn.** `reference videos → extractors → per-video features →
aggregate across many videos → StyleProfile.json`
The StyleProfile is the user's preference context: distributions (median ± spread),
not single-video values.

**Phase B — Edit.** `raw footage → same extractors → ContentFeatures.json →
director (StyleProfile + tool manifest → EditPlan) → executor → subsystems mutate
Timeline → render once → publish`

```
ingest ─┐
        ├─> extractors ─┬─(Phase A)─> aggregate ─> StyleProfile
        │               └─(Phase B)─> ContentFeatures ─┐
                                                        v
                                                  templates/ ─┐
                                                (match_template│
                                                 + fill_template)
                                                              v
                             tool manifest ──> director ──> EditPlan
                                                              │ (validate / fallback)
                                                              v
                                     executor ─> subsystems ─> Timeline ─> renderer ─> mp4
```

`templates/` (`match_template` + `fill_template`) is deterministic — no LLM,
same as the heuristic fallback. `match_template` picks the best-fitting
builtin `Template` for a `ContentFeatures`/`StyleProfile` pair and
`fill_template` pours real shots into its slots to produce an `EditPlan`
(via `scipy`'s Hungarian algorithm for a globally-optimal slot<->shot
assignment). The director's heuristic fallback (`director/heuristic.py`)
tries this template fill first and only drops to its own bare rule-based
plan when no template scores above a minimum fit threshold. When the LLM is
active, the matched template's name is included in the director's brief so
the model knows which structure is already in play — but the LLM is free
to override the template's plan entirely; templates never constrain it.

## The four extractors (shared by both phases)
1. **Pacing** (PySceneDetect + frame diff): cut list, median shot length, motion curve.
2. **Framing** (OpenCV): aspect, face count/position, shot scale (close/wide), static vs moving camera.
3. **Audio** (librosa): BPM, beat times, RMS energy curve, speech-vs-music flag.
4. **Text** (OCR): on-screen text position, timing vs cuts, style class (karaoke vs static title).

## Subsystems (pure functions over the Timeline)
`cutter`, `text_adder`, `emoji_adder`, `effects`, `transitions`, `reframe`,
`music`. Each is `(Timeline, params) -> Timeline`, writing instructions only.
The **renderer** is the single composite pass. The **executor** walks the
`EditPlan` and dispatches ops by name using the **tool manifest** (the map of
what each subsystem accepts).

---

## Data schemas (Pydantic v2 — the contracts between every stage)

```
MediaAsset      path, type, duration, width, height, fps, codec, audio_channels
Shot            id, source, in, out, dur, motion, brightness, sharpness,
                faces, scale ("close"|"wide")
StyleProfile    aspect, shot_len_median, shot_len_spread, cut_on_beat: bool,
                caption_style_freq: {karaoke: float, static: float},
                caption_density, text_amount, effect_freq, sample_count
ContentFeatures aspect, has_speech, music_bpm, shots: [Shot],
                flags/buckets (motion: low|med|high, is_vertical, has_face...)
EditOp          tool: str, params: dict            # tool must exist in the manifest
EditPlan        ops: [EditOp], confidence: float
Timeline        tracks of instructions (a plain Pydantic model that mirrors
                OpenTimelineIO's track/clip shape, not the `opentimelineio`
                package itself — see `models/timeline.py`); NOT media
```

`EditPlan` example:
```json
{ "ops": [
    { "tool": "cutter", "params": { "keep": ["s1","s2"], "sync": "beat" } },
    { "tool": "text",   "params": { "content": "3 months in", "style": "karaoke",
                                    "anchor": "top", "start": 0.0 } },
    { "tool": "emoji",  "params": { "glyph": "🔥", "at": 14.2 } },
    { "tool": "effect", "params": { "kind": "zoom_in", "shot": "s1" } } ],
  "confidence": 0.88 }
```

---

## Current state
Steps 0-8 are complete. The pipeline runs end-to-end with zero AI, the
template system (3 builtin templates, Hungarian-algorithm slot assignment),
7 subsystems (cutter, text, emoji, effect, transition, reframe, music),
a real LLM client (openai_client.py), an offline eval harness, a results
gallery, dry-run mode, multi-clip pooling, and beat-synced cuts.

## Next milestones
1. Split renderer.py into a package (video/overlays/audio/transitions).
2. More builtin templates (product showcase, before/after, tutorial).
3. User-created templates (reverse-engineer a reference video's cut
   rhythm into a reusable Template JSON).
4. Optional ML module: auto-reframe via subject tracking (the first place
   the no-ML approach strains — currently limited to center_crop /
   rule_of_thirds).
5. A lightweight UI for previewing plans before rendering.

## Testing rule
No test may require a real video or a render to check logic. Everything before the
renderer works on fixture JSON. Ship a `/fixtures` dir of sample features, shots,
and plans.

## Known traps (encode once, avoid forever)
- MoviePy v2 API is incompatible with v1 — pin v2, ignore v1 tutorials.
- MoviePy `TextClip` needs an explicit font path (no system font resolution on Linux).
- `ffmpeg` is a required system dependency, not a pip package.
- A single reference video is one data point — StyleProfile must aggregate many.
- Both `opencv-python` AND `opencv-python-headless` must be pinned `<5` (scenedetect
  pulls in the non-headless variant unpinned; 5.0 removed `CascadeClassifier`).
- Roboto has no emoji glyphs — emoji overlays need NotoColorEmoji or similar.
- `extract_pool` always sets `beat_times=[]` — pass `--music` for beat-sync on
  multi-clip edits.
