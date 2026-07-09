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
                             tool manifest ──> director ──> EditPlan
                                                              │ (validate / fallback)
                                                              v
                                     executor ─> subsystems ─> Timeline ─> renderer ─> mp4
```

## The four extractors (shared by both phases)
1. **Pacing** (PySceneDetect + frame diff): cut list, median shot length, motion curve.
2. **Framing** (OpenCV): aspect, face count/position, shot scale (close/wide), static vs moving camera.
3. **Audio** (librosa): BPM, beat times, RMS energy curve, speech-vs-music flag.
4. **Text** (OCR): on-screen text position, timing vs cuts, style class (karaoke vs static title).

## Subsystems (pure functions over the Timeline)
`cutter`, `text_adder`, `emoji_adder`, `effects`. Each is
`(Timeline, params) -> Timeline`, writing instructions only. The **renderer** is the
single composite pass. The **executor** walks the `EditPlan` and dispatches ops by
name using the **tool manifest** (the map of what each subsystem accepts).

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
Timeline        tracks of instructions (OTIO-backed); NOT media
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

## Build order (vertical slice, thin path first)
0. Repo skeleton + all Pydantic models above (+ tests that they validate).
1. `ingest.probe` → MediaAsset.
2. The four shared extractors (each pure + tested on a fixture).
3. Phase A `aggregate` → StyleProfile (across MULTIPLE videos).
4. Phase B `content.extract` → ContentFeatures.
5. Subsystems + single-pass renderer.
6. Director: brief builder, stubbed LLM interface, EditPlan validation, heuristic fallback.
7. Executor + end-to-end CLI (`learn`, `make`) running fully with the LLM stubbed.
8. Plan-level review + publish gallery.

Only after step 7 runs deterministically do we connect a real model.

## Testing rule
No test may require a real video or a render to check logic. Everything before the
renderer works on fixture JSON. Ship a `/fixtures` dir of sample features, shots,
and plans.

## Known traps (encode once, avoid forever)
- MoviePy v2 API is incompatible with v1 — pin v2, ignore v1 tutorials.
- MoviePy `TextClip` needs an explicit font path (no system font resolution on Linux).
- `ffmpeg` is a required system dependency, not a pip package.
- A single reference video is one data point — StyleProfile must aggregate many.
