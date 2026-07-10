# TheInfluencer

Autonomous short-form video editor (`autoedit`). See `AGENTS.md` for the full architecture,
data schemas, and build order.

## Status

Past P0-P8 (below), the pipeline has grown a real (optional) LLM client, a
richer subsystem/effect library, a deterministic template system, and an
offline eval harness — all still governed by AGENTS.md's inviolable laws
(the LLM only reads/writes JSON; every AI decision has a working, tested,
deterministic fallback):

- **Real LLM director.** `director/prompts.py` + `director/openai_client.py`
  add an optional, real OpenAI-compatible chat-completions client
  (`make_llm_client`) alongside the always-available zero-AI `stub_llm`.
  `cli.make`/`cli.eval` take `--api-key`/`--model`/`--base-url` (env
  `AUTOEDIT_API_KEY`) to opt in; omit them and the pipeline is unchanged.
  `director/cache.py` (`cached_llm_client`) disk-caches responses by a hash
  of the brief so repeat runs against the same footage skip the network.
- **More effects/transitions.** `subsystems/effects.py` adds `speed_ramp`,
  `flash`, `shake`, `ken_burns`, `blur_intro` (alongside the original
  `zoom_in`/`zoom_out`); `subsystems/transitions.py` adds `fade`/`whip_pan`
  between shots. Both are driven purely by `renderer.py`, per the "subsystems
  never render" rule.
- **Reframe + music.** `subsystems/reframe.py` normalizes aspect ratio/crop
  per shot (`center_crop`/`rule_of_thirds`/`fit`); `subsystems/music.py`
  mixes an external music bed under the video, and its real detected
  `beat_times`/`bpm` (`--music`) become the authoritative beat grid the
  director/`cutter` sync cuts against, overriding the raw footage's own
  (often silent/irrelevant) audio track.
- **Template system.** `models/template.py` + `templates/` add a small
  library of built-in, hand-authored `Template`s (slot roles, text slots,
  aspect ratio). `templates.matcher.match_template` scores footage against
  every template; `templates.filler.fill_template` deterministically turns
  the best match into an `EditPlan` (Hungarian-algorithm slot->shot
  assignment, duration budgeting via `cutter`'s `trim`, reframe/transition
  ops from the template itself). `heuristic_plan` tries this first, falling
  back to its original simple logic only when no template scores well
  enough.
- **Smarter captions.** `templates/captions.py` (`generate_caption_copy`)
  replaces static placeholder text ("TITLE"/"CTA") with deterministic,
  footage-derived copy (energy/motion/face-aware). `renderer.py` also
  renders `style="karaoke"` text overlays word-by-word instead of as one
  static block.
- **Duration budgeting.** `subsystems/cutter.py`'s `trim` param caps a kept
  shot's duration (never lengthens it); `templates/filler.py` derives caps
  from a slot's `duration`/`max_duration` and soft-penalizes shots shorter
  than `min_duration`. `review.check_plan` also flags any plan whose total
  duration falls outside a short-form sanity range (3s-90s), independent of
  the learned `StyleProfile`.
- **Multi-clip ingest.** `content.extract_pool(paths) -> ContentFeatures`
  merges several source clips into one feature set (globally unique shot
  ids, pooled aspect/speech/music/motion); `cli.make` accepts multiple
  footage paths and routes to it automatically.
- **Offline eval harness.** `director/eval.py` (`run_eval`) calls a real LLM
  directly against a directory of hand-authored `EvalCase`s with **no**
  heuristic fallback, to measure the model itself rather than the
  system's safety net — exposed as `autoedit eval`. `cli.make --dry-run`
  writes the directed `{plan, review}` JSON instead of rendering, for fast
  iteration on the director alone.

See each module's own docstring for the exact contract; the summary above is
intentionally terse — this file is a map, not the spec (that's AGENTS.md).

**P8 — Review + publish.** `autoedit/review.py` adds a plan-level sanity
check and a minimal results gallery, wired into `make`:

- `review.check_plan(plan, style, *, features=None) -> ReviewResult` — a
  **text-only** check (`EditPlan`/`StyleProfile` JSON, never a Timeline or a
  render): caption density, text amount, and implied cut pacing, each
  compared against the learned `StyleProfile`'s own values within a ±50%
  tolerance (or a small absolute epsilon when the profile's value is ~0).
  Pass `features` (the same `ContentFeatures` the plan was built from) for
  an accurate check using real per-shot durations; without it, durations
  fall back to a generic per-shot estimate that's deliberately *not*
  derived from `style` itself (so the pacing check stays meaningful, not
  circular). Returns a `ReviewResult` (`passed`, `warnings`, `metrics`).
- `review.GalleryEntry` / `build_gallery` / `append_gallery_entry` — a
  minimal, self-contained (no JS) HTML gallery: one card per rendered
  video with its confidence and PASS/WARN badge. `append_gallery_entry`
  keeps a small `gallery.json` manifest alongside `gallery.html` so
  results persist and accumulate across multiple `make` runs.
- `cli.make` now always runs `check_plan` and prints any warnings to
  stderr, and accepts an optional `-g/--gallery <dir>` flag to append the
  run to a gallery.

Tests cover the exact ask directly (`check_plan` flagging an
over-captioned plan, both with the duration estimate and with real
`features`-derived durations), a passing case for contrast, the cut-pacing
check, gallery HTML rendering/escaping/persistence, and the CLI wiring
(`-g` writes a gallery; a forced over-captioned plan prints `[review]`
warnings to stderr).

**P7 — Executor + end-to-end.** `autoedit/executor.py` and `autoedit/cli.py`
close the loop: raw footage in, an mp4 out, with zero AI required.

- `executor.build_initial_timeline(features) -> Timeline` seeds a video
  track with one item per shot, at its natural (uncut) position, carrying
  `shot`/`source`/`in`/`out` in its payload — everything `cutter`,
  `effects`, and the renderer need downstream.
- `executor.execute(plan, timeline) -> Timeline` walks an `EditPlan`'s ops
  through `subsystems.TOOL_MANIFEST` in order (defensively raising
  `ExecutorError` on an unknown tool, even though a plan that passed
  `director.validate_plan` can't produce one).
- `executor.run(features, plan, output_path)` ties it together: seed ->
  execute -> `renderer.render`, called exactly once, per
  `.cursor/rules/subsystems.mdc`.
- `style/extract_video_features.py` adds the missing Phase A wiring (all
  four shared extractors -> one `VideoFeatures`, mirroring what
  `content.extract` already does for Phase B) so `learn` has something to
  aggregate.
- `cli.py` — `learn <videos...> [-o style_profile.json]` extracts +
  aggregates reference videos into a `StyleProfile` JSON file;
  `make <video> [-s style.json] [-o out.mp4]` runs `content.extract ->
  director.direct -> executor.run` end to end. `make` always calls
  `director.direct` with its default stub LLM — this is what makes the
  whole CLI run with zero AI: the stub always fails validation, `direct`
  always falls back to `heuristic_plan`, and the pipeline still produces a
  real mp4. Installed as a console script (`autoedit learn ...` /
  `autoedit make ...`) via `pyproject.toml`'s `[project.scripts]`.

`tests/test_end_to_end.py` is the one deliberate exception to "no render in
tests" for this build step: it runs `learn` and `make` (including `cli.main`
itself) against real tiny fixture clips and asserts a real, playable mp4
comes out the other end — proving build order step 7's requirement
literally ("running fully with the LLM stubbed... must run start to finish
with zero AI"). `tests/test_executor.py` covers `build_initial_timeline`/
`execute` purely on fixture JSON, and `run`'s sequencing with
`renderer.render` monkeypatched out, same as every other subsystem test.

**P6 — Director.** On top of P0-P5, `autoedit/director/` implements the
ONE module allowed to call an LLM (per `.cursor/rules/director.mdc`),
turning `ContentFeatures` + `StyleProfile` + the tool manifest into a
validated `EditPlan`:

- `brief.build_brief(features, style, manifest) -> dict` — a pure, compact,
  JSON-serializable summary (quantized per-shot motion buckets, boolean
  flags, sorted tool names) — the entire, lossy "prompt payload."
- `llm.stub_llm(brief) -> Any` — the default LLM client: it deliberately
  returns an unusable response (`{"ops": [], "confidence": 0.0}`), so with
  AI "disabled" the pipeline always exercises the same fallback a real
  model's bad output would. `LLMClient` is just `Callable[[dict], Any]`, so
  swapping in a real model later is a one-line change at the call site.
- `validate.validate_plan(raw) -> EditPlan` — two-layer validation: does
  `raw` even parse as an `EditPlan`, and does every op's `tool` exist in
  `subsystems.TOOL_PARAMS_MANIFEST` with params matching that tool's own
  Pydantic params model (`CutterParams`, etc. from P5)? Raises
  `PlanValidationError` (never anything else) on failure.
- `heuristic.heuristic_plan(style, features) -> EditPlan` — the
  deterministic fallback. Always keeps every shot (beat-synced iff the
  style prefers it and music was detected), and only adds text/effect/emoji
  ops when the StyleProfile's own signals justify them — it never invents
  caption copy, since real text generation is an LLM-only concern.
- `director.direct(features, style, *, llm=stub_llm, confidence_threshold=0.6)
  -> EditPlan` — the full orchestration: brief -> `llm(brief)` ->
  `validate_plan` -> `heuristic_plan` fallback on ANY failure (malformed
  output, unknown tool, bad params, low confidence, or `llm` itself
  raising). The system never crashes on bad model output, per AGENTS.md's
  inviolable law #4.

Tests cover the stub returning outright garbage (`None`, a bare string, an
unknown tool, invalid params, sub-threshold confidence, a raising callable)
and confirm `direct` always still yields a valid `EditPlan` (`heuristic_plan`'s
own output, byte-for-byte).

**P5 — Subsystems + renderer.** `autoedit/subsystems/` and
`autoedit/renderer.py` implement the execution side of the pipeline —
everything an `EditPlan` can be turned into, short of an actual executor:

- Four subsystems, each a pure `(Timeline, params) -> Timeline` function that
  only rewrites instructions (never opens/decodes/writes media):
  `cutter.cut` (keep/reorder/retime shots already on the video track, with an
  optional beat-sync that snaps cut points to the nearest beat within
  tolerance), `text_adder.add_text`, `emoji_adder.add_emoji`, and
  `effects.apply_effect` (which resolves a named shot's *current*, post-cut
  span off the video track, or accepts an explicit span).
- `autoedit/subsystems/__init__.py` exposes `TOOL_MANIFEST`, the
  `EditOp.tool` name -> subsystem function map a future executor will
  dispatch an `EditPlan` through.
- `autoedit/renderer.py` is split for testability: `build_render_plan` is a
  pure Timeline -> `RenderPlan` translation (fully tested, no file IO or
  MoviePy at all) and `render(timeline, output_path)` is the actual single
  composite pass, driving MoviePy 2.x to produce one mp4 (concatenate cut
  segments, apply per-shot effects/transitions/reframe, composite text/emoji
  overlays and any music bed, encode once). `render()` itself needs real
  source media, so per AGENTS.md's testing rule it's covered separately by
  `tests/test_render_smoke.py` (`@pytest.mark.render`, opt-in/slow — every
  effect, transition, karaoke caption, music mix, and multi-source clip gets
  a real render there) rather than the default fixture-JSON suite. `TextClip`
  uses explicit vendored fonts (`fonts/Roboto-Regular.ttf` for text,
  `fonts/NotoColorEmoji-Regular.ttf` for emoji — system font resolution can't be
  relied on, per `.cursor/rules/subsystems.mdc` and AGENTS.md's known traps).

## Setup

Requires on `PATH`: `ffmpeg`/`ffprobe` (e.g. `brew install ffmpeg`) and
`tesseract` (e.g. `brew install tesseract`).

```bash
pip install -e ".[dev]"
```

## Usage

```bash
# Learn a style from videos you like the editing of:
autoedit learn ref1.mp4 ref2.mp4 ref3.mp4 -o style_profile.json

# Auto-edit your own raw footage to match (zero AI, heuristic/template director):
autoedit make my_footage.mp4 -s style_profile.json -o out.mp4

# Pour several clips into one edit (shots pooled across all of them):
autoedit make clip1.mp4 clip2.mp4 clip3.mp4 -s style_profile.json -o out.mp4

# Mix in a music bed and beat-sync cuts against ITS beats, not the footage's:
autoedit make my_footage.mp4 --music song.mp3 -o out.mp4

# Append this run to a simple local HTML results gallery:
autoedit make my_footage.mp4 -g gallery/ -o out.mp4

# Use a real LLM director instead of the zero-AI stub (env: AUTOEDIT_API_KEY):
autoedit make my_footage.mp4 --api-key sk-... -o out.mp4

# Inspect what the director would decide without rendering an mp4:
autoedit make my_footage.mp4 --dry-run -o plan.json

# Score a real LLM directly (no heuristic fallback) against fixture cases:
autoedit eval --api-key sk-... --cases-dir fixtures/eval_cases
```

`make` works even without `-s` (falls back to a neutral built-in
`StyleProfile`), and always runs with zero AI by default (the director's
LLM is stubbed — see `autoedit/director/llm.py`). Passing `--api-key` (or
setting `AUTOEDIT_API_KEY`) swaps in a real model via
`director.make_llm_client`; `--cache-dir` disk-caches its responses by
brief so repeat runs skip the network. Run `autoedit make -h` / `autoedit
eval -h` for the full flag list (`--model`, `--base-url`,
`--confidence-threshold`, etc.) — see `autoedit/cli.py`'s module docstring
for the rationale behind `--dry-run` vs. `eval`.

## Test

```bash
pytest                    # everything, including the real render path (slower)
pytest -m "not render"    # fast unit loop: skip the real MoviePy render path
pytest -m render          # just the real MoviePy render path (needs ffmpeg + fixtures)
```

`render` is a registered marker (`pyproject.toml`) on `tests/test_render_smoke.py`
(every effect, transition, karaoke caption, music mix, multi-source clip gets
a real render there) — it runs by default with plain `pytest` since it's the
only place a genuine defect in the MoviePy composite path would surface; use
`-m "not render"` for a fast inner loop while iterating on everything else.

Every stage before the renderer is tested against fixture JSON in
`/fixtures`, with one deliberate exception: `ingest.probe`, the four
extractors, and `content.extract` are the seam between real files and that
JSON-only world, so their tests exercise tiny (a few KB to ~200KB) real
fixtures under `fixtures/media/` instead — see each fixture's docstring in
`tests/conftest.py` for exactly how it was generated (all synthetic/rendered
except one public-domain NASA astronaut portrait used for face detection).
`content.extract`'s tests pair those real end-to-end runs with monkeypatched
unit tests (fixed `PacingFeatures`/`FramingFeatures`/`AudioFeatures`) that
exercise its assembly/quantization logic deterministically. `style.aggregate`
operates purely on already-computed feature JSON, so `test_aggregate.py`
goes back to plain fixtures (`fixtures/video_features_*.json`). The
subsystems and `renderer.build_render_plan` are pure `Timeline` -> `Timeline`
(or -> `RenderPlan`) functions, so their tests build small `Timeline` objects
in-memory — no media, no fixtures, no render. The director never touches
media or a Timeline at all — its tests work purely over
`ContentFeatures`/`StyleProfile`/`EditPlan` JSON, with no real LLM ever
called (`direct()` is exercised with hand-crafted `llm` callables, including
ones that return garbage or raise, to prove the heuristic fallback always
holds). `tests/test_end_to_end.py` is the sole exception for the executor/
CLI layer — see the P7 status note above.

## Layout

```
autoedit/
  ingest.py       # probe(path) -> MediaAsset via ffprobe (+ typed errors)
  extractors/     # pacing, framing, audio, text — pure measurement functions
  style/
    aggregate.py               # VideoFeatures + aggregate(videos) -> StyleProfile (Phase A)
    extract_video_features.py  # runs all 4 extractors -> one VideoFeatures, for `learn`
  content/
    extract.py      # extract(path)/extract_pool(paths) -> ContentFeatures (Phase B)
  subsystems/       # (Timeline, params) -> Timeline, dispatched via TOOL_MANIFEST:
    cutter.py        # keep/reorder/retime/trim shots, optional beat-sync
    text_adder.py     # static or karaoke text overlays
    emoji_adder.py    # emoji overlays
    effects.py        # zoom_in/out, speed_ramp, flash, shake, ken_burns, blur_intro
    transitions.py    # fade, whip_pan between shots
    reframe.py         # per-shot aspect/crop normalization (center_crop/rule_of_thirds/fit)
    music.py            # mixes an external music bed under the video
  renderer.py       # build_render_plan (pure) + render (the one MoviePy composite pass)
  templates/        # the deterministic template system:
    __init__.py       # TEMPLATE_REGISTRY, load_template, list_templates
    builtin/            # built-in Template JSON (punchy_beat_montage, etc.)
    matcher.py         # match_template(features) -> best-scoring Template
    filler.py           # fill_template(template, features) -> EditPlan
    captions.py         # generate_caption_copy: footage-derived text for text slots
  director/          # the ONE module allowed to call an LLM:
    brief.py           # build_brief(features, style, manifest) -> dict
    llm.py               # LLMClient protocol + stub_llm (always-invalid, zero-AI default)
    openai_client.py      # make_llm_client: real OpenAI-compatible chat-completions client
    prompts.py             # SYSTEM_PROMPT, few-shot examples, build_messages
    cache.py                # cached_llm_client: disk cache keyed by brief hash
    validate.py              # validate_plan(raw) -> EditPlan | PlanValidationError
    heuristic.py              # heuristic_plan: template-first, simple-logic fallback
    eval.py                    # run_eval: offline LLM scoring, NO heuristic fallback
    director.py                # direct(): brief -> llm -> validate -> heuristic fallback
  executor.py       # build_initial_timeline + execute + run (dispatch -> render once)
  review.py         # check_plan (plan-level sanity check) + gallery HTML output
  cli.py            # learn <videos...>, make <video...>, eval -- see cli.py's docstring
  models/           # Pydantic v2 contracts (MediaAsset, Shot, ContentFeatures,
                     # StyleProfile, Template, Timeline, EditOp/EditPlan, ...)
fonts/            # Vendored Roboto-Regular.ttf (text) + Noto Color Emoji (emoji), both
                  # for TextClip -- see AGENTS.md's "known traps"
fixtures/         # Sample JSON/media for every model and eval case, used by tests
  media/          # Tiny real clip/audio fixtures for ingest + extractor tests
  video_features_*.json  # Per-video extractor-output fixtures for aggregate()
  eval_cases/     # EvalCase JSON fixtures for `autoedit eval` / director/eval.py
tests/            # One test module per model/module: valid load + rejection cases;
                  # test_render_smoke.py is the one place the real MoviePy render runs
```
