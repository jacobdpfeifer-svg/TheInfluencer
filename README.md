# TheInfluencer

Autonomous short-form video editor (`autoedit`). See `AGENTS.md` for the full architecture,
data schemas, and build order.

## Status

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
  segments, apply per-shot effects, composite text/emoji overlays, encode
  once). Per AGENTS.md's testing rule, `render()` itself needs real source
  media and is intentionally not exercised by the automated suite — it was
  instead smoke-tested by hand against a fixture clip. `TextClip` uses an
  explicit vendored font (`fonts/Roboto-Regular.ttf`, Apache-2.0) rather than
  relying on system font resolution, per `.cursor/rules/subsystems.mdc`.

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

# Auto-edit your own raw footage to match:
autoedit make my_footage.mp4 -s style_profile.json -o out.mp4
```

`make` works even without `-s` (falls back to a neutral built-in
`StyleProfile`), and always runs with zero AI (the director's LLM is
stubbed — see `autoedit/director/llm.py`).

## Test

```bash
pytest
```

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
    aggregate.py  # VideoFeatures + aggregate(videos) -> StyleProfile (Phase A)
  content/
    extract.py    # extract(path) -> ContentFeatures (Phase B)
  subsystems/     # cutter, text_adder, emoji_adder, effects (+ TOOL_MANIFEST)
  renderer.py      # build_render_plan (pure) + render (the one composite pass)
  director/       # build_brief, stub_llm, validate_plan, heuristic_plan, direct
  executor.py     # build_initial_timeline + execute + run (dispatch -> render once)
  review.py       # check_plan (plan-level sanity check) + gallery HTML output
  cli.py          # learn <videos...> -> StyleProfile, make <video> -> mp4
  models/         # Pydantic v2 contracts
fonts/            # Vendored Roboto-Regular.ttf (Apache-2.0) for TextClip
fixtures/         # Sample JSON for every model, used by tests
  media/          # Tiny real clip/audio fixtures for ingest + extractor tests
  video_features_*.json  # Per-video extractor-output fixtures for aggregate()
tests/            # One test module per model/module: valid load + rejection cases
```
