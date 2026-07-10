"""Audio mixing — combines music-bed excerpts under the video clip.

`_mix_audio` is the only function here; it keeps the video clip's own
dialogue/ambient audio intact (as one layer of a `CompositeAudioClip`) and
mixes external music-bed `AudioInstruction` excerpts in alongside it.
"""

from __future__ import annotations

from typing import Any

from .plan import AudioInstruction


def _mix_audio(clip: Any, audio_instructions: list[AudioInstruction]) -> Any:
    """Mix `audio_instructions` (music-bed excerpts) UNDER `clip`'s own audio, if any.

    A talking-head shot's own dialogue survives (`clip.audio` is kept as one
    of the layers); the music bed(s) just play alongside it at their own
    `volume`. A clip with no embedded audio at all (e.g. silent b-roll) just
    gets the music bed as its only audio.
    """
    from moviepy import AudioFileClip, CompositeAudioClip
    from moviepy.audio.fx import MultiplyVolume

    music_layers = []
    for instruction in audio_instructions:
        excerpt = AudioFileClip(instruction.source).subclipped(instruction.in_, instruction.out_)
        excerpt = excerpt.with_effects([MultiplyVolume(instruction.volume)]).with_start(instruction.start)
        music_layers.append(excerpt)

    layers = ([clip.audio] if clip.audio is not None else []) + music_layers
    if not layers:
        return clip
    mixed = layers[0] if len(layers) == 1 else CompositeAudioClip(layers)
    return clip.with_audio(mixed)
