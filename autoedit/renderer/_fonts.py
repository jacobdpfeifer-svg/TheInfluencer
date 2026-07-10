"""Font-path constants shared across the renderer submodules.

Defined here (rather than in __init__.py) so submodules can import them
directly without a circular dependency.
"""

from pathlib import Path

# `.cursor/rules/subsystems.mdc`: TextClip needs an explicit font-file path —
# never rely on system font resolution (it fails on Linux). This file is
# vendored in-repo (Apache-2.0, see fonts/Roboto-LICENSE.txt).
DEFAULT_FONT_PATH = Path(__file__).resolve().parent.parent.parent / "fonts" / "Roboto-Regular.ttf"

# Roboto has NO emoji glyphs — text drawn from it renders blank for emoji
# codepoints. Emoji overlays need a dedicated emoji-capable font instead;
# this one is vendored in-repo (SIL OFL 1.1, see fonts/NotoColorEmoji-LICENSE.txt).
EMOJI_FONT_PATH = Path(__file__).resolve().parent.parent.parent / "fonts" / "NotoColorEmoji-Regular.ttf"
