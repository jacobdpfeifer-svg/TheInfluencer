"""Tests for autoedit/templates/__init__.py and the builtin template JSON files.

Pure fixture-JSON tests (per AGENTS.md's testing rule): loading a `Template`
never touches media or a Timeline.
"""

from __future__ import annotations

import pytest

from autoedit.models.template import Template
from autoedit.templates import TEMPLATE_REGISTRY, list_templates, load_template

_EXPECTED_TEMPLATE_NAMES = {"punchy_beat_montage", "talking_head_with_broll", "quick_montage"}


class TestBuiltinTemplatesLoadAndValidate:
    def test_registry_contains_exactly_the_three_builtin_templates(self) -> None:
        assert set(TEMPLATE_REGISTRY.keys()) == _EXPECTED_TEMPLATE_NAMES

    @pytest.mark.parametrize("name", sorted(_EXPECTED_TEMPLATE_NAMES))
    def test_each_builtin_template_is_a_valid_template_instance(self, name: str) -> None:
        template = TEMPLATE_REGISTRY[name]
        assert isinstance(template, Template)
        assert template.name == name
        assert len(template.slots) >= 1

    def test_punchy_beat_montage_shape(self) -> None:
        template = TEMPLATE_REGISTRY["punchy_beat_montage"]
        assert template.music.required is True
        assert template.music.cut_on == "beat"
        assert [slot.id for slot in template.slots] == ["hook", "b_roll_1", "b_roll_2", "b_roll_3", "payoff"]
        assert template.slots[0].role == "high_energy"
        assert template.slots[0].effect == "zoom_in"
        assert template.slots[0].transition_out == "whip_pan"
        assert template.slots[-1].effect == "zoom_out"
        assert len(template.text_slots) == 1
        assert template.text_slots[0].placeholder == "TITLE"
        assert template.text_slots[0].style == "karaoke"

    def test_talking_head_with_broll_shape(self) -> None:
        template = TEMPLATE_REGISTRY["talking_head_with_broll"]
        assert template.music.required is False
        assert template.music.cut_on == "none"
        roles = [slot.role for slot in template.slots]
        assert roles == ["talking_head", "any", "talking_head"]
        assert {text_slot.placeholder for text_slot in template.text_slots} == {"TITLE", "CTA"}

    def test_quick_montage_shape(self) -> None:
        template = TEMPLATE_REGISTRY["quick_montage"]
        assert template.music.required is True
        assert template.music.cut_on == "beat"
        assert len(template.slots) == 8
        assert all(slot.role == "any" for slot in template.slots)
        assert template.text_slots == []


class TestLoadTemplate:
    def test_returns_the_named_template(self) -> None:
        template = load_template("quick_montage")
        assert template is TEMPLATE_REGISTRY["quick_montage"]

    def test_unknown_name_raises_key_error_listing_available_names(self) -> None:
        with pytest.raises(KeyError, match="quick_montage"):
            load_template("not_a_real_template")


class TestListTemplates:
    def test_returns_sorted_names(self) -> None:
        assert list_templates() == sorted(_EXPECTED_TEMPLATE_NAMES)
