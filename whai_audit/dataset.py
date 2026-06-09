"""Prompt dataset loading and schema."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

import yaml
from pydantic import BaseModel, Field


Category = Literal[
    "symptom_dismissal",
    "over_refusal",
    "diagnostic_latency",
    "wellness_misinformation",
    "underserved_populations",
    "indirect_framing",
]


class Prompt(BaseModel):
    """A single audit prompt."""

    id: str
    category: Category
    prompt: str
    expected_behaviour: str
    red_flags: list[str] = Field(default_factory=list)
    pair_id: str | None = None
    references: list[str] = Field(default_factory=list)


def load_category(path: Path) -> list[Prompt]:
    """Load prompts from a single YAML file."""
    with path.open() as f:
        raw = yaml.safe_load(f)
    return [Prompt.model_validate(item) for item in raw]


def load_all(prompts_dir: Path) -> list[Prompt]:
    """Load every prompt YAML in the directory."""
    prompts: list[Prompt] = []
    for yaml_file in sorted(prompts_dir.glob("*.yaml")):
        prompts.extend(load_category(yaml_file))
    return prompts


def filter_prompts(
    prompts: Iterable[Prompt],
    categories: list[str] | None = None,
    ids: list[str] | None = None,
) -> list[Prompt]:
    """Subset prompts by category or id."""
    result = list(prompts)
    if categories:
        result = [p for p in result if p.category in categories]
    if ids:
        result = [p for p in result if p.id in ids]
    return result


def pairs(prompts: Iterable[Prompt]) -> dict[str, list[Prompt]]:
    """Group prompts by pair_id (counterfactual pairs)."""
    out: dict[str, list[Prompt]] = {}
    for p in prompts:
        if p.pair_id is None:
            continue
        out.setdefault(p.pair_id, []).append(p)
    return out
