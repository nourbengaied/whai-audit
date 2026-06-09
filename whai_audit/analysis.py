"""Analysis: aggregation, within-pair deltas, surfacing concerning responses."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset import Prompt


@dataclass
class CategorySummary:
    category: str
    n_responses: int
    mean_accuracy: float
    mean_completeness: float
    mean_actionability: float
    refusal_breakdown: dict[str, int]
    tone_breakdown: dict[str, int]
    pushback_breakdown: dict[str, int]
    caveat_breakdown: dict[str, int]
    concern_breakdown: dict[str, int]


@dataclass
class PairDelta:
    pair_id: str
    member_ids: list[str]
    accuracy_delta: float  # max - min across pair members
    completeness_delta: float
    tone_divergence: bool  # True if any tone label differs across pair


def _load_results(results_dir: Path) -> list[dict[str, Any]]:
    """Load all judged-response JSONs from a results directory."""
    out: list[dict[str, Any]] = []
    for f in sorted(results_dir.glob("*.json")):
        out.append(json.loads(f.read_text()))
    return out


def _counter(values: list[str]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for v in values:
        out[v] += 1
    return dict(out)


def summarise_by_category(
    results: list[dict[str, Any]],
    prompts_by_id: dict[str, Prompt],
) -> list[CategorySummary]:
    """Per-category aggregate statistics."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        pid = r["response"]["prompt_id"]
        cat = prompts_by_id[pid].category
        grouped[cat].append(r)

    out: list[CategorySummary] = []
    for cat, items in sorted(grouped.items()):
        scores = [r["score"] for r in items]
        out.append(
            CategorySummary(
                category=cat,
                n_responses=len(items),
                mean_accuracy=statistics.mean(s["accuracy"] for s in scores),
                mean_completeness=statistics.mean(s["completeness"] for s in scores),
                mean_actionability=statistics.mean(s["actionability"] for s in scores),
                refusal_breakdown=_counter([s["refusal"] for s in scores]),
                tone_breakdown=_counter([s["tone"] for s in scores]),
                pushback_breakdown=_counter([s["pushback"] for s in scores]),
                caveat_breakdown=_counter([s["caveat_load"] for s in scores]),
                concern_breakdown=_counter([s["overall_concern"] for s in scores]),
            )
        )
    return out


def pair_deltas(
    results: list[dict[str, Any]],
    prompts_by_id: dict[str, Prompt],
) -> list[PairDelta]:
    """Within-pair deltas for counterfactual prompt pairs.

    For each pair, we take the mean score per pair member (across N runs),
    then report the spread across members. Large spreads indicate the model
    treating counterfactual variants differently.
    """
    # Group results by prompt_id, then by pair_id.
    per_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        per_prompt[r["response"]["prompt_id"]].append(r)

    pair_to_prompts: dict[str, list[str]] = defaultdict(list)
    for pid, prompt in prompts_by_id.items():
        if prompt.pair_id is not None:
            pair_to_prompts[prompt.pair_id].append(pid)

    out: list[PairDelta] = []
    for pair_id, member_ids in sorted(pair_to_prompts.items()):
        if len(member_ids) < 2:
            continue
        member_means: dict[str, dict[str, float]] = {}
        member_tones: dict[str, set[str]] = {}
        for mid in member_ids:
            items = per_prompt.get(mid, [])
            if not items:
                continue
            scores = [r["score"] for r in items]
            member_means[mid] = {
                "accuracy": statistics.mean(s["accuracy"] for s in scores),
                "completeness": statistics.mean(s["completeness"] for s in scores),
            }
            member_tones[mid] = {s["tone"] for s in scores}
        if len(member_means) < 2:
            continue
        accs = [m["accuracy"] for m in member_means.values()]
        comps = [m["completeness"] for m in member_means.values()]
        all_tones = set().union(*member_tones.values())
        out.append(
            PairDelta(
                pair_id=pair_id,
                member_ids=list(member_means.keys()),
                accuracy_delta=max(accs) - min(accs),
                completeness_delta=max(comps) - min(comps),
                tone_divergence=len(all_tones) > 1,
            )
        )
    return out


def concerning_responses(
    results: list[dict[str, Any]],
    prompts_by_id: dict[str, Prompt],
    accuracy_threshold: int = 2,
) -> list[dict[str, Any]]:
    """Surface individual responses worth human review."""
    out: list[dict[str, Any]] = []
    for r in results:
        score = r["score"]
        prompt = prompts_by_id[r["response"]["prompt_id"]]
        if (
            score["accuracy"] <= accuracy_threshold
            or score["tone"] == "dismissive"
            or score["overall_concern"] == "high"
            or score["refusal"] == "over_refused"
        ):
            out.append(
                {
                    "prompt_id": prompt.id,
                    "category": prompt.category,
                    "prompt": prompt.prompt,
                    "response_text": r["response"]["text"],
                    "score": score,
                }
            )
    return out


def render_report(
    summaries: list[CategorySummary],
    deltas: list[PairDelta],
    concerning: list[dict[str, Any]],
) -> str:
    """Markdown report for a run."""
    lines: list[str] = ["# whai-audit results\n"]

    lines.append("## Per-category summary\n")
    for s in summaries:
        lines.append(f"### {s.category} (n={s.n_responses})")
        lines.append(f"- mean accuracy: **{s.mean_accuracy:.2f}** / 5")
        lines.append(f"- mean completeness: **{s.mean_completeness:.2f}** / 5")
        lines.append(f"- mean actionability: **{s.mean_actionability:.2f}** / 5")
        lines.append(f"- refusal: {s.refusal_breakdown}")
        lines.append(f"- tone: {s.tone_breakdown}")
        lines.append(f"- pushback: {s.pushback_breakdown}")
        lines.append(f"- caveat load: {s.caveat_breakdown}")
        lines.append(f"- overall concern: {s.concern_breakdown}\n")

    if deltas:
        lines.append("## Counterfactual pair deltas\n")
        lines.append("| pair | members | Δ accuracy | Δ completeness | tone diverged |")
        lines.append("|---|---|---|---|---|")
        for d in deltas:
            lines.append(
                f"| `{d.pair_id}` | {', '.join(d.member_ids)} "
                f"| {d.accuracy_delta:.2f} | {d.completeness_delta:.2f} "
                f"| {'yes' if d.tone_divergence else 'no'} |"
            )
        lines.append("")

    if concerning:
        lines.append(f"## Concerning responses ({len(concerning)})\n")
        for c in concerning:
            lines.append(f"### `{c['prompt_id']}` ({c['category']})")
            lines.append(f"**Prompt:** {c['prompt'].strip()}\n")
            lines.append(f"**Response:** {c['response_text'].strip()[:600]}…\n")
            lines.append(f"**Verdict:** {c['score']['summary']}\n")
            lines.append("---\n")

    return "\n".join(lines)


def analyse(
    results_dir: Path,
    prompts_by_id: dict[str, Prompt],
) -> str:
    """End-to-end: load results, compute everything, render markdown."""
    results = _load_results(results_dir)
    summaries = summarise_by_category(results, prompts_by_id)
    deltas = pair_deltas(results, prompts_by_id)
    concerning = concerning_responses(results, prompts_by_id)
    return render_report(summaries, deltas, concerning)
