"""Pilot run: small smoke test of the full pipeline.

Run with:
    ANTHROPIC_API_KEY=sk-... python examples/run_pilot.py
"""

from pathlib import Path

import anthropic

from whai_audit.analysis import analyse
from whai_audit.dataset import filter_prompts, load_all
from whai_audit.evaluator import judge_batch
from whai_audit.runner import run_batch


PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
CACHE_DIR = Path(__file__).parent.parent / ".cache"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "pilot"

# Pilot config: small to keep cost low.
MODEL = "claude-opus-4-7"
JUDGE = "claude-opus-4-7"
N_RUNS = 3
PILOT_IDS = [
    "sd_001_m",
    "sd_001_f",  # the MI counterfactual pair
    "or_001",  # mechanism of EC
    "wm_002",  # birth-control-causes-infertility myth
    "dl_003",  # adenomyosis presentation
    "if_001",  # vaginismus framed obliquely
]


def main() -> None:
    client = anthropic.Anthropic()
    prompts = filter_prompts(load_all(PROMPTS_DIR), ids=PILOT_IDS)
    prompts_by_id = {p.id: p for p in prompts}

    print(f"[pilot] {len(prompts)} prompts × {N_RUNS} runs against {MODEL}")
    responses = run_batch(
        prompts=prompts,
        model=MODEL,
        n=N_RUNS,
        temperature=1.0,
        cache_dir=CACHE_DIR,
        client=client,
    )

    print(f"[pilot] judging {len(responses)} responses with {JUDGE}")
    judge_batch(
        prompts_by_id=prompts_by_id,
        responses=responses,
        judge_model=JUDGE,
        results_dir=RESULTS_DIR,
    )

    report = analyse(RESULTS_DIR, prompts_by_id)
    out = RESULTS_DIR / "report.md"
    out.write_text(report)
    print(f"[pilot] wrote {out}")


if __name__ == "__main__":
    main()
