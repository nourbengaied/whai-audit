"""Command-line entry point for whai-audit.

Usage:
    python -m whai_audit.cli run \\
        --prompts-dir prompts \\
        --model claude-opus-4-7 \\
        --n 5 \\
        --category symptom_dismissal \\
        --cache .cache \\
        --results results/run_001

    python -m whai_audit.cli judge \\
        --prompts-dir prompts \\
        --responses-from .cache \\
        --judge-model claude-opus-4-7 \\
        --results results/run_001

    python -m whai_audit.cli analyse \\
        --prompts-dir prompts \\
        --results results/run_001 \\
        --out results/run_001/report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anthropic

from .analysis import analyse
from .dataset import filter_prompts, load_all
from .evaluator import judge_batch
from .runner import Response, run_batch


def cmd_run(args: argparse.Namespace) -> None:
    prompts = load_all(Path(args.prompts_dir))
    prompts = filter_prompts(
        prompts,
        categories=args.category or None,
        ids=args.id or None,
    )
    print(f"[run] {len(prompts)} prompts × {args.n} runs against {args.model}")
    responses = run_batch(
        prompts=prompts,
        model=args.model,
        n=args.n,
        temperature=args.temperature,
        cache_dir=Path(args.cache) if args.cache else None,
        max_workers=args.workers,
    )
    print(f"[run] {len(responses)} responses collected")


def cmd_judge(args: argparse.Namespace) -> None:
    prompts = {p.id: p for p in load_all(Path(args.prompts_dir))}
    cache = Path(args.responses_from)
    responses: list[Response] = []
    for f in cache.glob("*.json"):
        d = json.loads(f.read_text())
        if d.get("prompt_id") in prompts:
            responses.append(Response.from_dict(d))
    print(f"[judge] {len(responses)} responses to score")
    judged = judge_batch(
        prompts_by_id=prompts,
        responses=responses,
        judge_model=args.judge_model,
        results_dir=Path(args.results),
        max_workers=args.workers,
    )
    print(f"[judge] {len(judged)} responses scored → {args.results}")


def cmd_analyse(args: argparse.Namespace) -> None:
    prompts = {p.id: p for p in load_all(Path(args.prompts_dir))}
    report = analyse(Path(args.results), prompts)
    out_path = Path(args.out) if args.out else Path(args.results) / "report.md"
    out_path.write_text(report)
    print(f"[analyse] wrote {out_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="whai-audit")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run prompts through a model")
    run.add_argument("--prompts-dir", default="prompts")
    run.add_argument("--model", default="claude-opus-4-7")
    run.add_argument("--n", type=int, default=5)
    run.add_argument("--temperature", type=float, default=1.0)
    run.add_argument("--category", nargs="*", help="Restrict to categories")
    run.add_argument("--id", nargs="*", help="Restrict to prompt IDs")
    run.add_argument("--cache", default=".cache", help="Response cache dir")
    run.add_argument("--workers", type=int, default=10, help="Parallel API calls")
    run.set_defaults(func=cmd_run)

    judge = sub.add_parser("judge", help="Run the judge on cached responses")
    judge.add_argument("--prompts-dir", default="prompts")
    judge.add_argument("--responses-from", default=".cache")
    judge.add_argument("--judge-model", default="claude-opus-4-7")
    judge.add_argument("--results", required=True)
    judge.add_argument("--workers", type=int, default=10, help="Parallel judge calls")
    judge.set_defaults(func=cmd_judge)

    an = sub.add_parser("analyse", help="Render a markdown report")
    an.add_argument("--prompts-dir", default="prompts")
    an.add_argument("--results", required=True)
    an.add_argument("--out", default=None)
    an.set_defaults(func=cmd_analyse)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
