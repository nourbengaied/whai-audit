"""Run prompts against the Anthropic API with caching."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import anthropic

from .dataset import Prompt


@dataclass
class Response:
    """A single model response to a prompt."""

    prompt_id: str
    model: str
    run_index: int
    text: str
    stop_reason: str | None
    usage: dict[str, int]
    elapsed_s: float
    timestamp: float

    @classmethod
    def from_dict(cls, d: dict) -> Response:
        return cls(**d)


def _cache_key(prompt_id: str, model: str, run_index: int, temperature: float) -> str:
    """Stable cache key including the sampling temperature."""
    blob = f"{prompt_id}|{model}|{run_index}|{temperature}"
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.json"


def run_prompt(
    client: anthropic.Anthropic,
    prompt: Prompt,
    model: str,
    run_index: int,
    temperature: float = 1.0,
    max_tokens: int = 1024,
    cache_dir: Path | None = None,
) -> Response:
    """Run a single prompt, returning cached result if available."""
    key = _cache_key(prompt.id, model, run_index, temperature)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = _cache_path(cache_dir, key)
        if cached.exists():
            return Response.from_dict(json.loads(cached.read_text()))

    _MODELS_WITHOUT_TEMPERATURE = ("claude-opus-4-7", "claude-opus-4-8")

    start = time.time()
    create_kwargs: dict = dict(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt.prompt}],
    )
    if not any(model.startswith(m) for m in _MODELS_WITHOUT_TEMPERATURE):
        create_kwargs["temperature"] = temperature
    msg = client.messages.create(**create_kwargs)
    elapsed = time.time() - start

    # Concatenate text blocks; ignore tool-use blocks (none expected here).
    text = "".join(
        block.text for block in msg.content if getattr(block, "type", None) == "text"
    )

    response = Response(
        prompt_id=prompt.id,
        model=model,
        run_index=run_index,
        text=text,
        stop_reason=msg.stop_reason,
        usage={
            "input_tokens": msg.usage.input_tokens,
            "output_tokens": msg.usage.output_tokens,
        },
        elapsed_s=elapsed,
        timestamp=start,
    )

    if cache_dir is not None:
        _cache_path(cache_dir, key).write_text(json.dumps(asdict(response), indent=2))

    return response


def run_batch(
    prompts: list[Prompt],
    model: str,
    n: int = 5,
    temperature: float = 1.0,
    cache_dir: Path | None = None,
    client: anthropic.Anthropic | None = None,
    max_workers: int = 10,
) -> list[Response]:
    """Run each prompt n times, in parallel."""
    client = client or anthropic.Anthropic()
    tasks = [(prompt, i) for prompt in prompts for i in range(n)]

    results: list[Response] = [None] * len(tasks)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                run_prompt,
                client=client,
                prompt=prompt,
                model=model,
                run_index=i,
                temperature=temperature,
                cache_dir=cache_dir,
            ): idx
            for idx, (prompt, i) in enumerate(tasks)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results
