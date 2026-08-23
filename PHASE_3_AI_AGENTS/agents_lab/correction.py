"""Self-correction, auto-retry, and structured-output repair (notebook 11).

Agents fail in small, recoverable ways: a tool errors, JSON comes back malformed,
a schema field is missing. This module handles those without giving up:

- ``with_retry`` — re-run a flaky callable a few times (generic).
- ``repair_json`` — when the model's JSON won't parse or is missing fields, ask
  it to fix its own output against the schema.
- ``self_correct`` — validate an answer with a checker; if it fails, feed the
  error back and let the model try again.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable

from study_buddy import get_provider


def with_retry(
    func: Callable[[], Any],
    max_attempts: int = 3,
    delay_s: float = 0.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Any:
    """Call ``func`` up to ``max_attempts`` times, re-raising the last error."""
    last: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except exceptions as exc:
            last = exc
            if attempt < max_attempts and delay_s:
                time.sleep(delay_s)
    raise last  # type: ignore[misc]


def _loads(text: str) -> Any:
    m = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
    return json.loads(m.group() if m else text)


def repair_json(
    text: str,
    schema: dict[str, Any] | None = None,
    provider: str = "ollama",
    model: str | None = None,
    tracker: Any | None = None,
) -> Any:
    """Parse JSON, asking the model to repair it if the first parse fails."""
    try:
        return _loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    instruction = "Fix this into valid JSON. Reply with ONLY the JSON."
    if schema:
        instruction += f" It must match this JSON schema:\n{json.dumps(schema)}"
    resp = get_provider(provider).chat(
        [{"role": "system", "content": instruction}, {"role": "user", "content": text}],
        model=model,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    if tracker is not None:
        tracker.add(resp)
    return _loads(resp.text)


def self_correct(
    task: str,
    validator: Callable[[str], "tuple[bool, str]"],
    provider: str = "ollama",
    model: str | None = None,
    max_attempts: int = 3,
    tracker: Any | None = None,
) -> str:
    """Answer, validate, and retry with the error as feedback until it passes."""
    llm = get_provider(provider)
    answer = ""
    feedback = ""
    for _ in range(max_attempts):
        prompt = task if not feedback else f"{task}\n\nYour previous answer failed: {feedback}\nFix it."
        resp = llm.chat(
            [{"role": "system", "content": "Answer the task correctly."}, {"role": "user", "content": prompt}],
            model=model,
            temperature=0.0,
        )
        if tracker is not None:
            tracker.add(resp)
        answer = resp.text
        ok, feedback = validator(answer)
        if ok:
            break
    return answer
