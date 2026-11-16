"""Structured, local activity instrumentation for workflow, tool, and memory events.

Tool and memory adapters should call these helpers around their real operations.
The logger never invents events: an event is written only when a caller invokes it.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

DEFAULT_ACTIVITY_PATH = Path(__file__).resolve().parents[1] / "data" / "activity_events.jsonl"


def _safe_summary(value: Any, limit: int = 1200) -> str:
    """Create a bounded display summary; avoid persisting arbitrary objects."""
    try:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = str(value)
    return text[:limit]


def log_activity(
    activity_type: str,
    operation: str,
    *,
    status: str = "success",
    name: str = "",
    duration_ms: float | None = None,
    input_summary: Any = None,
    output_summary: Any = None,
    error: Any = None,
    run_id: str = "",
    metadata: dict | None = None,
    path: str | Path | None = None,
) -> dict:
    """Append one real activity event to JSONL and return the saved record."""
    if activity_type not in {"workflow", "tool", "memory"}:
        raise ValueError("activity_type must be workflow, tool, or memory")
    record = {
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "activity_type": activity_type,
        "operation": str(operation),
        "name": str(name or operation),
        "status": str(status),
        "duration_ms": round(float(duration_ms), 3) if duration_ms is not None else None,
        "input_summary": _safe_summary(input_summary) if input_summary is not None else "",
        "output_summary": _safe_summary(output_summary) if output_summary is not None else "",
        "error": _safe_summary(error, 800) if error else "",
        "run_id": str(run_id or ""),
        "metadata": metadata or {},
    }
    destination = Path(path or os.getenv("EVOMONITOR_ACTIVITY_LOG", str(DEFAULT_ACTIVITY_PATH)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return record


def load_activity_events(path: str | Path | None = None) -> list[dict]:
    """Read valid activity records; skip incomplete trailing/corrupt lines."""
    source = Path(path or os.getenv("EVOMONITOR_ACTIVITY_LOG", str(DEFAULT_ACTIVITY_PATH)))
    if not source.exists():
        return []
    events = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    events.append(row)
            except json.JSONDecodeError:
                continue
    return events


@contextmanager
def track_activity(
    activity_type: str,
    operation: str,
    *,
    name: str = "",
    input_summary: Any = None,
    run_id: str = "",
    metadata: dict | None = None,
    path: str | Path | None = None,
) -> Iterator[dict]:
    """Time a real operation and log success/failure when its context exits."""
    context = {"output": None}
    started = time.perf_counter()
    try:
        yield context
    except Exception as exc:
        log_activity(
            activity_type, operation, status="error", name=name,
            duration_ms=(time.perf_counter() - started) * 1000,
            input_summary=input_summary, error=exc, run_id=run_id,
            metadata=metadata, path=path,
        )
        raise
    else:
        log_activity(
            activity_type, operation, status="success", name=name,
            duration_ms=(time.perf_counter() - started) * 1000,
            input_summary=input_summary, output_summary=context.get("output"),
            run_id=run_id, metadata=metadata, path=path,
        )


def log_tool_activity(operation: str, **kwargs) -> dict:
    """Call immediately after/beside a real tool operation."""
    return log_activity("tool", operation, **kwargs)


def log_memory_activity(operation: str, **kwargs) -> dict:
    """Call immediately after/beside a real memory read/write/retrieval."""
    return log_activity("memory", operation, **kwargs)



def execute_tracked_tool(tool_callable, *, name: str, inputs: Any = None, run_id: str = "", metadata: dict | None = None):
    """Execute a real tool callable and record its actual result or exception."""
    with track_activity(
        "tool", "invoke", name=name, input_summary=inputs,
        run_id=run_id, metadata=metadata,
    ) as event:
        result = tool_callable()
        event["output"] = result
    return result


def execute_tracked_memory(memory_callable, *, operation: str, name: str, inputs: Any = None, run_id: str = "", metadata: dict | None = None):
    """Execute a real memory read/write/retrieval callable and record its result."""
    if operation not in {"read", "write", "retrieve", "delete", "search"}:
        raise ValueError("Unsupported memory operation")
    with track_activity(
        "memory", operation, name=name, input_summary=inputs,
        run_id=run_id, metadata=metadata,
    ) as event:
        result = memory_callable()
        event["output"] = result
    return result
