import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Any

from langfuse import Langfuse


class Tracer:
    def __init__(
        self, directory: Path, enabled: bool | None = None, run_kind: str = "interactive"
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "events.jsonl"
        self.run_kind = os.getenv("LAB3_RUN_KIND", run_kind)
        self.version = version("lab3-postgres-agent")
        active = os.getenv("LANGFUSE_ENABLED", "false").lower() == "true"
        self.client = Langfuse() if (active if enabled is None else enabled) else None

    def event(self, name: str, **values: Any) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {"name": name, "timestamp": time.time(), **values},
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )

    @contextmanager
    def span(self, name: str, kind: str = "span", **values: Any) -> Iterator[Any]:
        started = time.perf_counter()
        self.event(name, phase="start", **values)

        if not self.client:
            try:
                yield None
            except BaseException as error:
                self.event(name, phase="error", error=str(error))
                raise
            finally:
                self.event(name, phase="end", duration_ms=(time.perf_counter() - started) * 1000)

            return

        with self.client.start_as_current_observation(
            name=name,
            as_type=kind,
            input=values.get("input"),
            model=values.get("model"),
            model_parameters=values.get("model_parameters"),
            metadata={k: v for k, v in values.items() if k != "input"},
            trace_context=(
                {"trace_id": values["trace_id"], "parent_span_id": values["parent_span_id"]}
                if values.get("trace_id") and values.get("parent_span_id")
                else None
            ),
        ) as observation:
            if values.get("session_id"):
                observation.update_trace(session_id=values["session_id"])

            try:
                yield observation
            except BaseException as error:
                observation.update(level="ERROR", status_message=str(error))
                raise
            finally:
                self.event(name, phase="end", duration_ms=(time.perf_counter() - started) * 1000)

    def score(self, trace_id: str | None, name: str, value: float) -> None:
        self.event("score", trace_id=trace_id, metric=name, value=value)

        if self.client and trace_id:
            self.client.create_score(trace_id=trace_id, name=name, value=value)

    def flush(self) -> None:
        if self.client:
            self.client.flush()
