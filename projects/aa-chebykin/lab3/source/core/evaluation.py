import asyncio
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import uuid
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from itertools import combinations
from pathlib import Path
from typing import Any

import httpx
import psycopg
from psycopg import sql
from pydantic import BaseModel, Field, model_validator

from core.agent import CustomAgent, TurnResult
from core.config import Settings, refresh_model
from core.pi import PiAgent
from core.tools import session_schema
from core.tracing import Tracer


class TurnSpec(BaseModel):
    input: str
    expected_output: str
    expected_rows: list[list[Any]] | None = None
    state_sql: str | None = None
    answer_groups: list[list[str]] = Field(default_factory=list)
    ordered: bool = False
    history_dependent: bool = False
    expected_tools: list[str] = Field(default_factory=list)
    relevant_documents: list[str] = Field(default_factory=list)
    must_compact: bool = False
    scored: bool = True
    reference_sql: list[str] = Field(default_factory=list)
    allow_extra_columns: bool = False


class TaskSpec(BaseModel):
    id: str
    type: str
    difficulty: str
    turns: list[TurnSpec]

    @model_validator(mode="after")
    def validate_checks(self) -> "TaskSpec":
        if self.type not in {"single", "dialogue"} or not self.turns:
            raise ValueError("Task needs type single/dialogue and at least one turn")

        for turn in self.turns:
            if turn.scored and turn.expected_rows is None and not turn.answer_groups:
                raise ValueError(f"Scored turn in {self.id} needs executable or textual criteria")

        return self


def load_tasks(path: Path) -> list[TaskSpec]:
    tasks = [
        TaskSpec.model_validate(value) for value in json.loads(path.read_text(encoding="utf-8"))
    ]
    identifiers = [task.id for task in tasks]

    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate task IDs")

    return tasks


def rows_match(
    actual: list, expected: list, ordered: bool = False, allow_extra_columns: bool = False
) -> bool:
    if allow_extra_columns and actual and expected and len(actual[0]) > len(expected[0]):
        # Дополнительный id допустим только там, где критерий явно разрешает лишние столбцы.
        return any(
            rows_match([[row[index] for index in indices] for row in actual], expected, ordered)
            for indices in combinations(range(len(actual[0])), len(expected[0]))
        )

    # Decimal и JSON-числа приводим к сопоставимому представлению, дубликаты сохраняем.
    def rows(values: list) -> list[str]:
        return [
            json.dumps(
                [str(cell) if cell is not None else None for cell in row],
                ensure_ascii=False,
                sort_keys=True,
            )
            for row in values
        ]

    left, right = rows(actual), rows(expected)

    return left == right if ordered else Counter(left) == Counter(right)


def score_turn(spec: TurnSpec, result: TurnResult, settings: Settings, session_id: str) -> dict:
    text_ok = all(
        any(word.casefold() in result.answer.casefold() for word in group)
        for group in spec.answer_groups
    )
    rows_ok = True

    if spec.expected_rows is not None and result.status == "ok":
        if spec.state_sql:
            try:
                with psycopg.connect(settings.admin_url) as connection:
                    connection.execute(
                        sql.SQL("SET search_path TO {}, pg_catalog").format(
                            sql.Identifier(session_schema(session_id))
                        )
                    )
                    actual = connection.execute(spec.state_sql).fetchall()

                rows_ok = rows_match(actual, spec.expected_rows, spec.ordered)
            except (psycopg.errors.UndefinedTable, psycopg.errors.UndefinedColumn):
                rows_ok = False
        else:
            outputs = [
                call["result"].get("data", {}).get("rows")
                for call in result.tool_calls
                if call["name"] == "execute_sql" and call["result"].get("status") == "ok"
            ]
            rows_ok = bool(outputs) and rows_match(
                outputs[-1] or [], spec.expected_rows, spec.ordered, spec.allow_extra_columns
            )

    called = {call["name"] for call in result.tool_calls}
    tools_ok = set(spec.expected_tools) <= called
    compacted = result.history_compacted
    success = result.status == "ok" and text_ok and rows_ok and (compacted or not spec.must_compact)
    scores: dict[str, Any] = {
        "task_success": float(success),
        "tool_correctness": float(tools_ok) if spec.expected_tools else None,
        "history_dependent": spec.history_dependent,
        "compacted": compacted,
        "scored": spec.scored,
    }
    scores.update(
        latency_ms=result.latency_ms,
        cost_rub=result.cost_rub,
        steps=result.steps,
        tool_calls=len(result.tool_calls),
        max_context=result.max_context,
    )
    scores.update(
        {
            name: value
            for name, value in result.usage.items()
            if name in {"input_tokens", "output_tokens"}
        }
    )

    ranking = []

    for call in result.tool_calls:
        if call["name"] == "search_docs" and call["result"].get("status") == "ok":
            for chunk in call["result"].get("data", []):
                document = chunk["document_id"]

                if document not in ranking:
                    ranking.append(document)

    if spec.relevant_documents:
        relevant = set(spec.relevant_documents)

        for k in (5, 10):
            found = set(ranking[:k]) & relevant
            scores[f"recall@{k}"] = len(found) / len(relevant)
            scores[f"precision@{k}"] = len(found) / k

        scores["mrr@10"] = next(
            (1 / i for i, doc in enumerate(ranking[:10], 1) if doc in relevant), 0
        )
        sources = set(ranking)
        sources.update(
            call["result"].get("data", {}).get("document_id")
            for call in result.tool_calls
            if call["name"] == "read_doc_page" and call["result"].get("status") == "ok"
        )
        citations = set(re.findall(r"\[(pg17-p\d{4})\]", result.answer))
        scores["citation_validity"] = float(bool(citations) and citations <= sources)
        scores["task_success"] *= scores["citation_validity"]

    if spec.history_dependent:
        scores["history_success"] = scores["task_success"]
        key = "history_after_compaction" if compacted else "history_without_compaction"
        scores[key] = scores["task_success"]

    return scores


def task_metrics(records: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record["task_id"], []).append(record)

    metrics = []
    for task_id, turns in grouped.items():
        first = turns[0]
        scored = [turn for turn in turns if turn["scores"]["scored"]]
        complete = len({turn["turn"] for turn in turns}) == first.get("expected_turns", len(turns))
        tool_errors = sum(
            call["result"].get("status") == "error"
            for turn in turns
            for call in turn["result"].get("tool_calls", [])
        )
        failed = any(turn["result"]["status"] != "ok" for turn in turns)
        limits = sum(
            turn["result"]["status"] == "timeout"
            or (
                turn["result"]["status"] != "ok"
                and any(
                    word in turn["result"].get("answer", "").lower()
                    for word in ("maximum agent", "maximum llm", "maximum pi", "recursion limit")
                )
            )
            for turn in turns
        )
        costs = [turn["result"]["cost_rub"] for turn in turns]
        dependent = [
            turn["scores"]["history_success"]
            for turn in scored
            if "history_success" in turn["scores"]
        ]
        task = {
            "task_id": task_id,
            "session_id": first.get("session_id"),
            "type": first["type"],
            "difficulty": first["difficulty"],
            "complete": complete,
            "completed_turns": len({turn["turn"] for turn in turns}),
            "expected_turns": first.get("expected_turns", len(turns)),
            "task_success": float(
                complete
                and bool(scored)
                and all(turn["scores"]["task_success"] for turn in scored)
                and not failed
            ),
            "has_error": bool(failed or tool_errors),
            "terminal_error": failed,
            "has_limit": bool(limits),
            "tool_errors": tool_errors,
            "cost_rub": sum(costs) if all(cost is not None for cost in costs) else None,
            "latency_ms": sum(turn["result"]["latency_ms"] for turn in turns),
            "history_success": statistics.mean(dependent) if dependent else None,
            "trace_ids": [
                turn["result"].get("trace_id") for turn in turns if turn["result"].get("trace_id")
            ],
        }
        for name in ("input_tokens", "output_tokens"):
            values = [turn["result"]["usage"].get(name) for turn in turns]
            task[name] = (
                sum(values)
                if all(value is not None for value in values)
                and not any(turn["result"]["usage"].get("partial") for turn in turns)
                else None
            )
        for name in (
            "history_after_compaction",
            "history_without_compaction",
            "tool_correctness",
            "recall@5",
            "precision@5",
            "mrr@10",
        ):
            values = [
                turn["scores"][name] for turn in scored if turn["scores"].get(name) is not None
            ]
            task[name] = statistics.mean(values) if values else None
        task["tool_calls"] = sum(len(turn["result"].get("tool_calls", [])) for turn in turns)
        for name in ("steps", "max_context"):
            values = [turn["result"].get(name) for turn in turns]
            task[name] = (
                (sum(values) if name == "steps" else max(values))
                if all(value is not None for value in values)
                else None
            )
        metrics.append(task)
    return metrics


def aggregate(records: list[dict]) -> dict:
    scored = [r for r in records if r["scores"]["scored"]]

    def average(values: list) -> float | None:
        available = [v for v in values if v is not None]

        return statistics.mean(available) if available else None

    tasks: dict[str, list] = {}

    for record in records:
        tasks.setdefault(record["task_id"], []).append(record)

    per_task = task_metrics(records)
    result = {
        "tasks": len(tasks),
        "scored_turns": len(scored),
        "task_success_rate": average([task["task_success"] for task in per_task]),
        "error_task_rate": average([task["has_error"] for task in per_task]),
        "terminal_error_task_rate": average([task["terminal_error"] for task in per_task]),
        "limit_task_rate": average([task["has_limit"] for task in per_task]),
        "error_or_limit_task_rate": average(
            [task["has_error"] or task["has_limit"] for task in per_task]
        ),
        "incomplete_tasks": sum(not task["complete"] for task in per_task),
        "turn_success_rate": average([r["scores"]["task_success"] for r in scored]),
        "mean_cost_per_task_rub": sum(r["result"]["cost_rub"] or 0 for r in records) / len(tasks)
        if tasks and all(r["result"]["cost_rub"] is not None for r in records)
        else None,
        "mean_latency_per_task_ms": sum(r["result"]["latency_ms"] for r in records) / len(tasks)
        if tasks
        else None,
        "error_turn_rate": average([r["result"]["status"] != "ok" for r in scored]),
        "mean_input_tokens": average([r["result"]["usage"].get("input_tokens") for r in records]),
        "mean_output_tokens": average([r["result"]["usage"].get("output_tokens") for r in records]),
        "history_success": average([r["scores"].get("history_success") for r in scored]),
        "history_after_compaction": average(
            [r["scores"].get("history_after_compaction") for r in scored]
        ),
        "history_without_compaction": average(
            [r["scores"].get("history_without_compaction") for r in scored]
        ),
    }

    for metric in (
        "tool_correctness",
        "citation_validity",
        "recall@5",
        "precision@5",
        "recall@10",
        "precision@10",
        "mrr@10",
        "steps",
        "tool_calls",
        "max_context",
    ):
        result["mean_" + metric] = average([r["scores"].get(metric) for r in scored])

    return result


def manifest(settings: Settings, path: Path, profile: str) -> dict:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=settings.root, check=False
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        cwd=settings.root,
        check=False,
    ).stdout.strip()
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "profile": profile,
        "git_revision": revision,
        "dirty": bool(dirty),
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_sha256": {
            str(file.relative_to(settings.root)): hashlib.sha256(file.read_bytes()).hexdigest()
            for file in sorted(
                [
                    *settings.root.glob("source/core/*.py"),
                    settings.root / "pi/extension.ts",
                    settings.root / "resources/prompts.toml",
                    settings.root / "resources/database.sql",
                ]
            )
        },
        "config": settings.model_dump(mode="json"),
        "python_lock_sha256": hashlib.sha256(
            (settings.root / "source/uv.lock").read_bytes()
        ).hexdigest(),
        "currency": "RUB",
        "model": settings.model.id,
        "run_kind": os.getenv("LAB3_RUN_KIND", "evaluation"),
        "versions": {
            name: version(name)
            for name in (
                "langgraph",
                "openai",
                "mcp",
                "langfuse",
                "psycopg",
                "qdrant-client",
            )
        },
    }

    for name, artifact in {
        "index": settings.data / "corpus/05_index/index.json",
        "pi_lock": settings.root / "pi/package-lock.json",
    }.items():
        result[name + "_sha256"] = (
            hashlib.sha256(artifact.read_bytes()).hexdigest() if artifact.exists() else None
        )

    return result


def _save_turn(output: Path, record: dict, tracer: Tracer) -> None:
    # Каждая готовая реплика сразу сохраняется, не ожидая конца эксперимента.
    with (output / "turns.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    trace_id = record["result"]["trace_id"]
    scores = record["scores"]
    with (output / "scores.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "task_id": record["task_id"],
                    "turn": record["turn"],
                    "trace_id": trace_id,
                    **scores,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    if scores["scored"]:
        for name, value in scores.items():
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                tracer.score(trace_id, name, float(value))


def _budget_exhausted(
    spent: float,
    budget: float,
    reserve: float,
    tracer: Tracer,
    task: str,
    turn: int | None = None,
) -> bool:
    if spent + reserve <= budget:
        return False
    tracer.event("budget_stop", spent_rub=spent, budget_rub=budget, task=task, turn=turn)
    return True


def _finalize_run(
    output: Path,
    records: list[dict],
    tasks: list[TaskSpec],
    spent: float,
    profile: str,
    tracer: Tracer,
) -> dict:
    run_id = output.name
    per_task = task_metrics(records)
    (output / "tasks.json").write_text(
        json.dumps(per_task, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for task in per_task:
        with tracer.span(
            "task_result",
            session_id=task["session_id"],
            run_id=run_id,
            profile=profile,
            input={
                "task_id": task["task_id"],
                "type": task["type"],
                "difficulty": task["difficulty"],
                "turn_trace_ids": task["trace_ids"],
            },
        ) as observation:
            if observation:
                observation.update(output=task)
                observation.update_trace(tags=[profile, "task-result", run_id])
            for name, value in task.items():
                if isinstance(value, (float, int)):
                    tracer.score(observation.trace_id if observation else None, name, float(value))

    summary = aggregate(records)
    summary["spent_rub"] = (
        sum(r["result"]["cost_rub"] for r in records)
        if all(r["result"]["cost_rub"] is not None for r in records)
        else None
    )
    summary["budget_accounted_rub"] = spent
    summary["requested_tasks"] = len(tasks)
    summary["not_started_task_ids"] = [
        task.id for task in tasks if task.id not in {r["task_id"] for r in records}
    ]
    summary["complete"] = all(
        {r["turn"] for r in records if r["task_id"] == task.id} == set(range(len(task.turns)))
        for task in tasks
    )
    summary["by_type"] = {
        kind: aggregate([r for r in records if r["type"] == kind])
        for kind in {r["type"] for r in records}
    }
    summary["by_difficulty"] = {
        kind: aggregate([r for r in records if r["difficulty"] == kind])
        for kind in {r["difficulty"] for r in records}
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = [
        f"# Прогон {run_id}",
        "",
        f"Профиль: {profile}.",
        "",
        "| Метрика | Значение |",
        "|---|---|",
    ]
    report.extend(
        f"| {name} | {value} |" for name, value in summary.items() if not isinstance(value, dict)
    )
    report.extend(["", "## Неудачные проверяемые реплики", ""])

    for record in records:
        if record["scores"]["scored"] and not record["scores"]["task_success"]:
            trace = record["result"]["trace_id"]
            link = (
                f"http://localhost:3000/project/postgres-agents/traces/{trace}"
                if trace
                else "нет trace"
            )
            report.append(f"- {record['task_id']}, реплика {record['turn']}: {link}")

    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    with tracer.span(
        "dataset_run", run_id=run_id, profile=profile, input={"tasks": len(tasks)}
    ) as span:
        if span:
            span.update(output=summary)
            span.update_trace(tags=[profile, "dataset-run", run_id])

            for name, value in summary.items():
                if isinstance(value, (float, int)):
                    tracer.score(span.trace_id, name, float(value))

    return summary


async def run_dataset(
    settings: Settings,
    profile: str,
    task_ids: list[str] | None = None,
    max_cost: float | None = None,
    label: str = "benchmark",
) -> Path:
    if not os.getenv("POLZA_API_KEY"):
        raise ValueError("Set POLZA_API_KEY in lab3/.env before starting a paid evaluation")

    if profile not in settings.evaluation["profiles"]:
        raise ValueError(f"Unknown profile: {profile}")

    if label not in {"pilot", "benchmark", "stress"}:
        raise ValueError("label must be pilot, benchmark or stress")

    budget = max_cost if max_cost is not None else settings.evaluation["max_cost_rub"]
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("Evaluation budget must be finite and positive")

    if settings.model.base_url.startswith("https://polza.ai/"):
        await asyncio.to_thread(refresh_model, settings)

    dataset = settings.root / "resources/tasks.json"
    tasks = load_tasks(dataset)

    if task_ids:
        unknown = set(task_ids) - {task.id for task in tasks}

        if unknown:
            raise ValueError(f"Unknown task IDs: {unknown}")

        tasks = [task for task in tasks if task.id in task_ids]

    run_id = (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + profile + "-" + uuid.uuid4().hex[:6]
    )
    output = settings.root / "results" / "08_experiments" / label / run_id
    output.mkdir(parents=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest(settings, dataset, profile), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "dataset.json").write_text(
        json.dumps([task.model_dump() for task in tasks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tracer = Tracer(output, run_kind="evaluation")
    # Артефакты прогона пишутся постепенно: прерывание не уничтожает готовые реплики.
    records = []
    spent = 0.0
    # Резерв на реплику учитывает максимальные запросы и разрешённые повторы API.
    turn_reserve = (
        settings.agent["max_llm_calls"]
        * (settings.agent["api_max_retries"] + 1)
        * (
            settings.context.max_input_tokens * settings.model.input_price
            + settings.model.max_output_tokens * settings.model.output_price
        )
        / 1e6
    )
    headers = {"Authorization": f"Bearer {os.getenv('LAB3_TOOL_TOKEN', '')}"}

    try:
        async with httpx.AsyncClient(
            base_url=settings.tool_url, headers=headers, timeout=60
        ) as http:
            warmup = await http.post(
                "/tools/search_docs",
                headers={"x-lab3-session": run_id + ":warmup"},
                json={"arguments": {"query": "PostgreSQL", "limit": 1}},
            )
            warmup.raise_for_status()

            if warmup.json().get("status") != "ok":
                raise RuntimeError(f"Retrieval preflight failed: {warmup.json().get('error')}")

            for task in tasks:
                session_id = run_id + ":" + task.id
                if _budget_exhausted(spent, budget, turn_reserve, tracer, task.id):
                    break

                response = await http.post(f"/sessions/{session_id}/reset")
                response.raise_for_status()
                agent = (
                    PiAgent(settings, session_id, tracer)
                    if profile == "pi"
                    else CustomAgent(
                        settings,
                        session_id,
                        tracer,
                        strategy="trim" if profile == "custom-trim" else "tiered",
                    )
                )

                index = 0
                result = None
                try:
                    await agent.start()

                    for index, turn in enumerate(task.turns):
                        if _budget_exhausted(spent, budget, turn_reserve, tracer, task.id, index):
                            break

                        result = None
                        result = await agent.respond(turn.input)
                        spent += result.cost_rub if result.cost_rub is not None else turn_reserve
                        scores = await asyncio.to_thread(
                            score_turn, turn, result, settings, session_id
                        )
                        record = {
                            "task_id": task.id,
                            "turn": index,
                            "expected_turns": len(task.turns),
                            "session_id": session_id,
                            "type": task.type,
                            "difficulty": task.difficulty,
                            "input": turn.input,
                            "expected_output": turn.expected_output,
                            "criteria": turn.model_dump(exclude={"input"}),
                            "result": asdict(result),
                            "scores": scores,
                        }
                        records.append(record)

                        _save_turn(output, record, tracer)
                except Exception as error:
                    # Ошибка запуска или grader не должна исчезнуть из итогов задачи.
                    if not any(r["task_id"] == task.id and r["turn"] == index for r in records):
                        turn = task.turns[index]
                        if result is None:
                            result = TurnResult(
                                str(error), "startup_or_runner_error", cost_rub=None
                            )
                            spent += turn_reserve
                        else:
                            result.status = "evaluation_error"
                            result.answer += f"\nEvaluation failed: {error}"

                        with tracer.span(
                            "task_error", session_id=session_id, input=turn.input, error=str(error)
                        ) as observation:
                            if observation:
                                observation.update(
                                    level="ERROR", status_message=str(error), output=result.answer
                                )
                                result.trace_id = observation.trace_id
                        record = {
                            "task_id": task.id,
                            "turn": index,
                            "expected_turns": len(task.turns),
                            "session_id": session_id,
                            "type": task.type,
                            "difficulty": task.difficulty,
                            "input": turn.input,
                            "expected_output": turn.expected_output,
                            "criteria": turn.model_dump(exclude={"input"}),
                            "result": asdict(result),
                            "scores": score_turn(turn, result, settings, session_id),
                        }
                        records.append(record)
                        _save_turn(output, record, tracer)
                finally:
                    await agent.close()
    finally:
        try:
            _finalize_run(output, records, tasks, spent, profile, tracer)
        finally:
            tracer.flush()

    return output


def compare_runs(runs: list[Path], output: Path) -> dict:
    if len(runs) != 3 or len({path.resolve() for path in runs}) != 3:
        raise ValueError("Pass three distinct runs: custom-trim, custom-tiered and pi")

    loaded = []
    for path in runs:
        metadata = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        records = [
            json.loads(line)
            for line in (path / "turns.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        loaded.append((path, metadata, summary, records))

    profiles = {item[1]["profile"] for item in loaded}
    if profiles != {"custom-trim", "custom-tiered", "pi"}:
        raise ValueError("Comparison needs one run of each profile")
    if any(
        output.resolve() == item[0].resolve()
        or output.resolve() in item[0].resolve().parents
        or item[0].resolve() in output.resolve().parents
        for item in loaded
    ):
        raise ValueError("Report output must be separate from source runs")

    # Сравниваем условия, но не стратегии контекста, которые и являются предметом опыта.
    signatures = []
    for _, metadata, _, records in loaded:
        config = metadata["config"]
        signatures.append(
            {
                "dataset": metadata["dataset_sha256"],
                "model": config["model"],
                "context": {
                    key: value for key, value in config["context"].items() if key != "strategy"
                },
                "source": metadata.get("source_sha256"),
                "agent": config["agent"],
                "python_lock": metadata.get("python_lock_sha256"),
                "pi_lock": metadata.get("pi_lock_sha256"),
                "tools": config["tools"],
                "retrieval": config["retrieval"],
                "tasks": sorted({record["task_id"] for record in records}),
            }
        )
    comparable = all(value == signatures[0] for value in signatures[1:])
    complete = all(item[2]["complete"] for item in loaded)
    task_count = len(signatures[0]["tasks"])
    trace_ids_present = all(
        record["result"].get("trace_id") for _, _, _, records in loaded for record in records
    )
    result = {
        "comparable": comparable,
        "complete": complete,
        "tasks_per_run": task_count,
        "meets_dataset_size": 15 <= task_count <= 30,
        "trace_ids_present": trace_ids_present,
        "runs": {
            metadata["profile"]: {"path": str(path.resolve()), "summary": summary}
            for path, metadata, summary, _ in loaded
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = [
        "# Сравнение агентов",
        "",
        f"Одинаковые условия: {comparable}. Полные прогоны: {complete}.",
        "",
    ]
    if not comparable or not complete or not result["meets_dataset_size"] or not trace_ids_present:
        report += [
            "Сравнение предварительное. Для итогового отчёта нужны одинаковые полные прогоны "
            "на 15–30 задачах и идентификаторы traces всех реплик.",
            "",
        ]

    metrics = (
        "task_success_rate",
        "history_success",
        "history_after_compaction",
        "history_without_compaction",
        "mean_cost_per_task_rub",
        "mean_latency_per_task_ms",
        "error_task_rate",
        "terminal_error_task_rate",
        "limit_task_rate",
        "mean_steps",
        "mean_tool_calls",
        "mean_max_context",
        "mean_recall@5",
        "mean_precision@5",
        "mean_mrr@10",
    )
    report += ["| Профиль | Срез | Метрика | Значение |", "|---|---|---|---|"]
    for _, metadata, summary, _ in loaded:
        slices = {"все": summary}
        for group in ("by_type", "by_difficulty"):
            slices.update({f"{group}/{key}": value for key, value in summary[group].items()})
        for label, values in slices.items():
            report.extend(
                f"| {metadata['profile']} | {label} | {name} | {values.get(name)} |"
                for name in metrics
            )

    # Кандидаты для ручного разбора не объявляются доказанными причинами ошибок.
    analysis = [
        "# Разбор ошибок",
        "",
        "Категории ниже являются предварительными. Подтвердите причину по trace и ответу.",
        "",
        "Возможные категории: потеря истории, retrieval/цитирование, SQL/инструмент, "
        "лимит или API, несовпадение с критерием проверки.",
        "",
    ]
    cases = []
    for _, metadata, _, records in loaded:
        profile = metadata["profile"]
        failures = [
            record
            for record in records
            if record["scores"]["scored"] and not record["scores"]["task_success"]
        ]
        failed_tasks = {record["task_id"] for record in failures}
        analysis += [f"## {profile}", "", f"Неудачных задач: {len(failed_tasks)}.", ""]
        if len(failed_tasks) < 5:
            analysis += [
                "Меньше пяти неудачных задач. Для требуемого разбора нужны дополнительные "
                "согласованные стресс-сценарии либо явное объяснение ограничения.",
                "",
            ]
        for record in failures:
            scores, answer = record["scores"], record["result"]
            category = (
                "лимит или API"
                if answer["status"] != "ok"
                else "потеря истории"
                if scores.get("history_dependent")
                else "retrieval/цитирование"
                if scores.get("citation_validity") == 0
                else "SQL/инструмент или критерий проверки"
            )
            trace_id = answer.get("trace_id")
            base = os.getenv("LANGFUSE_BASE_URL", "http://localhost:3000").rstrip("/")
            project = metadata["config"]["evaluation"].get("langfuse_project_id", "postgres-agents")
            link = f"{base}/project/{project}/traces/{trace_id}" if trace_id else "нет trace"
            case = {
                "profile": profile,
                "task_id": record["task_id"],
                "turn": record["turn"],
                "suggested_category": category,
                "confirmed_category": None,
                "trace_url": link,
                "record": record,
            }
            cases.append(case)
            analysis += [
                f"### {record['task_id']}, реплика {record['turn']}",
                "",
                f"Предварительная категория: {category}.",
                "",
                f"Trace: {link}",
                "",
                f"Запрос: {record['input']}",
                "",
                f"Критерий: {record['expected_output']}",
                "",
                f"Ответ: {answer['answer']}",
                "",
                "Подтверждённая причина и скриншот: заполнить после просмотра trace.",
                "",
            ]

    report += [
        "",
        "## Выводы по измерениям",
        "",
        "По таблицам сравните качество, цену и время, затем проверьте причины по traces. "
        "Разница между trim и tiered характеризует стратегию целиком, "
        "а не изолированный вклад каждого уровня.",
        "",
        "Автоматические оценки не заменяют проверку SQL-ответов и причин ошибок.",
    ]
    for filename, lines in (("report.md", report), ("error_analysis.md", analysis)):
        (output / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "error_cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
