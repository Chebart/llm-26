import asyncio
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import httpx
import typer
import uvicorn

from core.agent import CustomAgent
from core.config import ROOT, Settings, load_settings, refresh_model
from core.evaluation import compare_runs, load_tasks, rows_match, run_dataset
from core.pi import PiAgent
from core.retrieval import Retriever
from core.tools import ToolService, create_tool_app
from core.tracing import Tracer

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
ConfigOption = Annotated[Path | None, typer.Option("--config", help="Path to config.toml")]


def show(value: object) -> None:
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2, default=str))


@app.command("init-env")
def init_env() -> None:
    path = ROOT / ".env"

    if path.exists():
        raise typer.BadParameter(
            ".env already exists; edit it explicitly, it will not be overwritten"
        )

    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    text = text.replace("replace-with-local-random-token", secrets.token_urlsafe(32))
    path.write_text(text, encoding="utf-8")
    typer.echo(f"Created {path}; set POLZA_API_KEY before paid model calls")


@app.command()
def doctor(config: ConfigOption = None) -> None:
    settings = load_settings(config)
    node = os.getenv("LAB3_NODE") or shutil.which("node")
    if not node and os.name == "nt":
        node = str(settings.runtime / "node/node.exe")
    show(
        {
            "python": sys.version,
            "root": settings.root,
            "node": node,
            "docker_cli": shutil.which("docker"),
            "polza_key_present": bool(os.getenv("POLZA_API_KEY")),
            "database_url_present": bool(settings.database_url),
            "langfuse_enabled": os.getenv("LANGFUSE_ENABLED", "false"),
            "model": settings.model.id,
            "counting": "model chat template estimate",
            "tokenizer": settings.context.tokenizer,
            "tokenizer_revision": settings.context.tokenizer_revision,
            "estimate_margin": settings.context.estimate_margin,
            "input_budget": settings.context.max_input_tokens,
            "output_reserve": settings.model.max_output_tokens,
        }
    )


@app.command()
def models(config: ConfigOption = None) -> None:
    settings = load_settings(config)
    response = httpx.get(settings.model.base_url + "/models", timeout=30)
    response.raise_for_status()
    catalog = response.json()
    settings.checks.mkdir(parents=True, exist_ok=True)
    (settings.checks / "model-catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False), encoding="utf-8"
    )
    show(
        [
            m
            for m in catalog.get("data", [])
            if m["id"]
            in {
                "z-ai/glm-5.3-flash",
                "deepseek/deepseek-v4.1-flash",
            }
        ]
    )


@app.command()
def prepare(rebuild: bool = False, config: ConfigOption = None) -> None:
    show(Retriever(load_settings(config)).prepare(rebuild))


@app.command()
def index(reset: bool = False, config: ConfigOption = None) -> None:
    show(Retriever(load_settings(config)).build(reset))


@app.command("serve-tools")
def serve_tools(config: ConfigOption = None) -> None:
    settings = load_settings(config)

    if not os.getenv("LAB3_TOOL_TOKEN"):
        raise typer.BadParameter("Run lab3 init-env or set LAB3_TOOL_TOKEN")

    uvicorn.run(create_tool_app(settings), host=settings.tools["host"], port=settings.tools["port"])


@app.command("export-index")
def export_index(config: ConfigOption = None) -> None:
    show(Retriever(load_settings(config)).export_index())


@app.command("verify-data")
def verify_data(config: ConfigOption = None) -> None:
    show(Retriever(load_settings(config)).verify())


@app.command()
def search(query: str, limit: int = 5, config: ConfigOption = None) -> None:
    show(Retriever(load_settings(config)).search(query, limit))


@app.command()
def reset(session: Annotated[str, typer.Option("--session")], config: ConfigOption = None) -> None:
    settings = load_settings(config)
    show({"session": session, "schema": ToolService(settings).reset(session)})


@app.command()
def validate(config: ConfigOption = None) -> None:
    settings = load_settings(config)
    tasks = load_tasks(settings.root / "resources/tasks.json")
    show(
        {
            "tasks": len(tasks),
            "turns": sum(len(task.turns) for task in tasks),
            "scored_turns": sum(turn.scored for task in tasks for turn in task.turns),
            "dialogues": sum(task.type == "dialogue" for task in tasks),
        }
    )


@app.command("check-dataset")
def check_dataset(config: ConfigOption = None) -> None:
    settings = load_settings(config)
    tasks = load_tasks(settings.root / "resources/tasks.json")
    service = ToolService(settings)
    checks = []

    for task in tasks:
        session = "reference-" + uuid.uuid4().hex[:12]
        service.reset(session)

        for index, turn in enumerate(task.turns):
            actual = None

            for query in turn.reference_sql:
                actual = service.execute_sql(query, session)["rows"]

            if turn.expected_rows is None:
                continue

            if turn.state_sql:
                actual = service.execute_sql(turn.state_sql, session)["rows"]

            ok = actual is not None and rows_match(actual, turn.expected_rows, turn.ordered)
            checks.append({"task": task.id, "turn": index, "ok": ok})

    settings.checks.mkdir(parents=True, exist_ok=True)
    (settings.checks / "dataset-check.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    show({"checked_sql_criteria": len(checks), "failures": [c for c in checks if not c["ok"]]})

    if any(not check["ok"] for check in checks):
        raise typer.Exit(1)


async def chat_session(profile: str, session: str, config: Path | None) -> None:
    settings = load_settings(config)

    if not os.getenv("POLZA_API_KEY"):
        raise ValueError("Set POLZA_API_KEY in .env")

    if settings.model.base_url.startswith("https://polza.ai/"):
        await asyncio.to_thread(refresh_model, settings)

    if profile not in settings.evaluation["profiles"]:
        raise ValueError("Profile must be custom-trim, custom-tiered or pi")

    tracer = Tracer(settings.sessions / ("chat-" + uuid.uuid4().hex[:8]))
    agent = (
        PiAgent(settings, session, tracer)
        if profile == "pi"
        else CustomAgent(
            settings,
            session,
            tracer,
            strategy="trim" if profile == "custom-trim" else "tiered",
        )
    )

    try:
        await agent.start()
        typer.echo(
            f"Session: {session}; /exit to finish. "
            f"Reset is explicit: lab3 reset --session {session}"
        )

        while True:
            message = await asyncio.to_thread(input, "Вы: ")

            if message.strip() == "/exit":
                break

            result = await agent.respond(message)
            typer.echo(result.answer)
            show(
                {
                    "status": result.status,
                    "usage": result.usage,
                    "cost_rub": result.cost_rub,
                    "latency_ms": result.latency_ms,
                }
            )
    finally:
        await agent.close()


@app.command()
def chat(
    profile: str = "custom-tiered", session: str = "demo", config: ConfigOption = None
) -> None:
    asyncio.run(chat_session(profile, session, config))


@app.command()
def evaluate(
    profile: str = "custom-tiered",
    tasks: str = "",
    max_cost: float | None = None,
    label: str = "benchmark",
    max_steps: int | None = None,
    reasoning: str | None = None,
    config: ConfigOption = None,
) -> None:
    settings = load_settings(config)

    if reasoning is not None:
        if reasoning not in {"none", "low", "medium", "high"}:
            raise typer.BadParameter("reasoning must be none, low, medium or high")

        settings.model.reasoning_effort = reasoning

    if max_steps is not None:
        if max_steps < 1:
            raise typer.BadParameter("max_steps must be positive")

        settings.agent["max_steps"] = max_steps
        settings.agent["max_llm_calls"] = min(settings.agent["max_llm_calls"], max_steps * 2)

    path = asyncio.run(
        run_dataset(settings, profile, tasks.split(",") if tasks else None, max_cost, label)
    )
    typer.echo(str(path))


@app.command()
def compare(
    runs: Annotated[list[Path], typer.Argument(help="Three explicit evaluation run directories")],
    output: Path | None = None,
    config: ConfigOption = None,
) -> None:
    settings = load_settings(config)
    target = output or settings.root / "results/09_report"
    result = compare_runs(runs, target)
    show({"output": target, "comparable": result["comparable"], "complete": result["complete"]})


def wait_for_tools(settings: Settings, process: subprocess.Popen, log: Path) -> None:
    deadline = time.monotonic() + 60
    headers = {"Authorization": f"Bearer {os.environ['LAB3_TOOL_TOKEN']}"}

    with httpx.Client(base_url=settings.tool_url, headers=headers, timeout=2) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Tool server exited. See {log}")
            try:
                response = client.get("/health")
                if response.status_code == 200 and response.json().get("status") == "ok":
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)

    raise TimeoutError(f"Tool server did not become ready in 60 seconds. See {log}")


@app.command()
def run(
    prepare_only: bool = False,
    max_cost: float | None = None,
    tasks: str = "",
    config: ConfigOption = None,
) -> None:
    settings = load_settings(config)
    budget = max_cost if max_cost is not None else settings.evaluation["max_cost_rub"]
    if not 0 < budget < float("inf"):
        raise typer.BadParameter("max-cost must be finite and positive")

    selected = tasks.split(",") if tasks else None
    available = {task.id for task in load_tasks(settings.root / "resources/tasks.json")}
    if selected and set(selected) - available:
        raise typer.BadParameter(f"Unknown tasks: {set(selected) - available}")

    required = ["LAB3_TOOL_TOKEN"]
    if not prepare_only:
        required += ["POLZA_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]
        os.environ["LANGFUSE_ENABLED"] = "true"
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise typer.BadParameter(f"Set these variables in .env: {', '.join(missing)}")
    if not shutil.which("docker"):
        raise typer.BadParameter("Install Docker Desktop and start its engine")
    if not prepare_only:
        node = os.getenv("LAB3_NODE") or shutil.which("node")
        if not node and os.name == "nt":
            node = str(settings.runtime / "node/node.exe")
        if not node or not Path(node).is_file():
            raise typer.BadParameter("Install Node.js before running the Pi comparison")
        if not (settings.root / "pi/node_modules/@earendil-works/pi-coding-agent").is_dir():
            raise typer.BadParameter("Install Pi dependencies: npm ci --prefix pi")

    # Существующий сервер не трогаем, чтобы не завершить чужой процесс.
    if not prepare_only:
        with socket.socket() as probe:
            probe.settimeout(1)
            if probe.connect_ex((settings.tools["host"], settings.tools["port"])) == 0:
                raise typer.BadParameter("Tool port is busy. Stop the previous serve-tools process")

    def command(arguments: list[str]) -> None:
        subprocess.run(arguments, cwd=settings.root, check=True)

    cli = [sys.executable, "-m", "core.cli"]
    config_args = ["--config", str(config.resolve())] if config else []
    typer.echo("[1/4] Docker")
    command(["docker", "compose", "up", "-d", "--wait", "--wait-timeout", "180"])
    command(
        [
            "docker",
            "compose",
            "exec",
            "-T",
            "postgres",
            "psql",
            "-U",
            "labadmin",
            "-d",
            "lab3",
            "-v",
            "ON_ERROR_STOP=1",
            "-f",
            "/docker-entrypoint-initdb.d/01-database.sql",
        ]
    )

    typer.echo("[2/4] Prepare data and check tools. No chat model calls.")
    stages = ["prepare"]
    if not (settings.data / "05_index/index.json").exists():
        stages.append("index")
    stages.extend(["verify-data", "validate", "check-dataset"])
    for stage in stages:
        command([*cli, stage, *config_args])
    command([*cli, "smoke", "--retrieval", *config_args])
    if prepare_only:
        typer.echo("Preparation complete. No paid model calls were made.")
        return

    # Сервер принадлежит этому запуску и завершается даже при ошибке или Ctrl+C.
    settings.checks.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    log = settings.checks / f"tools-{stamp}.log"
    label = "pilot" if selected else "benchmark"
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            [*cli, "serve-tools", *config_args],
            cwd=settings.root,
            stdout=output,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        try:
            wait_for_tools(settings, process, log)
            typer.echo(
                f"[3/4] Three profiles. Budget per profile: {budget:g} RUB. "
                f"Total configured budget: {3 * budget:g} RUB."
            )
            runs = []
            for profile in ("custom-trim", "custom-tiered", "pi"):
                typer.echo(f"Starting {profile}")
                path = asyncio.run(run_dataset(settings, profile, selected, budget, label))
                typer.echo(str(path))
                summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
                if not summary["complete"]:
                    raise RuntimeError(
                        f"Incomplete run: {path}. Inspect its report before retrying"
                    )
                runs.append(path)

            typer.echo("[4/4] Comparison")
            target = settings.root / "results/09_report"
            if selected:
                target /= "pilot-" + stamp
            result = compare_runs(runs, target)
            typer.echo(f"Report: {target}")
            if not result["comparable"]:
                raise RuntimeError("Run conditions differ. See comparison.json")
        finally:
            if process.poll() is None:
                # Uvicorn обрабатывает Ctrl+Break и успевает отправить накопленные traces.
                if os.name == "nt":
                    try:
                        process.send_signal(signal.CTRL_BREAK_EVENT)
                    except OSError:
                        process.terminate()
                else:
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


async def smoke_check(config: Path | None, retrieval: bool) -> dict:
    settings = load_settings(config)
    session = "smoke-" + uuid.uuid4().hex[:8]
    service = ToolService(settings)
    service.reset(session)
    results = []

    for name, args in [
        ("inspect_schema", {}),
        ("execute_sql", {"sql": "SELECT count(*) AS n FROM products"}),
        ("execute_sql", {"sql": "SELECT missing FROM products"}),
        ("execute_sql", {"sql": "SELECT count(*) AS n FROM products"}),
        ("explain_sql", {"sql": "SELECT * FROM products WHERE price > 500"}),
    ]:
        results.append(await service.call(name, args, session))

    if retrieval:
        results.append(
            await service.call("search_docs", {"query": "GENERATED ALWAYS IDENTITY"}, session)
        )

    ok = all(r["status"] == ("error" if i == 2 else "ok") for i, r in enumerate(results))
    output = {"ok": ok, "session": session, "results": results}
    settings.checks.mkdir(parents=True, exist_ok=True)
    (settings.checks / "smoke.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return output


@app.command()
def smoke(retrieval: bool = False, config: ConfigOption = None) -> None:
    result = asyncio.run(smoke_check(config, retrieval))
    show(result)

    if not result["ok"]:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
