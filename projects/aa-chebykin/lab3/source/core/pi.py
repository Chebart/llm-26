import asyncio
import json
import os
import shutil
import time
import uuid
from pathlib import Path

import httpx

from core.agent import TurnResult
from core.config import Settings
from core.tracing import Tracer


class PiAgent:
    def __init__(self, settings: Settings, session_id: str, tracer: Tracer) -> None:
        self.settings, self.session_id, self.tracer = settings, session_id, tracer
        self.directory = settings.sessions / "pi" / uuid.uuid4().hex
        self.process: asyncio.subprocess.Process | None = None
        self.stderr_task: asyncio.Task | None = None
        self.turn_file = self.directory / "turn.json"
        self.headers = {"Authorization": f"Bearer {os.getenv('LAB3_TOOL_TOKEN', '')}"}
        self.history_compacted = False
        self.turn_messages: list[dict] = []

    def _write_runtime_config(self) -> None:
        provider = self.settings.pi["provider"]
        # Pi обращается к шлюзу напрямую; расширение пишет generations через SDK.
        model = self.settings.model
        models = {
            "providers": {
                provider: {
                    "baseUrl": model.base_url,
                    "api": "openai-completions",
                    "apiKey": "POLZA_API_KEY",
                    "compat": {"supportsDeveloperRole": False, "supportsReasoningEffort": False},
                    "models": [
                        {
                            "id": model.id,
                            "name": model.id,
                            "reasoning": False,
                            "input": ["text"],
                            "contextWindow": self.settings.context.max_input_tokens
                            + model.max_output_tokens
                            + 4096,
                            "maxTokens": model.max_output_tokens,
                            "cost": {
                                "input": model.input_price,
                                "output": model.output_price,
                                "cacheRead": model.cache_price,
                                "cacheWrite": model.input_price,
                            },
                        }
                    ],
                }
            }
        }
        (self.directory / "models.json").write_text(json.dumps(models), encoding="utf-8")
        (self.directory / "settings.json").write_text(
            json.dumps(
                {
                    "compaction": {
                        "enabled": True,
                        # Pi дополнительно вычитает 4096 в clampMaxTokensToContext.
                        "reserveTokens": model.max_output_tokens + 4096,
                        "keepRecentTokens": self.settings.pi["keep_recent_tokens"],
                    },
                    "retry": {"enabled": False},
                    "quietStartup": True,
                }
            ),
            encoding="utf-8",
        )
        # Этот файл связывает все запросы Pi, включая compaction, с текущей репликой.
        self.turn_file.write_text("{}", encoding="utf-8")

    def _environment(self) -> dict[str, str]:
        model = self.settings.model
        return {
            **os.environ,
            "PI_CODING_AGENT_DIR": str(self.directory),
            "PI_OFFLINE": "1",
            "PI_TELEMETRY": "0",
            "LAB3_TOOL_URL": self.settings.tool_url,
            "LAB3_SESSION_ID": self.session_id,
            "LAB3_TURN_CONTEXT": str(self.turn_file),
            "LAB3_MAX_STEPS": str(self.settings.agent["max_steps"]),
            "LAB3_MAX_LLM_CALLS": str(self.settings.agent["max_llm_calls"]),
            "LAB3_TEMPERATURE": str(model.temperature),
            "LAB3_USER_ID": self.settings.evaluation["user_id"],
            "LAB3_AGENT_VERSION": self.tracer.version,
            "LAB3_MODEL_ID": model.id,
            "LAB3_RUN_KIND": self.tracer.run_kind,
            "LAB3_REASONING_EFFORT": model.reasoning_effort,
            "LAB3_PROVIDER": model.provider or "",
        }

    def _command(self, node: str, cli: Path) -> list[str]:
        model = self.settings.model
        provider = self.settings.pi["provider"]
        return [
            node,
            str(cli),
            "--mode",
            "rpc",
            "--provider",
            provider,
            "--model",
            model.id,
            "--thinking",
            "off",
            "--no-builtin-tools",
            "--no-extensions",
            "--no-skills",
            "--no-prompt-templates",
            "--no-themes",
            "--no-context-files",
            "--offline",
            "-e",
            str(self.settings.root / "pi/extension.ts"),
            "--append-system-prompt",
            self.settings.prompts()["agent"]["system"],
            "--session-dir",
            str(self.directory / "sessions"),
        ]

    async def start(self) -> None:
        node = os.getenv("LAB3_NODE") or shutil.which("node")

        if not node:
            local = self.settings.runtime / "node/node.exe"
            node = str(local) if local.exists() else None

        if not node:
            raise RuntimeError("Install Node.js >=22.19 or set LAB3_NODE")

        cli = self.settings.root / "pi/node_modules/@earendil-works/pi-coding-agent/dist/cli.js"

        if not cli.exists():
            raise RuntimeError("Install Pi dependencies: cd pi; npm ci --ignore-scripts")

        # Проверяем установку до создания runtime-артефактов.
        self.directory.mkdir(parents=True)
        self._write_runtime_config()
        self.process = await asyncio.create_subprocess_exec(
            *self._command(node, cli),
            cwd=self.directory,
            env=self._environment(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=4 * 1024 * 1024,
        )
        self.stderr_task = asyncio.create_task(self._stderr())

    async def _stderr(self) -> None:
        if self.process and self.process.stderr:
            while line := await self.process.stderr.readline():
                with (self.directory / "stderr.log").open("ab") as target:
                    target.write(line)

    async def respond(self, user_message: str) -> TurnResult:
        if self.process is None or self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("Pi is not started")

        started = time.perf_counter()
        self.turn_messages = []
        result = TurnResult("", "running", steps=None, max_context=None, cost_rub=None)
        turn_id = uuid.uuid4().hex

        with self.tracer.span(
            "user_turn", input=user_message, agent="pi", session_id=self.session_id, turn_id=turn_id
        ) as observation:
            context = {"turn_id": turn_id}

            if observation:
                observation.update_trace(
                    session_id=self.session_id,
                    input=user_message,
                    user_id=self.settings.evaluation["user_id"],
                    tags=[
                        "pi",
                        self.settings.model.id,
                        self.tracer.run_kind,
                        "lab3-" + self.tracer.version,
                    ],
                )
                result.trace_id = observation.trace_id
                context.update(trace_id=observation.trace_id, parent_span_id=observation.id)

            self.turn_file.write_text(json.dumps(context), encoding="utf-8")

            async with httpx.AsyncClient(
                base_url=self.settings.tool_url, headers=self.headers, timeout=30
            ) as http:
                response = await http.post(f"/sessions/{self.session_id}/turn", json=context)
                response.raise_for_status()
                before = len((await http.get(f"/sessions/{self.session_id}/events")).json())

                try:
                    await self._run_turn(user_message, turn_id, result)
                except Exception as error:
                    result.status = "timeout" if isinstance(error, TimeoutError) else "error"
                    result.answer = f"Pi остановлен: {error}"

                    if observation:
                        observation.update(level="WARNING", status_message=str(error))

                    await self.close()

                events = (await http.get(f"/sessions/{self.session_id}/events")).json()[before:]
                result.tool_calls = [{"name": e["tool_name"], "result": e} for e in events]

            if observation:
                observation.update(output=result.answer)
                observation.update_trace(output=result.answer)

        self._read_billing(result, turn_id)

        result.latency_ms = (time.perf_counter() - started) * 1000
        result.history_compacted = self.history_compacted

        return result

    async def _run_turn(self, user_message: str, turn_id: str, result: TurnResult) -> None:
        iteration_count = 0

        async with asyncio.timeout(self.settings.agent["turn_timeout_seconds"]):
            self.process.stdin.write(
                (
                    json.dumps({"type": "prompt", "id": turn_id, "message": user_message}) + "\n"
                ).encode()
            )
            await self.process.stdin.drain()

            while True:
                line = await self.process.stdout.readline()

                if not line:
                    raise RuntimeError(f"Pi exited; inspect {self.directory / 'stderr.log'}")

                event = json.loads(line)

                with (self.directory / "rpc.jsonl").open("ab") as target:
                    target.write(line)

                if event.get("type") == "turn_start":
                    iteration_count += 1

                    if iteration_count > self.settings.agent["max_steps"]:
                        raise RuntimeError("Maximum Pi iterations reached")

                if event.get("type") == "response" and not event.get("success", True):
                    raise RuntimeError(event.get("error", "Pi RPC command failed"))

                if self.handle_event(event, result):
                    break

    def _read_billing(self, result: TurnResult, turn_id: str) -> None:
        billing = self.directory / "billing.jsonl"

        if billing.exists():
            charges = {
                entry["request_id"]: entry["cost_rub"]
                for line in billing.read_text(encoding="utf-8").splitlines()
                if (entry := json.loads(line)).get("turn_id") == turn_id
            }
            expected = (result.steps or 0) + sum(
                bool(e.get("result")) for e in result.context_events
            )

            if len(charges) >= expected and expected and not result.usage.get("partial"):
                result.usage["estimated_cost_rub"] = result.cost_rub
                result.cost_rub = sum(charges.values())
                result.usage["cost_source"] = "gateway"
            else:
                result.usage["cost_source"] = "catalog_estimate"
        else:
            result.usage["cost_source"] = "catalog_estimate"

    def handle_event(self, event: dict, result: TurnResult) -> bool:
        if event.get("type") == "agent_end":
            self.turn_messages.extend(event.get("messages", []))
        elif event.get("type") == "compaction_end":
            result.context_events.append({"level": "pi.compaction", **event})
            compact = event.get("result") or {}

            with self.tracer.span(
                "pi.compaction",
                reason=event.get("reason"),
                tokens_before=compact.get("tokensBefore"),
                tokens_after=compact.get("estimatedTokensAfter"),
                error=event.get("errorMessage"),
            ) as observation:
                if observation:
                    observation.update(output=compact.get("summary"))

            if event.get("result") and not event.get("aborted"):
                self.history_compacted = True
        elif event.get("type") == "agent_settled":
            self._read_result({"messages": self.turn_messages}, result)

            if event.get("aborted"):
                result.status = "error"
                result.cost_rub = None
                result.usage["partial"] = True

            return True

        return False

    def _read_result(self, event: dict, result: TurnResult) -> None:
        assistants = [m for m in event.get("messages", []) if m.get("role") == "assistant"]

        if not assistants:
            result.status, result.answer = "error", "Pi returned no assistant messages"

            return

        last = assistants[-1]
        result.answer = "\n".join(
            c.get("text", "") for c in last.get("content", []) if c.get("type") == "text"
        )
        result.status = "error" if last.get("stopReason") in {"error", "aborted"} else "ok"

        if result.status == "error":
            result.answer += last.get("errorMessage", "")

        result.steps = len(assistants)

        if all(m.get("usage") for m in assistants):
            result.usage = {
                "input_tokens": sum(
                    m["usage"].get("input", 0) + m["usage"].get("cacheRead", 0) for m in assistants
                ),
                "output_tokens": sum(m["usage"].get("output", 0) for m in assistants),
            }
            # Каталог Pi обычно считает USD; здесь числа в models.json явно заданы в RUB.
            result.cost_rub = sum(m["usage"].get("cost", {}).get("total", 0) for m in assistants)

            if not all("total" in m["usage"].get("cost", {}) for m in assistants):
                result.cost_rub = None

        for event in result.context_events:
            compact = event.get("result") or {}
            usage = compact.get("usage")

            if usage and result.cost_rub is not None:
                result.usage["input_tokens"] += (
                    usage.get("input", 0) + usage.get("cacheRead", 0) + usage.get("cacheWrite", 0)
                )
                result.usage["output_tokens"] += usage.get("output", 0)
                result.cost_rub += usage.get("cost", {}).get("total", 0)
            elif compact:
                result.cost_rub = None
                result.usage["partial"] = True

        if result.status == "error" or any(
            e.get("aborted") or e.get("errorMessage") for e in result.context_events
        ):
            # Нулевой usage в оборванном потоке не подтверждает бесплатный запрос.
            result.cost_rub = None
            result.usage["partial"] = True

    async def close(self) -> None:
        if self.process and self.process.returncode is None:
            drain = asyncio.create_task(self.process.stdout.read()) if self.process.stdout else None

            # Abort отменяет запрос до shutdown; один EOF в Pi сначала ждёт extensions.
            if self.process.stdin and not self.process.stdin.is_closing():
                try:
                    self.process.stdin.write(b'{"type":"abort","id":"shutdown-abort"}\n')
                    await asyncio.wait_for(self.process.stdin.drain(), 2)
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    pass
                finally:
                    self.process.stdin.close()

            try:
                await asyncio.wait_for(self.process.wait(), 10)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()

            if drain and (remaining := await drain):
                with (self.directory / "rpc.jsonl").open("ab") as target:
                    target.write(remaining)

        if self.stderr_task:
            await self.stderr_task

        self.tracer.flush()
