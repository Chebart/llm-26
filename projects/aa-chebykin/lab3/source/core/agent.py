import asyncio
import json
import os
import time
import uuid
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import TypedDict

import httpx
from langgraph.graph import END, START, StateGraph
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from openai import AsyncOpenAI

from core.config import Settings
from core.context import ContextManager, ConversationStore
from core.tools import MCP_TOOLS, ToolService
from core.tracing import Tracer


@dataclass
class TurnResult:
    answer: str
    status: str
    trace_id: str | None = None
    usage: dict = field(default_factory=dict)
    cost_rub: float | None = 0
    latency_ms: float = 0
    steps: int | None = 0
    tool_calls: list[dict] = field(default_factory=list)
    context_events: list[dict] = field(default_factory=list)
    max_context: int | None = 0
    history_compacted: bool = False


class AgentState(TypedDict):
    messages: list[dict]
    prepared: list[dict]
    steps: int
    finished: bool


class CustomAgent:
    def __init__(
        self,
        settings: Settings,
        session_id: str,
        tracer: Tracer,
        strategy: str | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.settings, self.session_id, self.tracer = settings, session_id, tracer
        if (
            settings.context.tokenizer == "deepseek-ai/DeepSeek-V4.1-Flash"
            and settings.model.id != "deepseek/deepseek-v4.1-flash"
        ):
            raise ValueError(
                "Change context.tokenizer and its revision to match the selected model"
            )

        self.store = ConversationStore(settings.sessions / "history.sqlite3")
        # Журнал хранится полностью; модель получает отдельное представление истории.
        context = settings.context.model_dump()
        context["tokenizer_cache_dir"] = settings.data / "models/context-tokenizers"
        context["reasoning_effort"] = settings.model.reasoning_effort

        if strategy:
            context["strategy"] = strategy

        self.context = ContextManager(**context)
        self.service = ToolService(settings, tracer)
        self.schemas = self.service.schemas()
        self.prompts = settings.prompts()
        # SDK отвечает за протокол API, а LangGraph — за цикл и условия остановки.
        self.client = client or AsyncOpenAI(
            base_url=settings.model.base_url,
            api_key=os.getenv("POLZA_API_KEY") or "not-configured",
            timeout=settings.model.timeout_seconds,
            max_retries=settings.agent["api_max_retries"],
        )
        self.stack = AsyncExitStack()
        self.mcp: ClientSession | None = None
        self.result = TurnResult("", "pending")
        self.turn_id = ""
        self.deadline = 0.0
        self.history_compacted = False
        self.llm_calls = 0
        self.graph = self._graph()

    async def _complete(
        self, messages: list[dict], max_tokens: int, tools: list[dict] | None = None
    ) -> tuple[dict, dict]:
        # Один формат запроса используется основным агентом и суммаризатором.
        parameters = {
            "model": self.settings.model.id,
            "messages": messages,
            "temperature": self.settings.model.temperature,
            "max_tokens": max_tokens,
            "extra_body": {"reasoning": {"effort": self.settings.model.reasoning_effort}},
        }

        if self.settings.model.provider:
            parameters["extra_body"]["provider"] = {
                "only": [self.settings.model.provider],
                "allow_fallbacks": False,
            }

        if tools:
            parameters["tools"] = tools

        try:
            response = await self.client.chat.completions.create(**parameters)
        except (Exception, asyncio.CancelledError):
            self.result.cost_rub = None
            self.result.usage["partial"] = True
            raise

        # Native SDK сохраняет дополнительные поля провайдера, включая reasoning_content.
        message = response.choices[0].message.model_dump(exclude_none=True)
        usage = {}

        if response.usage:
            raw = response.usage.model_dump()
            usage = {
                "input_tokens": raw.get("prompt_tokens", 0),
                "output_tokens": raw.get("completion_tokens", 0),
                "total_tokens": raw.get("total_tokens", 0),
                "gateway_cost_rub": raw.get("cost_rub"),
                "input_token_details": {
                    "cache_read": (raw.get("prompt_tokens_details") or {}).get(
                        "cached_tokens", raw.get("prompt_cache_hit_tokens", 0)
                    )
                },
            }

        return message, usage

    async def start(self) -> None:
        headers = {
            "Authorization": f"Bearer {os.getenv('LAB3_TOOL_TOKEN', '')}",
            "x-lab3-session": self.session_id,
        }
        http = await self.stack.enter_async_context(httpx.AsyncClient(headers=headers, timeout=60))
        streams = await self.stack.enter_async_context(
            streamable_http_client(
                self.settings.tool_url + "/mcp/",
                http_client=http,
            )
        )
        self.mcp = await self.stack.enter_async_context(ClientSession(streams[0], streams[1]))
        await self.mcp.initialize()

    def _usage(self, usage: dict) -> float | None:
        if not usage:
            self.result.usage["partial"] = True

            return None

        for key in ("input_tokens", "output_tokens", "total_tokens"):
            self.result.usage[key] = self.result.usage.get(key, 0) + usage.get(key, 0)

        cached = (usage.get("input_token_details") or {}).get("cache_read", 0)
        estimate = (
            (usage.get("input_tokens", 0) - cached) * self.settings.model.input_price
            + cached * self.settings.model.cache_price
            + usage.get("output_tokens", 0) * self.settings.model.output_price
        ) / 1_000_000
        self.result.usage["estimated_cost_rub"] = (
            self.result.usage.get("estimated_cost_rub", 0) + estimate
        )
        billed = usage.get("gateway_cost_rub")
        self.result.usage["cost_source"] = "gateway" if billed is not None else "catalog_estimate"

        return float(billed) if billed is not None else estimate

    async def _call_model(
        self,
        messages: list[dict],
        max_tokens: int,
        tools: list[dict] | None = None,
        *,
        name: str = "llm",
        emergency: bool | None = None,
    ) -> dict:
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Turn deadline reached before model request")
        if self.llm_calls >= self.settings.agent["max_llm_calls"]:
            raise RuntimeError("Maximum LLM calls reached, including summarization")
        self.llm_calls += 1

        # И обычный ответ, и сводка проходят одинаковые лимиты, tracing и учёт.
        with self.tracer.span(
            name,
            kind="generation",
            input={"messages": messages, "tools": tools or []},
            model=self.settings.model.id,
            emergency=emergency,
            provider=self.settings.model.provider,
            model_parameters={
                "temperature": self.settings.model.temperature,
                "max_tokens": max_tokens,
                "reasoning_effort": self.settings.model.reasoning_effort,
            },
        ) as observation:
            answer, usage = await self._complete(messages, max_tokens, tools)
            cost = self._usage(usage)
            self.result.cost_rub = (
                self.result.cost_rub + cost
                if self.result.cost_rub is not None and cost is not None
                else None
            )
            if observation:
                observation.update(
                    output=answer,
                    usage_details={
                        key: usage.get(key, 0) for key in ("input_tokens", "output_tokens")
                    },
                    metadata={"cost_rub": cost, "currency": "RUB"},
                )
        return answer

    async def _summarize(self, content: str, emergency: bool) -> str:
        limit = self.settings.context.summary_tokens // (2 if emergency else 1)
        messages = [
            {"role": "system", "content": self.prompts["summary"]["system"]},
            {"role": "user", "content": content},
        ]
        answer = await self._call_model(messages, limit, name="summarize", emergency=emergency)
        content = answer.get("content", "")
        if not content:
            raise ValueError("Summarizer returned no text; use a larger output budget")
        return str(content)

    async def _prepare(self, state: AgentState) -> dict:
        self.result.max_context = max(
            self.result.max_context or 0,
            self.context.count(state["messages"], self.schemas),
        )
        prepared = await self.context.prepare(
            state["messages"], self.prompts["agent"]["system"], self.schemas, self._summarize
        )
        self.result.max_context = max(self.result.max_context or 0, prepared.tokens)
        # ?????? ???? ?????? ????????, ????? scores ? trace metadata ?????????? ???????????.
        if prepared.events:
            directory = self.settings.sessions / "context"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{self.turn_id}-{uuid.uuid4().hex[:8]}.json"
            path.write_text(
                json.dumps(
                    {
                        "session_id": self.session_id,
                        "turn_id": self.turn_id,
                        "tools": self.schemas,
                        "events": prepared.events,
                        "final_window": prepared.messages,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            for event in prepared.events:
                event.pop("window_before", None)
                event.pop("window_after", None)
                event["artifact"] = str(path.relative_to(self.settings.root))

        self.result.context_events.extend(prepared.events)
        self.history_compacted = self.history_compacted or bool(prepared.events)
        self.tracer.event(
            "context",
            session_id=self.session_id,
            turn_id=self.turn_id,
            tokens=prepared.tokens,
            limit=self.context.limit,
            counting_method=prepared.counting_method,
        )

        with self.tracer.span(
            "context",
            tokens=prepared.tokens,
            limit=self.context.limit,
            counting_method=prepared.counting_method,
        ):
            pass

        for event in prepared.events:
            with self.tracer.span("context." + event["level"], **event) as observation:
                if observation and (event.get("error") or event.get("warning")):
                    observation.update(
                        level="WARNING", status_message=event.get("error") or event["warning"]
                    )

        return {"prepared": prepared.messages}

    async def _generate(self, state: AgentState) -> dict:
        if state["steps"] >= self.settings.agent["max_steps"]:
            raise RuntimeError("Maximum agent iterations reached")

        message = await self._call_model(
            state["prepared"], self.settings.model.max_output_tokens, self.schemas
        )

        self.store.append(self.session_id, self.turn_id, [message])
        self.result.steps = state["steps"] + 1

        return {
            "messages": state["messages"] + [message],
            "steps": state["steps"] + 1,
            "finished": not bool(message.get("tool_calls")),
        }

    async def _execute_tools(self, state: AgentState) -> dict:
        messages = list(state["messages"])

        for call in messages[-1].get("tool_calls", []):
            name = call["function"]["name"]

            try:
                args = json.loads(call["function"]["arguments"])

                with self.tracer.span("tool.client." + name, input=args):
                    if name in MCP_TOOLS:
                        if self.mcp is None:
                            raise RuntimeError("MCP client is not connected; call start()")

                        response = await asyncio.wait_for(
                            self.mcp.call_tool(name, args),
                            timeout=self.settings.agent["tool_timeout_seconds"],
                        )
                        payload = response.structuredContent

                        if payload is None:
                            payload = json.loads(
                                next(c.text for c in response.content if c.type == "text")
                            )
                    else:
                        payload = await asyncio.wait_for(
                            self.service.call(name, args, self.session_id, call["id"]),
                            timeout=self.settings.agent["tool_timeout_seconds"],
                        )
            except Exception as error:
                payload = {
                    "status": "error",
                    "tool_name": name,
                    "error": {"code": type(error).__name__, "message": str(error)},
                }

            self.result.tool_calls.append(
                {"name": name, "arguments": call["function"]["arguments"], "result": payload}
            )
            message = {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(payload, ensure_ascii=False, default=str),
            }
            messages.append(message)
            self.store.append(self.session_id, self.turn_id, [message])

        return {"messages": messages}

    async def _step(self, state: AgentState) -> dict:
        with self.tracer.span("iteration", step=state["steps"] + 1) as observation:
            if observation:
                # Серверные spans MCP должны быть детьми этой же итерации.
                async with httpx.AsyncClient(timeout=10) as http:
                    response = await http.post(
                        self.settings.tool_url + f"/sessions/{self.session_id}/turn",
                        headers={"Authorization": f"Bearer {os.getenv('LAB3_TOOL_TOKEN', '')}"},
                        json={
                            "trace_id": observation.trace_id,
                            "parent_span_id": observation.id,
                            "turn_id": self.turn_id,
                        },
                    )
                    response.raise_for_status()

            changes = await self._generate(state)

            if not changes["finished"]:
                changes.update(await self._execute_tools({**state, **changes}))

            return changes

    def _graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("prepare", self._prepare)
        graph.add_node("step", self._step)
        graph.add_edge(START, "prepare")
        graph.add_edge("prepare", "step")
        graph.add_conditional_edges(
            "step",
            lambda state: "end" if state["finished"] else "prepare",
            {"end": END, "prepare": "prepare"},
        )
        return graph.compile()

    def _repair_history(self) -> None:
        # Закрываем незавершённые вызовы в журнале, чтобы следующая реплика была валидна.
        history = self.store.read(self.session_id)
        answered = {m.get("tool_call_id") for m in history if m["role"] == "tool"}
        pending = [
            call for m in history for call in m.get("tool_calls", []) if call["id"] not in answered
        ]
        repair = [
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps({"status": "error", "error": "Turn interrupted"}),
            }
            for call in pending
        ]
        repair.append({"role": "assistant", "content": self.result.answer})
        self.store.append(self.session_id, self.turn_id, repair)

    async def respond(self, user_message: str) -> TurnResult:
        self.turn_id = uuid.uuid4().hex
        self.result = TurnResult("", "running")
        self.llm_calls = 0
        started = time.perf_counter()
        self.deadline = time.monotonic() + self.settings.agent["turn_timeout_seconds"]
        self.store.append(
            self.session_id, self.turn_id, [{"role": "user", "content": user_message}]
        )

        with self.tracer.span(
            "user_turn",
            input=user_message,
            session_id=self.session_id,
            agent="custom",
            turn_id=self.turn_id,
        ) as observation:
            if observation:
                observation.update_trace(
                    session_id=self.session_id,
                    tags=[
                        "custom",
                        self.context.strategy,
                        self.settings.model.id,
                        self.tracer.run_kind,
                        "lab3-" + self.tracer.version,
                    ],
                    user_id=self.settings.evaluation["user_id"],
                    input=user_message,
                )
                self.result.trace_id = observation.trace_id

            try:
                state = await asyncio.wait_for(
                    self.graph.ainvoke(
                        {
                            "messages": self.store.read(self.session_id),
                            "prepared": [],
                            "steps": 0,
                            "finished": False,
                        },
                        config={"recursion_limit": self.settings.agent["max_steps"] * 3 + 3},
                    ),
                    timeout=self.settings.agent["turn_timeout_seconds"],
                )
                self.result.answer = str(state["messages"][-1].get("content", ""))
                self.result.status = "ok"
            except (Exception, asyncio.CancelledError) as error:
                if isinstance(error, asyncio.CancelledError):
                    self.result.status = "cancelled"
                else:
                    self.result.status = "timeout" if isinstance(error, TimeoutError) else "error"
                self.result.answer = f"Агент остановлен: {error or self.result.status}"
                self._repair_history()
                self.tracer.event(
                    "agent_error",
                    level="WARNING",
                    error=str(error),
                    session_id=self.session_id,
                    turn_id=self.turn_id,
                )

                if observation:
                    observation.update(level="WARNING", status_message=str(error))
                if isinstance(error, asyncio.CancelledError):
                    raise

            if observation:
                observation.update(output=self.result.answer)
                observation.update_trace(output=self.result.answer)

        self.result.latency_ms = (time.perf_counter() - started) * 1000
        self.result.history_compacted = self.history_compacted

        return self.result

    async def close(self) -> None:
        await self.stack.aclose()
        await self.client.close()
        self.tracer.flush()
