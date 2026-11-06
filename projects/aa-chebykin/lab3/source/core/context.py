import copy
import json
import math
import sqlite3
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, PreTrainedTokenizerBase


@lru_cache(maxsize=4)
def load_context_tokenizer(
    repository: str, revision: str, cache_dir: Path | None
) -> PreTrainedTokenizerBase:
    files = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
    directory = cache_dir / repository.replace("/", "--") / revision if cache_dir else None

    # Загружаются только tokenizer и шаблон; веса модели не нужны.
    if directory is None or not all((directory / name).is_file() for name in files):
        directory = Path(
            snapshot_download(
                repository,
                revision=revision,
                allow_patterns=[*files, "LICENSE"],
                local_dir=directory,
            )
        )

    return AutoTokenizer.from_pretrained(directory, local_files_only=True)


Summarizer = Callable[[str, bool], Awaitable[str]]


class ContextOverflow(ValueError):
    pass


@dataclass
class PreparedContext:
    messages: list[dict]
    tokens: int
    events: list[dict] = field(default_factory=list)
    counting_method: str = "model tokenizer and chat template estimate"


class ConversationStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

        with sqlite3.connect(path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS events ("
                "id INTEGER PRIMARY KEY, session_id TEXT, turn_id TEXT, message TEXT)"
            )

    def append(self, session_id: str, turn_id: str, messages: list[dict]) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.executemany(
                "INSERT INTO events(session_id, turn_id, message) VALUES (?, ?, ?)",
                [(session_id, turn_id, json.dumps(m, ensure_ascii=False)) for m in messages],
            )

    def read(self, session_id: str) -> list[dict]:
        with sqlite3.connect(self.path) as connection:
            return [
                json.loads(row[0])
                for row in connection.execute(
                    "SELECT message FROM events WHERE session_id=? ORDER BY id", (session_id,)
                )
            ]


class ContextManager:
    def __init__(
        self,
        tokenizer: str,
        tokenizer_revision: str,
        max_input_tokens: int = 8192,
        keep_recent_turns: int = 3,
        strategy: str = "tiered",
        estimate_margin: float = 1.3,
        summary_tokens: int = 512,
        tokenizer_cache_dir: Path | None = None,
        reasoning_effort: str = "none",
    ) -> None:
        self.limit = max_input_tokens
        self.keep_recent = keep_recent_turns
        self.strategy = strategy
        self.margin = estimate_margin
        self.summary_tokens = summary_tokens
        self.encoding = load_context_tokenizer(tokenizer, tokenizer_revision, tokenizer_cache_dir)
        self.reasoning_effort = reasoning_effort
        self.counting_method = (
            f"{tokenizer}@{tokenizer_revision}, chat template, margin={estimate_margin}"
        )
        self.summary = ""
        self.covered_messages = 0

    def count(self, messages: list[dict], tools: list[dict]) -> int:
        # Шаблон учитывает роли, native tools и служебные токены DeepSeek.
        payload = self.encoding.apply_chat_template(
            messages,
            tools=tools or None,
            tokenize=False,
            add_generation_prompt=True,
            thinking_mode="chat" if self.reasoning_effort == "none" else "thinking",
            reasoning_effort="high" if self.reasoning_effort == "none" else self.reasoning_effort,
        )

        tokens = len(self.encoding.encode(payload, add_special_tokens=False))
        return math.ceil(tokens * self.margin)

    async def prepare(
        self,
        history: list[dict],
        system: str,
        tools: list[dict],
        summarize: Summarizer | None = None,
    ) -> PreparedContext:
        prefix = [{"role": "system", "content": system}]
        if self.summary and len(history) >= self.covered_messages:
            prefix.append({"role": "system", "content": "Сводка истории:\n" + self.summary})

        # Меняем только копию окна; сводку фиксируем после успешной подготовки.
        messages = prefix + copy.deepcopy(history[self.covered_messages :])
        tokens = self.count(messages, tools)
        events = []
        if tokens <= self.limit:
            return PreparedContext(messages, tokens, events, self.counting_method)

        for level, shrink in (
            ("MessageTrim", self._trim),
            ("ToolResultPurge", self._purge),
            ("MessageCollapse", self._collapse),
        ):
            before = tokens
            window_before = copy.deepcopy(messages)
            tokens, details = shrink(messages, tools, tokens)
            if details:
                events.append(
                    {
                        "level": level,
                        "before": before,
                        "after": tokens,
                        "window_before": window_before,
                        "window_after": copy.deepcopy(messages),
                        **details,
                    }
                )
            if tokens <= self.limit:
                return PreparedContext(messages, tokens, events, self.counting_method)
            if self.strategy == "trim":
                raise ContextOverflow("MessageTrim cannot fit protected turns; increase budget")

        last_user = max(i for i, m in enumerate(history) if m["role"] == "user")
        source = copy.deepcopy(history[self.covered_messages : last_user])
        if self.summary:
            source.insert(0, {"role": "system", "content": self.summary})

        if summarize and source:
            for emergency in (False, True):
                level = "EmergencyLLMSummarize" if emergency else "LLMSummarize"
                before = tokens
                window_before = copy.deepcopy(messages)
                try:
                    summary = await self._summarize_bounded(source, summarize, emergency)
                    current = max(i for i, m in enumerate(messages) if m["role"] == "user")
                    candidate = (
                        messages[:1]
                        + [
                            {
                                "role": "system",
                                "content": "Сводка предыдущей истории:\n" + summary,
                            }
                        ]
                        + messages[current:]
                    )
                    tokens = self.count(candidate, tools)
                    messages = candidate
                    events.append(
                        {
                            "level": level,
                            "before": before,
                            "after": tokens,
                            "window_before": window_before,
                            "window_after": copy.deepcopy(messages),
                        }
                    )

                    if tokens <= self.limit:
                        self.summary, self.covered_messages = summary, last_user
                        return PreparedContext(messages, tokens, events, self.counting_method)
                    source = [{"role": "system", "content": summary}]
                except Exception as error:
                    events.append(
                        {
                            "level": level,
                            "before": before,
                            "after": tokens,
                            "error": str(error),
                            "fallback": "deterministic",
                        }
                    )

        # При сбое сводки оставляем текущую задачу и целые пары инструментов.
        current = max(i for i, m in enumerate(messages) if m["role"] == "user")
        fallback = messages[:1] + messages[current:]
        fallback_tokens = self.count(fallback, tools)
        if fallback_tokens <= self.limit:
            events.append(
                {
                    "level": "DeterministicFallback",
                    "before": tokens,
                    "after": fallback_tokens,
                    "warning": "Earlier context omitted",
                    "window_before": copy.deepcopy(messages),
                    "window_after": copy.deepcopy(fallback),
                }
            )
            return PreparedContext(fallback, fallback_tokens, events, self.counting_method)

        raise ContextOverflow(
            f"Protected context requires {tokens} estimated tokens, budget {self.limit}. "
            "Shorten the current request or increase max_input_tokens."
        )

    def _trim(self, messages: list[dict], tools: list[dict], tokens: int) -> tuple[int, dict]:
        positions = [i for i, m in enumerate(messages) if m["role"] == "user"]
        removable = max(0, len(positions) - self.keep_recent - 1)
        removed = 0

        # Удаляем завершённые ходы целиком, не разрывая assistant/tool.
        for _ in range(removable):
            positions = [i for i, m in enumerate(messages) if m["role"] == "user"]
            removed += positions[1] - positions[0]
            del messages[positions[0] : positions[1]]
            tokens = self.count(messages, tools)
            if tokens <= self.limit:
                break

        return tokens, {"removed_messages": removed} if removed else {}

    def _purge(self, messages: list[dict], tools: list[dict], tokens: int) -> tuple[int, dict]:
        purged = 0
        for message in messages:
            content = str(message.get("content", ""))
            if message["role"] != "tool" or len(content) < 800:
                continue

            try:
                result = json.loads(content)
                brief = {
                    key: result[key]
                    for key in ("status", "error", "artifact_id", "tool_name", "truncated")
                    if key in result
                }
                brief["notice"] = "Подробный результат исключён из контекста"
                message["content"] = json.dumps(brief, ensure_ascii=False)
            except (ValueError, TypeError):
                message["content"] = content[:400] + "\n[Результат сокращён]"
            purged += 1

        return (self.count(messages, tools), {"purged_results": purged}) if purged else (tokens, {})

    def _collapse(self, messages: list[dict], tools: list[dict], tokens: int) -> tuple[int, dict]:
        users = [i for i, m in enumerate(messages) if m["role"] == "user"]
        old = messages[1 : users[-1]] if len(users) > 1 else []
        if not old:
            return tokens, {}

        # Это выдержки журнала, а не автоматически извлечённые факты.
        journal = [
            {
                "role": m["role"],
                "tool": m.get("name"),
                "excerpt": str(m.get("content", ""))[:120],
            }
            for m in old
        ]
        messages[1 : users[-1]] = [
            {
                "role": "system",
                "content": "Журнал старых действий:\n" + json.dumps(journal, ensure_ascii=False),
            }
        ]
        return self.count(messages, tools), {"collapsed_messages": len(old)}

    async def _summarize_bounded(
        self, history: list[dict], summarize: Summarizer, emergency: bool
    ) -> str:
        text = json.dumps(history, ensure_ascii=False)
        tokens = self.encoding.encode(text, add_special_tokens=False)
        chunk_size = max(64, int((self.limit - self.summary_tokens - 512) / self.margin / 2))
        summary = ""

        # Каждый async-вызов завершается до следующего; отмена не оставляет фоновой работы.
        for offset in range(0, len(tokens), chunk_size):
            chunk = self.encoding.decode(tokens[offset : offset + chunk_size])
            prefix = self.encoding.decode(
                self.encoding.encode(summary, add_special_tokens=False)[:chunk_size]
            )
            summary = await summarize(f"<history>\n{prefix}\n{chunk}\n</history>", emergency)
            if not summary:
                raise ValueError("Summarizer returned no text")

        return summary
