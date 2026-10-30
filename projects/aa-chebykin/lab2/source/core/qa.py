import re
import time
import tomllib
from pathlib import Path

import httpx
from jinja2 import Environment, StrictUndefined

from core.config import RESOURCES_ROOT, WORK_ROOT, Settings, read_json
from core.ingest import read_markdown
from core.store import Retriever


class RagAssistant:
    def __init__(self, settings: Settings, directory: Path) -> None:
        # Retriever и HTTP-клиент переиспользуются между вопросами одного процесса.
        self.settings = settings
        self.model_artifact = read_json(RESOURCES_ROOT / "assets.lock.json")["llm"]
        self.retriever = Retriever(settings, directory)
        self.http = httpx.Client(base_url=settings.llm_url, timeout=settings.llm_timeout)
        # TOML объединяет два обычных текстовых шаблона; Jinja2 подставляет переменные.
        with (RESOURCES_ROOT / "prompts.toml").open("rb") as source:
            prompts = tomllib.load(source)
        environment = Environment(undefined=StrictUndefined, autoescape=False)
        self.templates = {name: environment.from_string(text) for name, text in prompts.items()}

    def messages(self, question: str, contexts: list[dict]) -> list[dict]:
        return [
            {"role": "system", "content": self.templates["system"].render()},
            {
                "role": "user",
                "content": self.templates["user"].render(question=question, contexts=contexts),
            },
        ]

    def count_prompt_tokens(self, messages: list[dict]) -> int:
        # Chat template и токенизацию выполняет сам llama.cpp для загруженной Qwen.
        rendered = self.http.post(
            "/apply-template",
            json={
                "messages": messages,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        rendered.raise_for_status()
        tokenized = self.http.post(
            "/tokenize", json={"content": rendered.json()["prompt"], "add_special": True}
        )
        tokenized.raise_for_status()
        return len(tokenized.json()["tokens"])

    def ask(self, question: str) -> dict:
        started = time.perf_counter()

        # Ищем короткие чанки, затем расширяем найденные документы до исходных страниц.
        contexts = self.retriever.search_documents(question, self.settings.top_k)
        contexts = [
            {**context, "text": read_markdown(WORK_ROOT / context["markdown_path"])}
            for context in contexts
        ]
        retrieval_seconds = time.perf_counter() - started

        # Оставляем место для ответа и убираем наименее релевантные страницы при переполнении.
        messages = self.messages(question, contexts)
        budget = self.settings.llm_context_size - self.settings.max_answer_tokens - 64
        prompt_tokens = self.count_prompt_tokens(messages)
        while prompt_tokens > budget and contexts:
            contexts.pop()
            messages = self.messages(question, contexts)
            prompt_tokens = self.count_prompt_tokens(messages)
        if prompt_tokens > budget:
            raise ValueError("Question exceeds the language model context budget. Shorten it.")
        # Генерация идёт через локальный REST; параметры sampling записываются в отчёт.
        response = self.http.post(
            "/v1/chat/completions",
            json={
                "model": self.settings.llm_model,
                "messages": messages,
                "temperature": self.settings.temperature,
                "top_p": self.settings.llm_top_p,
                "top_k": self.settings.llm_top_k,
                "min_p": self.settings.llm_min_p,
                "max_tokens": self.settings.max_answer_tokens,
                "seed": 42,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        response.raise_for_status()
        completion = response.json()
        answer = completion["choices"][0]["message"]["content"] or ""

        # Номера ссылок сопоставляем только с действительно переданными фрагментами.
        used = sorted({int(value) for value in re.findall(r"\[(\d+)\]", answer)})
        citations = [
            {
                "id": number,
                "source": context["source"],
                "page": context["page"],
                "page_label": context["page_label"],
                "section": context["section"],
                "document_id": context["document_id"],
                "chunk_id": context["chunk_id"],
                "snippet": context["text"],
            }
            for number, context in enumerate(contexts, 1)
            if number in used
        ]
        # Сохраняем точный промпт и usage, чтобы ответ можно было разобрать на защите.
        return {
            "question": question,
            "answer": answer,
            "citations": citations,
            "invalid_citations": [number for number in used if not 1 <= number <= len(contexts)],
            "context_pages": [context["page"] for context in contexts],
            "retrieval_seconds": retrieval_seconds,
            "total_seconds": time.perf_counter() - started,
            "usage": completion.get("usage", {}),
            "finish_reason": completion["choices"][0]["finish_reason"],
            "model": completion.get("model", self.settings.llm_model),
            "model_artifact": self.model_artifact,
            "collection": self.retriever.info["collection"],
            "prompt_tokens_counted": prompt_tokens,
            "messages": messages,
        }

    def close(self) -> None:
        self.http.close()
        self.retriever.close()
