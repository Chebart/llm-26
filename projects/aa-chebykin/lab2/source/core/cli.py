import json
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

import httpx
import typer
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.assets import download_assets, start_llm
from core.checks import check_api, verify_models
from core.config import RESULTS_ROOT, Settings, load_settings, read_json, write_json
from core.evaluation import evaluate_qa, evaluate_retrieval, run_experiments
from core.ingest import build_index, existing_index, prepare_corpus
from core.qa import RagAssistant
from core.store import Retriever, load_to_qdrant

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


def configured(config: Path | None, overrides: Path | None) -> Settings:
    return load_settings(config, read_json(overrides) if overrides else None)


@app.command("download", help="Скачать PDF, модели и готовый движок llama.cpp.")
def download(config: Path | None = None) -> None:
    download_assets(load_settings(config))


@app.command("start-llm", help="Запустить локальную Qwen в фоне; журналы находятся в data/logs.")
def launch_llm(config: Path | None = None, gpu_layers: int = 14, port: int = 8080) -> None:
    print(f"llama.cpp PID: {start_llm(load_settings(config), gpu_layers, port)}")


@app.command("verify", help="Проверить размерность, нормы и двунаправленность Giga.")
def verify(config: Path | None = None) -> None:
    verify_models(load_settings(config))


@app.command("check-api", help="Проверить индекс и реальные HTTP-запросы к работающему сервису.")
def smoke(config: Path | None = None, overrides: Path | None = None) -> None:
    check_api(configured(config, overrides))


@app.command(help="Сохранить сырой и очищенный Markdown для всех страниц корпуса.")
def prepare(config: Path | None = None, rebuild: bool = False) -> None:
    print(prepare_corpus(load_settings(config), rebuild))


@app.command(help="Построить чанки и dense/sparse-векторы на диске.")
def build(config: Path | None = None, overrides: Path | None = None, rebuild: bool = False) -> None:
    print(build_index(configured(config, overrides), rebuild))


@app.command(help="Загрузить готовые артефакты в Qdrant или выполнить reset коллекции.")
def load(
    config: Path | None = None, overrides: Path | None = None, drop_and_reindex: bool = False
) -> None:
    settings = configured(config, overrides)
    print(
        json.dumps(load_to_qdrant(settings, existing_index(settings), drop_and_reindex), indent=2)
    )


@app.command(help="Оценить retrieval; с --qa также вызвать локальную LLM.")
def evaluate(config: Path | None = None, overrides: Path | None = None, qa: bool = False) -> None:
    settings = configured(config, overrides)
    directory = existing_index(settings)
    if qa:
        result = evaluate_qa(settings, directory)
        print({key: value for key, value in result.items() if key not in ["questions", "settings"]})
    else:
        result = evaluate_retrieval(settings, directory)
        write_json(RESULTS_ROOT / "retrieval_check.json", result)
        print(json.dumps(result["metrics"], indent=2))


@app.command(help="Сравнить варианты индекса и сохранить один общий retrieval-отчёт.")
def experiments(config: Path | None = None, variants: Path | None = None) -> None:
    print(json.dumps(run_experiments(load_settings(config), variants), indent=2))


@app.command(help="Показать найденные страницы без генерации ответа LLM.")
def search(question: str, config: Path | None = None, overrides: Path | None = None) -> None:
    # Поиск не требует LLM-сервера; encoder освобождается после запроса.
    settings = configured(config, overrides)
    retriever = Retriever(settings, existing_index(settings))
    try:
        print(
            json.dumps(
                retriever.search_documents(question, settings.top_k), ensure_ascii=False, indent=2
            )
        )
    finally:
        retriever.close()


@app.command(help="Ответить на один вопрос с источниками через весь RAG-пайплайн.")
def ask(
    question: str,
    config: Path | None = None,
    overrides: Path | None = None,
    output: Path | None = None,
) -> None:
    # Один вопрос проходит весь RAG-пайплайн, а точный ответ можно сохранить в JSON.
    settings = configured(config, overrides)
    assistant = RagAssistant(settings, existing_index(settings))
    try:
        answer = assistant.ask(question)
    finally:
        assistant.close()
    if output:
        write_json(output, answer)

    # В консоли показываем человеку ответ и проверяемые ссылки на страницы PDF.
    print(answer["answer"])
    for citation in answer["citations"]:
        print(f"[{citation['id']}] PDF page {citation['page']}: {citation['section']}")
        print(f"{citation['source']}#page={citation['page']}")


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


def create_api(settings: Settings, directory: Path) -> FastAPI:
    # Запросы выполняются последовательно, чтобы encoder не загружался конкурентно.
    inference_lock = Lock()

    @asynccontextmanager
    async def lifespan(api: FastAPI):
        # Один помощник и его ресурсы живут столько же, сколько процесс FastAPI.
        api.state.assistant = RagAssistant(settings, directory)
        yield
        api.state.assistant.close()

    api = FastAPI(title="Local PostgreSQL RAG", lifespan=lifespan)

    @api.get("/health")
    def health() -> dict:
        return {"status": "ok", "collection": settings.collection_prefix}

    @api.post("/ask")
    def answer(payload: Question) -> dict:
        try:
            with inference_lock:
                return api.state.assistant.ask(payload.question)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except httpx.HTTPError as error:
            raise HTTPException(
                status_code=503, detail="Local model server is unavailable"
            ) from error

    return api


@app.command(help="Запустить RAG REST API.")
def serve(config: Path | None = None, overrides: Path | None = None, port: int = 8000) -> None:
    settings = configured(config, overrides)
    uvicorn.run(create_api(settings, existing_index(settings)), host="127.0.0.1", port=port)


if __name__ == "__main__":
    app()
