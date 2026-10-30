import json
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = ROOT / "data"
RESOURCES_ROOT = ROOT / "resources"
RESULTS_ROOT = ROOT / "results"
CONVERTED_ROOT = WORK_ROOT / "02_converted"
CLEANED_ROOT = WORK_ROOT / "03_cleaned"
INDEX_ROOT = WORK_ROOT / "04_index"
CACHE_ROOT = WORK_ROOT / ".cache"
LLAMA_ROOT = WORK_ROOT / "llama_cpp"
LLM_MODEL_ROOT = WORK_ROOT / "models/qwen3-4b"
LOGS_ROOT = WORK_ROOT / "logs"


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Источник данных и физические номера страниц оригинального PDF.
    corpus_pdf: str
    corpus_url: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)

    # Разбиение и оба представления текста; токены чанков считаются токенизатором Giga.
    splitter: Literal["recursive", "markdown_recursive"] = "markdown_recursive"
    chunk_size: int = Field(default=200, ge=100, le=1000)
    overlap: int = Field(default=30, ge=0)
    include_headings: bool = True
    vectorization: Literal["dense", "sparse", "hybrid"] = "hybrid"
    embedding_model: str
    embedding_repo: str
    embedding_revision: str
    embedding_device: Literal["cpu", "cuda"] = "cuda"
    embedding_batch_size: int = Field(default=4, ge=1)
    embedding_max_tokens: int = Field(default=1024, ge=128)
    query_instruction: str
    bm25_language: str = "russian"
    bm25_k: float = Field(default=1.2, gt=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    bm25_avg_len: float = Field(default=256, gt=0)

    # Серверный индекс и получение кандидатов для поиска.
    qdrant_url: str = "http://127.0.0.1:6333"
    collection_prefix: str = "postgres17"
    hnsw_m: int = Field(default=16, ge=4)
    ef_construction: int = Field(default=100, ge=4)
    ef_search: int = Field(default=64, ge=1)
    full_scan_threshold: int = Field(default=10, ge=10)
    top_k: int = Field(default=3, ge=1)
    prefetch_k: int = Field(default=40, ge=10)

    # У локального генератора свой токенизатор и отдельный бюджет контекста.
    llm_url: str = "http://127.0.0.1:8080"
    llm_model: str = "qwen3-4b-instruct-2507"
    llm_context_size: int = Field(default=4096, ge=1024)
    max_answer_tokens: int = Field(default=512, ge=32)
    llm_timeout: float = 240
    temperature: float = Field(default=0.7, ge=0, le=2)
    llm_top_p: float = Field(default=0.8, gt=0, le=1)
    llm_top_k: int = Field(default=20, ge=0)
    llm_min_p: float = Field(default=0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_limits(self) -> "Settings":
        # Некорректные параметры останавливаем до загрузки моделей или создания коллекции.
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        if self.chunk_size + 2 > self.embedding_max_tokens:
            raise ValueError("embedding_max_tokens must accommodate chunk_size and special tokens")
        if self.page_end < self.page_start:
            raise ValueError("page_end must be at least page_start")
        if self.prefetch_k < self.top_k:
            raise ValueError("prefetch_k must be at least top_k")
        if self.max_answer_tokens >= self.llm_context_size:
            raise ValueError("max_answer_tokens must be smaller than llm_context_size")
        return self

    def path(self, relative: str) -> Path:
        # Относительные пути данных разрешаются внутри рабочего каталога data.
        path = Path(relative)
        return path if path.is_absolute() else WORK_ROOT / path

    def index_parameters(self) -> dict:
        # Сверяем параметры сборки, чтобы не использовать индекс с другими настройками.
        names = [
            "splitter",
            "chunk_size",
            "overlap",
            "include_headings",
            "embedding_repo",
            "embedding_revision",
            "embedding_model",
            "embedding_device",
            "embedding_max_tokens",
            "bm25_language",
            "bm25_k",
            "bm25_b",
            "bm25_avg_len",
            "hnsw_m",
            "ef_construction",
            "full_scan_threshold",
        ]
        return {name: getattr(self, name) for name in names}


def load_settings(path: Path | None = None, overrides: dict | None = None) -> Settings:
    # TOML задаёт базу, JSON overrides заменяет отдельные поля эксперимента.
    with (path or RESOURCES_ROOT / "config.toml").open("rb") as source:
        data = tomllib.load(source)
    data.update(overrides or {})
    return Settings.model_validate(data)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    # Атомарная замена не оставляет наполовину записанный отчёт при прерывании процесса.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, values: list[dict]) -> None:
    # Одна строка соответствует одному чанку или sparse-вектору; порядок строк важен.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values), encoding="utf-8"
    )
    temporary.replace(path)
