import hashlib
import os
import ssl
import subprocess
import zipfile
from pathlib import Path

import httpx
import truststore
from fastembed import SparseTextEmbedding
from huggingface_hub import hf_hub_download, snapshot_download

from core.config import (
    CACHE_ROOT,
    LLAMA_ROOT,
    LLM_MODEL_ROOT,
    LOGS_ROOT,
    RESOURCES_ROOT,
    ROOT,
    Settings,
    read_json,
    write_json,
)


def verify_hash(path: Path, expected: str) -> None:
    # Проверяем содержимое, а не только наличие или размер файла.
    with path.open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    if actual != expected:
        raise RuntimeError(f"SHA256 mismatch for {path}; remove it and download again.")


def download(url: str, target: Path) -> Path:
    # Повторный запуск использует готовый файл; незавершённая загрузка имеет суффикс .part.
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target

    temporary = target.with_suffix(target.suffix + ".part")
    with httpx.stream(
        "GET",
        url,
        follow_redirects=True,
        timeout=180,
        verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
    ) as response:
        response.raise_for_status()
        with temporary.open("wb") as output:
            for block in response.iter_bytes(1024 * 1024):
                output.write(block)
    temporary.replace(target)
    return target


def download_assets(settings: Settings) -> None:
    # Ревизии моделей и контрольные суммы лежат в одном lock-файле.
    assets = read_json(RESOURCES_ROOT / "assets.lock.json")
    truststore.inject_into_ssl()

    # Сохраняем оригинальный PDF без изменений и проверяем контрольную сумму источника.
    pdf = download(assets["corpus"]["url"], settings.path(settings.corpus_pdf))
    verify_hash(pdf, assets["corpus"]["sha256"])
    write_json(pdf.with_suffix(".json"), assets["corpus"])

    # Hub сам скачивает набор файлов encoder и отдельный GGUF по точным commit SHA.
    snapshot_download(
        assets["embeddings"]["repository"],
        revision=assets["embeddings"]["revision"],
        local_dir=settings.path(settings.embedding_model),
        ignore_patterns=["*.bin", "*.onnx", "*.gguf", "*.h5"],
    )
    model = Path(
        hf_hub_download(
            assets["llm"]["repository"],
            assets["llm"]["filename"],
            revision=assets["llm"]["revision"],
            local_dir=LLM_MODEL_ROOT,
        )
    )
    verify_hash(model, assets["llm"]["sha256"])

    # Для BM25 нужны только готовые языковые ресурсы FastEmbed, а не нейросетевые веса.
    SparseTextEmbedding(
        "Qdrant/bm25",
        language=settings.bm25_language,
        cache_dir=str(CACHE_ROOT / "fastembed"),
    )

    # Уже установленную версию движка не распаковываем поверх работающих DLL.
    release = assets["llama_cpp"]["release"]
    release_path = LLAMA_ROOT / "release.json"
    installed = (
        (LLAMA_ROOT / "llama-server.exe").exists()
        and release_path.exists()
        and read_json(release_path)["tag"] == release
    )
    if not installed:
        # Установленному движку архивы не нужны: повторный download не возвращает удалённые ZIP.
        for filename, checksum in assets["llama_cpp"]["files"].items():
            archive = download(
                f"https://github.com/ggml-org/llama.cpp/releases/download/{release}/{filename}",
                CACHE_ROOT / "downloads" / filename,
            )
            verify_hash(archive, checksum)
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(LLAMA_ROOT)
    write_json(release_path, {"tag": release, "files": assets["llama_cpp"]["files"]})


def start_llm(settings: Settings, gpu_layers: int = 14, port: int = 8080) -> int:
    # Пути вычисляем от проекта, поэтому команду можно вызывать из любого каталога.
    assets = read_json(RESOURCES_ROOT / "assets.lock.json")
    engine = LLAMA_ROOT / "llama-server.exe"
    model = LLM_MODEL_ROOT / assets["llm"]["filename"]
    if not engine.exists() or not model.exists():
        raise FileNotFoundError("Сначала выполните lab2 download")

    # Один процесс llama.cpp обслуживает запросы через локальный совместимый REST API.
    arguments = [
        str(engine),
        "-m",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--alias",
        settings.llm_model,
        "--ctx-size",
        str(settings.llm_context_size),
        "--parallel",
        "1",
        "--gpu-layers",
        str(gpu_layers),
        "--batch-size",
        "256",
        "--ubatch-size",
        "128",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--flash-attn",
        "on",
        "--reasoning-budget",
        "0",
        "--jinja",
        "--threads",
        "6",
        "--fit",
        "off",
    ]

    # На Windows фоновый сервер не создаёт дополнительное окно консоли.
    logs = LOGS_ROOT
    logs.mkdir(parents=True, exist_ok=True)
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with (
        (logs / "llama.stdout.log").open("wb") as output,
        (logs / "llama.stderr.log").open("wb") as errors,
    ):
        process = subprocess.Popen(
            arguments, cwd=ROOT, stdout=output, stderr=errors, creationflags=flags
        )
    (logs / "llama.pid").write_text(str(process.pid), encoding="utf-8")
    return process.pid
