import hashlib
import re
import time
import unicodedata
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pymupdf
import pymupdf.layout
import pymupdf4llm
import yaml
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from transformers import PreTrainedTokenizerBase

from core.config import (
    CLEANED_ROOT,
    CONVERTED_ROOT,
    INDEX_ROOT,
    WORK_ROOT,
    Settings,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from core.embeddings import Encoders, load_tokenizer


def normalize_markdown(text: str) -> str:
    # Приводим эквивалентные Unicode-символы к одному представлению.
    text = unicodedata.normalize("NFC", text).replace("\u00ad", "").replace("\u00a0", " ")

    # Очищаем пробелы и пустые строки, сохраняя SQL, регистр и Markdown-разметку.
    text = "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").splitlines())
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def prepare_corpus(settings: Settings, rebuild: bool = False) -> Path:
    # Сохраняем входные параметры и хеш PDF для проверки пригодности готовых файлов.
    pdf_path = settings.path(settings.corpus_pdf)
    if not pdf_path.exists():
        raise FileNotFoundError("Сначала скачайте данные командой lab2 download")
    with pdf_path.open("rb") as source:
        checksum = hashlib.file_digest(source, "sha256").hexdigest()
    identity = {
        "pdf_sha256": checksum,
        "pages": [settings.page_start, settings.page_end],
        "converter": "pymupdf4llm-0.3.4-layout-1.27.1",
        "header": False,
        "footer": False,
        "use_ocr": False,
    }
    output = CLEANED_ROOT
    manifest_path = output / "manifest.json"

    # Кеш пригоден, только если сохранились и очищенные, и исходные Markdown-файлы.
    if manifest_path.exists() and not rebuild:
        manifest = read_json(manifest_path)
        if all(manifest.get(key) == value for key, value in identity.items()) and all(
            (WORK_ROOT / row["markdown_path"]).exists()
            and row.get("converted_markdown_path")
            and (WORK_ROOT / row["converted_markdown_path"]).exists()
            and hashlib.sha256(
                (WORK_ROOT / row["converted_markdown_path"]).read_text(encoding="utf-8").encode()
            ).hexdigest()
            == row.get("converted_text_sha256")
            and hashlib.sha256(read_markdown(WORK_ROOT / row["markdown_path"]).encode()).hexdigest()
            == row["text_sha256"]
            for row in manifest["documents"]
        ):
            return manifest_path
    output.mkdir(parents=True, exist_ok=True)
    converted_output = CONVERTED_ROOT
    converted_output.mkdir(parents=True, exist_ok=True)
    documents = []

    # Готовый анализатор извлекает структуру PDF без OCR; номера страниц здесь нулевые.
    with pymupdf.open(pdf_path) as pdf:
        if settings.page_end > len(pdf):
            raise ValueError(f"PDF has only {len(pdf)} pages")
        toc = pdf.get_toc()
        pages = list(range(settings.page_start - 1, settings.page_end))
        converted = pymupdf4llm.to_markdown(
            pdf,
            pages=pages,
            page_chunks=True,
            header=False,
            footer=False,
            use_ocr=False,
            show_progress=False,
        )
        for page_index, page in zip(pages, converted, strict=True):
            # Фиксируем точный результат конвертера до собственной нормализации.
            doc_id = f"pg17-p{page_index + 1:04d}"
            converted_path = converted_output / f"{doc_id}.md"
            converted_path.write_text(page["text"], encoding="utf-8")

            # Из оглавления восстанавливаем происхождение текста на каждой странице.
            hierarchy = {}
            for level, heading, start_page in toc:
                if start_page > page_index + 1:
                    continue
                hierarchy = {key: value for key, value in hierarchy.items() if key < level}
                hierarchy[level] = normalize_markdown(heading).strip()
            section = " / ".join(hierarchy.values())
            headings = [heading for _, heading, number in toc if number == page_index + 1]
            body = normalize_markdown(page["text"])

            # Путь из оглавления остаётся в YAML; тело содержит только очищенный текст источника.
            md_path = output / f"{doc_id}.md"
            metadata = {
                "document_id": doc_id,
                "source": settings.corpus_url,
                "source_title": "PostgreSQL 17.11, русский перевод",
                "page": page_index + 1,
                "page_label": pdf[page_index].get_label(),
                "section": section,
                "page_headings": headings,
                "pdf_path": pdf_path.relative_to(WORK_ROOT).as_posix(),
                "markdown_path": md_path.relative_to(WORK_ROOT).as_posix(),
                "converted_markdown_path": converted_path.relative_to(WORK_ROOT).as_posix(),
                "converted_text_sha256": hashlib.sha256(page["text"].encode()).hexdigest(),
                "downloaded_at": datetime.fromtimestamp(pdf_path.stat().st_mtime, UTC).isoformat(),
                "processed_at": datetime.now(UTC).isoformat(),
                "text_sha256": hashlib.sha256(body.encode()).hexdigest(),
                "characters": len(body),
            }
            md_path.write_text(
                "---\n"
                + yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False)
                + "---\n\n"
                + body,
                encoding="utf-8",
            )
            documents.append(metadata)

    # Проверяем объём реальных страниц и только затем публикуем готовый manifest.
    text_pages = sum(row["characters"] >= 200 for row in documents)
    if text_pages < 50:
        raise ValueError(f"Only {text_pages} pages with at least 200 characters; need at least 50")
    write_json(
        manifest_path,
        {**identity, "text_pages": text_pages, "documents": documents},
    )
    write_jsonl(output / "documents.jsonl", documents)
    return manifest_path


def read_markdown(path: Path) -> str:
    # YAML нужен для происхождения документа, но не должен попадать в эмбеддинг текста.
    content = path.read_text(encoding="utf-8")
    return content.split("---\n", 2)[2].lstrip() if content.startswith("---\n") else content


def split_documents(
    settings: Settings, manifest: dict, tokenizer: PreTrainedTokenizerBase
) -> list[dict]:
    # Первый библиотечный сплиттер выделяет секции по h1–h3 и сохраняет их названия.
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=True,
    )
    chunks = []
    for document in manifest["documents"]:
        text = read_markdown(WORK_ROOT / document["markdown_path"])
        if not settings.include_headings and settings.splitter == "recursive":
            text = re.sub(r"^#{1,6}\s+.*$", "", text, flags=re.MULTILINE)
        if len(text.strip()) < 30:
            continue

        # В режиме recursive предварительное выделение Markdown-секций пропускается.
        sections = (
            markdown_splitter.split_text(text) if settings.splitter == "markdown_recursive" else []
        )
        parts = [(part.page_content, part.metadata) for part in sections] or [(text, {})]
        chunk_number = 0
        for section_text, headings in parts:
            # В текст возвращаем только ближайший заголовок, распознанный в самом Markdown.
            nearest_heading = headings.get(next(reversed(headings), ""), "")
            prefix = (
                nearest_heading + "\n\n" if settings.include_headings and nearest_heading else ""
            )
            prefix_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
            budget = settings.chunk_size - prefix_tokens - 8
            if budget <= settings.overlap:
                raise ValueError(
                    "Headings leave too little space; increase chunk_size or reduce overlap."
                )
            # Второй готовый сплиттер считает длину именно токенизатором Giga.
            recursive = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                tokenizer,
                chunk_size=budget,
                chunk_overlap=settings.overlap,
                separators=["\n\n", "\n", " ", ""],
            )
            for body in recursive.split_text(section_text):
                content = prefix + body
                token_count = len(tokenizer.encode(content, add_special_tokens=False))
                if token_count > settings.chunk_size:
                    raise ValueError(f"Chunk exceeds token budget: {token_count}")
                # Стабильный UUID позволяет повторно загружать чанк через upsert без дублей.
                identity = (
                    f"{document['source']}:{document['document_id']}:{chunk_number}:{content}"
                )
                chunks.append(
                    {
                        **document,
                        "chunk_id": str(uuid.uuid5(uuid.NAMESPACE_URL, identity)),
                        "chunk_number": chunk_number,
                        "headings": headings,
                        "text": content,
                        "tokens": token_count,
                    }
                )
                chunk_number += 1
    return chunks


def build_index(settings: Settings, rebuild: bool = False) -> Path:
    # При изменении параметров или документов перезаписываем тот же рабочий индекс.
    started = time.perf_counter()
    manifest_path = prepare_corpus(settings)
    manifest = read_json(manifest_path)
    output = INDEX_ROOT
    source_checksums = {row["document_id"]: row["text_sha256"] for row in manifest["documents"]}
    if (output / "index.json").exists() and not rebuild:
        info = read_json(output / "index.json")
        expected = ["chunks.jsonl", "dense.npy", "sparse.jsonl"]
        if all((output / name).exists() for name in expected):
            if (
                info["parameters"] == settings.index_parameters()
                and info["source_checksums"] == source_checksums
            ):
                print(f"Reusing index artifacts: {output.name}")
                return output
    output.mkdir(parents=True, exist_ok=True)
    (output / "index.json").unlink(missing_ok=True)

    # Один и тот же набор чанков используется обеими ветвями гибридного поиска.
    tokenizer = load_tokenizer(settings)
    chunks = split_documents(settings, manifest, tokenizer)
    if not chunks:
        raise ValueError("No chunks produced")
    encoders = Encoders(settings)
    try:
        dense = encoders.encode_dense([chunk["text"] for chunk in chunks])
        sparse = encoders.encode_sparse([chunk["text"] for chunk in chunks])
        if not np.isfinite(dense).all():
            raise ValueError("Embedding model produced non-finite vectors")
    finally:
        encoders.close()

    # Manifest записывается последним: его наличие означает успешную полную сборку.
    np.save(output / "dense.npy", dense)
    write_jsonl(output / "chunks.jsonl", chunks)
    write_jsonl(output / "sparse.jsonl", sparse)
    info = {
        "source_checksums": source_checksums,
        "corpus_manifest": manifest_path.relative_to(WORK_ROOT).as_posix(),
        "collection": settings.collection_prefix,
        "parameters": settings.index_parameters(),
        "settings": settings.model_dump(),
        "chunk_count": len(chunks),
        "dimension": dense.shape[1],
        "token_min": min(chunk["tokens"] for chunk in chunks),
        "token_max": max(chunk["tokens"] for chunk in chunks),
        "build_seconds": time.perf_counter() - started,
    }
    write_json(output / "index.json", info)
    return output


def load_artifacts(directory: Path) -> tuple[dict, list[dict], np.ndarray, list[dict]]:
    # Чанки и оба массива векторов имеют одинаковый порядок строк.
    return (
        read_json(directory / "index.json"),
        read_jsonl(directory / "chunks.jsonl"),
        np.load(directory / "dense.npy"),
        read_jsonl(directory / "sparse.jsonl"),
    )


def existing_index(settings: Settings) -> Path:
    # Чтение индекса не должно запускать подготовку документов или менять файлы.
    manifest_path = CLEANED_ROOT / "manifest.json"
    if not manifest_path.exists() or not (INDEX_ROOT / "index.json").exists():
        raise FileNotFoundError("Сначала выполните lab2 prepare, lab2 build и lab2 load")
    manifest = read_json(manifest_path)
    if manifest["pages"] != [settings.page_start, settings.page_end] or any(
        row["source"] != settings.corpus_url or row["pdf_path"] != settings.corpus_pdf
        for row in manifest["documents"]
    ):
        raise ValueError(
            "Источник или диапазон страниц изменился. Выполните lab2 prepare и lab2 build"
        )
    info = read_json(INDEX_ROOT / "index.json")
    checksums = {row["document_id"]: row["text_sha256"] for row in manifest["documents"]}
    if info["parameters"] != settings.index_parameters() or info["source_checksums"] != checksums:
        raise ValueError(
            "Параметры или документы изменились. "
            "Выполните lab2 build и lab2 load --drop-and-reindex"
        )
    return INDEX_ROOT
