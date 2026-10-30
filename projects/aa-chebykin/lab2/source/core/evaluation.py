import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from core.config import (
    RESOURCES_ROOT,
    RESULTS_ROOT,
    WORK_ROOT,
    Settings,
    read_json,
    write_json,
)
from core.ingest import build_index
from core.qa import RagAssistant
from core.store import Retriever, load_to_qdrant


def retrieval_metrics(ranked: list[str], relevant: set[str], ks: tuple[int, ...] = (5, 10)) -> dict:
    # Единица оценки — уникальная исходная страница, а не меняющийся номер чанка.
    if not relevant:
        raise ValueError("Relevance labels must not be empty")
    if len(set(ranked)) != len(ranked):
        raise ValueError("Ranked documents must be unique")
    # Для Recall знаменатель — число эталонов, для Precision — фиксированное k.
    result = {}
    for k in ks:
        hits = len(set(ranked[:k]) & relevant)
        result[f"recall@{k}"] = hits / len(relevant)
        result[f"precision@{k}"] = hits / k
    result["mrr@10"] = next(
        (1 / rank for rank, doc in enumerate(ranked[:10], 1) if doc in relevant), 0.0
    )
    return result


def evaluate_retrieval(settings: Settings, directory: Path, questions: Path | None = None) -> dict:
    # Эталонная разметка должна ссылаться на страницы подготовленного корпуса.
    dataset = read_json(questions or RESOURCES_ROOT / "questions.json")
    retriever = Retriever(settings, directory)
    available = {
        document["document_id"]
        for document in read_json(WORK_ROOT / retriever.info["corpus_manifest"])["documents"]
    }
    rows = []
    try:
        for item in dataset:
            relevant = set(item["relevant_documents"])
            if not relevant <= available:
                raise ValueError(
                    f"Unknown relevance labels for {item['id']}: {relevant - available}"
                )
            # Время включает векторизацию вопроса и поиск; первый запрос загружает модель.
            started = time.perf_counter()
            hits = retriever.search_documents(item["question"], 10)
            elapsed = time.perf_counter() - started
            ranked = [hit["document_id"] for hit in hits]
            metrics = retrieval_metrics(ranked, relevant)
            rows.append(
                {
                    **item,
                    "ranked_documents": ranked,
                    "ranked_chunks": [hit["chunk_id"] for hit in hits],
                    "metrics": metrics,
                    "seconds": elapsed,
                }
            )
    finally:
        retriever.close()

    # Сохраняем и средние значения, и подробности каждого вопроса для проверки ошибок.
    metric_names = rows[0]["metrics"]
    return {
        "evaluated_at": datetime.now(UTC).isoformat(),
        "collection": settings.collection_prefix,
        "settings": settings.model_dump(),
        "metrics": {
            name: statistics.mean(row["metrics"][name] for row in rows) for name in metric_names
        },
        "latency_mean_seconds": statistics.mean(row["seconds"] for row in rows),
        "latency_warm_mean_seconds": statistics.mean(row["seconds"] for row in rows[1:]),
        "questions": rows,
    }


def run_experiments(settings: Settings, variants_path: Path | None = None) -> dict:
    variants = read_json(variants_path or RESOURCES_ROOT / "experiments.json")
    results = []
    details = []

    # Каждый вариант занимает тот же каталог и коллекцию; результаты сохраняются в отчёте.
    for variant in variants:
        values = settings.model_dump() | variant["overrides"]
        current = Settings.model_validate(values)
        print(f"Experiment: {variant['name']}", flush=True)
        directory = build_index(current)
        loading = load_to_qdrant(current, directory, drop_and_reindex=True)
        result = evaluate_retrieval(current, directory)
        result.update(
            {
                "name": variant["name"],
                "build": read_json(directory / "index.json"),
                "load": loading,
                "artifact_bytes": sum(
                    path.stat().st_size for path in directory.iterdir() if path.is_file()
                ),
            }
        )
        details.append(result)
        results.append(
            {
                key: result[key]
                for key in [
                    "name",
                    "collection",
                    "metrics",
                    "latency_mean_seconds",
                    "latency_warm_mean_seconds",
                    "artifact_bytes",
                ]
            }
            | {
                "chunk_count": result["build"]["chunk_count"],
                "build_seconds": result["build"]["build_seconds"],
            }
        )
    # Правило выбора фиксировано: Recall, затем MRR, затем размер артефактов.
    hybrid = [row for row in results if row["name"].startswith("hybrid_")]
    best = max(
        hybrid,
        key=lambda row: (
            row["metrics"]["recall@5"],
            row["metrics"]["mrr@10"],
            -row["artifact_bytes"],
        ),
    )
    selected = next(variant for variant in variants if variant["name"] == best["name"])
    # После сравнения оставляем рабочий индекс лучшего hybrid, а не последнего варианта.
    chosen = Settings.model_validate(settings.model_dump() | selected["overrides"])
    load_to_qdrant(chosen, build_index(chosen), drop_and_reindex=True)
    summary = {
        "results": results,
        "best_hybrid": best["name"],
        "selection_rule": "Recall@5, then MRR@10, then fewer artifact bytes; exploratory set",
        "best_overrides": selected["overrides"],
    }
    write_json(RESULTS_ROOT / "retrieval.json", {**summary, "variants": details})
    return summary


def evaluate_qa(settings: Settings, directory: Path) -> dict:
    # Для смыслового разбора сохраняем настоящий ответ и промпт каждого вопроса.
    assistant = RagAssistant(settings, directory)
    results = []
    try:
        for item in read_json(RESOURCES_ROOT / "questions.json"):
            print(f"QA: {item['id']}", flush=True)
            response = assistant.ask(item["question"])
            # Ключевые слова показывают покрытие, но не доказывают фактическую правильность.
            text = response["answer"].casefold()
            groups = item["answer_keywords"]
            coverage = sum(
                any(word.casefold() in text for word in group) for group in groups
            ) / len(groups)
            cited = {citation["document_id"] for citation in response["citations"]}
            valid = bool(cited) and not response["invalid_citations"]
            results.append(
                {
                    **item,
                    **response,
                    "keyword_coverage": coverage,
                    "citation_format_ok": valid,
                    "gold_citation_hit": bool(cited & set(item["relevant_documents"])),
                }
            )
    finally:
        assistant.close()
    result = {
        "settings": settings.model_dump(),
        "collection": settings.collection_prefix,
        "keyword_coverage_mean": statistics.mean(row["keyword_coverage"] for row in results),
        "citation_format_rate": statistics.mean(row["citation_format_ok"] for row in results),
        "gold_citation_hit_rate": statistics.mean(row["gold_citation_hit"] for row in results),
        "truncated_answers": sum(row["finish_reason"] == "length" for row in results),
        "note": "Keyword coverage and citation checks are diagnostics, not a factuality judge.",
        "questions": results,
    }
    write_json(RESULTS_ROOT / "qa.json", result)
    return result
