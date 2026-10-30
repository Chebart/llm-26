import importlib.metadata
import platform
import time
from contextlib import closing

import httpx
import numpy as np
import psutil
import torch

from core.config import RESOURCES_ROOT, RESULTS_ROOT, Settings, read_json, write_json
from core.embeddings import Encoders, load_tokenizer
from core.ingest import existing_index, load_artifacts
from core.store import connect


def verify_models(settings: Settings) -> dict:
    # Используем реальные веса и токенизатор, а не фиктивные векторы тестового стенда.
    tokenizer = load_tokenizer(settings)
    encoders = Encoders(settings)
    try:
        # Смысловой запрос должен быть ближе к описанию ROLLBACK, чем к CREATE TABLE.
        query = encoders.encode_dense(["Как отменить изменения транзакции?"], query=True)
        documents = encoders.encode_dense(
            [
                "Команда ROLLBACK отменяет изменения транзакции.",
                "Команда CREATE TABLE создаёт новую таблицу.",
            ]
        )
        similarities = (query @ documents.T)[0]
        assert similarities[0] > similarities[1]
        assert np.allclose(np.linalg.norm(documents, axis=1), 1, atol=1e-4)

        # Проверяем двунаправленность: изменение суффикса влияет на первый токен.
        model = encoders.dense_model()[0].auto_model
        inputs = tokenizer(
            ["Начало транзакции SELECT", "Начало транзакции ROLLBACK"],
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        assert inputs["input_ids"][0, 0] == inputs["input_ids"][1, 0]
        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state
        first_token_delta = (hidden[0, 0] - hidden[1, 0]).float().norm().item()
        assert first_token_delta > 1e-4, (
            "First token must depend on later tokens in a bidirectional model"
        )
        example = "Чем WHERE отличается от HAVING?"

        # Фиксируем устройство, версии, наблюдаемые нормы и настоящий пример токенизации.
        report = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_logical": psutil.cpu_count(),
            "cpu_physical": psutil.cpu_count(logical=False),
            "ram_bytes": psutil.virtual_memory().total,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "gpu_vram_bytes": (
                torch.cuda.get_device_properties(0).total_memory
                if torch.cuda.is_available()
                else None
            ),
            "torch_cuda": torch.version.cuda,
            "packages": {
                name: importlib.metadata.version(name)
                for name in [
                    "torch",
                    "sentence-transformers",
                    "transformers",
                    "qdrant-client",
                    "fastembed",
                    "pymupdf4llm",
                    "pymupdf",
                    "pymupdf-layout",
                ]
            },
            "embedding_shape": list(documents.shape),
            "embedding_norms": np.linalg.norm(documents, axis=1).tolist(),
            "semantic_test_cosines": similarities.tolist(),
            "first_token_delta_after_suffix_change": first_token_delta,
            "all_attention_layers_bidirectional": all(
                not layer.self_attn.is_causal for layer in model.layers
            ),
            "tokenizer_class": tokenizer.__class__.__name__,
            "vocab_size": tokenizer.vocab_size,
            "token_example": {
                "text": example,
                "tokens": tokenizer.tokenize(example),
                "ids": tokenizer.encode(example, add_special_tokens=False),
            },
            "peak_embedding_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
            if torch.cuda.is_available()
            else 0,
        }
        write_json(RESULTS_ROOT / "environment.json", report)
        print(report)
    finally:
        encoders.close()

    return report


def check_api(settings: Settings) -> dict:
    # Проверяем согласованность артефактов и лимиты всех чанков до HTTP-запросов.
    directory = existing_index(settings)
    info, chunks, dense, sparse = load_artifacts(directory)
    tokenizer = load_tokenizer(settings)
    assert len(chunks) == len(dense) == len(sparse) == info["chunk_count"]
    assert len({row["chunk_id"] for row in chunks}) == len(chunks)
    assert np.isfinite(dense).all()
    assert np.allclose(np.linalg.norm(dense, axis=1), 1, atol=1e-3)
    assert all(
        len(tokenizer.encode(row["text"], add_special_tokens=False)) <= settings.chunk_size
        for row in chunks
    )
    # Здесь нужен именно серверный Qdrant: локальный режим не проверяет построение HNSW.
    with closing(connect(settings)) as client:
        collection = client.get_collection(info["collection"])
        assert client.count(info["collection"], exact=True).count == len(chunks)
        assert collection.indexed_vectors_count > 0
        storage = {
            "collection": info["collection"],
            "status": collection.status.value,
            "qdrant_version": client.info().version,
            "points_count": collection.points_count,
            "indexed_vectors_count": collection.indexed_vectors_count,
            "hnsw": collection.config.hnsw_config.model_dump(),
        }

    # Сверяем реально загруженный GGUF, а не только имя из конфигурации клиента.
    with httpx.Client(base_url=settings.llm_url, timeout=30) as model_client:
        response = model_client.get("/props")
        response.raise_for_status()
        properties = response.json()
        model_runtime = {
            key: properties[key] for key in ["model_path", "model_alias", "build_info"]
        }
        assert model_runtime["model_alias"] == settings.llm_model
        assert model_runtime["model_path"].endswith(
            read_json(RESOURCES_ROOT / "assets.lock.json")["llm"]["filename"]
        )

    # Один вопрос с ответом и два отрицательных примера проверяют разные ветви поведения.
    cases = [
        {
            "question": (
                "Как выбрать строки только из родительской таблицы cities, "
                "исключив унаследовавшую её capitals?"
            ),
            "expected": "Use FROM ONLY cities and cite the documentation.",
        },
        {
            "question": (
                "Как настроить логическую репликацию PostgreSQL? Какие параметры нужно задать?"
            ),
            "expected": "Report missing replication instructions in this corpus.",
        },
        {
            "question": "Как испечь яблочный пирог?",
            "expected": "Refuse the unrelated question rather than inventing a recipe.",
        },
    ]
    # Ответы сохраняются целиком, чтобы смысл отказов можно было проверить вручную.
    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=300) as client:
        health = client.get("/health")
        health.raise_for_status()
        invalid = client.post("/ask", json={"question": ""})
        assert invalid.status_code == 422
        results = []
        for case in cases:
            print(case["question"], flush=True)
            started = time.perf_counter()
            response = client.post("/ask", json={"question": case["question"]})
            response.raise_for_status()
            result = response.json()
            results.append(
                {
                    **case,
                    "status_code": response.status_code,
                    "http_seconds": time.perf_counter() - started,
                    "response": result,
                }
            )
        report = {
            "collection": settings.collection_prefix,
            "storage": storage,
            "model_runtime": model_runtime,
            "health": health.json(),
            "empty_question_status": invalid.status_code,
            "cases": results,
        }
        write_json(RESULTS_ROOT / "api_smoke.json", report)
        assert "only" in results[0]["response"]["answer"].casefold()
        assert results[0]["response"]["citations"]
        assert not results[0]["response"]["invalid_citations"]
        write_json(RESULTS_ROOT / "example.json", results[0]["response"])
        print("API smoke checks passed; inspect the saved negative-case answers.")

    return report
