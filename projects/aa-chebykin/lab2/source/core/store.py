import time
from contextlib import closing
from pathlib import Path

from qdrant_client import QdrantClient, models

from core.config import Settings, read_json, write_json
from core.embeddings import Encoders
from core.ingest import load_artifacts


def connect(settings: Settings) -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, timeout=120, check_compatibility=False)


def load_to_qdrant(settings: Settings, directory: Path, drop_and_reindex: bool = False) -> dict:
    # Проверяем согласованность чанков и двух представлений до изменения коллекции.
    started = time.perf_counter()
    info, chunks, dense, sparse = load_artifacts(directory)
    if not len(chunks) == len(dense) == len(sparse):
        raise ValueError("Artifact lengths differ. Rebuild the index before loading it.")
    collection = info["collection"]
    with closing(connect(settings)) as client:
        # Reset затрагивает только коллекцию лабораторной, указанную в конфигурации.
        exists = client.collection_exists(collection)
        if exists and drop_and_reindex:
            client.delete_collection(collection)
            exists = False
        if not exists:
            # Dense использует cosine/HNSW, а sparse получает актуальный IDF от Qdrant.
            client.create_collection(
                collection,
                vectors_config={
                    "dense": models.VectorParams(
                        size=info["dimension"], distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
                hnsw_config=models.HnswConfigDiff(
                    m=settings.hnsw_m,
                    ef_construct=settings.ef_construction,
                    full_scan_threshold=settings.full_scan_threshold,
                ),
                optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1),
            )
            client.create_payload_index(collection, "document_id", models.PayloadSchemaType.KEYWORD)
        # Общий UUID связывает оба вектора с текстом и происхождением одного чанка.
        points = [
            models.PointStruct(
                id=chunk["chunk_id"],
                payload=chunk,
                vector={
                    "dense": dense[index].tolist(),
                    "bm25": models.SparseVector(**sparse[index]),
                },
            )
            for index, chunk in enumerate(chunks)
        ]
        for offset in range(0, len(points), 64):
            client.upsert(collection, points[offset : offset + 64], wait=True)

        # Повторный upsert не должен увеличивать число логических документов.
        count = client.count(collection, exact=True).count
        if count != len(chunks):
            raise RuntimeError(
                f"Collection has {count} points, expected {len(chunks)}; use --drop-and-reindex"
            )
        result = {
            "collection": collection,
            "point_count": count,
            "load_seconds": time.perf_counter() - started,
            "qdrant_version": client.info().version,
            "hnsw": client.get_collection(collection).config.hnsw_config.model_dump(),
        }
    write_json(directory / "load.json", result)
    return result


class Retriever:
    def __init__(
        self, settings: Settings, directory: Path, encoders: Encoders | None = None
    ) -> None:
        # Один клиент и один набор encoder переиспользуются между поисковыми запросами.
        self.settings = settings
        self.info = read_json(directory / "index.json")
        self.encoders = encoders or Encoders(settings)
        self.client = connect(settings)

    def search(self, question: str, limit: int | None = None) -> list[dict]:
        limit = limit or self.settings.top_k
        if not question.strip():
            raise ValueError("Question must not be empty")
        mode = self.settings.vectorization

        # Векторизуем только ветви выбранного режима; cosine и BM25 score не складываем.
        dense = (
            self.encoders.encode_dense([question], query=True)[0].tolist()
            if mode != "sparse"
            else None
        )
        sparse = (
            models.SparseVector(**self.encoders.encode_sparse([question], query=True)[0])
            if mode != "dense"
            else None
        )
        search_params = models.SearchParams(hnsw_ef=self.settings.ef_search, exact=False)
        kwargs = {"collection_name": self.info["collection"], "limit": limit, "with_payload": True}
        if mode == "hybrid":
            # Оба поиска и RRF выполняются внутри Qdrant, без самописного слияния рангов.
            response = self.client.query_points(
                **kwargs,
                prefetch=[
                    models.Prefetch(
                        query=dense,
                        using="dense",
                        limit=max(self.settings.prefetch_k, limit),
                        params=search_params,
                    ),
                    models.Prefetch(
                        query=sparse, using="bm25", limit=max(self.settings.prefetch_k, limit)
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
            )
        else:
            response = self.client.query_points(
                **kwargs,
                query=dense if mode == "dense" else sparse,
                using="dense" if mode == "dense" else "bm25",
                search_params=search_params,
            )
        return [{**point.payload, "score": point.score} for point in response.points]

    def close(self) -> None:
        self.client.close()
        self.encoders.close()

    def search_documents(self, question: str, limit: int) -> list[dict]:
        # Несколько соседних чанков одной страницы не должны занять все места в контексте.
        candidates = self.search(question, limit=max(limit * 4, self.settings.prefetch_k))
        unique = {}
        for candidate in candidates:
            unique.setdefault(candidate["document_id"], candidate)
        return list(unique.values())[:limit]
