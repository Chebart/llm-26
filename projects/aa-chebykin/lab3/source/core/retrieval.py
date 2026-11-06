import hashlib
import json
import os
import re
import unicodedata
import uuid
from importlib.metadata import version
from typing import Any

import httpx
from qdrant_client import QdrantClient, models

from core.config import Settings

try:
    import pymupdf4llm
    import torch
    from fastembed import SparseTextEmbedding
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer
except ImportError:
    pymupdf4llm = None
    SparseTextEmbedding = None
    MarkdownHeaderTextSplitter = None
    RecursiveCharacterTextSplitter = None
    SentenceTransformer = None


def normalize_markdown(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\u00ad", "").replace("\u00a0", " ")
    text = "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").splitlines())

    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


class Retriever:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.config = settings.retrieval
        self.directory = settings.data / "corpus"
        self._dense: Any = None
        self._sparse: Any = None
        self._client: QdrantClient | None = None

    def _require_dependencies(self) -> None:
        if SentenceTransformer is None:
            raise RuntimeError("Install retrieval dependencies: uv sync --extra rag --extra dev")

    def _encoders(self, offline: bool = True) -> tuple[Any, Any]:
        self._require_dependencies()

        if self._dense is None:
            torch.set_num_threads(self.config["embedding_threads"])
            self._dense = SentenceTransformer(
                self.config["embedding_repo"],
                revision=self.config["embedding_revision"],
                cache_folder=str(self.settings.data / "models"),
                device=self.config["embedding_device"],
                trust_remote_code=True,
                local_files_only=offline,
            )
            self._dense.max_seq_length = 1024

        if self._sparse is None:
            self._sparse = SparseTextEmbedding(
                "Qdrant/bm25",
                language="russian",
                k=1.2,
                b=0.75,
                avg_len=256,
                cache_dir=str(self.settings.data / "models/bm25"),
                local_files_only=offline,
            )

        return self._dense, self._sparse

    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(
                url=os.environ.get("LAB3_QDRANT_URL", "http://127.0.0.1:6335"),
                timeout=60,
            )

        return self._client

    def prepare(self, rebuild: bool = False) -> dict:
        self._require_dependencies()
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "04_chunks").mkdir(exist_ok=True)
        pdf = self.directory / "01_pdf/postgres-17.11-ru.pdf"
        pdf.parent.mkdir(exist_ok=True)

        if not pdf.exists():
            with httpx.stream(
                "GET", self.config["corpus_url"], follow_redirects=True, timeout=120
            ) as response:
                response.raise_for_status()
                temporary = pdf.with_suffix(".part")

                with temporary.open("wb") as target:
                    for chunk in response.iter_bytes():
                        target.write(chunk)

                temporary.replace(pdf)

        checksum = hashlib.sha256(pdf.read_bytes()).hexdigest()

        if checksum != self.config["corpus_sha256"]:
            raise ValueError(
                "PDF checksum differs from resources/config.toml; verify the input PDF"
            )

        identity = {
            "pdf_sha256": checksum,
            "config": self.config,
            "converter": version("pymupdf4llm"),
            "splitter": version("langchain-text-splitters"),
        }
        manifest = self.directory / "04_chunks/manifest.json"

        if manifest.exists() and not rebuild:
            saved = json.loads(manifest.read_text(encoding="utf-8"))

            if saved["identity"] != identity:
                raise ValueError("Corpus configuration changed; use prepare --rebuild explicitly")

            return saved

        dense, _ = self._encoders(offline=False)
        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            dense.tokenizer,
            chunk_size=self.config["chunk_tokens"],
            chunk_overlap=self.config["overlap_tokens"],
        )
        headings = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
            strip_headers=False,
        )
        converted = self.directory / "02_converted"
        cleaned = self.directory / "03_cleaned"
        converted.mkdir(exist_ok=True)
        cleaned.mkdir(exist_ok=True)
        chunks = []

        for page in range(self.config["page_start"], self.config["page_end"] + 1):
            document_id = f"pg17-p{page:04d}"
            raw = pymupdf4llm.to_markdown(str(pdf), pages=[page - 1], use_ocr=False)
            (converted / f"{document_id}.md").write_text(raw, encoding="utf-8")
            text = normalize_markdown(raw)
            (cleaned / f"{document_id}.md").write_text(text, encoding="utf-8")

            for section in headings.split_text(text):
                for index, content in enumerate(splitter.split_text(section.page_content)):
                    chunk_id = str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL, f"{checksum}:{page}:{len(chunks)}:{index}:{content}"
                        )
                    )
                    chunks.append(
                        {
                            "chunk_id": chunk_id,
                            "document_id": document_id,
                            "page": page,
                            "source": self.config["corpus_url"],
                            "text": content,
                            "headings": section.metadata,
                        }
                    )

        (self.directory / "04_chunks/chunks.json").write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        result = {"identity": identity, "chunk_count": len(chunks)}
        manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        return result

    def build(self, reset: bool = False) -> dict:
        (self.directory / "05_index").mkdir(parents=True, exist_ok=True)
        manifest_path = self.directory / "04_chunks/manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError("Run lab3 prepare before lab3 index")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        if manifest["identity"]["config"] != self.config:
            raise ValueError("Stale corpus configuration; run prepare --rebuild")

        chunks = json.loads((self.directory / "04_chunks/chunks.json").read_text(encoding="utf-8"))
        dense, sparse = self._encoders()
        texts = [chunk["text"] for chunk in chunks]
        vectors = dense.encode(
            texts, batch_size=4, normalize_embeddings=True, show_progress_bar=True
        )
        sparse_vectors = list(sparse.embed(texts))
        client, collection = self.client(), self.config["collection"]

        if client.collection_exists(collection) and reset:
            client.delete_collection(collection)

        if not client.collection_exists(collection):
            client.create_collection(
                collection,
                vectors_config={
                    "dense": models.VectorParams(
                        size=len(vectors[0]),
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "bm25": models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                    )
                },
            )

        for offset in range(0, len(chunks), 64):
            client.upsert(
                collection,
                points=[
                    models.PointStruct(
                        id=chunk["chunk_id"],
                        payload=chunk,
                        vector={
                            "dense": vectors[i].tolist(),
                            "bm25": models.SparseVector(
                                indices=sparse_vectors[i].indices.tolist(),
                                values=sparse_vectors[i].values.tolist(),
                            ),
                        },
                    )
                    for i, chunk in enumerate(chunks[offset : offset + 64], start=offset)
                ],
                wait=True,
            )

        count = client.count(collection, exact=True).count

        if count != len(chunks):
            raise ValueError("Index contains stale points; use index --reset explicitly")

        result = {
            "collection": collection,
            "count": count,
            "corpus": manifest,
            "chunks_sha256": hashlib.sha256(
                (self.directory / "04_chunks/chunks.json").read_bytes()
            ).hexdigest(),
        }
        (self.directory / "05_index/index.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.export_index()

        return result

    def export_index(self) -> dict:
        # Сохраняем реальные точки Qdrant: векторы и provenance можно показать без UI.
        path = self.directory / "05_index/vectors.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        count, offset = 0, None

        with path.open("w", encoding="utf-8") as output:
            while True:
                points, offset = self.client().scroll(
                    self.config["collection"],
                    limit=64,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True,
                )

                for point in points:
                    output.write(
                        json.dumps(
                            point.model_dump(mode="json", exclude_none=True), ensure_ascii=False
                        )
                        + "\n"
                    )
                    count += 1

                if offset is None:
                    break

        result = {
            "collection": self.config["collection"],
            "points": count,
            "file": "05_index/vectors.jsonl",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        (self.directory / "05_index/vector-export.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        return result

    def verify(self) -> dict:
        self._require_dependencies()

        # Проверяем сохранённые стадии без повторного скачивания и пересчёта эмбеддингов.
        pdf = self.directory / "01_pdf/postgres-17.11-ru.pdf"
        checksum = hashlib.sha256(pdf.read_bytes()).hexdigest()
        if checksum != self.config["corpus_sha256"]:
            raise ValueError("Source PDF checksum changed")

        chunks = json.loads((self.directory / "04_chunks/chunks.json").read_text(encoding="utf-8"))
        tokenizer = AutoTokenizer.from_pretrained(
            self.config["embedding_repo"],
            revision=self.config["embedding_revision"],
            cache_dir=str(self.settings.data / "models"),
            trust_remote_code=True,
            local_files_only=True,
        )
        lengths = [len(tokenizer.encode(chunk["text"])) for chunk in chunks]
        if max(lengths) > self.config["chunk_tokens"]:
            raise ValueError("A chunk exceeds the configured embedding tokenizer limit")

        pages = range(self.config["page_start"], self.config["page_end"] + 1)
        for page in pages:
            for stage in ("02_converted", "03_cleaned"):
                if not (self.directory / stage / f"pg17-p{page:04d}.md").is_file():
                    raise FileNotFoundError(f"Missing {stage} page {page}")

        if any(
            chunk["document_id"] != f"pg17-p{chunk['page']:04d}"
            or chunk["source"] != self.config["corpus_url"]
            for chunk in chunks
        ):
            raise ValueError("Chunk provenance is inconsistent")

        index = json.loads((self.directory / "05_index/index.json").read_text(encoding="utf-8"))
        if (
            hashlib.sha256((self.directory / "04_chunks/chunks.json").read_bytes()).hexdigest()
            != index["chunks_sha256"]
        ):
            raise ValueError("Chunks differ from the indexed artifact")

        count = self.client().count(self.config["collection"], exact=True).count
        if count != len(chunks):
            raise ValueError("Qdrant point count differs from the prepared chunks")

        example = next(chunk for chunk in chunks if chunk["document_id"] == "pg17-p0059")
        ids = tokenizer.encode(example["text"])
        with (self.directory / "05_index/vectors.jsonl").open(encoding="utf-8") as stream:
            vector = json.loads(stream.readline())
        result = {
            "ok": True,
            "pdf_sha256": checksum,
            "pages": len(pages),
            "chunks": len(chunks),
            "max_chunk_tokens": max(lengths),
            "tokenizer": self.config["embedding_repo"],
            "special_tokens_per_sequence": tokenizer.num_special_tokens_to_add(False),
            "dense_dimension": len(vector["vector"]["dense"]),
            "example": {
                "document_id": example["document_id"],
                "chunk_id": example["chunk_id"],
                "token_ids": ids[:32],
                "token_pieces": tokenizer.convert_ids_to_tokens(ids[:32]),
            },
            "stages": {
                stage: f"data/corpus/{stage}"
                for stage in ("01_pdf", "02_converted", "03_cleaned", "04_chunks", "05_index")
            },
        }
        self.settings.checks.mkdir(parents=True, exist_ok=True)
        (self.settings.checks / "pipeline.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        return result

    def search(self, query: str, limit: int = 5) -> list[dict]:
        path = self.directory / "05_index/index.json"

        if not path.exists():
            raise FileNotFoundError("Retrieval is not indexed; run lab3 prepare and lab3 index")

        index = json.loads(path.read_text(encoding="utf-8"))
        checksum = hashlib.sha256(
            (self.directory / "04_chunks/chunks.json").read_bytes()
        ).hexdigest()

        if checksum != index["chunks_sha256"]:
            raise ValueError("Chunk artifact changed after indexing; rebuild explicitly")

        if index["corpus"]["identity"]["config"] != self.config:
            raise ValueError("Index configuration differs; rebuild the corpus and index explicitly")

        dense, sparse = self._encoders()

        if len(dense.tokenizer.encode(self.config["query_instruction"] + query)) > 1024:
            raise ValueError("Search query exceeds the embedding tokenizer limit (1024)")

        vector = dense.encode(
            [query], prompt=self.config["query_instruction"], normalize_embeddings=True
        )[0].tolist()
        lexical = next(iter(sparse.query_embed(query)))
        response = self.client().query_points(
            self.config["collection"],
            prefetch=[
                models.Prefetch(query=vector, using="dense", limit=self.config["prefetch_k"]),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=lexical.indices.tolist(), values=lexical.values.tolist()
                    ),
                    using="bm25",
                    limit=self.config["prefetch_k"],
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )

        return [{**point.payload, "score": point.score} for point in response.points]

    def read_page(self, document_id: str, offset: int = 0, limit: int = 6000) -> dict:
        if not re.fullmatch(r"pg17-p\d{4}", document_id):
            raise ValueError("Use a document_id returned by search_docs, e.g. pg17-p0059")

        path = self.directory / "03_cleaned" / f"{document_id}.md"

        if not path.exists():
            raise FileNotFoundError(f"Page {document_id} is not in the prepared corpus")

        text = path.read_text(encoding="utf-8")
        end = min(offset + limit, len(text))

        return {
            "document_id": document_id,
            "page": int(document_id[-4:]),
            "source": self.config["corpus_url"],
            "text": text[offset:end],
            "next_offset": end if end < len(text) else None,
            "total_chars": len(text),
        }
