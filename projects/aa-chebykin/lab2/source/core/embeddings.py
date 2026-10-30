import gc
import os

import numpy as np
import torch
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from core.config import CACHE_ROOT, Settings


def load_tokenizer(settings: Settings) -> PreTrainedTokenizerBase:
    # Берём токенизатор именно скачанного encoder; интернет при индексировании не нужен.
    return AutoTokenizer.from_pretrained(
        settings.path(settings.embedding_model), trust_remote_code=True, local_files_only=True
    )


class Encoders:
    def __init__(self, settings: Settings) -> None:
        # Модели загружаются лениво: BM25-only не должен занимать память весами Giga.
        self.settings = settings
        self.dense: SentenceTransformer | None = None
        self.sparse: SparseTextEmbedding | None = None

    def dense_model(self) -> SentenceTransformer:
        if self.dense is None:
            # На CPU используем FP32, на небольшой GPU — FP16 с готовым SDPA attention.
            device = self.settings.embedding_device
            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(
                    'CUDA is unavailable. Install CUDA PyTorch or set embedding_device="cpu".'
                )
            torch.set_num_threads(min(6, os.cpu_count() or 1))
            self.dense = SentenceTransformer(
                str(self.settings.path(self.settings.embedding_model)),
                trust_remote_code=True,
                local_files_only=True,
                device=device,
                model_kwargs={
                    "dtype": torch.float16 if device == "cuda" else torch.float32,
                    "attn_implementation": "sdpa",
                },
            )
            self.dense.max_seq_length = self.settings.embedding_max_tokens
        return self.dense

    def sparse_model(self) -> SparseTextEmbedding:
        # Стемминг, стоп-слова, хеширование терминов и BM25 реализует готовый FastEmbed.
        if self.sparse is None:
            self.sparse = SparseTextEmbedding(
                "Qdrant/bm25",
                cache_dir=str(CACHE_ROOT / "fastembed"),
                language=self.settings.bm25_language,
                k=self.settings.bm25_k,
                b=self.settings.bm25_b,
                avg_len=self.settings.bm25_avg_len,
                local_files_only=True,
            )
        return self.sparse

    def encode_dense(self, texts: list[str], query: bool = False) -> np.ndarray:
        model = self.dense_model()

        # Запрос с инструкцией не должен молча обрезаться по максимальной длине encoder.
        if query:
            for text in texts:
                tokens = model.tokenizer.encode(self.settings.query_instruction + text)
                if len(tokens) > model.max_seq_length:
                    raise ValueError(
                        "Question exceeds the embedding model token budget. Shorten it."
                    )
        # Для документов инструкция не добавляется; pooling и L2-нормализация библиотечные.
        return model.encode(
            texts,
            prompt=self.settings.query_instruction if query else "",
            batch_size=self.settings.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=not query,
            convert_to_numpy=True,
        ).astype(np.float32)

    def encode_sparse(self, texts: list[str], query: bool = False) -> list[dict]:
        # Документные TF-веса и поисковый sparse-вектор строятся разными API FastEmbed.
        model = self.sparse_model()
        encoded = model.query_embed(texts) if query else model.embed(texts)
        return [
            {"indices": vector.indices.tolist(), "values": vector.values.tolist()}
            for vector in encoded
        ]

    def close(self) -> None:
        # Между этапами освобождаем модель и циклические ссылки перед очисткой CUDA-кеша.
        self.dense = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
