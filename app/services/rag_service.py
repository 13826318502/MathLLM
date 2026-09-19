"""Local RAG over a Chroma vector store.

Embeddings are computed by an injectable embedder and passed to Chroma
explicitly, so the store never has to persist or rebuild an embedding function.

The default embedder is ``BAAI/bge-small-zh-v1.5`` via fastembed, a small
Chinese retrieval model. Chroma's bundled MiniLM model is English-only and
mis-ranks Chinese text, so it is not used.

Build the index with:
    python -m app.services.rag_service
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from app.core.config import Settings, settings

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
_TEXT_SUFFIXES = {".md", ".txt"}

_CLIENTS: dict[str, Any] = {}
_EMBEDDERS: dict[str, "Embedder"] = {}
_INDEX_LOCK = threading.Lock()


class KnowledgeBaseError(RuntimeError):
    """Raised when the knowledge base is missing or cannot be read."""


class Embedder(Protocol):
    """Asymmetric embedder: documents and queries may be encoded differently."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class RetrievedChunk:
    content: str
    source: str
    distance: float


class FastEmbedEmbedder:
    """Lazily loaded fastembed model, shared per process via get_default_embedder."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [
            [float(value) for value in vector]
            for vector in self._load().embed(list(texts))
        ]

    def embed_query(self, text: str) -> list[float]:
        model = self._load()
        if hasattr(model, "query_embed"):
            return [float(value) for value in next(iter(model.query_embed([text])))]
        return self.embed_documents([text])[0]


def get_default_embedder(settings: Settings) -> Embedder:
    """Return the process-wide embedder for the configured model."""
    embedder = _EMBEDDERS.get(settings.rag_embedding_model)
    if embedder is None:
        embedder = FastEmbedEmbedder(settings.rag_embedding_model)
        _EMBEDDERS[settings.rag_embedding_model] = embedder
    return embedder


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks, packing whole paragraphs first."""
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []
    overlap = max(0, min(overlap, chunk_size // 2))
    paragraphs = [part.strip() for part in normalized.split("\n\n") if part.strip()]

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if not current:
            current = paragraph
        elif len(current) + len(paragraph) + 2 <= chunk_size:
            current = f"{current}\n\n{paragraph}"
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{paragraph}" if tail else paragraph
        while len(current) > chunk_size:
            chunks.append(current[:chunk_size])
            current = (
                current[chunk_size - overlap :] if overlap else current[chunk_size:]
            )
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _load_documents(knowledge_dir: Path) -> list[tuple[str, str]]:
    """Return (source, text) pairs for every supported file, sorted by name."""
    documents: list[tuple[str, str]] = []
    for path in sorted(knowledge_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in _TEXT_SUFFIXES:
            source = path.relative_to(knowledge_dir).as_posix()
            documents.append((source, path.read_text(encoding="utf-8")))
    return documents


def _normalize(vectors: list[list[float]]) -> list[list[float]]:
    """L2-normalize embeddings so Chroma's default L2 space ranks like cosine."""
    normalized: list[list[float]] = []
    for vector in vectors:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            normalized.append(list(vector))
        else:
            normalized.append([value / norm for value in vector])
    return normalized


def _client(persist_dir: str) -> Any:
    client = _CLIENTS.get(persist_dir)
    if client is None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        _CLIENTS[persist_dir] = client
    return client


def index_knowledge(settings: Settings, *, embedder: Embedder | None = None) -> int:
    """Rebuild the vector store from the knowledge directory; return chunk count."""
    embedder = embedder or get_default_embedder(settings)
    knowledge_dir = Path(settings.knowledge_dir)
    if not knowledge_dir.is_dir():
        raise KnowledgeBaseError(f"知识库目录不存在：{knowledge_dir}")

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str]] = []
    for source, text in _load_documents(knowledge_dir):
        for index, chunk in enumerate(split_text(text)):
            ids.append(f"{source}#{index}")
            documents.append(chunk)
            metadatas.append({"source": source})

    client = _client(settings.rag_persist_dir)
    try:
        client.delete_collection(settings.rag_collection)
    except Exception:
        pass
    collection = client.create_collection(settings.rag_collection)
    if documents:
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=_normalize(embedder.embed_documents(documents)),
        )
    return len(documents)


def index_size(settings: Settings) -> int:
    """Return the number of chunks in the existing store, or 0 when absent."""
    if not Path(settings.rag_persist_dir).exists():
        return 0
    try:
        collection = _client(settings.rag_persist_dir).get_collection(
            settings.rag_collection
        )
        return int(collection.count())
    except Exception:
        return 0


def has_index(settings: Settings) -> bool:
    """Return True when a non-empty vector store is already built."""
    return index_size(settings) > 0


def ensure_index(settings: Settings, *, embedder: Embedder | None = None) -> int:
    """Build the vector store when missing and return the chunk count.

    Safe to call concurrently: the first caller builds, later callers reuse
    the result. An existing non-empty index is never rebuilt.
    """
    if has_index(settings):
        return index_size(settings)
    with _INDEX_LOCK:
        if has_index(settings):
            return index_size(settings)
        return index_knowledge(settings, embedder=embedder)


def search(
    settings: Settings,
    query: str,
    *,
    top_k: int | None = None,
    embedder: Embedder | None = None,
) -> list[RetrievedChunk]:
    """Return the most similar knowledge chunks, or raise when unavailable."""
    query = (query or "").strip()
    if not query:
        return []
    if not Path(settings.rag_persist_dir).exists():
        raise KnowledgeBaseError("知识库尚未建立，请先运行索引")

    embedder = embedder or get_default_embedder(settings)
    client = _client(settings.rag_persist_dir)
    try:
        collection = client.get_collection(settings.rag_collection)
    except Exception as exc:
        raise KnowledgeBaseError("知识库尚未建立，请先运行索引") from exc
    if collection.count() == 0:
        return []

    result = collection.query(
        query_embeddings=_normalize([embedder.embed_query(query)]),
        n_results=top_k or settings.rag_top_k,
        include=["documents", "metadatas", "distances"],
    )
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    chunks: list[RetrievedChunk] = []
    for content, metadata, distance in zip(documents, metadatas, distances):
        chunks.append(
            RetrievedChunk(
                content=content or "",
                source=str((metadata or {}).get("source", "unknown")),
                distance=float(distance),
            )
        )
    return chunks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MathLLM 知识库向量索引工具")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--check",
        action="store_true",
        help="只检查向量库是否存在，存在退出码 0，不存在退出码 1",
    )
    group.add_argument(
        "--ensure",
        action="store_true",
        help="仅在向量库缺失时建立索引",
    )
    args = parser.parse_args()

    if args.check:
        if has_index(settings):
            print(f"向量库已建立：{index_size(settings)} 个片段")
            raise SystemExit(0)
        print("向量库尚未建立")
        raise SystemExit(1)

    if args.ensure and has_index(settings):
        print(f"向量库已存在，跳过索引：{index_size(settings)} 个片段")
        raise SystemExit(0)

    count = index_knowledge(settings)
    print(
        f"已索引 {count} 个片段 -> "
        f"{settings.rag_persist_dir} / {settings.rag_collection}"
    )
