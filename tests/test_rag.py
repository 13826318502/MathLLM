from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from app.agent.tools import ToolContext, call_tool
from app.core.config import settings
from app.services import rag_service
from app.services.rag_service import (
    KnowledgeBaseError,
    RetrievedChunk,
    index_knowledge,
    search,
    split_text,
)


class HashingEmbedder:
    """Deterministic character-bucket embedder so tests need no model download."""

    def __init__(self, dimensions: int = 128) -> None:
        self.dimensions = dimensions

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for char in text:
            vector[ord(char) % self.dimensions] += 1.0
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FakeClient:
    async def complete(self, messages, **kwargs) -> dict:
        return {"choices": [{"message": {"content": ""}}]}

    async def complete_json(self, messages, *, max_tokens=None, schema=None) -> dict:
        return {"choices": [{"message": {"content": ""}}]}


class SplitTextTest(unittest.TestCase):
    def test_empty_text(self) -> None:
        self.assertEqual(split_text("   "), [])

    def test_short_text_is_one_chunk(self) -> None:
        self.assertEqual(split_text("判别式决定根的个数"), ["判别式决定根的个数"])

    def test_long_paragraph_is_windowed(self) -> None:
        chunks = split_text("x" * 1200, chunk_size=500, overlap=50)
        self.assertGreaterEqual(len(chunks), 3)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 500)

    def test_paragraphs_are_packed(self) -> None:
        text = "第一段内容。" * 3 + "\n\n" + "第二段内容。" * 3
        chunks = split_text(text, chunk_size=500, overlap=50)
        self.assertEqual(len(chunks), 1)


class RagServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = Path(self._tmp.name)
        self.knowledge_dir = root / "knowledge"
        self.knowledge_dir.mkdir()
        (self.knowledge_dir / "discriminant.md").write_text(
            "判别式 Δ = b^2 - 4ac。判别式大于零有两个实根。判别式等于零有重根。",
            encoding="utf-8",
        )
        (self.knowledge_dir / "derivative.md").write_text(
            "导数表示瞬时变化率。导数为零的点是驻点。二阶导数判断极值。",
            encoding="utf-8",
        )
        self.settings = replace(
            settings,
            knowledge_dir=str(self.knowledge_dir),
            rag_persist_dir=str(root / "chroma"),
            rag_collection="test_knowledge",
            rag_top_k=2,
        )
        self.embedder = HashingEmbedder()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_index_and_search_returns_best_match(self) -> None:
        count = index_knowledge(self.settings, embedder=self.embedder)
        self.assertGreater(count, 0)
        chunks = search(
            self.settings,
            "判别式",
            embedder=self.embedder,
        )
        self.assertTrue(chunks)
        self.assertEqual(chunks[0].source, "discriminant.md")

    def test_search_missing_store_raises(self) -> None:
        empty = replace(self.settings, rag_persist_dir=str(Path(self._tmp.name) / "nope"))
        with self.assertRaises(KnowledgeBaseError):
            search(empty, "判别式", embedder=self.embedder)

    def test_index_missing_directory_raises(self) -> None:
        empty = replace(self.settings, knowledge_dir=str(Path(self._tmp.name) / "nope"))
        with self.assertRaises(KnowledgeBaseError):
            index_knowledge(empty, embedder=self.embedder)

    def test_blank_query_returns_empty(self) -> None:
        index_knowledge(self.settings, embedder=self.embedder)
        self.assertEqual(search(self.settings, "   ", embedder=self.embedder), [])

    def test_has_index_false_before_build(self) -> None:
        self.assertFalse(rag_service.has_index(self.settings))
        self.assertEqual(rag_service.index_size(self.settings), 0)

    def test_ensure_index_builds_when_missing(self) -> None:
        count = rag_service.ensure_index(self.settings, embedder=self.embedder)
        self.assertGreater(count, 0)
        self.assertTrue(rag_service.has_index(self.settings))
        self.assertEqual(rag_service.index_size(self.settings), count)

    def test_ensure_index_keeps_existing_store(self) -> None:
        first = rag_service.ensure_index(self.settings, embedder=self.embedder)

        class ExplodingEmbedder:
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                raise AssertionError("existing index must not be rebuilt")

            def embed_query(self, text: str) -> list[float]:
                raise AssertionError("existing index must not be rebuilt")

        second = rag_service.ensure_index(self.settings, embedder=ExplodingEmbedder())
        self.assertEqual(second, first)


class KnowledgeToolTest(unittest.IsolatedAsyncioTestCase):
    def _ctx(self) -> ToolContext:
        return ToolContext(client=FakeClient(), settings=settings)

    async def test_reports_missing_knowledge_base(self) -> None:
        with patch.object(
            rag_service,
            "search",
            side_effect=KnowledgeBaseError("知识库尚未建立，请先运行索引"),
        ):
            result = await call_tool("search_knowledge", {"query": "判别式"}, self._ctx())
        self.assertFalse(result.success)
        self.assertIn("知识库", result.error or "")

    async def test_returns_documents_on_success(self) -> None:
        chunks = [
            RetrievedChunk(content="Δ = b^2 - 4ac", source="discriminant.md", distance=0.1)
        ]
        with patch.object(rag_service, "search", return_value=chunks):
            result = await call_tool("search_knowledge", {"query": "判别式"}, self._ctx())
        self.assertTrue(result.success)
        self.assertEqual(result.data["documents"][0]["source"], "discriminant.md")
        self.assertIn("b^2", result.data["documents"][0]["content"])

    async def test_empty_result_is_failure(self) -> None:
        with patch.object(rag_service, "search", return_value=[]):
            result = await call_tool("search_knowledge", {"query": "无关问题"}, self._ctx())
        self.assertFalse(result.success)
        self.assertIn("没有找到", result.error or "")


if __name__ == "__main__":
    unittest.main()
