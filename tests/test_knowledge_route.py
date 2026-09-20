from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from app.api.dependencies import get_settings
from app.api.main import app
from app.core.config import settings


class KnowledgeDocumentRouteTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = Path(self._tmp.name)
        (root / "01-判别式.md").write_text("# 判别式\n\nΔ = b^2 - 4ac", encoding="utf-8")
        (root / "sub").mkdir()
        (root / "sub" / "02-导数.md").write_text("导数", encoding="utf-8")
        (root / "secret.py").write_text("print('x')", encoding="utf-8")
        self.settings = replace(settings, knowledge_dir=self._tmp.name)
        app.dependency_overrides[get_settings] = lambda: self.settings
        self.http = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides.clear()
        self._tmp.cleanup()

    def test_lists_markdown_documents(self) -> None:
        response = self.http.get("/api/knowledge/documents")
        self.assertEqual(response.status_code, 200)
        names = [item["name"] for item in response.json()]
        self.assertIn("01-判别式.md", names)
        self.assertIn("sub/02-导数.md", names)
        self.assertNotIn("secret.py", names)

    def test_returns_document_content(self) -> None:
        response = self.http.get("/api/knowledge/documents/01-判别式.md")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["name"], "01-判别式.md")
        self.assertIn("b^2 - 4ac", payload["content"])

    def test_missing_document_is_404(self) -> None:
        response = self.http.get("/api/knowledge/documents/nope.md")
        self.assertEqual(response.status_code, 404)

    def test_unsupported_suffix_is_rejected(self) -> None:
        response = self.http.get("/api/knowledge/documents/secret.py")
        self.assertEqual(response.status_code, 400)

    def test_path_traversal_is_rejected(self) -> None:
        response = self.http.get("/api/knowledge/documents/..%2Fsecret.py")
        self.assertIn(response.status_code, (400, 404))


if __name__ == "__main__":
    unittest.main()
