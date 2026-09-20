"""Knowledge-base document browsing endpoints.

Lets the web UI list and render the Markdown files under ``knowledge/`` so the
retrieved sources can be read in full instead of only as snippets.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.dependencies import get_settings
from app.core.config import Settings

router = APIRouter()

_TEXT_SUFFIXES = {".md", ".txt"}


class KnowledgeDocumentSummary(BaseModel):
    name: str
    size: int
    modified: str


class KnowledgeDocumentContent(BaseModel):
    name: str
    content: str


def _knowledge_root(settings: Settings) -> Path:
    return Path(settings.knowledge_dir)


def _safe_path(settings: Settings, name: str) -> Path:
    root = _knowledge_root(settings).resolve()
    candidate = (root / name).resolve()
    if root != candidate and root not in candidate.parents:
        raise HTTPException(status_code=400, detail="非法的文档路径")
    if candidate.suffix.lower() not in _TEXT_SUFFIXES:
        raise HTTPException(status_code=400, detail="不支持的文件类型")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="未找到该文档")
    return candidate


@router.get("/knowledge/documents", response_model=list[KnowledgeDocumentSummary])
async def knowledge_documents(
    settings: Settings = Depends(get_settings),
) -> list[KnowledgeDocumentSummary]:
    """List every Markdown/text document in the knowledge directory."""
    root = _knowledge_root(settings)
    if not root.is_dir():
        return []
    documents: list[KnowledgeDocumentSummary] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        stat = path.stat()
        documents.append(
            KnowledgeDocumentSummary(
                name=path.relative_to(root).as_posix(),
                size=stat.st_size,
                modified=datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(timespec="seconds"),
            )
        )
    return documents


@router.get("/knowledge/documents/{name:path}", response_model=KnowledgeDocumentContent)
async def knowledge_document(
    name: str,
    settings: Settings = Depends(get_settings),
) -> KnowledgeDocumentContent:
    """Return the raw Markdown of one knowledge document."""
    path = _safe_path(settings, name)
    return KnowledgeDocumentContent(
        name=name, content=path.read_text(encoding="utf-8")
    )
