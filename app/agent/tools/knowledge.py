"""search_knowledge: retrieve math concepts from the local Chroma store."""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, ConfigDict, Field

from app.agent.schema import ToolResult
from app.agent.tools.base import ToolContext, ToolSpec
from app.services import rag_service


class SearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)


async def _search(args: SearchInput, ctx: ToolContext) -> ToolResult:
    try:
        chunks = await asyncio.to_thread(
            rag_service.search,
            ctx.settings,
            args.query,
            top_k=args.top_k,
        )
    except rag_service.KnowledgeIndexMissing as exc:
        return ToolResult(
            success=False,
            error=str(exc),
            data={"documents": [], "reason": "index_missing"},
        )
    except rag_service.KnowledgeEmbeddingError as exc:
        return ToolResult(
            success=False,
            error=f"检索失败：{exc}",
            data={"documents": [], "reason": "embedding_error"},
        )
    except rag_service.KnowledgeBaseError as exc:
        return ToolResult(
            success=False,
            error=str(exc),
            data={"documents": [], "reason": "retrieval_error"},
        )
    if not chunks:
        # Retrieval itself worked; the knowledge base just has nothing relevant.
        # This is reported as success so callers can tell it apart from a
        # timeout or a missing index.
        return ToolResult(success=True, data={"documents": [], "reason": "no_match"})
    return ToolResult(
        success=True,
        data={
            "documents": [
                {
                    "content": chunk.content,
                    "source": chunk.source,
                    "distance": chunk.distance,
                    "rank": rank,
                }
                for rank, chunk in enumerate(chunks, start=1)
            ]
        },
    )


SPEC = ToolSpec(
    name="search_knowledge",
    description="从数学知识库检索概念、定理和定义的依据，返回带来源的片段。",
    input_model=SearchInput,
    handler=_search,
    timeout=30.0,
)
