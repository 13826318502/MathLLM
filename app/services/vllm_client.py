"""Async client for the vLLM OpenAI-compatible API."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.core.config import Settings


class VLLMServiceError(RuntimeError):
    """A safe, user-facing error raised when vLLM cannot be reached."""


class VLLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _timeout(self) -> httpx.Timeout:
        return httpx.Timeout(
            connect=self.settings.connect_timeout,
            read=self.settings.read_timeout,
            write=30.0,
            pool=10.0,
        )

    def _headers(self) -> dict[str, str]:
        if not self.settings.api_key:
            return {}
        return {"Authorization": f"Bearer {self.settings.api_key}"}

    async def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        body = await response.aread()
        detail = body.decode("utf-8", errors="replace").strip()
        if len(detail) > 400:
            detail = detail[:400] + "..."
        raise VLLMServiceError(
            f"模型服务返回 HTTP {response.status_code}"
            + (f": {detail}" if detail else "")
        )

    async def list_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(
                base_url=self.settings.vllm_base_url,
                timeout=self._timeout(),
            ) as client:
                response = await client.get("/models", headers=self._headers())
                await self._raise_for_status(response)
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise VLLMServiceError("无法连接或解析 vLLM 的模型接口") from exc

        data = payload.get("data", []) if isinstance(payload, dict) else []
        return [
            str(item.get("id"))
            for item in data
            if isinstance(item, dict) and item.get("id")
        ]

    async def complete(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        payload = {
            "model": self.settings.model_name,
            "messages": messages,
            "stream": False,
            "temperature": 0.0,
            "max_tokens": self.settings.max_output_tokens,
        }
        try:
            async with httpx.AsyncClient(
                base_url=self.settings.vllm_base_url,
                timeout=self._timeout(),
            ) as client:
                response = await client.post(
                    "/chat/completions",
                    json=payload,
                    headers=self._headers(),
                )
                await self._raise_for_status(response)
                result = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise VLLMServiceError("无法获取模型回答") from exc
        if not isinstance(result, dict):
            raise VLLMServiceError("模型返回了无效响应")
        return result

    async def stream_raw(
        self,
        messages: list[dict[str, str]],
    ) -> AsyncIterator[dict[str, Any]]:
        payload = {
            "model": self.settings.model_name,
            "messages": messages,
            "stream": True,
            "temperature": 0.0,
            "max_tokens": self.settings.max_output_tokens,
        }
        try:
            async with httpx.AsyncClient(
                base_url=self.settings.vllm_base_url,
                timeout=self._timeout(),
            ) as client:
                async with client.stream(
                    "POST",
                    "/chat/completions",
                    json=payload,
                    headers=self._headers(),
                ) as response:
                    await self._raise_for_status(response)
                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line or line.startswith(":"):
                            continue
                        if line.startswith("data:"):
                            line = line[5:].strip()
                        if line == "[DONE]":
                            break
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(event, dict):
                            yield event
        except VLLMServiceError:
            raise
        except httpx.HTTPError as exc:
            raise VLLMServiceError("模型流式请求失败") from exc
