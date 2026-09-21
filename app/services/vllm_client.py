"""Async client for the vLLM OpenAI-compatible API."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.agent.schema import TokenUsage
from app.core.config import ModelConfig
from app.services.stream_service import extract_usage


class VLLMServiceError(RuntimeError):
    """A safe, user-facing error raised when vLLM cannot be reached."""


class VLLMClient:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.usage = TokenUsage()

    def reset_usage(self) -> None:
        """Clear accumulated token usage so one agent run reports only its own."""
        self.usage = TokenUsage()

    def _record_usage(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        parsed = extract_usage(payload)
        if parsed is None:
            return
        self.usage.add(*parsed)

    def _timeout(self) -> httpx.Timeout:
        return httpx.Timeout(
            connect=self.config.connect_timeout,
            read=self.config.read_timeout,
            write=30.0,
            pool=10.0,
        )

    def _headers(self) -> dict[str, str]:
        if not self.config.api_key:
            return {}
        return {"Authorization": f"Bearer {self.config.api_key}"}

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
                base_url=self.config.base_url,
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

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens or self.config.max_output_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        try:
            async with httpx.AsyncClient(
                base_url=self.config.base_url,
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
        self._record_usage(result)
        return result

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Request a JSON-only completion.

        Defaults to JSON object mode, which guarantees syntactically valid JSON
        but not schema conformance. When ``schema`` is provided the backend is
        asked for schema-constrained output; vLLM honours it, while some
        OpenAI-compatible servers silently ignore it.
        """
        if schema is None:
            response_format: dict[str, Any] = {"type": "json_object"}
        else:
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema},
            }
        return await self.complete(
            messages,
            max_tokens=max_tokens,
            response_format=response_format,
        )

    async def complete_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        *,
        tool_choice: Any = "auto",
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Ask the model to pick a tool and fill its arguments.

        The engine (vLLM with a tool-call parser, or a cloud API) is responsible
        for emitting a well-formed ``tool_calls`` payload. Servers that do not
        understand ``tools`` raise ``VLLMServiceError``, which callers turn into
        a structured-JSON fallback.
        """
        return await self.complete(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
        )

    async def stream_raw(
        self,
        messages: list[dict[str, str]],
    ) -> AsyncIterator[dict[str, Any]]:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": True,
            "temperature": 0.0,
            "max_tokens": self.config.max_output_tokens,
            # Servers that understand this report token usage on the last chunk.
            # Servers that do not simply ignore the unknown field.
            "stream_options": {"include_usage": True},
        }
        try:
            async with httpx.AsyncClient(
                base_url=self.config.base_url,
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
                            self._record_usage(event)
                            yield event
        except VLLMServiceError:
            raise
        except httpx.HTTPError as exc:
            raise VLLMServiceError("模型流式请求失败") from exc
