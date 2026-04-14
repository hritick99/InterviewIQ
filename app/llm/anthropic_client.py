import time
import uuid
from typing import AsyncIterator

from app.config import settings
from app.llm.base import LLMClient

ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"


class AnthropicClient(LLMClient):
    def __init__(self) -> None:
        import anthropic as _anthropic
        self._client = _anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

    async def complete(
        self,
        prompt: str,
        system: str = "",
        json_mode: bool = False,
        task_type: str = "general",
    ) -> str:
        from app.observability.logger import log_llm_call

        request_id = str(uuid.uuid4())
        start = time.perf_counter()

        kwargs: dict = {
            "model": ANTHROPIC_MODEL,
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = await self._client.messages.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        log_llm_call(
            model=ANTHROPIC_MODEL,
            task_type=task_type,
            tokens_in=response.usage.input_tokens,
            tokens_out=response.usage.output_tokens,
            latency_ms=latency_ms,
            request_id=request_id,
        )
        return response.content[0].text if response.content else ""

    async def stream(
        self,
        prompt: str,
        system: str = "",
        task_type: str = "general",
    ) -> AsyncIterator[str]:
        kwargs: dict = {
            "model": ANTHROPIC_MODEL,
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text
