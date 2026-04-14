import time
import uuid
from typing import AsyncIterator

from app.config import settings
from app.llm.base import LLMClient

GROQ_MODEL = "llama-3.1-8b-instant"


class GroqClient(LLMClient):
    def __init__(self) -> None:
        import groq as _groq
        self._client = _groq.AsyncGroq(api_key=settings.GROQ_API_KEY)

    async def complete(
        self,
        prompt: str,
        system: str = "",
        json_mode: bool = False,
        task_type: str = "general",
    ) -> str:
        from app.observability.logger import log_llm_call

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = {"model": GROQ_MODEL, "messages": messages, "max_tokens": 2048}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        request_id = str(uuid.uuid4())
        start = time.perf_counter()
        response = await self._client.chat.completions.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        usage = response.usage
        log_llm_call(
            model=GROQ_MODEL,
            task_type=task_type,
            tokens_in=usage.prompt_tokens if usage else 0,
            tokens_out=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
            request_id=request_id,
        )
        return response.choices[0].message.content or ""

    async def stream(
        self,
        prompt: str,
        system: str = "",
        task_type: str = "general",
    ) -> AsyncIterator[str]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        stream = await self._client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=2048,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
