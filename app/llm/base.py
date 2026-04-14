from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional


class LLMClient(ABC):
    """Abstract base for all LLM provider implementations.

    The rest of the application must only import this class — never provider SDKs directly.
    """

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system: str = "",
        json_mode: bool = False,
        task_type: str = "general",
    ) -> str:
        """Return a complete response string."""
        ...

    @abstractmethod
    async def stream(
        self,
        prompt: str,
        system: str = "",
        task_type: str = "general",
    ) -> AsyncIterator[str]:
        """Yield response tokens one by one."""
        ...


def get_llm_client(provider: Optional[str] = None) -> LLMClient:
    from app.config import settings

    p = (provider or settings.LLM_PROVIDER).lower()
    if p == "groq":
        from app.llm.groq_client import GroqClient
        return GroqClient()
    if p == "anthropic":
        from app.llm.anthropic_client import AnthropicClient
        return AnthropicClient()
    raise ValueError(f"Unknown LLM provider: {p!r}. Choose 'groq' or 'anthropic'.")
