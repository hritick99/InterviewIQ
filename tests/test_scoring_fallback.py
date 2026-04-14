"""Test: Scoring agent continues and returns fallback state when LLM returns malformed JSON.

Injects a mock LLM client that always returns invalid JSON.
Verifies the graph completes without raising an exception.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


class BrokenLLMClient:
    """Always returns malformed JSON."""

    async def complete(self, prompt, system="", json_mode=False, task_type="general"):
        return "THIS IS NOT JSON {{{{ broken !!!!"

    async def stream(self, prompt, system="", task_type="general"):
        yield "token"


SAMPLE_SEGMENTS = [
    {
        "segment_index": 0,
        "topic": "python",
        "skills_mentioned": ["python"],
        "sentiment": "positive",
        "confidence_in_answer": "high",
        "raw_text": "I have five years of Python experience.",
    }
]


@pytest.mark.asyncio
async def test_scoring_agent_fallback_on_malformed_llm():
    from app.scoring import agent as agent_module

    with patch.object(agent_module, "get_llm_client", return_value=BrokenLLMClient()):
        state = await agent_module.run_scoring_agent(
            candidate_id="test_candidate",
            tenant_id="test_tenant",
            segments=SAMPLE_SEGMENTS,
            rubric={},
        )

    # Graph must complete — no exception raised
    assert state is not None

    # Fallbacks must have been used
    assert len(state["errors"]) > 0, "Expected fallback errors to be recorded"

    # Audit log must be present for every node
    node_names = {entry["node"] for entry in state["audit_log"]}
    assert "technical_depth" in node_names
    assert "consistency" in node_names
    assert "behavioural" in node_names
    assert "synthesis" in node_names

    # Final verdict must be present (fallback verdict)
    assert "overall_score" in state["final_verdict"]
    assert state["final_verdict"]["hire_recommendation"] in ("strong_yes", "yes", "maybe", "no")
