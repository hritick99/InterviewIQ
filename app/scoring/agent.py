"""LangGraph multi-step scoring workflow.

State is fully typed. Every node has a fallback — a malformed LLM response never
kills the graph. Every run produces a full audit_log.

Graph: START → technical_depth → consistency → behavioural → synthesis → END
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.llm.base import get_llm_client


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class TechnicalScore(TypedDict):
    topic: str
    depth_score: int         # 1-5
    evidence_segment_ids: list[str]


class ConsistencyFlag(TypedDict):
    description: str
    segment_ids: list[str]


class BehaviouralScore(TypedDict):
    segment_index: int
    star_completeness: int   # 0-3
    notes: str


class FinalVerdict(TypedDict):
    overall_score: float     # 0-10
    hire_recommendation: str  # strong_yes|yes|maybe|no
    strengths: list[str]
    concerns: list[str]
    segment_citations: list[str]


class AuditEntry(TypedDict):
    node: str
    timestamp: str
    prompt: str
    output: str
    fallback_used: bool


class ScoringState(TypedDict):
    candidate_id: str
    tenant_id: str
    segments: list[dict]           # raw segment dicts from DB
    rubric: dict                   # tenant-configurable rubric
    technical_scores: list[TechnicalScore]
    consistency_flags: list[ConsistencyFlag]
    behavioural_scores: list[BehaviouralScore]
    final_verdict: FinalVerdict
    audit_log: list[AuditEntry]
    errors: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _segments_summary(segments: list[dict]) -> str:
    lines = []
    for s in segments:
        lines.append(
            f"[Segment {s['segment_index']}] Topic: {s['topic']} | "
            f"Skills: {', '.join(s.get('skills_mentioned', []))} | "
            f"Confidence: {s.get('confidence_in_answer', '?')}\n"
            f"{s['raw_text'][:400]}"
        )
    return "\n\n".join(lines)


def _audit(node: str, prompt: str, output: str, fallback: bool) -> AuditEntry:
    return AuditEntry(
        node=node,
        timestamp=datetime.utcnow().isoformat(),
        prompt=prompt[:2000],   # truncate for storage
        output=output[:2000],
        fallback_used=fallback,
    )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

async def technical_depth_node(state: ScoringState) -> dict:
    """Assess depth of technical answers. Fallback: depth_score=1 for all topics."""
    llm = get_llm_client()
    rubric_focus = state["rubric"].get("technical_topics", [])
    summary = _segments_summary(state["segments"])

    prompt = f"""You are scoring a candidate's technical depth.

Segments:
{summary}

{"Focus on these topics: " + ", ".join(rubric_focus) if rubric_focus else "Identify all technical topics."}

For each technical topic mentioned, respond with JSON array:
[
  {{"topic": "...", "depth_score": 1-5, "evidence_segment_ids": ["0", "1"]}}
]

depth_score meaning:
1 = surface mention only
2 = basic explanation
3 = solid understanding
4 = demonstrates design reasoning
5 = expert-level with trade-off analysis
"""

    fallback = False
    output_raw = ""
    try:
        output_raw = await llm.complete(prompt=prompt, json_mode=True, task_type="node_scoring")
        data = json.loads(output_raw)
        if not isinstance(data, list):
            data = data.get("topics", data.get("scores", []))
        scores = [
            TechnicalScore(
                topic=item.get("topic", "unknown"),
                depth_score=max(1, min(5, int(item.get("depth_score", 1)))),
                evidence_segment_ids=[str(x) for x in item.get("evidence_segment_ids", [])],
            )
            for item in data
        ]
    except Exception as e:
        fallback = True
        scores = [
            TechnicalScore(topic=s["topic"], depth_score=1, evidence_segment_ids=[str(s["segment_index"])])
            for s in state["segments"]
            if s.get("topic") and s["topic"] != "unknown"
        ]
        state["errors"].append(f"technical_depth_node fallback: {e}")

    return {
        "technical_scores": scores,
        "audit_log": state["audit_log"] + [_audit("technical_depth", prompt, output_raw, fallback)],
        "errors": state["errors"],
    }


async def consistency_node(state: ScoringState) -> dict:
    """Detect contradictions across segments. Fallback: no flags."""
    llm = get_llm_client()
    summary = _segments_summary(state["segments"])

    prompt = f"""Review these interview segments for contradictions.

Segments:
{summary}

Identify any contradictions (e.g., claims X years of experience but can't explain basics).
Respond with JSON array (empty array if no contradictions found):
[
  {{"description": "...", "segment_ids": ["0", "3"]}}
]
"""

    fallback = False
    output_raw = ""
    try:
        output_raw = await llm.complete(prompt=prompt, json_mode=True, task_type="node_scoring")
        data = json.loads(output_raw)
        if not isinstance(data, list):
            data = data.get("contradictions", data.get("flags", []))
        flags = [
            ConsistencyFlag(
                description=item.get("description", ""),
                segment_ids=[str(x) for x in item.get("segment_ids", [])],
            )
            for item in data
        ]
    except Exception as e:
        fallback = True
        flags = []
        state["errors"].append(f"consistency_node fallback: {e}")

    return {
        "consistency_flags": flags,
        "audit_log": state["audit_log"] + [_audit("consistency", prompt, output_raw, fallback)],
        "errors": state["errors"],
    }


async def behavioural_node(state: ScoringState) -> dict:
    """Score STAR completeness for behavioural answers. Fallback: score=0."""
    llm = get_llm_client()
    rubric_focus = state["rubric"].get("behavioural_topics", [])
    summary = _segments_summary(state["segments"])

    prompt = f"""Score STAR (Situation, Task, Action, Result) completeness for each behavioural answer.

Segments:
{summary}

{"Look for behavioural answers about: " + ", ".join(rubric_focus) if rubric_focus else "Identify behavioural/situational answers."}

For each relevant segment, respond with JSON array:
[
  {{"segment_index": 0, "star_completeness": 0-3, "notes": "..."}}
]

star_completeness: 0=no STAR, 1=partial, 2=most components, 3=full STAR
"""

    fallback = False
    output_raw = ""
    try:
        output_raw = await llm.complete(prompt=prompt, json_mode=True, task_type="node_scoring")
        data = json.loads(output_raw)
        if not isinstance(data, list):
            data = data.get("scores", data.get("behavioural", []))
        b_scores = [
            BehaviouralScore(
                segment_index=int(item.get("segment_index", 0)),
                star_completeness=max(0, min(3, int(item.get("star_completeness", 0)))),
                notes=item.get("notes", ""),
            )
            for item in data
        ]
    except Exception as e:
        fallback = True
        b_scores = [
            BehaviouralScore(segment_index=s["segment_index"], star_completeness=0, notes="fallback")
            for s in state["segments"]
        ]
        state["errors"].append(f"behavioural_node fallback: {e}")

    return {
        "behavioural_scores": b_scores,
        "audit_log": state["audit_log"] + [_audit("behavioural", prompt, output_raw, fallback)],
        "errors": state["errors"],
    }


async def synthesis_node(state: ScoringState) -> dict:
    """Aggregate all node outputs into a final verdict. Fallback: default verdict."""
    llm = get_llm_client()

    tech_summary = json.dumps(state["technical_scores"], indent=2)
    consistency_summary = json.dumps(state["consistency_flags"], indent=2)
    behavioural_summary = json.dumps(state["behavioural_scores"], indent=2)

    prompt = f"""Synthesise a hiring verdict from these assessment results.

Technical scores:
{tech_summary}

Consistency flags:
{consistency_summary}

Behavioural scores:
{behavioural_summary}

Rubric weights: {json.dumps(state["rubric"].get("weights", {}))}

Respond with JSON:
{{
  "overall_score": 0.0-10.0,
  "hire_recommendation": "strong_yes|yes|maybe|no",
  "strengths": ["..."],
  "concerns": ["..."],
  "segment_citations": ["0", "1"]
}}
"""

    fallback = False
    output_raw = ""
    try:
        output_raw = await llm.complete(prompt=prompt, json_mode=True, task_type="synthesis")
        data = json.loads(output_raw)
        verdict = FinalVerdict(
            overall_score=max(0.0, min(10.0, float(data.get("overall_score", 5.0)))),
            hire_recommendation=data.get("hire_recommendation", "maybe"),
            strengths=data.get("strengths", []),
            concerns=data.get("concerns", []),
            segment_citations=[str(x) for x in data.get("segment_citations", [])],
        )
    except Exception as e:
        fallback = True
        verdict = FinalVerdict(
            overall_score=5.0,
            hire_recommendation="maybe",
            strengths=[],
            concerns=["Scoring agent error — manual review required"],
            segment_citations=[],
        )
        state["errors"].append(f"synthesis_node fallback: {e}")

    return {
        "final_verdict": verdict,
        "audit_log": state["audit_log"] + [_audit("synthesis", prompt, output_raw, fallback)],
        "errors": state["errors"],
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_graph() -> StateGraph:
    graph = StateGraph(ScoringState)
    graph.add_node("technical_depth", technical_depth_node)
    graph.add_node("consistency", consistency_node)
    graph.add_node("behavioural", behavioural_node)
    graph.add_node("synthesis", synthesis_node)

    graph.add_edge(START, "technical_depth")
    graph.add_edge("technical_depth", "consistency")
    graph.add_edge("consistency", "behavioural")
    graph.add_edge("behavioural", "synthesis")
    graph.add_edge("synthesis", END)

    return graph.compile()


_compiled_graph = None


def _get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _build_graph()
    return _compiled_graph


async def run_scoring_agent(
    candidate_id: str,
    tenant_id: str,
    segments: list[dict],
    rubric: dict,
) -> ScoringState:
    initial: ScoringState = {
        "candidate_id": candidate_id,
        "tenant_id": tenant_id,
        "segments": segments,
        "rubric": rubric,
        "technical_scores": [],
        "consistency_flags": [],
        "behavioural_scores": [],
        "final_verdict": {},
        "audit_log": [],
        "errors": [],
    }
    graph = _get_graph()
    result = await graph.ainvoke(initial)
    return result
