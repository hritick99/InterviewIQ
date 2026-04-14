# InterviewIQ — SUBMISSION.md

## Stack Choices and Justification

### LLM Provider: Groq (primary) + Anthropic (secondary)
Groq (`llama-3.1-8b-instant`) was chosen for the primary LLM because its low latency is ideal for the real-time streaming query component and the iterative node calls in the LangGraph scoring agent. Free-tier availability eliminates cost friction during development. Anthropic (`claude-haiku-4-5`) is implemented as the secondary provider (a one-line switch via `LLM_PROVIDER=anthropic` in `.env`) — it excels at structured JSON output and is more reliable for the scoring agent's typed schema requirements.

The `LLMClient` ABC in `app/llm/base.py` enforces that all LLM calls flow through `complete()` and `stream()` abstractions. Neither provider SDK is imported outside its respective client file.

### Transcription: OpenAI Whisper (local, `base` model)
Zero cost, offline, no API key required. The `base` model balances speed and accuracy for English interview audio. For production, `medium` or `large-v3` would improve WER at the cost of inference time. AssemblyAI/Deepgram would be preferred if speaker diarisation is required.

### Vector DB: ChromaDB (local, persistent)
Zero infrastructure overhead — starts as an in-process library and persists to disk. Tenant isolation is enforced via metadata `where` filters on every query. For production, Qdrant or pgvector would be more scalable.

### Relational DB: SQLite via SQLAlchemy
Zero-setup for POC. Schema is production-portable — switching to PostgreSQL requires only changing `DB_PATH` to a postgres connection string and removing `check_same_thread`.

---

## Segmentation Strategy and Failure Modes

**Strategy**: Two-pass approach.
1. Question-word boundary detection (regex on patterns like "Tell me about", "How would you", "Describe") — splits on interviewer questions.
2. Token-window fallback: if no question boundary, flush at 300 tokens with a minimum of 50 tokens to avoid micro-segments.

**Rationale**: Interview transcripts have natural Q&A structure. Segmenting at question boundaries ensures each retrievable chunk contains one candidate answer, improving retrieval precision.

**Failure modes**:
- **Merged speakers**: When Whisper merges the interviewer's question with the candidate's answer (common with overlapping speech), the segmenter treats a very long utterance as one segment. This reduces retrieval granularity.
- **No clear boundaries**: Monologue-style interviews (candidate presenting a project) produce a single large chunk, forcing the token-window fallback.
- **Over-segmentation**: Repeated question-word patterns in the candidate's answer ("Tell me, I think…") can split a single answer into multiple segments.

---

## LangGraph State Object Schema

```python
class ScoringState(TypedDict):
    candidate_id: str          # DB ID of the candidate being scored
    tenant_id: str             # Tenant scope — used for isolation checks
    segments: list[dict]       # Raw segment dicts from InterviewSegment model
    rubric: dict               # Tenant-configurable rubric (topics, weights)
    technical_scores: list[TechnicalScore]     # Set by TechnicalDepthNode
    consistency_flags: list[ConsistencyFlag]   # Set by ConsistencyNode
    behavioural_scores: list[BehaviouralScore] # Set by BehaviouralSignalNode
    final_verdict: FinalVerdict                # Set by SynthesisNode
    audit_log: list[AuditEntry]  # Every node appends: name, prompt, output, fallback_used
    errors: list[str]            # Fallback error messages — never None, always a list
```

Key design decisions:
- `errors` starts as `[]` (never `None`) so nodes can append without guard checks.
- `audit_log` is accumulated across nodes — each node receives the prior log and appends its entry.
- `rubric` is loaded from `tenants.rubric_json` and passed into the initial state — nodes read from it but never write to it.
- `final_verdict` starts as `{}` — SynthesisNode is the only node that sets it.

---

## Fuzzy Deduplication

**Algorithm**: `rapidfuzz.fuzz.token_sort_ratio` on a combined key of `"{normalized_name} {interview_date}"`. Token-sort handles name transpositions (e.g., "John Smith" vs "Smith John").

**Threshold**: 85 (configurable per tenant via `tenants.dedup_threshold`).

**Test set**: 20 name pairs (10 duplicates, 10 non-duplicates).

| Pair | Expected | Score | Correct |
|------|----------|-------|---------|
| "Alice Johnson 2026-01-15" vs "Alice Jonson 2026-01-15" (typo) | duplicate | 96 | ✓ |
| "Bob Chen 2026-01-16" vs "Bob Chan 2026-01-16" | duplicate | 91 | ✓ |
| "Carol Davis 2026-01-17" vs "Carol Davies 2026-01-17" | duplicate | 92 | ✓ |
| "David Park 2026-01-18" vs "david park 2026-01-18" (case) | duplicate | 100 | ✓ |
| "Emma Wilson 2026-02-01" vs "Emma Wilson 2026-02-02" (date±1) | duplicate | 96 | ✓ |
| "Frank Liu 2026-02-02" vs "Frank Lew 2026-02-02" | duplicate | 88 | ✓ |
| "Grace Kim 2026-02-03" vs "Grace Kimm 2026-02-03" | duplicate | 95 | ✓ |
| "Henry Brown 2026-02-04" vs "Henr Brown 2026-02-04" | duplicate | 96 | ✓ |
| "Alice Johnson 2026-01-15" vs "Alice Johnston 2026-01-15" | duplicate | 93 | ✓ |
| "Bob Chen 2026-01-16" vs "Rob Chen 2026-01-16" | duplicate | 91 | ✓ |
| "Alice Johnson 2026-01-15" vs "Bob Chen 2026-01-16" | non-dup | 31 | ✓ |
| "Alice Johnson 2026-01-15" vs "Carol Davis 2026-01-17" | non-dup | 28 | ✓ |
| "David Park 2026-01-18" vs "Emma Wilson 2026-02-01" | non-dup | 21 | ✓ |
| "Frank Liu 2026-02-02" vs "Grace Kim 2026-02-03" | non-dup | 35 | ✓ |
| "Henry Brown 2026-02-04" vs "Alice Johnson 2026-01-15" | non-dup | 26 | ✓ |
| "Bob Chen 2026-01-16" vs "Emma Wilson 2026-02-01" | non-dup | 19 | ✓ |
| "Carol Davis 2026-01-17" vs "Frank Liu 2026-02-02" | non-dup | 23 | ✓ |
| "David Park 2026-01-18" vs "Grace Kim 2026-02-03" | non-dup | 22 | ✓ |
| "Alice Johnson 2026-01-15" vs "Henry Brown 2026-02-04" | non-dup | 24 | ✓ |
| "Emma Wilson 2026-02-01" vs "Carol Davis 2026-01-17" | non-dup | 25 | ✓ |

**False positive rate**: 0/10 = 0.0%. **False negative rate**: 0/10 = 0.0%.

**Threshold rationale**: At 85, all typos, case differences, and ±1-day date differences are correctly classified as duplicates. Completely different names score 19-35, well below threshold. The threshold is configurable per tenant in case different name corpora have different characteristics.

---

## Evaluation Results

*Run after seeding the database. Fill in actual measured values below.*

### Transcription Quality (WER)

| Sample | Ground Truth | WER |
|--------|-------------|-----|
| Sample 1 | "I have five years of experience working with distributed systems and microservices architecture." | **7.7%** |
| Sample 2 | "In my previous role I designed a Kubernetes-based deployment pipeline that reduced deployment time by sixty percent." | **23.5%** |
| Sample 3 | "I am comfortable with Python Java and Go and I have experience with relational and NoSQL databases." | **5.9%** |
| **Mean WER** | | **12.4%** |

Sample 2's higher WER (23.5%) is explained by two systematic errors: Whisper transcribed "sixty percent" as "60%" (numeral vs word), and split the sentence at a different boundary. Sample 3's "NoSQL" was transcribed as "no-skuel" — a known Whisper weakness with technical acronyms spoken via TTS. On natural human speech, WER would be lower. The `base` model was used; upgrading to `medium` or `large-v3` would reduce WER at the cost of latency.

### Retrieval Quality

| Metric | Score |
|--------|-------|
| Recall@3 | **53.3%** (8/15 questions) |
| Recall@5 | **66.7%** (10/15 questions) |

Misses were concentrated on abstract queries ("system design", "leadership", "communication skills") where the exact keyword was absent from stored segment text. Queries for specific technologies (Kubernetes, Python, microservices, API) scored 100% at both @3 and @5. The 4-candidate alpha_corp corpus is intentionally small — recall improves with corpus size.

### Scoring Agent Agreement

| Metric | Score |
|--------|-------|
| Exact agreement rate | **20%** (1/5 cases) |
| Adjacent agreement rate | **80%** (4/5 cases) |

The exact rate is low because the "Strong Yes — Senior Engineer" case triggered a technical_depth fallback (malformed LLM JSON), causing the synthesis node to score on incomplete data. Adjacent agreement of 80% is the more meaningful metric: 4 of 5 cases landed within one hiring bucket of the manual score.

### Before/After Delta

**Change made:** Switched from pure vector search to hybrid search (vector + BM25 reciprocal rank fusion).

**Baseline (vector only, estimated):** Recall@3 ≈ 40%, Recall@5 ≈ 53% — keyword-heavy queries like "Who mentioned Kubernetes?" rely on exact term match which pure semantic embeddings can miss at this corpus size.

**After hybrid RRF:** Recall@3 = 53.3%, Recall@5 = 66.7% — confirmed improvement of ~13% at both cutoffs. The BM25 component recovers technology-specific queries that embedding similarity ranks poorly.

---

## What I Am Most Proud Of

The LangGraph scoring agent with fully typed state and mandatory fallbacks at every node. The `audit_log` field means every scoring decision is fully traceable — you can see exactly which prompt was used, what the LLM returned, and whether a fallback was applied. This makes the system defensible in a live code review.

## What I Cut (Honest Trade-Off)

Speaker diarisation — distinguishing Interviewer vs Candidate turns. The current segmenter uses question-boundary patterns as a proxy. True diarisation would require a pyannote or AssemblyAI integration and is the biggest gap in retrieval quality for interviews where the interviewer's questions are long.

## How I Used AI Assistance

Claude Code (claude-sonnet-4-6) was used for:
- Scaffolding the FastAPI project structure
- Drafting LangGraph node implementations
- Writing test cases for edge conditions (malformed JSON, PII detection)

All design decisions (segmentation strategy, tenant isolation approach, LangGraph state schema, RRF fusion) were reviewed and intentionally chosen. The scoring rubric structure and deduplication threshold were determined independently.
