# InterviewIQ POC — RUNBOOK

## Startup

```bash
cp .env.example .env
# Fill GROQ_API_KEY in .env
pip install -r requirements.txt
uvicorn app.main:app --reload
# API docs: http://localhost:8000/docs
```

Seed the database:
```bash
python seed/seed.py
```

---

## Scenario 1: Transcription Quality Degrades

**Symptoms**: WER is high, segments contain garbled text, retrieval quality drops.

**Diagnosis**:
1. Check `logs/audio_ingestion.jsonl` — look for `transcription_latency` spikes.
2. Run `python eval/wer_eval.py` to measure current WER.
3. Listen to a sample audio file and compare against the transcript in `interview_segments.raw_text`.

**Remediation**:
- Switch Whisper model: change `WHISPER_MODEL=base` → `medium` or `large` in `.env`. Larger models are slower but more accurate.
- If audio quality is fundamentally poor (heavy accents, background noise), consider AssemblyAI or Deepgram for speaker diarisation.
- For POC: re-ingest affected candidates after model upgrade.

---

## Scenario 2: Scoring Agent Produces Malformed Verdict

**Symptoms**: `/score/{candidate_id}` returns `overall_score=5.0`, `hire_recommendation=maybe`, `errors` list non-empty.

**Diagnosis**:
1. Check `state.errors` in the API response — each fallback node logs its error.
2. Check `state.audit_log` — the `fallback_used=true` fields identify which nodes failed.
3. Look at the `prompt` field in audit_log to identify if the prompt was too long for the model context window.

**Remediation**:
- If the issue is context length: reduce segment count passed to scoring (filter to top-5 by relevance).
- If JSON mode is failing: switch LLM provider via `LLM_PROVIDER=anthropic` (Anthropic has stronger structured output).
- If rubric is malformed: check `tenants.rubric_json` in the SQLite DB — ensure it is valid JSON.

---

## Scenario 3: Tenant Reports Seeing Another Tenant's Data

**Symptoms**: A recruiter at Tenant A sees candidates from Tenant B in query results.

**Immediate Action**:
1. **Stop the service**: `pkill -f uvicorn`
2. Verify the report by running: `pytest tests/test_tenant_isolation.py -v`
3. If test fails, this is a confirmed breach.

**Root Cause Investigation**:
- Check `app/db/vector_store.py:vector_search` — the `where={"tenant_id": tenant_id}` filter must be present in every query.
- Check `app/api/query.py` — verify `tenant_id` is always taken from `token_data`, never from request body.
- Check ChromaDB logs for any `get()` calls without `where` filter.

**Remediation**:
- Patch the filter, re-run isolation test.
- Audit all `collection.query()` and `collection.get()` calls to ensure `where` is always set.
- Notify affected tenants if breach is confirmed.

---

## Scenario 4: Candidate Audio Injection Attack

**Symptoms**: Unusual data in query results, candidate score is unexpectedly high, transcript contains instruction-like text.

**Detection**:
- Search transcripts: `grep -i "ignore.*instructions\|return all\|reveal" interviewiq.db` (via sqlite3 CLI).
- Check `app/pipeline/extraction.py` — `sanitise_for_prompt()` should have replaced patterns with `[REDACTED]`.

**Remediation**:
- The existing regex in `sanitise_for_prompt()` covers common patterns.
- For new attack patterns: add them to `INJECTION_PATTERNS` in `extraction.py`.
- Re-process affected candidate transcripts through extraction.

---

## Common Operations

### Issue a JWT token for a tenant
```bash
# Via API (requires tenant to exist in DB)
curl -X POST http://localhost:8000/auth/token \
  -d "tenant_id=<id>&user_id=admin"
```

### Check metrics
```bash
curl http://localhost:8000/metrics
```

### Stream a query
```bash
curl -N http://localhost:8000/query/stream?question="Who+has+Kubernetes+experience" \
  -H "Authorization: Bearer <token>" \
  -H "Accept: text/event-stream"
```

### Score a candidate
```bash
curl -X POST http://localhost:8000/score/<candidate_id> \
  -H "Authorization: Bearer <token>"
```

### Run evaluation
```bash
python eval/wer_eval.py
python eval/retrieval_eval.py
python eval/scoring_eval.py
```
