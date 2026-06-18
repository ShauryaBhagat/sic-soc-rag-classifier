# SIC/SOC Classifier — RAG + LLM Pipeline

> Built as a personal project to deepen hands-on experience with GenAI
> engineering — directly mirrors the LLM-integrated classification
> architecture I collaborated on at the Office for National Statistics.

**[→ Live Demo](https://sic-soc-rag-classifier.streamlit.app)**

---

## What it does

Classifies free-text industry descriptions into
[UK SIC 2007](https://www.ons.gov.uk/methodology/classificationsandstandards/ukstandardindustrialclassificationofeconomicactivities/uksic2007)
codes using a production-style RAG + LLM pipeline.

**Example:**
- Input: `"GP doctor working in an NHS health centre seeing patients"`
- Output: `SIC 86210 — General medical practice activities` (HIGH confidence)

---

## Architecture

Free-text input
│
▼
Input guardrail ── rejects empty / numeric / too-short inputs
│             logs rejections to audit file
▼
Vector retrieval ── 15,957 ONS SIC activity descriptions
│              embedded with OpenAI text-embedding-3-small
│              stored in ChromaDB
│              returns top-5 candidates by similarity
▼
LLM classification ── GPT-4o-mini (temperature=0)
│                prompt-injected with retrieved candidates
│                returns: SIC code + confidence + reasoning
▼
Confidence gate
├── HIGH / MEDIUM → accept prediction ✓
└── AMBIGUOUS → agentic retry with chain-of-thought
│
├── resolved → accept ✓
└── still ambiguous → human review ⚠
▼
Output guardrail ── validates predicted code against ONS master list
│
▼
Langfuse trace ── full observability: input, retrieval, LLM calls,
confidence score, latency, token usage, audit log

---

## Stack

| Component | Technology |
|-----------|------------|
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Vector store | ChromaDB (in-memory for stateless cloud deployment) |
| Orchestration | LangChain |
| Observability | Langfuse (traces, spans, confidence scores) |
| UI | Streamlit |
| Deployment | Streamlit Community Cloud |
| Governance | Input guardrails, output validation, JSON audit log |

---

## Key design decisions

**In-memory ChromaDB** — correct pattern for stateless cloud deployment.
No persistent disk dependency; rebuilt at startup via `@st.cache_resource`
(shared across users, not rebuilt per request). Cost: ~$0.003, ~60s.

**temperature=0** — deterministic LLM output. Essential for reproducible
classification; non-zero temperature would make audit logs unreproducible.

**Agentic retry with chain-of-thought** — on AMBIGUOUS confidence, the
pipeline reformulates the prompt asking the model to reason step by step
before committing. After 2 attempts it escalates to human review rather
than returning a low-confidence prediction silently.

**Audit log (JSON Lines)** — every classification decision logged with
timestamp, input, candidates considered, prediction, confidence, retry
count, and outcome. Full traceability for governance and compliance.

---

## Observability

All pipeline steps traced in Langfuse:

- `input_validation` span — guardrail decision + reason
- `vector_retrieval` span — candidate count + scores  
- `llm_call` generation — full prompt, response, token usage, latency
- `output_guardrail` span — validation result
- `confidence_score` — numeric metric (HIGH=1.0, MEDIUM=0.5, AMBIGUOUS=0.0)

---

## Local setup

```bash
git clone https://github.com/[your-username]/sic-soc-rag-classifier
cd sic-soc-rag-classifier
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your API keys
python ingest.py              # build vector store (~$0.003)
streamlit run app.py          # launch UI
```

**Required environment variables** (see `.env.example`):

OPENAI_API_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=

---

## Project structure

├── app.py                    # Streamlit UI
├── ingest.py                 # Embed SIC codes into ChromaDB
├── retrieve.py               # Vector similarity search + guardrails
├── classify.py               # LLM classification + agentic retry
├── eval.py                   # Evaluation metrics
├── data/
│   ├── SIC_2007_index.xlsx   # ONS SIC 2007 source data
│   ├── chroma_store/         # ChromaDB vector store (local only)
│   ├── classification_results.csv
│   └── eval_report.json
├── docker-compose.yml        # Local reproducibility
├── requirements.txt          # Pinned dependencies
└── .env.example              # Environment variable template

---

## Evaluation results

Run `python eval.py` after classification to generate a report:

── Classification Outcomes ──────────────────────────────
Classified:                X  (X%)
Escalated to human review: X  (X%)
── Confidence Distribution ──────────────────────────────
HIGH       X  (X%)
MEDIUM     X  (X%)
AMBIGUOUS  X  (X%)
── Agentic Retry Stats ──────────────────────────────────
Avg LLM attempts per run:  X
Runs that triggered retry: X

---

*Built by Shaurya Bhagat · [LinkedIn](https://linkedin.com/in/shaurya-bhagat-a11460166)*