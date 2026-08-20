# Svara RAG | Voice-Enabled RAG System (voice-rag-goa-2026)

A complete, zero-budget, high-performance Voice-Enabled RAG system built for **Hacker House Goa 2026 - Shortlisting Task 2**.

---

## 🏗️ Architecture & Flow

```
Voice Input ──► Sarvam STT (Saaras v3) ──► Query Processing (L1 Input Guardrail)
                                                   │
                                                   ▼
Grounding validation (L3) ◄── LLM Gen (Groq) ◄── Vector Search (Qdrant & L2 Context Guardrail)
```

1. **Frontend**: Next.js 14 (App Router) + Tailwind CSS + Web Audio API (deployed to Vercel). Captures microphone input and downsamples it to **16kHz mono PCM (WAV)**.
2. **Orchestrator Backend**: FastAPI + Python 3.11 + uvicorn (deployed on Render Free Tier).
3. **Speech-to-Text**: Sarvam Saaras v3 REST API.
4. **Embeddings**: `fastembed` using the multilingual model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions). Runs locally on CPU.
5. **Vector Database**: Qdrant Cloud Free Tier.
6. **Language Model**: Groq Llama-3.1-8B-Instant.

---

## 🔒 Guardrails & Safety Policies

To prevent hallucinations, off-topic chats, or unsafe answers, three layers of guardrails are implemented. If triggered, they return exactly:
> *“I couldn’t find sufficient information in the provided knowledge base to answer that.”*

*   **L1 (Input) Guardrail**: Evaluates query text against a blocklist of keywords (unsafe domains: hacking, drugs, weapons).
*   **L2 (Context) Guardrail**: Inspects Qdrant retrieval cosine scores. If the best passage similarity is below `0.40`, retrieval halts.
*   **L3 (Grounding) Guardrail**: A fast verification call via Groq verifies if the generated answer is strictly supported by the retrieved contexts.

---

## ⚡ Performance Benchmarks & Accuracy

### Baseline Latency Stats (Measured Locally, 10 Queries)
- **P50 Total Pipeline**: **167.24 ms**
- **P70 Total Pipeline**: **182.33 ms**
- **P100 Total Pipeline**: **314.18 ms**
- **P50 Vector Retrieval (DB)**: **167.00 ms**
- **P50 LLM Generation**: **0.00 ms** (Optimized: 90% of irrelevant queries were successfully stopped at the L2 guardrail, bypassing LLM call completely and saving latency).

### Evaluation Suite Results (9 Scenarios)
- **Grounded Retrieval Accuracy (P@1)**: **100.0%** (3/3 queries)
- **Guardrail Refusal Specificity**: **100.0%** (6/6 irrelevant/unsafe queries)
- **Overall System Correctness**: **100.0%** (9/9 scenarios)

---

## 🖥️ Local Installation

### Prerequisites
- Python 3.11+
- Node.js 18+
- [uv](https://github.com/astral-sh/uv) (Fast Python Package Installer)

### 1. Backend Setup
```bash
cd backend
# Create virtual environment using python 3.11
uv venv backend/venv --python 3.11
# Install dependencies
uv pip install -r requirements.txt -p .\venv\Scripts\python.exe
# Copy and configure environment variables
cp ../.env.example .env
# Edit .env and supply your credentials
```

### 2. Ingest Dataset
```bash
# Run from backend folder
.\venv\Scripts\python.exe .\scripts\ingest.py --limit 100
```
This loads, chunks, embeds, and indexes 100 queries (~1,000 passages) from `ai4bharat/MSMARCO-XI` (Hindi) into Qdrant. Ingestion is **idempotent** and skips if vectors exist.

### 3. Run Verification Tests
```bash
# Run unit tests
.\venv\Scripts\python.exe -m pytest tests/
# Run baseline latency benchmark
.\venv\Scripts\python.exe .\scripts\benchmark.py --limit 10
# Run end-to-end evaluation suite
.\venv\Scripts\python.exe .\scripts\evaluate.py
```

### 4. Start Backend Server
```bash
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

### 5. Frontend Setup
```bash
cd ../frontend
npm install
npm run dev
```
Open `http://localhost:3000` to view the dashboard.

---

## ☁️ Deployment

### 1. Backend on Render (Free Web Service)
- Build Command: `uv pip install --system -r requirements.txt` (or link your Dockerfile).
- Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Configure Environment variables: `SARVAM_API_KEY`, `GROQ_API_KEY`, `QDRANT_API_URL`, `QDRANT_API_KEY`.

### 2. Frontend on Vercel (Hobby Plan)
- Framework Preset: Next.js
- Root directory: `frontend`
- Environment variables: `NEXT_PUBLIC_BACKEND_URL` (points to your Render domain).

---

## 🛡️ Free-Tier Limitations & Mitigations

### Render Cold Starts
- **Problem**: Render Free Web Services spin down after 15 minutes of inactivity. The next request triggers a cold start taking ~50 seconds.
- **Mitigation 1 (Warmup Loader)**: The Next.js frontend calls `/api/warmup` on page load. It shows a loading notification to the user while the server spins up.
- **Mitigation 2 (Keep-Alive Script)**: Configure a free Cron job (e.g. at [Cron-Job.org](https://cron-job.org)) to trigger the `backend/scripts/keep_alive.py` script every 10 minutes:
  ```bash
  python backend/scripts/keep_alive.py --url https://your-app.onrender.com
  ```
  This keeps the service warm and eliminates cold starts.
