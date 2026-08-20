# Voice-Enabled RAG Model (voice-rag-goa-2026)

A zero-budget, high-performance Voice-Enabled RAG system built for **Hacker House Goa 2026 - Shortlisting Task 2**.

---

## Architecture Overview

```
Voice Input ──► Sarvam STT (Saaras v3) ──► Query Processing (L1 Guardrail)
                                                   │
                                                   ▼
Grounding validation (L3) ◄── LLM Gen (Groq) ◄── Vector Search (Qdrant & L2 Guardrail)
```

- **Frontend**: Next.js 14, Tailwind CSS, Web Audio API (deployed to Vercel).
- **Backend**: FastAPI, Python 3.11, fastembed (deployed to Render free tier).
- **STT**: Sarvam Saaras v3 REST API.
- **LLM**: Groq Llama-3.1-8B-Instant.
- **Vector DB**: Qdrant Cloud (Free tier) with multilingual-e5-small embeddings.

---

## Directory Structure

```
├── backend/                  # FastAPI Web Server
│   ├── app/                  # Main server codebase
│   └── scripts/              # Dataset Ingestion & Benchmarking
├── frontend/                 # Next.js 14 Application
├── docs/                     # Documentation & PROGRESS.md
└── docker-compose.yml        # Local Qdrant container
```

---

## Local Setup

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- Node.js 18+

### Step 1: Start Qdrant Local Service
```bash
docker-compose up -d
```

### Step 2: Setup Backend
```bash
cd backend
python -m venv venv
source venv/Scripts/activate # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp ../.env.example .env
# Edit .env with your credentials
uvicorn app.main:app --reload --port 8000
```

### Step 3: Setup Frontend
```bash
cd frontend
npm install
npm run dev
```
