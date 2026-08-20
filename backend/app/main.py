import uuid
import time
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from qdrant_client import QdrantClient

from app.config import settings
from app.models import (
    HealthResponse,
    WarmupResponse,
    QueryRequest,
    QueryResponse,
    LatencyBreakdown,
    SourceDoc
)
from app.services.stt_service import stt_service
from app.services.embedding_service import embedding_service
from app.services.retrieval_service import retrieval_service
from app.services.llm_service import llm_service
from app.services.guardrails import guardrails

app = FastAPI(
    title="Voice-Enabled RAG Orchestrator",
    description="Backend API for speech-to-text, vector search, and grounded response generation.",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory latency logs
latency_history = []

def record_latency(stt_ms: Optional[float], retrieval_ms: float, llm_ms: float, total_ms: float):
    """
    Log and record latency metrics for percentile calculations.
    """
    latency_history.append({
        "stt": stt_ms,
        "retrieval": retrieval_ms,
        "llm": llm_ms,
        "total": total_ms,
        "timestamp": time.time()
    })
    logger.info(f"Latency log added - Total: {total_ms:.2f}ms (STT: {stt_ms}ms, Retrieval: {retrieval_ms:.2f}ms, LLM: {llm_ms:.2f}ms)")

@app.get("/api/health", response_model=HealthResponse)
def health_check():
    qdrant_connected = False
    try:
        client = retrieval_service.client
        client.get_collections()
        qdrant_connected = True
    except Exception:
        pass

    return HealthResponse(
        status="healthy",
        qdrant_connected=qdrant_connected,
        version="1.0.0"
    )

@app.get("/api/warmup", response_model=WarmupResponse)
def warmup():
    # Pre-initialize heavy services to save first-request latency on Render
    try:
        _ = embedding_service.model
        logger.info("Warmup: Embedded model loaded successfully.")
    except Exception as e:
        logger.error(f"Warmup model load error: {e}")
        
    return WarmupResponse(
        status="warmed_up",
        message="Backend services initialized and ready."
    )

@app.post("/api/query-text", response_model=QueryResponse)
def query_text(payload: QueryRequest):
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Received text query request.")
    start_total = time.perf_counter()

    query_str = payload.query
    stt_ms = 0.0

    # L1 Guardrail (Input check)
    if not guardrails.validate_input(query_str):
        total_ms = (time.perf_counter() - start_total) * 1000
        record_latency(None, 0.0, 0.0, total_ms)
        return QueryResponse(
            request_id=request_id,
            query=query_str,
            answer=guardrails.refusal_message,
            sources=[],
            latency=LatencyBreakdown(stt_ms=None, retrieval_ms=0.0, llm_ms=0.0, total_ms=total_ms),
            guardrail_refusal=True
        )

    # Retrieval Stage
    start_retrieval = time.perf_counter()
    sources = []
    try:
        sources = retrieval_service.search(query_str, limit=3)
    except Exception as e:
        logger.error(f"[{request_id}] Retrieval failure: {e}")
    retrieval_ms = (time.perf_counter() - start_retrieval) * 1000

    # L2 Guardrail (Context check)
    if not guardrails.validate_context(sources):
        total_ms = (time.perf_counter() - start_total) * 1000
        record_latency(None, retrieval_ms, 0.0, total_ms)
        return QueryResponse(
            request_id=request_id,
            query=query_str,
            answer=guardrails.refusal_message,
            sources=sources,
            latency=LatencyBreakdown(stt_ms=None, retrieval_ms=retrieval_ms, llm_ms=0.0, total_ms=total_ms),
            guardrail_refusal=True
        )

    # LLM Generation Stage
    start_llm = time.perf_counter()
    try:
        contexts = [doc.passage for doc in sources]
        answer = llm_service.generate_answer(query_str, contexts)
    except Exception as e:
        logger.error(f"[{request_id}] LLM Generation failure: {e}")
        answer = guardrails.refusal_message
    llm_ms = (time.perf_counter() - start_llm) * 1000

    # L3 Guardrail (Grounding validation)
    guardrail_refusal = False
    if answer != guardrails.refusal_message:
        is_grounded = guardrails.validate_grounding(query_str, answer, [doc.passage for doc in sources])
        if not is_grounded:
            logger.warning(f"[{request_id}] L3 Guardrail triggered: generated answer is not grounded.")
            answer = guardrails.refusal_message
            guardrail_refusal = True

    total_ms = (time.perf_counter() - start_total) * 1000
    record_latency(None, retrieval_ms, llm_ms, total_ms)

    return QueryResponse(
        request_id=request_id,
        query=query_str,
        answer=answer,
        sources=sources,
        latency=LatencyBreakdown(stt_ms=None, retrieval_ms=retrieval_ms, llm_ms=llm_ms, total_ms=total_ms),
        guardrail_refusal=guardrail_refusal
    )

@app.post("/api/query-audio", response_model=QueryResponse)
async def query_audio(file: UploadFile = File(...)):
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Received audio query request (filename: {file.filename}).")
    start_total = time.perf_counter()

    # Read audio bytes
    try:
        audio_bytes = await file.read()
    except Exception as e:
        logger.error(f"[{request_id}] Failed to read uploaded audio file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not read uploaded audio file."
        )

    # STT Stage
    start_stt = time.perf_counter()
    try:
        query_str = stt_service.transcribe(audio_bytes, file.filename)
    except Exception as e:
        logger.error(f"[{request_id}] STT transcription failed: {e}")
        total_ms = (time.perf_counter() - start_total) * 1000
        return QueryResponse(
            request_id=request_id,
            query="",
            answer="Transcribing voice input failed. Please try again.",
            sources=[],
            latency=LatencyBreakdown(stt_ms=(time.perf_counter() - start_stt) * 1000, retrieval_ms=0.0, llm_ms=0.0, total_ms=total_ms),
            guardrail_refusal=True
        )
    stt_ms = (time.perf_counter() - start_stt) * 1000

    # L1 Guardrail (Input check)
    if not guardrails.validate_input(query_str):
        total_ms = (time.perf_counter() - start_total) * 1000
        record_latency(stt_ms, 0.0, 0.0, total_ms)
        return QueryResponse(
            request_id=request_id,
            query=query_str,
            answer=guardrails.refusal_message,
            sources=[],
            latency=LatencyBreakdown(stt_ms=stt_ms, retrieval_ms=0.0, llm_ms=0.0, total_ms=total_ms),
            guardrail_refusal=True
        )

    # Retrieval Stage
    start_retrieval = time.perf_counter()
    sources = []
    try:
        sources = retrieval_service.search(query_str, limit=3)
    except Exception as e:
        logger.error(f"[{request_id}] Retrieval failure: {e}")
    retrieval_ms = (time.perf_counter() - start_retrieval) * 1000

    # L2 Guardrail (Context check)
    if not guardrails.validate_context(sources):
        total_ms = (time.perf_counter() - start_total) * 1000
        record_latency(stt_ms, retrieval_ms, 0.0, total_ms)
        return QueryResponse(
            request_id=request_id,
            query=query_str,
            answer=guardrails.refusal_message,
            sources=sources,
            latency=LatencyBreakdown(stt_ms=stt_ms, retrieval_ms=retrieval_ms, llm_ms=0.0, total_ms=total_ms),
            guardrail_refusal=True
        )

    # LLM Generation Stage
    start_llm = time.perf_counter()
    try:
        contexts = [doc.passage for doc in sources]
        answer = llm_service.generate_answer(query_str, contexts)
    except Exception as e:
        logger.error(f"[{request_id}] LLM Generation failure: {e}")
        answer = guardrails.refusal_message
    llm_ms = (time.perf_counter() - start_llm) * 1000

    # L3 Guardrail (Grounding validation)
    guardrail_refusal = False
    if answer != guardrails.refusal_message:
        is_grounded = guardrails.validate_grounding(query_str, answer, [doc.passage for doc in sources])
        if not is_grounded:
            logger.warning(f"[{request_id}] L3 Guardrail triggered: generated answer is not grounded.")
            answer = guardrails.refusal_message
            guardrail_refusal = True

    total_ms = (time.perf_counter() - start_total) * 1000
    record_latency(stt_ms, retrieval_ms, llm_ms, total_ms)

    return QueryResponse(
        request_id=request_id,
        query=query_str,
        answer=answer,
        sources=sources,
        latency=LatencyBreakdown(stt_ms=stt_ms, retrieval_ms=retrieval_ms, llm_ms=llm_ms, total_ms=total_ms),
        guardrail_refusal=guardrail_refusal
    )

@app.get("/api/metrics")
def get_metrics():
    """
    Computes real P50 / P70 / P100 latency percentiles based on history.
    """
    if not latency_history:
        return {
            "count": 0,
            "stt": {"p50": 0.0, "p70": 0.0, "p100": 0.0},
            "retrieval": {"p50": 0.0, "p70": 0.0, "p100": 0.0},
            "llm": {"p50": 0.0, "p70": 0.0, "p100": 0.0},
            "total": {"p50": 0.0, "p70": 0.0, "p100": 0.0}
        }

    total_lats = [log["total"] for log in latency_history]
    retrieval_lats = [log["retrieval"] for log in latency_history]
    llm_lats = [log["llm"] for log in latency_history]
    stt_lats = [log["stt"] for log in latency_history if log["stt"] is not None]

    def calc_stats(lats):
        if not lats:
            return {"p50": 0.0, "p70": 0.0, "p100": 0.0}
        return {
            "p50": round(float(np.percentile(lats, 50)), 2),
            "p70": round(float(np.percentile(lats, 70)), 2),
            "p100": round(float(np.percentile(lats, 100)), 2)
        }

    return {
        "count": len(latency_history),
        "stt": calc_stats(stt_lats),
        "retrieval": calc_stats(retrieval_lats),
        "llm": calc_stats(llm_lats),
        "total": calc_stats(total_lats)
    }
