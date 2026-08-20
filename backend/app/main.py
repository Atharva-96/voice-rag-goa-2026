import uuid
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import settings
from app.models import HealthResponse, WarmupResponse

app = FastAPI(
    title="Voice-Enabled RAG Orchestrator",
    description="Backend API for speech-to-text, vector search, and grounded response generation.",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint to verify system status and DB connectivity.
    """
    qdrant_connected = False
    try:
        # Simple ping to Qdrant to verify connection
        client = QdrantClient(url=settings.QDRANT_API_URL, api_key=settings.QDRANT_API_KEY)
        client.get_collections()
        qdrant_connected = True
    except Exception:
        # Do not let DB connection failure crash the health endpoint itself, return false in the payload
        pass

    return HealthResponse(
        status="healthy",
        qdrant_connected=qdrant_connected,
        version="1.0.0"
    )

@app.get("/api/warmup", response_model=WarmupResponse)
def warmup():
    """
    Warmup endpoint to initialize models/services on Render cold-start.
    """
    # This endpoint is explicitly triggered by the frontend to spin up Render serverless instances
    return WarmupResponse(
        status="warmed_up",
        message="Backend services initialized and ready."
    )
