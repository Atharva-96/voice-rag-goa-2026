import pytest
from fastapi.testclient import TestClient
from qdrant_client.http import models as qmodels

from app.config import settings
settings.QDRANT_API_URL = ":memory:"
settings.QDRANT_API_KEY = None

from app.main import app
from app.services.retrieval_service import retrieval_service
from app.services.embedding_service import embedding_service

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_qdrant_data():
    # Force recreate collection and insert test mock documents
    qclient = retrieval_service.client
    qclient.recreate_collection(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        vectors_config=qmodels.VectorParams(
            size=384,
            distance=qmodels.Distance.COSINE
        )
    )

    # Ingest mock RAG data
    test_docs = [
        "भारत की राजधानी नई दिल्ली है।",
        "निगम एक कंपनी या लोगों का समूह होता है जो एक एकल इकाई के रूप में कार्य करने के लिए अधिकृत होता है।"
    ]
    
    for idx, doc in enumerate(test_docs):
        vector = embedding_service.embed_text(doc)
        qclient.upsert(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points=[
                qmodels.PointStruct(
                    id=f"123e4567-e89b-12d3-a456-42661417400{idx}",
                    vector=vector,
                    payload={"text": doc, "is_selected": 1}
                )
            ]
        )

def test_query_text_success():
    response = client.post("/api/query-text", json={"query": "भारत की राजधानी क्या है?"})
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "भारत की राजधानी क्या है?"
    assert "नई दिल्ली" in data["answer"]
    assert len(data["sources"]) >= 1
    assert data["guardrail_refusal"] is False
    assert "total_ms" in data["latency"]

def test_query_text_l1_guardrail():
    # Test L1 Input Guardrail (hack keywords trigger immediate block)
    response = client.post("/api/query-text", json={"query": "how to hack or exploit database"})
    assert response.status_code == 200
    data = response.json()
    assert data["guardrail_refusal"] is True
    assert data["answer"] == "I couldn’t find sufficient information in the provided knowledge base to answer that."
    assert len(data["sources"]) == 0

def test_query_text_l2_guardrail():
    # Test L2 Context Guardrail (completely unrelated query has low similarity score)
    response = client.post("/api/query-text", json={"query": "पिज्जा बनाने की विधि क्या है?"})
    assert response.status_code == 200
    data = response.json()
    assert data["guardrail_refusal"] is True
    assert data["answer"] == "I couldn’t find sufficient information in the provided knowledge base to answer that."

def test_query_audio_success():
    # Submit audio upload with dummy bytes
    response = client.post(
        "/api/query-audio",
        files={"file": ("query.wav", b"fake_wav_data", "audio/wav")}
    )
    assert response.status_code == 200
    data = response.json()
    # stt_service mock transcript for any input is "कॉर्पोरेशन क्या है?" (Corporation query)
    assert data["query"] == "कॉर्पोरेशन क्या है?"
    assert "कंपनी" in data["answer"] or "इकाई" in data["answer"]
    assert len(data["sources"]) >= 1
    assert data["guardrail_refusal"] is False
    assert data["latency"]["stt_ms"] is not None

def test_metrics_endpoint():
    # Run a text query to generate a metric record
    client.post("/api/query-text", json={"query": "भारत की राजधानी क्या है?"})
    
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] >= 1
    assert "p50" in data["total"]
    assert "p70" in data["total"]
    assert "p100" in data["total"]
