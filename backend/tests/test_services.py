import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import settings
# Configure in-memory Qdrant for testing
settings.QDRANT_API_URL = ":memory:"
settings.QDRANT_API_KEY = None

from app.services.chunking import Chunker
from app.services.embedding_service import embedding_service
from app.services.retrieval_service import retrieval_service

def test_chunker_sentence_splitting():
    chunker = Chunker()
    text = "यह पहला वाक्य है। क्या यह दूसरा वाक्य है? Yes, it is."
    sentences = chunker.split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "यह पहला वाक्य है।"
    assert sentences[1] == "क्या यह दूसरा वाक्य है?"
    assert sentences[2] == "Yes, it is."

def test_chunker_grouping():
    chunker = Chunker(chunk_size=50, chunk_overlap=10)
    text = "यह एक बहुत लंबा वाक्य समूह है। हम इसे छोटे टुकड़ों में विभाजित करना चाहते हैं ताकि यह सीमा के भीतर रहे।"
    chunks = chunker.chunk_text(text, {"doc_id": "test_1"})
    
    assert len(chunks) >= 1
    for chunk in chunks:
        assert len(chunk["text"]) <= 50
        assert chunk["metadata"]["doc_id"] == "test_1"
        assert "chunk_index" in chunk["metadata"]

def test_embedding_dimensions():
    # Verify embedding dimensions for intfloat/multilingual-e5-small
    vector = embedding_service.embed_text("नमस्ते दुनिया")
    assert len(vector) == 384
    assert isinstance(vector[0], float)

def test_retrieval_service_flow():
    # Force recreate in-memory collection
    client = retrieval_service.client
    client.recreate_collection(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        vectors_config=qmodels.VectorParams(
            size=384,
            distance=qmodels.Distance.COSINE
        )
    )

    # Insert test mock points
    test_text = "भारत की राजधानी नई दिल्ली है।"
    vector = embedding_service.embed_text(test_text)
    
    client.upsert(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        points=[
            qmodels.PointStruct(
                id="123e4567-e89b-12d3-a456-426614174000",
                vector=vector,
                payload={"text": test_text, "language": "hi"}
            )
        ]
    )

    # Search query
    results = retrieval_service.search("भारत की राजधानी क्या है?", limit=1, min_score=0.2)
    assert len(results) == 1
    assert results[0].passage == test_text
    assert results[0].score > 0.3

def test_stt_service_mock():
    from unittest.mock import MagicMock, patch
    from app.services.stt_service import stt_service

    with patch("app.services.stt_service.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"transcript": "कॉर्पोरेशन क्या है?"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        transcript = stt_service.transcribe(b"fake_audio_bytes")
        assert transcript == "कॉर्पोरेशन क्या है?"

def test_llm_service_refusal_and_mock():
    from unittest.mock import MagicMock, patch
    from app.services.llm_service import llm_service
    
    # Empty context must trigger standard refusal
    ans_empty = llm_service.generate_answer("क्या भारत की राजधानी नई दिल्ली है?", [])
    assert ans_empty == llm_service.refusal_message

    # Non-empty context should return grounded answer using mocked Groq client
    mock_client = MagicMock()
    mock_chat = MagicMock()
    mock_completions = MagicMock()
    mock_completion = MagicMock()
    
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = "भारत की राजधानी नई दिल्ली है।"
    mock_completions.create.return_value = mock_completion
    mock_chat.completions = mock_completions
    mock_client.chat = mock_chat

    with patch.object(llm_service, "_client", mock_client):
        ans_grounded = llm_service.generate_answer(
            "भारत की राजधानी क्या है?", 
            ["भारत की राजधानी नई दिल्ली है।"]
        )
        assert "नई दिल्ली" in ans_grounded

