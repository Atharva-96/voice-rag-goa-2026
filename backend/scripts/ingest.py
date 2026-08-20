import os
import sys
import uuid
import argparse
from tqdm import tqdm
from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# Ensure backend directory is on the path so we can import services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.services.chunking import Chunker
from app.services.embedding_service import embedding_service

def parse_args():
    parser = argparse.ArgumentParser(description="Ingest MSMARCO-XI Hindi subset to Qdrant.")
    parser.add_argument("--limit", type=int, default=100, help="Number of queries to ingest (default: 100).")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for embedding and upserting (default: 50).")
    parser.add_argument("--force", action="store_true", help="Force ingestion even if collection already exists and has points.")
    return parser.parse_args()

def main():
    # Force sys.stdout to output utf-8 so Devanagari output doesn't crash Windows CLI
    sys.stdout.reconfigure(encoding='utf-8')
    args = parse_args()

    print(f"Connecting to Qdrant at {settings.QDRANT_API_URL}...")
    if settings.QDRANT_API_URL == ":memory:":
        client = QdrantClient(":memory:")
    elif settings.QDRANT_API_KEY:
        client = QdrantClient(url=settings.QDRANT_API_URL, api_key=settings.QDRANT_API_KEY)
    else:
        client = QdrantClient(url=settings.QDRANT_API_URL)

    collection_name = settings.QDRANT_COLLECTION_NAME

    # Idempotency Check
    try:
        collections_info = client.get_collections()
        existing_collections = [c.name for c in collections_info.collections]
        
        if collection_name in existing_collections:
            collection_info = client.get_collection(collection_name)
            points_count = collection_info.points_count
            print(f"Collection '{collection_name}' already exists with {points_count} points.")
            if points_count > 0 and not args.force:
                print("Skipping ingestion as the database already contains points. Use --force to override.")
                return
    except Exception as e:
        print(f"Error checking collections: {e}. Attempting to proceed...")

    # Create collection
    print(f"Creating collection '{collection_name}' with 384 dimensions...")
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(
            size=384,
            distance=qmodels.Distance.COSINE
        )
    )

    print("Loading MSMARCO-XI Hindi validation subset...")
    try:
        ds = load_dataset(
            "parquet",
            data_files="https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/hinval.parquet",
            split="train",
            streaming=True
        )
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    chunker = Chunker()
    points = []
    processed_queries = 0

    print(f"Processing and indexing first {args.limit} queries...")
    for item in tqdm(ds, total=args.limit, desc="Queries processed"):
        if processed_queries >= args.limit:
            break

        query = item.get("query")
        query_id = str(item.get("query_id"))
        gold_answer = item.get("Answer")
        passages = item.get("passages", {})
        
        translated_passages = passages.get("Translated_passages", [])
        is_selected_list = passages.get("is_selected", [])

        # Process each passage
        for p_idx, text in enumerate(translated_passages):
            is_selected = is_selected_list[p_idx] if p_idx < len(is_selected_list) else 0
            
            doc_metadata = {
                "query_id": query_id,
                "passage_index": p_idx,
                "is_selected": int(is_selected),
                "gold_answer": gold_answer,
                "source_query": query
            }

            # Chunk the passage text
            chunks = chunker.chunk_text(text, doc_metadata)
            
            for c_idx, chunk in enumerate(chunks):
                chunk_text = chunk["text"]
                chunk_meta = chunk["metadata"]
                
                # Generate determinisitic UUID to keep it idempotent
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{query_id}_{p_idx}_{c_idx}"))
                
                points.append({
                    "id": point_id,
                    "text": chunk_text,
                    "meta": chunk_meta
                })

        processed_queries += 1

    print(f"Generated {len(points)} total chunks. Embedding and uploading in batches of {args.batch_size}...")

    # Embed and upload in batches
    for i in tqdm(range(0, len(points), args.batch_size), desc="Batches uploaded"):
        batch = points[i : i + args.batch_size]
        batch_texts = [p["text"] for p in batch]
        
        # Batch embed (much faster!)
        try:
            batch_vectors = embedding_service.embed_batch(batch_texts)
        except Exception as e:
            print(f"\nEmbedding failed for batch starting at {i}: {e}. Retrying individually...")
            # Fallback to individual embedding to be robust
            batch_vectors = []
            for txt in batch_texts:
                try:
                    batch_vectors.append(embedding_service.embed_text(txt))
                except Exception as ex:
                    print(f"Skipping text due to embedding error: {ex}")
                    batch_vectors.append([0.0] * 384)

        # Build Qdrant PointStructs
        qdrant_points = []
        for p_idx, p in enumerate(batch):
            qdrant_points.append(
                qmodels.PointStruct(
                    id=p["id"],
                    vector=batch_vectors[p_idx],
                    payload={
                        "text": p["text"],
                        **p["meta"]
                    }
                )
            )

        # Upsert to Qdrant
        client.upsert(
            collection_name=collection_name,
            points=qdrant_points
        )

    print(f"Successfully ingested {processed_queries} queries ({len(points)} chunks) into collection '{collection_name}'.")

if __name__ == "__main__":
    main()
