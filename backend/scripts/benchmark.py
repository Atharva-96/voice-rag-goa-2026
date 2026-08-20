import os
import sys
import time
import argparse
import numpy as np
import requests
from tqdm import tqdm
from datasets import load_dataset
from fastapi.testclient import TestClient

# Ensure backend directory is on the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.main import app

def parse_args():
    parser = argparse.ArgumentParser(description="Voice RAG Latency & Retrieval Benchmarking Script.")
    parser.add_argument("--url", type=str, default=None, help="Url of the live FastAPI server (e.g. http://localhost:8000). If empty, benchmarks via in-memory TestClient.")
    parser.add_argument("--limit", type=int, default=30, help="Number of queries to run in the benchmark (default: 30).")
    return parser.parse_args()

def main():
    # Force UTF-8 encoding for standard output
    sys.stdout.reconfigure(encoding='utf-8')
    args = parse_args()

    print("Loading MSMARCO-XI validation subset for benchmark...")
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

    # Filter out first N queries
    queries = []
    iterator = iter(ds)
    for _ in range(args.limit):
        try:
            item = next(iterator)
            queries.append(item.get("query"))
        except StopIteration:
            break

    print(f"Loaded {len(queries)} benchmark queries.")

    total_latencies = []
    retrieval_latencies = []
    llm_latencies = []
    success_count = 0
    guardrail_triggers = 0

    if args.url:
        print(f"Benchmarking against live server at: {args.url}")
        # Warmup the live server first (critical for Render cold start)
        try:
            requests.get(f"{args.url}/api/warmup", timeout=30.0)
        except Exception as e:
            print(f"Warning: Warmup request failed: {e}")
            
        for q in tqdm(queries, desc="Benchmarking"):
            try:
                start = time.perf_counter()
                res = requests.post(f"{args.url}/api/query-text", json={"query": q}, timeout=30.0)
                elapsed = (time.perf_counter() - start) * 1000
                res.raise_for_status()
                data = res.json()
                
                total_latencies.append(data["latency"]["total_ms"])
                retrieval_latencies.append(data["latency"]["retrieval_ms"])
                llm_latencies.append(data["latency"]["llm_ms"])
                
                if data["guardrail_refusal"]:
                    guardrail_triggers += 1
                success_count += 1
            except Exception as e:
                print(f"Error querying live server: {e}")
    else:
        print("Benchmarking locally using in-memory FastAPI TestClient (no network overhead)...")
        # Ensure we use in-memory Qdrant for local test client if not configured otherwise
        settings.QDRANT_API_URL = ":memory:"
        settings.QDRANT_API_KEY = None
        
        # Pre-populate some dummy vectors to ensure retrieval has something to query and doesn't just return empty immediately
        from app.services.retrieval_service import retrieval_service
        from app.services.embedding_service import embedding_service
        from qdrant_client.http import models as qmodels
        
        qclient = retrieval_service.client
        qclient.recreate_collection(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(
                size=384,
                distance=qmodels.Distance.COSINE
            )
        )
        
        # Populate with a few sample paragraphs
        samples = [
            "भारत की राजधानी नई दिल्ली है।",
            "निगम एक कंपनी या लोगों का समूह होता है जो एक एकल इकाई के रूप में कार्य करने के लिए अधिकृत होता है।",
            "एआई (कृत्रिम बुद्धिमत्ता) कंप्यूटर विज्ञान का एक क्षेत्र है जो बुद्धिमान मशीनों के निर्माण पर केंद्रित है।"
        ]
        for idx, doc in enumerate(samples):
            v = embedding_service.embed_text(doc)
            qclient.upsert(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                points=[
                    qmodels.PointStruct(
                        id=f"223e4567-e89b-12d3-a456-42661417400{idx}",
                        vector=v,
                        payload={"text": doc, "is_selected": 1}
                    )
                ]
            )

        tclient = TestClient(app)
        
        for q in tqdm(queries, desc="Benchmarking"):
            try:
                res = tclient.post("/api/query-text", json={"query": q})
                assert res.status_code == 200
                data = res.json()
                
                total_latencies.append(data["latency"]["total_ms"])
                retrieval_latencies.append(data["latency"]["retrieval_ms"])
                llm_latencies.append(data["latency"]["llm_ms"])
                
                if data["guardrail_refusal"]:
                    guardrail_triggers += 1
                success_count += 1
            except Exception as e:
                print(f"Error querying local TestClient: {e}")

    # Compute percentiles
    def report_percentiles(name, data):
        if not data:
            print(f"{name:15} | No data collected")
            return
        p50 = np.percentile(data, 50)
        p70 = np.percentile(data, 70)
        p100 = np.percentile(data, 100)
        print(f"{name:15} | P50: {p50:8.2f}ms | P70: {p70:8.2f}ms | P100: {p100:8.2f}ms")

    print("\n" + "="*60)
    print("                 BENCHMARK RESULTS")
    print("="*60)
    print(f"Queries Run:          {success_count}/{len(queries)}")
    print(f"Guardrail Refusals:   {guardrail_triggers} ({guardrail_triggers/success_count*100:.1f}%)" if success_count else "Queries Run: 0")
    print("-"*60)
    report_percentiles("Total Pipeline", total_latencies)
    report_percentiles("Retrieval (DB)", retrieval_latencies)
    report_percentiles("LLM Gen", llm_latencies)
    print("="*60)

if __name__ == "__main__":
    main()
