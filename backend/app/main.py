import os
import time
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load .env explicitly from backend directory
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from groq import AsyncGroq
from app.core.vector_db import FastVectorEngine
from app.core.chunker import MSMARCOChunker

# 1. Initialize FastAPI instance
app = FastAPI(title="Voice-RAG Sub-200ms Engine")

# 2. Retrieve environment keys
groq_api_key = os.getenv("GROQ_API_KEY", "")
if not groq_api_key:
    print("⚠️ WARNING: GROQ_API_KEY is not set in .env")

groq_client = AsyncGroq(api_key=groq_api_key)
vector_engine = FastVectorEngine()
chunker = MSMARCOChunker()

@app.on_event("startup")
async def startup_event():
    """Seeds a baseline passage if the in-memory database is empty on start."""
    sample_doc = {
        "passage": "Goa is a state located on the southwestern coast of India within the Konkan region. Panaji is the state capital.",
        "query_id": "1001",
        "passage_id": "p_01",
        "language": "en",
    }
    chunks = chunker.process_record(sample_doc)
    vector_engine.index_chunks(chunks)
    print("✅ In-memory Vector DB initialized and ready.")

@app.get("/")
async def root():
    return {"status": "running", "message": "Voice-RAG Sub-200ms Engine active"}

@app.websocket("/ws/rag")
async def websocket_rag_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)

            start_ns = time.perf_counter_ns()
            query_text = payload.get("query", "")

            # Guardrail Check 1: Empty input
            if not query_text.strip():
                await websocket.send_json({"error": "Empty query received."})
                continue

            # Step 1: Vector DB Lookup (<20ms)
            retrieved_contexts, max_score = vector_engine.search(query_text, top_k=2)

            # Guardrail Check 2: Relevance thresholding
            if max_score < 0.60:
                total_latency_ms = (time.perf_counter_ns() - start_ns) / 1e6
                await websocket.send_json(
                    {
                        "answer": "I do not have enough context in the dataset to answer that question.",
                        "latency_ms": round(total_latency_ms, 2),
                        "guardrail_triggered": True,
                    }
                )
                continue

            # Step 2: Synthesis & LLM Inference via Groq (<80ms TTFT)
            context_str = "\n".join([c.get("parent_context", c.get("text", "")) for c in retrieved_contexts])
            system_prompt = (
                "You are an accurate assistant. Answer the user question using ONLY the provided context. "
                "If the answer is not present in the context, state that you do not know.\n\n"
                f"Context:\n{context_str}"
            )

            # --- THE FIX: Using the active 8B model on Groq ---
            response = await groq_client.chat.completions.create(
                model="openai/gpt-oss-20b", 
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query_text},
                ],
                temperature=0.1,
                max_tokens=150,
            )

            answer = response.choices[0].message.content
            total_latency_ms = (time.perf_counter_ns() - start_ns) / 1e6

            await websocket.send_json(
                {
                    "answer": answer,
                    "score": round(max_score, 3),
                    "latency_ms": round(total_latency_ms, 2),
                    "guardrail_triggered": False,
                }
            )

    except WebSocketDisconnect:
        print("WebSocket client disconnected.")