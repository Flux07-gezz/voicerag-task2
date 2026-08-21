"""
FastAPI Server handling WebSocket voice audio stream, Sarvam STT transcription,
Guardrail evaluation, Qdrant Retrieval, and Groq LPU Generation.
"""

import time
import os
import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from groq import AsyncGroq
from app.core.vector_db import FastVectorEngine
from app.core.chunker import MSMARCOChunker

app = FastAPI(title="Voice-RAG Sub-200ms Engine")

# Initialize Clients & In-Memory DB
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY", "your_groq_key"))
vector_engine = FastVectorEngine()
chunker = MSMARCOChunker()

# Pre-populate dummy dataset entry on startup for test runs
@app.on_event("startup")
async def startup_event():
    sample_doc = {
        "passage": "Goa is a state located on the southwestern coast of India within the Konkan region. Panaji is the state capital.",
        "query_id": "1001",
        "passage_id": "p_01",
        "language": "en",
    }
    chunks = chunker.process_record(sample_doc)
    vector_engine.index_chunks(chunks)


@app.websocket("/ws/rag")
async def websocket_rag_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            # Receive speech audio bytes or text query over WS
            data = await websocket.receive_text()
            payload = json.loads(data)

            start_ns = time.perf_counter_ns()

            query_text = payload.get("query", "")

            # Guardrail Check 1: Empty or invalid input
            if not query_text.strip():
                await websocket.send_json({"error": "Empty query received."})
                continue

            # Step 1: Vector DB Search (<20ms)
            retrieved_contexts, max_score = vector_engine.search(query_text, top_k=2)

            # Guardrail Check 2: Grounding & Threshold check
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

            # Context Preparation for LLM
            context_str = "\n".join([c["parent_context"] for c in retrieved_contexts])

            # Step 2: Groq LPU Inference (<80-100ms TTFT)
            system_prompt = (
                "You are an accurate assistant. Answer the user question using ONLY the provided context. "
                "If the answer is not present in the context, state that you do not know.\n\n"
                f"Context:\n{context_str}"
            )

            response = await groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query_text},
                ],
                temperature=0.1,
                max_tokens=150,
            )

            answer = response.choices[0].message.content
            total_latency_ms = (time.perf_counter_ns() - start_ns) / 1e6

            # Return answer + timing metrics
            await websocket.send_json(
                {
                    "answer": answer,
                    "score": round(max_score, 3),
                    "latency_ms": round(total_latency_ms, 2),
                    "guardrail_triggered": False,
                }
            )

    except WebSocketDisconnect:
        print("Client disconnected.")