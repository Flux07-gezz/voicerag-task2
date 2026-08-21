"""
STT Provider WebSocket Mapping & Proxy Handler for FastAPI.
Supports both Sarvam AI and ElevenLabs real-time streaming engines.
"""

import os
import json
import asyncio
import time
import base64
from pathlib import Path
from typing import AsyncGenerator
from dotenv import load_dotenv

# Load .env file explicitly from the backend directory
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from websockets.asyncio.client import connect as ws_connect
from groq import AsyncGroq

from app.core.vector_db import FastVectorEngine
from app.core.chunker import MSMARCOChunker

app = FastAPI(title="Voice-RAG Sub-200ms Proxy Pipeline")

# Initialize Groq client with loaded key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY was not found. Please verify backend/.env contains a valid key.")

groq_client = AsyncGroq(api_key=groq_api_key)
vector_engine = FastVectorEngine()

STT_PROVIDER = os.getenv("STT_PROVIDER", "sarvam").lower()  # 'sarvam' or 'elevenlabs'


# =====================================================================
# 1. SARVAM AI WEBSOCKET HANDLER
# =====================================================================
async def stream_sarvam_stt(client_ws: WebSocket) -> AsyncGenerator[str, None]:
    """
    Connects to Sarvam AI STT WebSocket, maps incoming client audio chunks,
    and yields finalized transcripts.
    """
    sarvam_url = f"wss://api.sarvam.ai/v1/stt/websocket?language_code=hi-IN&model=saaras:v3&sample_rate=16000"
    headers = {"api-subscription-key": os.getenv("SARVAM_API_KEY", "")}

    async with ws_connect(sarvam_url, additional_headers=headers) as sarvam_ws:
        async def forward_audio():
            try:
                while True:
                    data = await client_ws.receive_text()
                    payload = json.loads(data)

                    # Frontend sends raw PCM base64
                    if payload.get("event") == "media":
                        base64_pcm = payload["media"]["payload"]
                        # Map to Sarvam WebSocket input payload schema
                        sarvam_payload = {
                            "audio_data": base64_pcm,
                            "sample_rate": 16000
                        }
                        await sarvam_ws.send(json.dumps(sarvam_payload))
                    
                    elif payload.get("event") == "stop":
                        await sarvam_ws.send(json.dumps({"type": "flush"}))
                        break
            except Exception as e:
                print(f"Sarvam audio forward error: {e}")

        # Fire async task for bi-directional audio push
        asyncio.create_task(forward_audio())

        # Receive real-time STT events from Sarvam
        async for message in sarvam_ws:
            event = json.loads(message)
            
            # Map Sarvam output event schema to standardized text
            if event.get("type") in ["transcript", "speech_to_text"] and event.get("is_final", False):
                transcript_text = event.get("text", "").strip()
                if transcript_text:
                    yield transcript_text


# =====================================================================
# 2. ELEVENLABS WEBSOCKET HANDLER
# =====================================================================
async def stream_elevenlabs_stt(client_ws: WebSocket) -> AsyncGenerator[str, None]:
    """
    Connects to ElevenLabs Conversational WebSocket API, maps incoming audio chunks,
    and yields finalized user transcripts.
    """
    agent_id = os.getenv("ELEVENLABS_AGENT_ID", "")
    eleven_url = f"wss://api.elevenlabs.io/v1/convai/conversation?agent_id={agent_id}"

    async with ws_connect(eleven_url) as eleven_ws:
        async def forward_audio():
            try:
                while True:
                    data = await client_ws.receive_text()
                    payload = json.loads(data)

                    if payload.get("event") == "media":
                        base64_pcm = payload["media"]["payload"]
                        # Map to ElevenLabs User Audio Chunk schema
                        eleven_payload = {
                            "user_audio_chunk": base64_pcm
                        }
                        await eleven_ws.send(json.dumps(eleven_payload))
                        
                    elif payload.get("event") == "stop":
                        break
            except Exception as e:
                print(f"ElevenLabs audio forward error: {e}")

        asyncio.create_task(forward_audio())

        # Receive events from ElevenLabs
        async for message in eleven_ws:
            event = json.loads(message)
            event_type = event.get("type")

            # Map ElevenLabs transcript event structure
            if event_type == "user_transcript":
                transcript_text = event.get("user_transcription_event", {}).get("user_transcript", "").strip()
                if transcript_text:
                    yield transcript_text


# =====================================================================
# 3. MAIN FASTAPI WEBSOCKET ENDPOINT
# =====================================================================
@app.websocket("/ws/rag")
async def voice_rag_websocket(websocket: WebSocket):
    await websocket.accept()
    
    try:
        # Choose STT generator based on active configuration
        stt_generator = stream_sarvam_stt if STT_PROVIDER == "sarvam" else stream_elevenlabs_stt

        async for user_query in stt_generator(websocket):
            start_time = time.perf_counter_ns()

            # Execute Vector DB Search (<20ms)
            retrieved_chunks, max_score = vector_engine.search(user_query, top_k=2)

            # Guardrail Check: Relevance thresholding (<0.60 distance score)
            if max_score < 0.60:
                latency_ms = (time.perf_counter_ns() - start_time) / 1e6
                await websocket.send_json({
                    "transcript": user_query,
                    "answer": "I do not have sufficient context in the dataset to answer that accurately.",
                    "latency_ms": round(latency_ms, 2),
                    "guardrail_triggered": True
                })
                continue

            # Context Synthesis & LLM Call (<80ms TTFT)
            context = "\n".join([c["parent_context"] for c in retrieved_chunks])
            
            system_prompt = (
                "Answer using ONLY the provided context. If unsure, state that you don't know.\n\n"
                f"Context:\n{context}"
            )

            res = await groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.1,
                max_tokens=150
            )

            answer_text = res.choices[0].message.content
            latency_ms = (time.perf_counter_ns() - start_time) / 1e6

            # Return answer + transcript + benchmark latency to frontend
            await websocket.send_json({
                "transcript": user_query,
                "answer": answer_text,
                "latency_ms": round(latency_ms, 2),
                "guardrail_triggered": False
            })

    except WebSocketDisconnect:
        print("WebSocket client disconnected.")