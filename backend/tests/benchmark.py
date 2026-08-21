"""
Automated Latency Benchmark Suite for Voice-RAG Task 2 Pipeline.
Fires 100 queries via WebSockets to evaluate P50, P70, and P100 execution metrics.
"""

import asyncio
import json
import time
import numpy as np
import websockets

# Configuration
WS_URI = "ws://127.0.0.1:8000/ws/rag"
NUM_QUERIES = 100

# Mix of valid in-context queries and out-of-context guardrail triggers
TEST_QUERIES = [
    "Where is Goa located?",
    "What is the capital of Goa?",
    "Tell me about the southwestern coast of India.",
    "Which region is Goa in?",
    "What is the climate of Tokyo?",  # Out-of-bounds (guardrail check)
    "Who won the 1998 World Cup?",    # Out-of-bounds (guardrail check)
]


async def run_single_query(ws, query: str) -> dict:
    """Sends a query over WebSocket and tracks accurate millisecond round-trip latency."""
    payload = json.dumps({"query": query})
    
    t_start = time.perf_counter_ns()
    await ws.send(payload)
    response_raw = await ws.recv()
    t_end = time.perf_counter_ns()
    
    total_rtt_ms = (t_end - t_start) / 1e6
    response_data = json.loads(response_raw)
    response_data["client_rtt_ms"] = total_rtt_ms
    return response_data


async def benchmark_pipeline():
    print(f"🚀 Starting Latency Benchmark across {NUM_QUERIES} test queries...")
    print(f"Connecting to {WS_URI}...\n")
    
    latencies_client = []
    latencies_server = []
    guardrail_hits = 0
    
    try:
        async with websockets.connect(WS_URI) as ws:
            # Warmup request (discarded from metrics to avoid cold-start bias)
            print("🔥 Running 1 warmup query...")
            await run_single_query(ws, "Goa capital")
            print("Warmup complete. Running main test suite...\n")

            for i in range(1, NUM_QUERIES + 1):
                query = TEST_QUERIES[i % len(TEST_QUERIES)]
                res = await run_single_query(ws, query)
                
                client_ms = res.get("client_rtt_ms", 0)
                server_ms = res.get("latency_ms", 0)
                
                latencies_client.append(client_ms)
                latencies_server.append(server_ms)
                
                if res.get("guardrail_triggered", False):
                    guardrail_hits += 1

                # Live progress bar update every 10 queries
                if i % 10 == 0 or i == NUM_QUERIES:
                    print(f"  Progress: {i}/{NUM_QUERIES} queries completed | Last Query RTT: {client_ms:.2f}ms")

    except ConnectionRefusedError:
        print("\n❌ Error: Cannot connect to FastAPI server. Make sure `uvicorn app.main:app` is running!")
        return

    # Statistical Analysis
    p50_client = np.percentile(latencies_client, 50)
    p70_client = np.percentile(latencies_client, 70)
    p100_client = np.max(latencies_client)

    p50_server = np.percentile(latencies_server, 50)
    p70_server = np.percentile(latencies_server, 70)
    p100_server = np.max(latencies_server)

    # Display Report
    print("\n" + "=" * 55)
    print("📊 BENCHMARK LATENCY REPORT (100 QUERIES)")
    print("=" * 55)
    print(f"Total Queries Executed : {NUM_QUERIES}")
    print(f"Guardrail Rejections   : {guardrail_hits}/{NUM_QUERIES}")
    print("-" * 55)
    print("Metric     Server Pipeline Latency     Client Total RTT")
    print("-" * 55)
    print(f"P50  :     {p50_server:8.2f} ms             {p50_client:8.2f} ms")
    print(f"P70  :     {p70_server:8.2f} ms             {p70_client:8.2f} ms")
    print(f"P100 :     {p100_server:8.2f} ms             {p100_client:8.2f} ms")
    print("=" * 55)

    # Target Evaluation
    if p50_server < 200:
        print("\n✅ SUCCESS: P50 execution target (<200ms) achieved!")
    else:
        print("\n⚠️ WARNING: P50 latency exceeded 200ms threshold.")


if __name__ == "__main__":
    asyncio.run(benchmark_pipeline())