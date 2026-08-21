"""
Dataset processing script to download ai4bharat/MSMARCO-XI,
apply parent-child chunking, and index vectors for ultra-fast retrieval.
"""

import sys
import os
from pathlib import Path

# Add backend directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
from app.core.chunker import MSMARCOChunker
from app.core.vector_db import FastVectorEngine

def run_ingestion(dataset_name: str = "ai4bharat/MSMARCO-XI", lang_split: str = "hi", max_records: int = 20000):
    print(f"📥 Loading dataset '{dataset_name}' (split: {lang_split}, records: {max_records})...")
    
    # Load dataset split
    dataset = load_dataset(dataset_name, "default", split=f"train[:{max_records}]")
    
    chunker = MSMARCOChunker(child_chunk_size=128, child_overlap=32)
    vector_engine = FastVectorEngine()
    
    all_chunks = []
    print("✂️ Chunking passages using Parent-Child strategy...")
    
    for i, record in enumerate(dataset):
        chunks = chunker.process_record(record)
        all_chunks.extend(chunks)
        
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1}/{max_records} records ({len(all_chunks)} total chunks generated)...")

    print(f"⚡ Indexing {len(all_chunks)} vectors into In-Memory Qdrant DB...")
    
    # Batch indexing to maximize vector insertion speed
    batch_size = 500
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        vector_engine.index_chunks(batch)
        print(f"  Indexed batch {i // batch_size + 1}/{(len(all_chunks) + batch_size - 1) // batch_size}")

    print("✅ Ingestion complete! Vector index is ready for sub-200ms queries.")

if __name__ == "__main__":
    run_ingestion()