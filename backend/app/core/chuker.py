"""
Multi-strategy chunking implementation for ai4bharat/MSMARCO-XI dataset.
Combines Parent-Child chunking with rich metadata enrichment.
"""

from typing import List, Dict, Any
import uuid


class MSMARCOChunker:
    def __init__(self, child_chunk_size: int = 128, child_overlap: int = 32):
        self.child_chunk_size = child_chunk_size
        self.child_overlap = child_overlap

    def process_record(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Processes a raw record from MSMARCO-XI into child chunks mapped to parent context.
        """
        passage_text = record.get("passage", record.get("text", ""))
        query_id = record.get("query_id", "unknown")
        passage_id = record.get("passage_id", str(uuid.uuid4()))
        language = record.get("language", "hi")
        is_selected = record.get("is_selected", 0)

        words = passage_text.split()
        chunks = []

        # If text is small, maintain a single child chunk
        if len(words) <= self.child_chunk_size:
            chunk_text = passage_text
            chunks.append(
                self._create_payload(
                    chunk_text=chunk_text,
                    parent_text=passage_text,
                    query_id=query_id,
                    passage_id=passage_id,
                    language=language,
                    is_selected=is_selected,
                    chunk_index=0,
                )
            )
            return chunks

        # Sliding window parent-child splitting
        step = self.child_chunk_size - self.child_overlap
        for i in range(0, len(words), step):
            window_words = words[i : i + self.child_chunk_size]
            chunk_text = " ".join(window_words)

            chunks.append(
                self._create_payload(
                    chunk_text=chunk_text,
                    parent_text=passage_text,
                    query_id=query_id,
                    passage_id=passage_id,
                    language=language,
                    is_selected=is_selected,
                    chunk_index=i // step,
                )
            )

        return chunks

    def _create_payload(
        self,
        chunk_text: str,
        parent_text: str,
        query_id: str,
        passage_id: str,
        language: str,
        is_selected: int,
        chunk_index: int,
    ) -> Dict[str, Any]:
        return {
            "chunk_id": f"{passage_id}_{chunk_index}",
            "text": chunk_text,  # Vectors match against child chunk
            "parent_context": parent_text,  # LLM reads full parent context
            "metadata": {
                "query_id": query_id,
                "passage_id": passage_id,
                "language": language,
                "is_selected": is_selected,
                "chunk_index": chunk_index,
            },
        }