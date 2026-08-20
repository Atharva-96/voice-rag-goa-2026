import re
from typing import List, Dict, Any

class Chunker:
    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 150,
        min_chunk_len: int = 100
    ):
        """
        Multi-strategy chunking engine.
        - Splits text by sentence boundaries (supports English '.' and Hindi '।').
        - Groups sentences into chunks to respect chunk_size.
        - Appends adaptive overlap from preceding chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_len = min_chunk_len

    def split_sentences(self, text: str) -> List[str]:
        """
        Splits text into sentences supporting both English and Indic (Hindi) punctuation.
        """
        if not text:
            return []
        
        # Split by typical sentence terminators: '.', '।', '?', '!' followed by space or end of string
        # We use lookbehind and lookahead to avoid removing the punctuation
        sentence_end = re.compile(r'(?<=[.।?!])\s+')
        sentences = sentence_end.split(text.strip())
        return [s for s in sentences if s]

    def chunk_text(self, text: str, doc_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Chunks the document text using sentence-level grouping and adaptive overlap.
        Each chunk is returned as a dict with 'text' and 'metadata'.
        """
        if not doc_metadata:
            doc_metadata = {}
            
        sentences = self.split_sentences(text)
        if not sentences:
            return []

        chunks = []
        current_chunk = []
        current_length = 0

        for i, sentence in enumerate(sentences):
            sent_len = len(sentence)
            # If a single sentence exceeds the chunk_size, we split it into words
            if sent_len > self.chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split large sentence by words
                words = sentence.split(" ")
                word_chunk = []
                word_len = 0
                for word in words:
                    if word_len + len(word) + 1 > self.chunk_size:
                        if word_chunk:
                            chunks.append(" ".join(word_chunk))
                        word_chunk = [word]
                        word_len = len(word)
                    else:
                        word_chunk.append(word)
                        word_len += len(word) + 1
                if word_chunk:
                    chunks.append(" ".join(word_chunk))
                continue

            if current_length + sent_len + 1 > self.chunk_size:
                # Store the current chunk
                chunk_str = " ".join(current_chunk)
                if len(chunk_str) >= self.min_chunk_len:
                    chunks.append(chunk_str)
                
                # Generate overlap: backtrack to include previous sentences that fit within overlap limit
                overlap_chunk = []
                overlap_len = 0
                for prev_sent in reversed(current_chunk):
                    if overlap_len + len(prev_sent) + 1 <= self.chunk_overlap:
                        overlap_chunk.insert(0, prev_sent)
                        overlap_len += len(prev_sent) + 1
                    else:
                        break
                
                current_chunk = overlap_chunk
                current_length = overlap_len

            current_chunk.append(sentence)
            current_length += sent_len + 1

        if current_chunk:
            chunk_str = " ".join(current_chunk)
            if len(chunk_str) >= self.min_chunk_len or not chunks:
                chunks.append(chunk_str)

        # Build output objects with enriched metadata
        output_chunks = []
        for index, chunk_text in enumerate(chunks):
            metadata = doc_metadata.copy()
            metadata["chunk_index"] = index
            metadata["total_chunks"] = len(chunks)
            output_chunks.append({
                "text": chunk_text,
                "metadata": metadata
            })

        return output_chunks
