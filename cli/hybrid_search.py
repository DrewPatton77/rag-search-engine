import os
from search import InvertedIndex
from semantic_search import ChunkedSemanticSearch
import numpy as np

class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        self.idx.save()

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def min_max_norm_score(scores: list[int]) -> None:
    if len(scores) == 0:
        return

    scores = np.array(scores)

    if np.max(scores) == np.min(scores):
        for i in range(0, len(scores)):
            print(f"* 1.0")
        return

    normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    for score in normalized_scores:
        print(f"* {score:.4f}")
