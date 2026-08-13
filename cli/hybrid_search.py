import os
from search_functions import InvertedIndex, load_movies
from semantic_search import ChunkedSemanticSearch
import numpy as np
import constants

class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(constants.CACHE_INDEX_PATH):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        self.idx.save()
        self.idx.bm25_search(query, limit = limit)


    def weighted_search(self, query: str, alpha: float = 0.5, limit: int = 5):
        method = "min_max"
        return self.hybrid_scores_mapping(method, query, alpha=alpha, limit=limit)

    def rrf_search(self, query: str, k: int = 60, limit: int = 10) -> list[dict]:
        method = "rrf"
        return self.hybrid_scores_mapping(method, query, k=k, limit=limit)

    def hybrid_scores_mapping(self, method, query: str, alpha: float = 0.5, k: int = 60, limit: int = 10):
        if method == "None":
            raise Exception("method is None. Input method = 'min_max' for min-max normalization or 'rrf' for reciprocal rank fusion.")

        self._bm25_search(query, 500*limit)
        keyword_scores = [self.idx.score[key] for key in self.idx.score]
        if method == "min_max":
            keyword_scores = [self.idx.score[key] for key in self.idx.score]
            keyword_scores_normalized = min_max_norm_score(keyword_scores)
        if method == "rrf":
            keyword_scores_normalized = [i for i in range(0, len(self.idx.score))]
            #keyword_scores_normalized = rrf_score(keyword_scores)

        for i, key in enumerate(self.idx.score):
            self.idx.score[key] = keyword_scores_normalized[i]

        semantic_score_list = self.semantic_search.search_chunks(query, limit=500*limit)
        semantic_scores_normalized = norm_score(semantic_score_list, method=method)

        hybrid_scores_docmap = {}
        for id in self.idx.score:
            hybrid_scores_docmap[id] = {
                "id": id,
                "title": self.idx.docmap[id]['title'],
                'document': self.idx.docmap[id]['description'][:300],
                'keyword_score': self.idx.score[id],
                'semantic_score': 0.0,
            }
        for i in range(0, len(semantic_scores_normalized)):
            id = semantic_scores_normalized[i]['id']
            score = semantic_scores_normalized[i]['score']
            document = self.idx.docmap[id]
            if id not in hybrid_scores_docmap:
                hybrid_scores_docmap[id] = {
                    "id": id,
                    "title": self.idx.docmap[id]['title'],
                    'document': self.idx.docmap[id]['description'][:300],
                    'keyword_score': 0.0,
                    'semantic_score': score,
                }
            hybrid_scores_docmap[id]['semantic_score'] = score


        for id in hybrid_scores_docmap:
            keyword_score = hybrid_scores_docmap[id]['keyword_score']
            semantic_score = hybrid_scores_docmap[id]['semantic_score']
            if method == "min_max":
                hybrid_scores_docmap[id]['hybrid_score'] = hybrid_score(keyword_score, semantic_score, alpha=alpha, method=method)
            if method == "rrf":
                hybrid_scores_docmap[id]['hybrid_score'] = rrf_score(keyword_score) + rrf_score(semantic_score)

        sorted_data = dict(sorted(hybrid_scores_docmap.items(), key=lambda item: item[1]['hybrid_score'], reverse=True))
        top_results = {}
        for i, id in enumerate(sorted_data):
            if i < limit:
                top_results[id] = sorted_data[id]
        return top_results


def min_max_norm_score(scores: list[int]) -> None:
    if len(scores) == 0:
        return

    scores = np.array(scores)

    if np.max(scores) == np.min(scores):
        for i in range(0, len(scores)):
            print(f"* 1.0")
        return

    normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    return normalized_scores
    #for score in normalized_scores:
        #print(f"* {score:.4f}")

def norm_score(score_dict_list: list[dict], method: str | None = None) -> list[dict]:
    scores = [d.get("score") for d in score_dict_list]

    if method == None:
        raise Exception("method is None. Input method = 'min_max' for min-max normalization or 'rrf' for reciprocal rank fusion.")

    if method == "min_max":
        scores_normalized = min_max_norm_score(scores)

    if method == "rrf":
        scores_normalized = [i for i in range(0, len(scores))] # Ranked.
        #scores_normalized = rrf_score(scores)

    for i in range(0, len(scores_normalized)):
        score_dict_list[i]["score"] = scores_normalized[i]
    return score_dict_list

def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5, k: int = 60, method: str | None = None) -> float:
    if method == "min_max":
        return alpha * bm25_score + (1 - alpha) * semantic_score
    return bm25_score + semantic_score

def weighted_search(query: str, alpha: float = 0.5, limit: int = 5):
    documents = load_movies()
    hybrid = HybridSearch(documents['movies'])
    hybrid.weighted_search(query, alpha=alpha, limit=limit)

def rrf_score(rank: int, k: int = 60):
    return 1 / (k + rank)

def rrf_search(query, k: int = 60, limit: int = 5) -> None:
    documents = load_movies()
    hybrid = HybridSearch(documents['movies'])
    return hybrid.rrf_search(query, k=k, limit=limit)
