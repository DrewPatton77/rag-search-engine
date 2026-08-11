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

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

    def weighted_search(self, query: str, alpha: float = 0.5, limit: int = 5):
        self._bm25_search(query, 500*limit)
        keyword_scores = [self.idx.score[key] for key in self.idx.score]
        keyword_scores_normalized = min_max_norm_score(keyword_scores)
        for i, key in enumerate(self.idx.score):
            self.idx.score[key] = keyword_scores_normalized[i]

        semantic_score_list = self.semantic_search.search_chunks(query, limit=500*limit)
        semantic_scores_normalized = min_max_norm_score_semantics(semantic_score_list)

        #print(len(semantic_scores_normalized))
        #print(len(self.idx.score))
        #semantic -> list of dictionaries, id: id, score: score, etc...
        #keyword -> dictionary, id: score
        # Make a list of dictionaries like semantic
        hybrid_scores_docmap = {}
        for id in self.idx.score:
            hybrid_scores_docmap[id] = {
                'document': self.idx.docmap[id],
                'keyword_score': self.idx.score[id],
                'semantic_score': 0.0,
            }
        for i in range(0, len(semantic_scores_normalized)):
            id = semantic_scores_normalized[i]['id']
            score = semantic_scores_normalized[i]['score']
            document = self.idx.docmap[id]
            if id not in hybrid_scores_docmap:
                hybrid_scores_docmap[id] = {
                    'document': document,
                    'keyword_score': 0.0,
                    'semantic_score': score,
                }
            hybrid_scores_docmap[id]['semantic_score'] = score

        for id in hybrid_scores_docmap:
            keyword_score = hybrid_scores_docmap[id]['keyword_score']
            semantic_score = hybrid_scores_docmap[id]['semantic_score']
            hybrid_scores_docmap[id]['hybrid_score'] = hybrid_score(keyword_score, semantic_score, alpha=alpha)

        sorted_data = dict(sorted(hybrid_scores_docmap.items(), key=lambda item: item[1]['hybrid_score'], reverse=True))
        for i, id in enumerate(sorted_data):
            if i < limit:
                print(f"{i + 1}. {sorted_data[id]['document']['title']} (score: {sorted_data[id]['hybrid_score']})")






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

def min_max_norm_score_semantics(score_dict_list: list[dict]) -> list[dict]:
    scores = [d.get("score") for d in score_dict_list]
    scores_normalized = min_max_norm_score(scores)
    for i in range(0, len(scores_normalized)):
        score_dict_list[i]["score"] = scores_normalized[i]
    return score_dict_list

def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def weighted_search(query: str, alpha: float = 0.5, limit: int = 5):
    documents = load_movies()
    hybrid = HybridSearch(documents['movies'])
    hybrid.weighted_search(query, alpha=alpha, limit=limit)
