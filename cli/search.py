import json
from search_functions import load_movies, tokenize, filter_token, stem_token, InvertedIndex
import pickle


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False

def search_command(query: str) -> list[str]:

    iindex = InvertedIndex()
    iindex.load()

    query_tokens: list[str] = stem_token(filter_token(tokenize(query)))
    docs = []
    for token in query_tokens:
        if len(docs) >= 5:
            break

        doc_ids = iindex.get_documents(token)
        if doc_ids != None:
            for id in doc_ids:
                docs.append(iindex.docmap[id])

    return docs[:5]

    #data = load_movies()

    #movies_in_query = []

    #for movie in data['movies']:
    #    query_tokens: list[str] = stem_token(filter_tokens(tokenize(query)))
    #    title_tokens: list[str] = stem_token(filter_tokens(tokenize(movie['title'])))
    #    if has_matching_token(query_tokens, title_tokens) and movie['title'] not in movies_in_query:
    #        movies_in_query.append(movie['title'])

    #return movies_in_query

def build_command():
    iindex = InvertedIndex()
    iindex.build()
    iindex.save()
    iindex.load()

def tf_command(doc_id: int, term: str) -> None:
    iindex = InvertedIndex()
    iindex.load()
    tf = iindex.get_tf(doc_id, term)
    doc = iindex.docmap[doc_id]
    print(f"{term} has frequency: {tf} for movie: {doc['title']}")

def inv_df_command(term: str) -> None:
    iindex = InvertedIndex()
    iindex.load()
    inv_df = iindex.get_inv_df_score(term)
    print(f"Inverse document frequency of '{term}': {inv_df:.2f}")

def tfidf_command(doc_id: int, term: str) -> None:
    iindex = InvertedIndex()
    iindex.load()
    tf = iindex.get_tf(doc_id, term)
    inv_df = iindex.get_inv_df_score(term)
    tfidf = tf * inv_df
    print(f"TF-IDF score of '{term}' in document '{doc_id}:' {tfidf:.2f}")

def bm25_inv_df_command(term: str) -> float:
    iindex = InvertedIndex()
    iindex.load()
    return iindex.get_bm25(term)

def bm25tf_command(doc_id: int, term: str, k1: float, b: float) -> float:
    iindex = InvertedIndex()
    iindex.load()
    return iindex.get_bm25_tf(doc_id, term, k1=k1, b=b)
