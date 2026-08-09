from ast import Dict
from collections import Counter
import json
import string
from nltk.stem import PorterStemmer
from typing import Any, TypedDict
import pickle
import sys
import math

class Movie(TypedDict):
    id: int
    title: str
    descirption: str

class InvertedIndex():
    def __init__(self):
        self.index: Dict[str , set[int]] = {}
        self.docmap: Dict[str, Movie] = {}
        self.term_frequencies: Dict[int, Counter] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = stem_token(filter_token(tokenize(text)))
        counts = Counter()
        for token in tokens:
            if token not in self.index:
                self.index[token] = {doc_id}
            self.index[token].add(doc_id)
            counts.update([token])
        self.term_frequencies[doc_id] = counts

    def get_documents(self, term: str) -> list[int]:
        try:
            self.index[term]
        except:
            return None
        return sorted(list(self.index[term]))

    def build(self) -> None:
        data = load_movies()
        for movie in data['movies']:
            movie_data = f"{movie['title']} {movie['description']}"
            self.__add_document(movie["id"], movie_data)
            self.docmap[movie["id"]] = movie

    def save(self) -> None:
        with open(CACHE_INDEX_PATH, 'wb') as f:
            pickle.dump(self.index, f)
        with open(CACHE_DOCMAP_PATH, 'wb') as f:
            pickle.dump(self.docmap, f)
        with open(CACHE_TERM_FREQUENCY_PATH, 'wb') as f:
            pickle.dump(self.term_frequencies, f)

    def load(self) -> None:
        try:
            with open(CACHE_INDEX_PATH, 'rb') as f:
                self.index = pickle.load(f)
            with open(CACHE_DOCMAP_PATH, 'rb') as f:
                self.docmap = pickle.load(f)
            with open(CACHE_TERM_FREQUENCY_PATH, 'rb') as f:
                self.term_frequencies = pickle.load(f)
        except:
            print("Error: File does not exist")
            sys.exit(1)

    def get_tf(self, doc_id: int, term: str) -> int:
        return self.term_frequencies[doc_id][single_token(term)]

    def get_inv_df_score(self, term: str) -> int:
        term = single_token(term)

        total_doc_count: int = len(self.docmap)

        doc_ids = self.docmap.keys()
        term_match_doc_count: int = 0
        for doc_id in doc_ids:
            if self.get_tf(doc_id, term) != 0:
                term_match_doc_count += 1

        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def get_bm25(self, term: str) -> float:
        term = single_token(term)

        N: int = len(self.docmap)
        doc_ids = self.docmap.keys()
        df: int = 0
        for doc_id in doc_ids:
            if self.get_tf(doc_id, term) != 0:
                df += 1

        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id: int, term: str, k1: float = 1.5) -> float:
        tf = self.get_tf(doc_id, term)
        return (tf * (k1 + 1)) / (tf + k1)


DATA_PATH = "Data/movies.json"
STOPWORDS_PATH = "Data/stopwords.txt"
CACHE_INDEX_PATH = "cache/index.pkl"
CACHE_DOCMAP_PATH = "cache/docmap.pkl"
CACHE_TERM_FREQUENCY_PATH = "cache/term_frequencie.pkl"
BM25_K1 = 1.5


def load_movies() -> list[Movie]:
    with open(DATA_PATH, "r") as f:
        data = f.read()
    return json.loads(data)


def preprocess_text(text: str) -> str:
    return text.lower().translate(str.maketrans("","",string.punctuation))

def tokenize(text: str) -> list[str]:
    return preprocess_text(text).split()

def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        stopwords_preprocessed= f.read().splitlines()
    stopwords = []
    for stopword in stopwords_preprocessed:
        stopwords.append(preprocess_text(stopword))
    return stopwords

def filter_token(tokens: list[str]) -> list[str]:
    stopwords: list[str] = load_stopwords()
    filtered_tokens = []
    for token in tokens:
        if token not in stopwords:
            filtered_tokens.append(token)
    return filtered_tokens

def stem_token(tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer() # Create an instance of PorterStemmer

    stemmed_tokens = []
    for token in tokens:
        stemmed_tokens.append(stemmer.stem(token))
    return stemmed_tokens

def single_token(term: str) -> str:
    token = stem_token(filter_token(tokenize(term)))
    if len(token) != 1:
        raise Exception('Error: term should be a single token')
    return token[0]
