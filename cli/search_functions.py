from ast import Dict
import json
import string
from typing import Any, TypedDict
import pickle

class Movie(TypedDict):
    id: int
    title: str
    descirption: str

class InvertedIndex():
    def __init__(self):
        self.index: Dict[str , set[int]] = {}
        self.docmap: Dict[str, Movie] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = {doc_id}
            self.index[token].add(doc_id)


    def get_documents(self, term: str) -> list[int]:
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

DATA_PATH = "Data/movies.json"
STOPWORDS_PATH = "Data/stopwords.txt"
CACHE_INDEX_PATH = "cache/index.pkl"
CACHE_DOCMAP_PATH = "cache/docmap.pkl"

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
