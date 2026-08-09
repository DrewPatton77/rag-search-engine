import json
import string
from typing import Any, TypedDict

class Movie(TypedDict):
    id: int
    title: str
    descirption: str

DATA_PATH = "Data/movies.json"
STOPWORDS_PATH = "Data/stopwords.txt"

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
