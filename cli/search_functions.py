import json
import string
from typing import Any, TypedDict

class Movie(TypedDict):
    id: int
    title: str
    descirption: str

DATA_PATH = "Data/movies.json"
def load_movies() -> list[Movie]:
    with open(DATA_PATH, "r") as f:
        data = f.read()

    return json.loads(data)

def preprocess_text(text: str) -> str:
    return text.lower().translate(str.maketrans("","",string.punctuation))

def tokenize(text: str) -> list[str]:
    return preprocess_text(text).split()
