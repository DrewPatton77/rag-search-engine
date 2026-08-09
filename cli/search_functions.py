import json

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
