import json
from search_functions import load_movies, preprocess_text


def search_command(query: str) -> list[str]:

    data = load_movies()

    movies_in_query = []
    for movie in data['movies']:
        if preprocess_text(query) in preprocess_text(movie['title']):
            movies_in_query.append(movie['title'])

    return movies_in_query
