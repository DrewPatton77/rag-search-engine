import json
from search_functions import load_movies, tokenize


def search_command(query: str) -> list[str]:

    data = load_movies()

    movies_in_query = []
    for movie in data['movies']:
        query_tokens: list[str] = tokenize(query)
        title_tokens: list[str] = tokenize(movie['title'])
        for query_token in query_tokens:
            for title_token in title_tokens:
                if query_token in title_token and movie['title'] not in movies_in_query:
                    movies_in_query.append(movie['title'])

    return movies_in_query
