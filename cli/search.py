import json
from search_functions import load_movies
import string

def search_command(query: str) -> list[str]:

    data = load_movies()

    movies_in_query = []
    remove_table = str.maketrans("", "", string.punctuation)
    for movie in data['movies']:
        if query in movie['title'].lower().translate(remove_table):
            movies_in_query.append(movie['title'])

    return movies_in_query
