import json
from search_functions import load_movies, tokenize, load_stopwords
from nltk.stem import PorterStemmer

def filter_tokens(tokens: list[str]) -> list[str]:
    stopwords: list[str] = load_stopwords()
    filtered_tokens = []
    for token in tokens:
        if token not in stopwords:
            filtered_tokens.append(token)
    return filtered_tokens

def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False

def stem_token(tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer() # Create an instance of PorterStemmer

    stemmed_tokens = []
    for token in tokens:
        stemmed_tokens.append(stemmer.stem(token))
    return stemmed_tokens

def search_command(query: str) -> list[str]:

    data = load_movies()

    movies_in_query = []
    for movie in data['movies']:
        query_tokens: list[str] = stem_token(filter_tokens(tokenize(query)))
        title_tokens: list[str] = stem_token(filter_tokens(tokenize(movie['title'])))
        if has_matching_token(query_tokens, title_tokens) and movie['title'] not in movies_in_query:
            movies_in_query.append(movie['title'])

    return movies_in_query
