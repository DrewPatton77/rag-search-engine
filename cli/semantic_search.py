from sentence_transformers import SentenceTransformer
import numpy as np
import os
import constants
from search_functions import load_movies, Movie

class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str):
        text = text.strip()
        if text == "":
            raise ValueError("The text is empty")

        embedding = self.model.encode([text])
        return embedding[0]

    def build_embeddings(self, documents: Movie):
        self.documents = documents
        document_list = []
        for document in documents:
            self.document_map[document['id']] = document
            document_list.append(f"{document['title']}: {document['description']}")

        self.embeddings = self.model.encode(document_list, show_progress_bar = True)

        with open(constants.CACHE_MOVIE_EMBEDDINGS_PATH, 'wb') as f:
            np.save(f, self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents: Movie):
        self.documents = documents
        for document in documents:
            self.document_map[document['id']] = document

        if os.path.exists(constants.CACHE_MOVIE_EMBEDDINGS_PATH):
            with open(constants.CACHE_MOVIE_EMBEDDINGS_PATH, 'rb') as f:
                self.embeddings = np.load(f)
                if len(self.embeddings) == len(documents):
                    return self.embeddings

        return self.build_embeddings(documents)

def verify_model() -> None:
    sems = SemanticSearch()
    print(f"Model loaded: {sems.model}")
    print(f"Max sequence length: {sems.model.max_seq_length}")

def embed_text(text: str) -> None:
    sems = SemanticSearch()
    embedding = sems.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings() -> None:
    sems = SemanticSearch()
    documents = load_movies()
    movies = documents['movies']
    sems.load_or_create_embeddings(movies)
    print(f"Number of docs: {len(movies)}")
    print(
        f"Embeddings shape: {sems.embeddings.shape[0]} vectors in {sems.embeddings.shape[1]} dimensions"
    )

def embed_query_text(query: str) -> None:
    sems = SemanticSearch()
    embedding = sems.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")
