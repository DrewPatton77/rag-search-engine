from sentence_transformers import SentenceTransformer
import numpy as np
import os
import constants
from search_functions import load_movies, Movie
import re
import math

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

    def search(self, query: str, limit: int = 5):
        if len(self.embeddings) == 0:
            raise ValueError("No embeddings loaded. Call 'load_or_create_embeddings' first.")
        query_embedding = self.generate_embedding(query)

        cosine_similarity_list = []
        for i in range(0,len(self.embeddings)):
            cosine_similarity_list.append(cosine_similarity(self.embeddings[i], query_embedding))

        similarity_tuple_unsorted: list[tuple[float, Movie]] = zip(cosine_similarity_list, self.documents)
        similarity_tuple = sorted(similarity_tuple_unsorted, key = lambda t: t[0], reverse=True)

        top_results = []
        for i in range(0, int(limit)):
            top_results.append({
                "score": similarity_tuple[i][0],
                "title": similarity_tuple[i][1]['title'],
                "description": similarity_tuple[i][1]['description']
            })
        return top_results



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


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)

    prod_norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if prod_norm == 0:
        return 0.0

    return dot_product / prod_norm

def chunk_overlap(tokens: list[str], chunk_size: int, overlap: int = 0):
    num_tokens = len(tokens)
    i = 0
    chunks = []
    while i < num_tokens:
        chunk_tokens = tokens[i : i + chunk_size]
        if chunks and len(chunk_tokens) <= overlap:
            break

        chunks.append(chunk_tokens)
        i += chunk_size - overlap
    return chunks

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 0) -> None:
    text_len = len(text)
    words = text.split(" ")
    if overlap <= 0:
        word_chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]
    else:
        word_chunks = chunk_overlap(words, chunk_size, overlap=overlap)

    print(f"Chunking {text_len} characters")
    for i in range(0,len(word_chunks)):
        print(f"{i + 1}. {" ".join(word_chunks[i])}")

def semantic_chunk_text(text: str, max_chunk_size: int = 4, overlap: int = 0) -> None:
    text_len = len(text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if overlap <= 0:
        sentence_chunks = [sentences[i:i + max_chunk_size] for i in range(0, len(sentences), max_chunk_size)]
    else:
        sentence_chunks = chunk_overlap(sentences, max_chunk_size, overlap=overlap)

    print(f"Semantically chunking {text_len} characters")
    for i in range(0,len(sentence_chunks)):
        print(f"{i + 1}. {" ".join(sentence_chunks[i])}")
