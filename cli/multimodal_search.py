from PIL import Image
from sentence_transformers import SentenceTransformer
from typing import Any, TypedDict
import os
import numpy as np
import constants
from search_functions import InvertedIndex, load_movies

class Movie(TypedDict):
    id: int
    title: str
    descirption: str

class MultimodalSearch:
    def __init__(self, documents: list[Movie], model_name="clip-ViT-B-32") -> None:
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts: list[str] = []
        self.document_map: dict[int, Movie] = {}
        self.embeddings = []
        self.score: dict[int,dict[str, Any]] = {}

    def generate_texts(self) -> None:
        for doc in self.documents:
            self.document_map[doc['id']] = doc
            self.texts.append(f"{doc['title']}: {doc['description']}")

    def build_embeddings(self, documents: Movie):
        self.documents = documents
        self.generate_texts()

        self.embeddings = self.model.encode(self.texts, show_progress_bar = True)

        with open(constants.CACHE_TEXT_EMBEDDINGS_PATH, 'wb') as f:
            np.save(f, self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self):

        if os.path.exists(constants.CACHE_TEXT_EMBEDDINGS_PATH):
            with open(constants.CACHE_TEXT_EMBEDDINGS_PATH, 'rb') as f:
                self.embeddings = np.load(f)
                if len(self.embeddings) == len(self.documents):
                    return self.embeddings

        return self.build_embeddings(self.documents)


    def search_with_image(self, image_path: str):
        image_embed = self.embed_image(image_path)
        for i, doc in enumerate(self.documents):
            score = cosine_similarity(self.embeddings[i], image_embed)
            self.score[doc['id']] = {
                'id': doc['id'],
                'title': doc['title'],
                'description': doc['description'],
                'score': score,
            }
        sorted_scores = dict(sorted(self.score.items(), key=lambda item: item[1]['score'], reverse=True))
        top_scores = {}
        for i, id in enumerate(sorted_scores):
            if i < 5:
                top_scores[id] = sorted_scores[id]
        return top_scores

    def embed_image(self, image_path: str):
        image = Image.open(image_path)
        embedding = self.model.encode([image])
        return embedding[0]

def verify_image_embedding(image_path: str):
    documents = load_movies()
    mms = MultimodalSearch(documents)
    embedding = mms.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)

    prod_norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if prod_norm == 0:
        return 0.0

    return dot_product / prod_norm

def image_search_command(image_path: str):
    documents = load_movies()
    mms = MultimodalSearch(documents['movies'])
    mms.load_or_create_embeddings()
    top_scores = mms.search_with_image(image_path)
    return top_scores
