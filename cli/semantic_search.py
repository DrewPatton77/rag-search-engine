from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, text: str):
        if text.strip() == "":
            raise ValueError("The text is empty")

        embedding = self.model.encode([text])
        return embedding[0]

def embed_text(text: str):
    sems = SemanticSearch()
    embedding = sems.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_model():
    sems = SemanticSearch()
    print(f"Model loaded: {sems.model}")
    print(f"Max sequence length: {sems.model.max_seq_length}")
