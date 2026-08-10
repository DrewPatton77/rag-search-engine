from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

def verify_model():
    sems = SemanticSearch()
    print(f"Model loaded: {sems.model}")
    print(f"Max sequence length: {sems.model.max_seq_length}")
