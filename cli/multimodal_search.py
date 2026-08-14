from PIL import Image
from sentence_transformers import SentenceTransformer

class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)

    def image_embed(self, image_path: str):
        image = Image.open(image_path)
        embedding = self.model.encode([image])
        return embedding[0]

def verify_image_embedding(image_path: str):
    mms = MultimodalSearch()
    embedding = mms.image_embed(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")
