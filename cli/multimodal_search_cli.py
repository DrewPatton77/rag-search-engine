import argparse
from multimodal_search import verify_image_embedding

parser = argparse.ArgumentParser(description="Multimodal Search CLI")
subparser = parser.add_subparsers(dest="Command", help="Available commands")

verify_image_embedding_parser = subparser.add_parser("verify_image_embedding", help="Command to verify the image embedding")
verify_image_embedding_parser.add_argument("image_path", type=str, help="The image path to embed")

args = parser.parse_args()

image_path = args.image_path
verify_image_embedding(image_path)
