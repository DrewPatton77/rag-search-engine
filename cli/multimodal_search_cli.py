import argparse
from multimodal_search import *

parser = argparse.ArgumentParser(description="Multimodal Search CLI")
subparser = parser.add_subparsers(dest="command", help="Available commands")

verify_image_embedding_parser = subparser.add_parser("verify_image_embedding", help="Command to verify the image embedding")
verify_image_embedding_parser.add_argument("image_path", type=str, help="The image path to embed")

image_search_parser = subparser.add_parser("image_search", help="Search for movie that best matches a prompted image")
image_search_parser.add_argument("image_path", type=str, help="Search image")

args = parser.parse_args()

match args.command:

    case "verify_image_embedding":
        image_path = args.image_path
        verify_image_embedding(image_path)

    case "image_search":
        image_path = args.image_path
        top_scores = image_search_command(image_path)
        for i, id in enumerate(top_scores):
            print(f"{i + 1}. {top_scores[id]['title']} (similarity: {top_scores[id]['score']:.4f})")
            print(f"   {top_scores[id]['description'][:300]}...")
            print("")


    case _:
        parser.print_help()
