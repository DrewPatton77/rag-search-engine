import argparse
from sentence_transformers import SentenceTransformer
from semantic_search import *


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Print model information to verify that the embedding model is loaded")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embeds given text")
    embed_text_parser.add_argument("text", type=str, help="Text to be embedded")

    args = parser.parse_args()



    match args.command:

        case "verify":
            verify_model()

            pass

        case "embed_text":
            text = args.text
            embed_text(text)

            pass

        case _:
            parser.print_help()

            pass


if __name__ == "__main__":
    main()
