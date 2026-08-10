import argparse
from sentence_transformers import SentenceTransformer
from semantic_search import *
import constants


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Print model information to verify that the embedding model is loaded")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embeds given text")
    embed_text_parser.add_argument("text", type=str, help="Text to be embedded")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="verifies if the movie embeddings were completed successfully")

    embed_query_parser = subparsers.add_parser("embed_query", help="Embeds the query text as a vector in the embedding space")
    embed_query_parser.add_argument("query", type=str, help="Text to be embedded")

    args = parser.parse_args()



    match args.command:

        case "verify":
            verify_model()

            pass

        case "embed_text":
            text = args.text
            embed_text(text)

            pass

        case "verify_embeddings":
            verify_embeddings()

        case "embed_query":
            query = args.query
            embed_query_text(query)

        case _:
            parser.print_help()

            pass


if __name__ == "__main__":
    main()
