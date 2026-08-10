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

    search_parser = subparsers.add_parser("search", help="Does a semantic search")
    search_parser.add_argument("query", type=str, help="The query to do a semantic search on")
    search_parser.add_argument("-l", "--limit", type=int, nargs="?", default=5, help="The number of movies to return")

    chunk_parser = subparsers.add_parser("chunk", help="Chunks according to a set size")
    chunk_parser.add_argument("text", type=str, help="The text to chunk")
    chunk_parser.add_argument("-cs","--chunk-size", type=int, nargs="?", default=200, help="The set chunk-size")


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

        case "search":
            query = args.query
            limit = args.limit
            sems = SemanticSearch()
            documents = load_movies()
            movies = documents['movies']
            sems.load_or_create_embeddings(movies)
            if limit != None:
                top_results = sems.search(query, limit)
            else:
                top_results = sems.search(query)

            for i in range(0,len(top_results)):
                result = top_results[i]
                print(
                    f"{i + 1}. {result['title']} (score: {result['score']})\n  {result['description']}"
                )
                print("")

        case "chunk":
            chunk_size = args.chunk_size
            text = args.text
            text_len = len(text)
            text_split = text.split(" ")
            text_chunks = [text_split[i:i + chunk_size] for i in range(0, len(text_split), chunk_size)]

            print(f"Chunking {text_len} characters")
            for i in range(0,len(text_chunks)):
                print(f"{i + 1}. {" ".join(text_chunks[i])}")

        case _:
            parser.print_help()

            pass


if __name__ == "__main__":
    main()
