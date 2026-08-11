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
    chunk_parser.add_argument("-o", "--overlap", type=int, nargs="?", default=0, help="How much overlap each chunk should share")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunks according to sentence structure to preserve the natural language context")
    semantic_chunk_parser.add_argument("text", type=str, help="The text to chunk")
    semantic_chunk_parser.add_argument("-m", "--max-chunk-size", type=int, nargs="?", default=4, help="The maximum chunks size a chunk is allowed to be")
    semantic_chunk_parser.add_argument("-o", "--overlap", type=int, nargs="?", default=0, help="How much overlap each chunk should share")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Searches for the chunks that have the best cosine similarity score with the query")
    search_chunked_parser.add_argument("query", type=str, help="The query that to search from")
    search_chunked_parser.add_argument("-l", "--limit", type=int, nargs="?", default=5, help="Number of results to return")

    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="Embeds chunked movie descriptions")

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
            overlap = args.overlap
            text = args.text
            chunk_text(text, chunk_size, overlap)

        case "semantic_chunk":
            max_chunk_size = args.max_chunk_size
            overlap = args.overlap
            text = args.text
            semantic_chunk_text(text, max_chunk_size, overlap)

        case "embed_chunks":
            embed_chunks()

        case "search_chunked":
            query = args.query
            limit = args.limit
            search_chunked(query, limit=limit)

        case _:
            parser.print_help()

            pass


if __name__ == "__main__":
    main()
