import argparse
from search import *
import constants
from call_llm import *

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Builds inverted index for movies and caches the index mapping and document mapping")

    tf_parser = subparsers.add_parser("tf", help="Get the term frequency of a single term for a document id")
    tf_parser.add_argument("doc_id", type=int, help="The document id")
    tf_parser.add_argument("term", type=str, help="The term to use for counting")

    inv_df_parser = subparsers.add_parser("idf", help="Get the inverse document frequency score for a given term")
    inv_df_parser.add_argument("term", type=str, help="The term to use for the inverse document frequency score")

    tf_inv_df_parser = subparsers.add_parser("tfidf", help="Get the product of the term-frequency and inverse document frequency score")
    tf_inv_df_parser.add_argument("doc_id", type=int, help="The document id")
    tf_inv_df_parser.add_argument("term", type=str, help="The term to use for the tfidf score")

    bm25_parser = subparsers.add_parser("bm25idf", help="Get the bm25 inverse document frequency score")
    bm25_parser.add_argument("term", type=str, help="The term to use to calculate the bm25 inverse document frequency score")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get the term frequency saturation score for a given term and document id")
    bm25_tf_parser.add_argument("doc_id", type=int, help="The document id")
    bm25_tf_parser.add_argument("term", type=str, help="The term to use to calculate the term frequency saturation score for a given document id")
    bm25_tf_parser.add_argument("k1", type=float, nargs="?", default=constants.BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs="?", default=constants.BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("-l", "--limit", type=int, nargs="?", const=5, help="The number of movies to return")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            query = args.query
            movies: list[str] = search_command(query)
            for movie in movies:
                print(f"Title: {movie['title']} ID: {movie['id']}")

            pass

        case "build":
            print(f"Building...")
            build_command()
            print(f"Build completed successfully")

            pass

        case "tf":
            doc_id = args.doc_id
            term = args.term
            tf_command(doc_id, term)

        case "idf":
            term = args.term
            inv_df_command(term)

        case "tfidf":
            doc_id = args.doc_id
            term = args.term
            tfidf_command(doc_id, term)

        case "bm25idf":
            term = args.term
            bm25 = bm25_inv_df_command(term)
            print(f"BM25 Inv-DF score of '{args.term}': {bm25:.2f}")

        case "bm25tf":
            doc_id = args.doc_id
            term = args.term
            k1 = args.k1
            b = args.b
            bm25tf = bm25tf_command(doc_id, term, k1, b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")

        case "bm25search":
            query = args.query
            limit = args.limit
            if limit != None:
                bm25search_command(query, limit)
            else:
                bm25search_command(query)

        case _:
            parser.print_help()




if __name__ == "__main__":
    main()
