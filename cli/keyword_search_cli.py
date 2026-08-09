import argparse
from search import search_command, build_command, tf_command, inv_df_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Builds inverted index for movies and caches the index mapping and document mapping")

    tf_parser = subparsers.add_parser("tf", help="Get the term frequency of a single term for a document id")
    tf_parser.add_argument("doc_id", type=int, help="The document id")
    tf_parser.add_argument("term", type=str, help="The term to count")

    inv_df_parser = subparsers.add_parser("idf", help="Get the inverse document frequency score for a given term")
    inv_df_parser.add_argument("term", type=str, help="The term for the inverse document frequency score")

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

        case _:
            parser.print_help()




if __name__ == "__main__":
    main()
