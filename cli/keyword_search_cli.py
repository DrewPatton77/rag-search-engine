import argparse
from search import search_command, build_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    search_parser = subparsers.add_parser("build", help="Builds inverted index for movies and caches the index mapping and document mapping")
    #search_parser.add_argument("query", type=str, help="Search query")


    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            query = args.query
            movie_titles: list[str] = search_command(query)
            movie_titles_truncated: list[str] = movie_titles[:5]
            for i in range(0,len(movie_titles_truncated)):
                print(f"{i+1}. {movie_titles_truncated[i]}")

            pass

        case "build":
            print(f"Building...")
            build_command()

        case _:
            parser.print_help()




if __name__ == "__main__":
    main()
