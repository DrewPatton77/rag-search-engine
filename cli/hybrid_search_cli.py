import argparse
from hybrid_search import *

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize the cosine similarity and the BM25 scores using min-max normalization")
    normalize_parser.add_argument("scores", type=float, nargs="*", help="A list of scores to normalize using min-max normalization")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Weighs the search return")
    weighted_search_parser.add_argument("query", type=str, help="The query to search for")
    weighted_search_parser.add_argument("-a", "--alpha", type=float, nargs="?", default=0.5, help="The weight parameter. A proportional ratio between the keyword:semantic. 1 - gives a 100% keyword return, 0 - gives a 100% semantic return, 0.5 - 50/50 split, etc...")
    weighted_search_parser.add_argument("-l", "--limit", type=int, nargs="?", default=5, help="The number of results to return")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="Reciprocal Rank Fusion (rrf), instead of normalizing this method uses the rank of the score.")
    rrf_search_parser.add_argument("query", type=str, help="The query to search for")
    rrf_search_parser.add_argument("-k", type=int, nargs="?", default=60, help="A parameter that controls how much more weight we give to higher-ranked results vs lower-ranked ones. Where a lower k values like 20 gives more weight to top-ranked results, creating a steep drop-off in score. While higher k values like 100 creates a more gradual decline, giving lower-ranked results more influence")
    rrf_search_parser.add_argument("-l", "--limit", type=int, nargs="?", default=5, help="The number of results to return")

    args = parser.parse_args()

    match args.command:

        case "normalize":
            scores = args.scores
            min_max_norm_score(scores)

        case "weighted-search":
            query = args.query
            alpha = args.alpha
            limit = args.limit
            weighted_search(query, alpha=alpha, limit=limit)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
