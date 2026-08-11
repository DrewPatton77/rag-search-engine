import argparse
from hybrid_search import *

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize the cosine similarity and the BM25 scores using min-max normalization")
    normalize_parser.add_argument("scores", type=float, nargs="*", help="A list of scores to normalize using min-max normalization")


    args = parser.parse_args()

    match args.command:

        case "normalize":
            scores = args.scores
            min_max_norm_score(scores)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
