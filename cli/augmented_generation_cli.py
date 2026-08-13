import argparse
from hybrid_search import *
from call_llm import *

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser("summarize", help="Summarizes the search results with respect to the query")
    summarize_parser.add_argument("query", type=str, help="Search query for summary")

    args = parser.parse_args()

    match args.command:

        case "rag":
            query = args.query
            ranked_docs = rrf_search(query, k=60, limit=5)
            titles = ""
            for id in ranked_docs:
                titles += f"- {ranked_docs[id]['title']}\n"

            llm_response = RAG(query, titles)

            print("Search Results:")
            print(titles)
            print("")
            print("RAG Response:")
            print(f"{llm_response}")

        case "summarize":
            query = args.query
            ranked_docs = rrf_search(query, k=60, limit=5)
            titles = ""
            for id in ranked_docs:
                titles += f"- {ranked_docs[id]['title']}\n"

            llm_response = llm_summarize(query, titles)

            print("Search Results:")
            print(titles)
            print("")
            print("LLM Summary:")
            print(f"{llm_response}")

        case _:
           parser.print_help()

if __name__ == "__main__":
   main()
