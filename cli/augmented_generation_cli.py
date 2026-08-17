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

    citations_parser = subparsers.add_parser("citations", help="LLM generates a response with citations")
    citations_parser.add_argument("query", type=str, help="Search query for response and citations")

    question_parser = subparsers.add_parser("question", help="LLM will answer a question query")
    question_parser.add_argument("query", help="Question query for answer")

    args = parser.parse_args()

    match args.command:

        case "rag":
            query = args.query
            ranked_docs = rrf_search(query, k=60, limit=5)

            titles = format_titles(ranked_docs)
            doc_string = format_doc_str(ranked_docs)

            llm_response = RAG(query, doc_string)

            print("Search Results:")
            print(titles)
            print("")
            print("RAG Response:")
            print(f"{llm_response}")

        case "summarize":
            query = args.query
            ranked_docs = rrf_search(query, k=60, limit=5)

            titles = format_titles(ranked_docs)
            doc_string = format_doc_str(ranked_docs)

            llm_response = llm_summarize(query, doc_string)

            print("Search Results:")
            print(titles)
            print("")
            print("LLM Summary:")
            print(f"{llm_response}")

        case "citations":
            query = args.query
            ranked_docs = rrf_search(query, k=60, limit=5)

            titles = format_titles(ranked_docs)
            doc_string = format_doc_str(ranked_docs)

            llm_response = llm_citations(query, doc_string)

            print("Search Results:")
            print(titles)
            print("")
            print("LLM Answer:")
            print(f"{llm_response}")

        case "question":
            query = args.query
            ranked_docs = rrf_search(query, k=60, limit=5)

            titles = format_titles(ranked_docs)
            doc_string = format_doc_str(ranked_docs)

            llm_response = llm_question(query, doc_string)

            print("Search Results:")
            print(titles)
            print("")
            print("LLM Answer:")
            print(f"{llm_response}")

        case _:
           parser.print_help()

def format_titles(ranked_docs):
    titles = ""
    for id in ranked_docs:
        titles += f"- {ranked_docs[id]['title']}\n"
    return titles

def format_doc_str(ranked_docs):
    doc_list_str = ""
    for id in ranked_docs:
        doc_list_str += f"Movie ID: {id}\n"
        doc_list_str += f"Title: {ranked_docs[id]['title']}\n"
        doc_list_str += f"Description: {ranked_docs[id]['document']}...\n"
    return doc_list_str


if __name__ == "__main__":
   main()
