import argparse
from hybrid_search import *
from call_llm import *
from time import sleep
import json
from sentence_transformers import CrossEncoder
import pickle


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
    rrf_search_parser.add_argument("-e", "--enhance", type=str, choices=["spell","rewrite","expand"], help="Query enhancement method")
    rrf_search_parser.add_argument("-r", "--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Asks an LLM to do a reranking of the top-scored documents")
    rrf_search_parser.add_argument("-ev", "--evaluate", action="store_true", help="Calls an LLM to evaluate the search results")

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

        case "rrf-search":
            query = args.query
            k = args.k
            limit = args.limit
            method = args.enhance
            rerank_method = args.rerank_method
            evaluate = args.evaluate

            if method != None:
                enhanced_query = call_llm(query, method=method)
                print(f"Enhanced query ({method}): '{query}' -> '{enhanced_query}'\n")
                query += " " + enhanced_query

            if rerank_method != None:

                print(f"Re-ranking top 3 results using {rerank_method} method...")
                print(f"Reciprocal Rank Fusion Results for {query} (k={k}):")
                limit = 5 * limit
                ranked_docs = rrf_search(query, k=k, limit=limit)

                if rerank_method == "individual":
                    for i, id in enumerate(ranked_docs):
                        rerank = "User Safety: safe"
                        while rerank == "User Safety: safe":
                            rerank = call_llm(query, rerank_method=rerank_method, doc=ranked_docs[id])
                        ranked_docs[id]['ranking'] = float(rerank)
                        sleep(3)
                    reranked_docs = dict(sorted(ranked_docs.items(), key=lambda item: item[1]['score'], reverse=True))

                if rerank_method == "batch":
                    doc_list_str = ""
                    for id in ranked_docs:
                        doc_list_str += f"Movie ID: {id}\n"
                        doc_list_str += f"Title: {ranked_docs[id]['title']}\n"
                        doc_list_str += f"Description: {ranked_docs[id]['document']}...\n"

                    reranked_json = call_llm(query, rerank_method=rerank_method, doc_list_str=doc_list_str)
                    with open("cache/reranked_array.json", "w") as f:
                        f.write(reranked_json)
                    with open("cache/reranked_array.json", "r") as f:
                        data = f.read()

                    reranked_array = json.loads(data)
                    reranked_docs = {id: ranked_docs[id] for id in reranked_array}


                if rerank_method == "cross_encoder":
                    reranked_docs = cross_encoder_reranker(query, ranked_docs)
                    #debug_log(query, ranked_docs, reranked_docs)
                print_results_rrf(reranked_docs, rerank_method=rerank_method, limit=limit / 5)

            elif evaluate != None:
                ranked_docs = rrf_search(query, k=k, limit=limit)
                formatted_results = ""
                for i, id in enumerate(ranked_docs):
                    formatted_results += f"{i + 1}. {ranked_docs[id]['title']}\n"
                    formatted_results += f"Reciprocal Rank Fusion Score: {ranked_docs[id]['hybrid_score']:.3f}\n"
                    formatted_results += f"BM25 Rank: {ranked_docs[id]['keyword_score']}, Semantic Rank: {ranked_docs[id]['semantic_score']}\n"
                    formatted_results += f"{ranked_docs[id]['document']}...\n"

                reranked_json = call_llm(query, evaluate=evaluate, doc_list_str=formatted_results)
                print(f"rerank_json: {reranked_json}")
                with open("cache/rerank_array_eval.json", "w") as f:
                    f.write(reranked_json)
                with open("cache/rerank_array_eval.json", "r") as f:
                    data = f.read()
                rerank_array = json.loads(data)
                for i,id in enumerate(ranked_docs):
                    print(f"{i + 1}. {ranked_docs[id]['title']}: {rerank_array[i]}/3" )

            else:
                ranked_docs = rrf_search(query, k=k, limit=limit)
                print_results_rrf(ranked_docs, limit=limit)

        case _:
            parser.print_help()


def print_results_rrf(docs: dict, rerank_method: str | None = None, limit: int = 25) -> None:

    for i, id in enumerate(docs):
        if i < limit:
            print(f"{i + 1}. {docs[id]['title']}")
            if rerank_method != None:
                print(f"Re-rank Rank: {i + 1}")
            if rerank_method == "individual":
                print(f"Re-rank Score {docs[id]['score']}/10")
            if rerank_method == "cross_encoder":
                print(f"Cross Encoder Score: {docs[id]['score']}")
            else:
                print(f"Reciprocal Rank Fusion: {docs[id]['hybrid_score']:.3f}")
            print(f"BM25 Rank: {docs[id]['keyword_score']}, Semantic Rank: {docs[id]['semantic_score']}")
            print(f"{docs[id]['document']}...")
            print("")

def debug_log(query: str, ranked_docs, reranked_docs) -> None:
    print(f"DEBUG LOG -- Query: {query}")

    print("DEBUG LOG -- RRF search Results:")
    for i, id in enumerate(ranked_docs):
        print("")
        print(f"{i + 1}. {ranked_docs[id]['title']}")
        print(f"   Reciprocal Rank Fusion Score: {ranked_docs[id]['score']}")
        print(f"   BM25 Rank: {ranked_docs[id]['keyword_score']}, Semantic Rank: {ranked_docs[id]['semantic_score']}")
        print(f"   {ranked_docs[id]['document']}...")
    print("")
    print("DEBUG LOG ------")

    print(f"DEBUG LOG -- Final Results After Re-ranking:")
    for i, id in enumerate(reranked_docs):
        print("")
        print(f"{i + 1}. {reranked_docs[id]['title']}")
        print(f"   RRF Score: {reranked_docs[id]['score']}")
        print(f"   BM25 Rank: {reranked_docs[id]['keyword_score']}, Semantic Rank: {reranked_docs[id]['semantic_score']}")
        print(f"   {reranked_docs[id]['document']}...")
    print(f"===========END OF DEBUG LOG==============")


if __name__ == "__main__":
    main()
