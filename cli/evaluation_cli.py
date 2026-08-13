import argparse
import json
from hybrid_search import *


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument("--limit", type=int, default=5, help="Number of results to evaluate (k for precision@k, recall@k")

    args = parser.parse_args()
    limit = args.limit

    with open("data/golden_dataset.json", "rb") as f:
        data = json.load(f)


    metrics = []
    for i in range(0, len(data['test_cases'])):
        query = data['test_cases'][i]['query']
        relevant_docs = data['test_cases'][i]['relevant_docs']
        print(f"{i + 1}. Query : {query}")
        sorted_docs = rrf_search(query, k=60, limit=limit)

        retrieved_docs = []
        for id in sorted_docs:
            retrieved_docs.append(sorted_docs[id]['title'])

        relevant_retrieved = 0
        for doc in retrieved_docs:
            if doc in relevant_docs:
               relevant_retrieved += 1

        total_retrieved = len(retrieved_docs)
        total_relevant = len(relevant_docs)
        precision = relevant_retrieved / total_retrieved
        recall = relevant_retrieved / total_relevant
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        metrics.append(
            {
                'query': query,
                f'Precision@{limit}': precision,
                f'Recall@{limit}': recall,
                'F1 Score': f1,
                'Retrieved': retrieved_docs,
                'Relevant': relevant_docs,
            }
        )

    print(f'k={limit}')
    for metric in metrics:
        print(f"- Query: {metric['query']}")
        print(f"  - Precision@{limit}: {metric[f'Precision@{limit}']:.4f}")
        print(f"  - Recall@{limit}: {metric[f'Recall@{limit}']:.4f}")
        print(f'F1 Score: {metric['F1 Score']:.4f}')
        print(f"  - Retrieved: {metric['Retrieved']}")
        print(f"  - Relevant: {metric['Relevant']}")


if __name__ == "__main__":
    main()
