# RAG Search Engine

A command-line search engine over a movie dataset that builds hybrid retrieval from scratch: **BM25** keyword search, **semantic search** with sentence embeddings, and **reciprocal rank fusion (RRF)** to merge the two. Optional LLM steps handle query enhancement, re-ranking, and retrieval-augmented answers. It also supports image-based search.

## Features

| Area | What it does |
|---|---|
| Keyword search | Inverted index with tokenization, stopword removal and stemming (NLTK); TF, IDF, TF-IDF and full BM25 scoring (`k1 = 1.5`, `b = 0.75`) |
| Semantic search | `all-MiniLM-L6-v2` embeddings with cosine similarity; fixed-size and sentence-based chunking for long descriptions |
| Hybrid search | Min-max weighted fusion (`--alpha`) and reciprocal rank fusion (`-k`, default 60) |
| Query enhancement | LLM spelling correction, query rewriting, or query expansion |
| Re-ranking | LLM re-ranking (individual or batch) or a cross-encoder |
| RAG | Answers, summaries, and cited responses generated from the retrieved results |
| Multimodal | CLIP (`clip-ViT-B-32`) image search over the movie descriptions; LLM rewriting of a query from an image |
| Evaluation | Precision@k, recall@k and F1 against a golden dataset of queries and relevant movies |

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Put the data in `data/` (the directory is git-ignored):

- `data/movies.json`: movies with `id`, `title`, and `description`
- `data/stopwords.txt`
- `data/golden_dataset.json`: test cases for evaluation (`query`, `relevant_docs`)

The LLM features use [OpenRouter](https://openrouter.ai/). Create a `.env` file:

```
OPENROUTER_API_KEY=your_key_here
```

Build the index and embeddings once. They are cached in `cache/`.

```bash
uv run python cli/keyword_search_cli.py build
uv run python cli/semantic_search_cli.py verify_embeddings
```

## Usage

Run commands from the repository root.

```bash
# Keyword search (BM25)
uv run python cli/keyword_search_cli.py bm25search "space adventure" --limit 5

# Semantic search
uv run python cli/semantic_search_cli.py search "a robot learns to love" --limit 5

# Hybrid search with reciprocal rank fusion
uv run python cli/hybrid_search_cli.py rrf-search "family movie about lost toys" -k 60 --limit 5

# Hybrid search with LLM query enhancement and cross-encoder re-ranking
uv run python cli/hybrid_search_cli.py rrf-search "scary bear movie" --enhance rewrite --rerank-method cross_encoder

# Retrieval-augmented answer
uv run python cli/augmented_generation_cli.py rag "what are some good heist movies?"

# Evaluate retrieval quality
uv run python cli/evaluation_cli.py --limit 5
```

Run any CLI with `--help` to see all of its commands.

## How hybrid search works

1. **BM25** scores each document by term frequency (with saturation and length normalization) and inverse document frequency.
2. **Semantic search** embeds the query and the documents, then ranks by cosine similarity.
3. **RRF** ignores the raw scores, which are on different scales, and combines the two rankings: `score = Σ 1 / (k + rank)`.
4. Optionally, an LLM or cross-encoder re-ranks the top results.

## Project structure

```
cli/
├── keyword_search_cli.py          # BM25 / inverted index commands
├── semantic_search_cli.py         # embeddings, chunking, semantic search
├── hybrid_search_cli.py           # weighted and RRF search, enhancement, re-ranking
├── augmented_generation_cli.py    # RAG, summaries, citations
├── multimodal_search_cli.py       # CLIP image search
├── describe_image_cli.py          # LLM query rewriting from an image
├── evaluation_cli.py              # precision / recall / F1
└── *.py                           # search logic, LLM calls, constants
```

## Stack

Python · NLTK · NumPy · sentence-transformers · OpenAI SDK (via OpenRouter) · uv
