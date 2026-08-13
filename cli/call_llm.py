import os
from dotenv import load_dotenv
from openai import OpenAI
import system_prompt

def call_llm(query: str, method: str | None = None,  rerank_method: str | None = None, evaluate: bool | None = None, doc: dict | None = None, doc_list_str: str | None = None) -> str:

    client, model = llm_init()

    user_prompt = get_user_prompt(query, method=method, rerank_method=rerank_method, evaluate=evaluate, doc=doc, doc_list_str=doc_list_str)

    response = get_response(client, model, user_prompt)

    return response

def llm_init():
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        base_url = "https://openrouter.ai/api/v1",
        api_key = api_key,
    )

    model = "openrouter/free"

    return client, model

def get_response(client, model, user_prompt):
    messages = [
        {
            "role": "user", "content": user_prompt,
        }
    ]
    response = client.chat.completions.create(model=model, messages=messages)

    print("Response:")
    print(response.choices[0].message.content)
    print("-----")
    print(f"Prompt tokens: {response.usage.prompt_tokens}")
    print(f"Response tokens: {response.usage.completion_tokens}")

    return response.choices[0].message.content

def get_user_prompt(query: str, method: str | None = None,  rerank_method: str | None = None, evaluate: bool | None = None, doc: dict | None = None, doc_list_str: str | None = None):
    if method == "spell":
        user_prompt = f"""
        Fix any spelling errors in the user-provided movie search query below.
        Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
        Preserve punctuation and capitalization unless a change is required for a typo fix.
        If there are no spelling errors, or if you're unsure, output the original query unchanged.
        Output only the final query text, nothing else.
        User query: "{query}"
        """
    if method == "rewrite":
        user_prompt = f"""
        Rewrite the user-provided movie search query below to be more specific and searchable.

        Consider:
        - Common movie knowledge (famous actors, popular films)
        - Genre conventions (horror = scary, animation = cartoon)
        - Keep the rewritten query concise (under 10 words)
        - It should be a Google-style search query, specific enough to yield relevant results
        - Don't use boolean logic

        Examples:
        - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
        - "movie about bear in london with marmalade" -> "Paddington London marmalade"
        - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

        If you cannot improve the query, output the original unchanged.
        Output only the rewritten query text, nothing else.

        User query: "{query}"
        """

    if method == "expand":
        user_prompt = f"""Expand the user-provided movie search query below with related terms.

        Add synonyms and related concepts that might appear in movie descriptions.
        Keep expansions relevant and focused.
        Do not make the response too long, keep it short, relevant, focused, and at 20 tokens max.
        Output only the additional terms; they will be appended to the original query.

        Examples:
        - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
        - "action movie with bear" -> "action thriller bear chase fight adventure"
        - "comedy with bear" -> "comedy funny bear humor lighthearted"

        Include related professions, character types, and adjacent fields.

        User query: "{query}"
        """

    if rerank_method != None:
        if rerank_method == "individual":
            user_prompt = f"""Rate how well this movie matches the search query.

            Query: "{query}"
            Movie: {doc.get("title", "")} - {doc.get("document", "")}

            Consider:
            - Direct relevance to query
            - User intent (what they're looking for)
            - Content appropriateness

            Rate 0-10 (10 = perfect match).
            Do NOT respond with "User Safety: safe"
            Output ONLY the number in your response, no other text or explanation.

            Score:"""

        if rerank_method == "batch":
            user_prompt = f"""Rank the movies listed below by relevance to the following search query.

            Query: "{query}"

            Movies:
            {doc_list_str}

            Return the movie IDs in order of relevance, best match first.

            Your response must be a raw JSON array of integers.
            Do not wrap the JSON in Markdown. Do not use a ```json code block.
            Do not include any explanatory text.

            For example:
            [75, 12, 34, 2, 1]

            Ranking:"""

    if evaluate != None:
        user_prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

        Query: "{query}"

        Results:
        {chr(10).join(doc_list_str)}

        Scale:
        - 3: Highly relevant
        - 2: Relevant
        - 1: Marginally relevant
        - 0: Not relevant

        Do NOT give any numbers other than 0, 1, 2, or 3.

        Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

        [2, 0, 3, 2, 0, 1]"""

def RAG(query, titles):
    client, model = llm_init()
    user_prompt = f"""You are a RAG agent for Webflyx, a movie streaming service.
    Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
    Provide a comprehensive answer that addresses the user's query.

    Query: {query}

    Documents:
    {titles}

    Answer:"""
    response = get_response(client, model, user_prompt)

    return response

def llm_summarize(query, titles):
    client, model = llm_init()
    user_prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

    The goal is to provide comprehensive information so that users know what their options are.
    Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

    This should be tailored to Webflyx users. Webflyx is a movie streaming service.

    Query: {query}

    Search results:
    {titles}

    Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

    response = get_response(client, model, user_prompt)
    return response
