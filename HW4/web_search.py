"""
web_search.py

takes a query, fetches search results, and asks an LLM to produce a cited, Markdown-formatted answer based on those results
"""

from utilities.call_llm import call_llm
from utilities.call_ddg import call_ddg

SYSTEM = """\
1) Read the provided web search results.
2) Generate an answer to the question.
 a) Format with markdown.
 b) The answer should be clear, concise, and grounded.
 c) Provide citations or URLs.
3) Do not include requests for follow-ups.
"""

USER = """\
Question:
{question}

Search Results:
{searchResults}
"""


def DDGSearch(query: str) -> str:
    searchResults = call_ddg(query)

    response = call_llm(
        system_prompt=SYSTEM,
        user_prompt=USER.format(question=query, searchResults=searchResults),
    )

    return response
