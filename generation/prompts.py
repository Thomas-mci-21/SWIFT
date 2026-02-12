"""Prompt templates for SWIFT."""

# Unified prompt: generate judgment, rationale, and search_query in one call
UNIFIED_PROMPT = """\
Instructions:
1. You are provided with a STATEMENT and relevant KNOWLEDGE points.
2. Based on the KNOWLEDGE, assess the factual accuracy of the STATEMENT.
3. Think step-by-step and provide your reasoning.
4. Also generate a backup search query in case more evidence is needed.
5. Output ONLY a valid JSON object in the following format (no other text):
   {{
     "judgment": "True" or "False",
     "rationale": "Your step-by-step reasoning here",
     "search_query": "A Google search query to find additional evidence"
   }}

IMPORTANT:
- Always provide a useful search_query, even if you are confident in your judgment.
- The query should aim to find NEW information not already present in the KNOWLEDGE.
- Output ONLY the JSON object, with no additional text before or after it.

KNOWLEDGE:
{knowledge}

STATEMENT:
{claim}
"""

# Placeholder definitions
KNOWLEDGE_PLACEHOLDER = '{knowledge}'
CLAIM_PLACEHOLDER = '{claim}'


def format_prompt(claim: str, knowledge: str) -> str:
    """Format the unified prompt with claim and knowledge."""
    return UNIFIED_PROMPT.format(knowledge=knowledge, claim=claim)
