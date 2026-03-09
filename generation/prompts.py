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


# ============================================================
# Intervention query prompts for branched generation
# ============================================================

INTERVENTION_QUERY_PROMPT = """\
You are a fact-checking assistant. A claim is being verified with the following evidence and current judgment.

Claim: {claim}
Current Evidence:
{knowledge}
Current Judgment: {judgment}
Rationale: {rationale}

Your task: generate a Google search query for a specific retrieval strategy.

Strategy: {strategy_description}

Output ONLY the search query string, nothing else."""


STRATEGY_DESCRIPTIONS = {
    "support": "Find STRONGER supporting evidence that further confirms the current judgment. Look for authoritative sources, official data, or expert statements that reinforce the current conclusion.",
    "refute": "Find evidence that CONTRADICTS or CHALLENGES the current judgment. Look for counter-examples, opposing expert opinions, or data that undermines the current conclusion.",
    "resolve": "Find AUTHORITATIVE or DEFINITIVE evidence that can resolve any conflicts or ambiguities in the existing evidence. Look for official reports, meta-analyses, or primary sources that can serve as a tiebreaker.",
}


def format_prompt(claim: str, knowledge: str) -> str:
    """Format the unified prompt with claim and knowledge."""
    return UNIFIED_PROMPT.format(knowledge=knowledge, claim=claim)


def format_intervention_prompt(
    claim: str, knowledge: str, judgment: str, rationale: str, action: str
) -> str:
    """Format an intervention query prompt for a specific action type."""
    return INTERVENTION_QUERY_PROMPT.format(
        claim=claim,
        knowledge=knowledge,
        judgment=judgment,
        rationale=rationale,
        strategy_description=STRATEGY_DESCRIPTIONS[action],
    )
