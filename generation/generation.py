"""Phase 1: Self-Practicing data generation for SWIFT."""

import os
import sys
import json
import re
import pandas as pd
import argparse
from tqdm import tqdm
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from common.modeling import Model
from common.shared_config import serper_api_key
from search.query_serper import SerperAPI
from search.query_ddg import DuckDuckGoSearch
from generation.prompts import format_prompt


def extract_response(response: str) -> tuple[str, str, str]:
    """Extract judgment, rationale, search_query from LLM response.

    Returns:
        tuple: (judgment, rationale, search_query)
    """
    if not response:
        return '', '', ''

    # Try to extract JSON
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        try:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            judgment = str(data.get('judgment', '')).strip()
            rationale = str(data.get('rationale', '')).strip()
            search_query = str(data.get('search_query', '')).strip()
            return judgment, rationale, search_query
        except json.JSONDecodeError:
            pass

    return '', '', ''


def format_evidence(evidence_list: list[dict]) -> str:
    """Format accumulated evidence into structured text.

    Args:
        evidence_list: [{"query": str, "result": str}, ...]

    Returns:
        Formatted knowledge string
    """
    if not evidence_list:
        return "N/A"

    formatted = []
    for i, ev in enumerate(evidence_list, 1):
        formatted.append(f"[Evidence {i}]\nQuery: {ev['query']}\nFindings: {ev['result']}")
    return "\n\n".join(formatted)


def compute_verdict(judgment: str, label) -> int:
    """Compute verdict: 1 if judgment matches label, 0 otherwise."""
    judgment_lower = str(judgment).lower().strip()
    label_lower = str(label).lower().strip()

    # Normalize judgment
    if judgment_lower in ['true', 'yes', '1']:
        judgment_normalized = 'true'
    elif judgment_lower in ['false', 'no', '0']:
        judgment_normalized = 'false'
    else:
        judgment_normalized = judgment_lower

    return 1 if judgment_normalized == label_lower else 0


def generate_trajectory(
    claim_id: str,
    claim: str,
    label: str,
    rater: Model,
    searcher: SerperAPI,
    max_steps: int = 5,
    min_steps: int = 1,
    max_retries: int = 10,
) -> tuple[list[dict], dict]:
    """Generate a complete retrieval trajectory for one claim.

    Args:
        claim_id: Unique identifier for the claim
        claim: The claim text
        label: Ground truth label ('true' or 'false')
        rater: LLM model for generation
        searcher: Serper API for search
        max_steps: Maximum number of search rounds
        min_steps: Minimum rounds before saving samples (skip round 0)
        max_retries: Max retries per step

    Returns:
        List of training samples (dicts)
    """
    samples = []
    evidence_list = []
    total_usage = {'input_tokens': 0, 'output_tokens': 0}

    for step in range(max_steps):
        # Format current knowledge
        knowledge = format_evidence(evidence_list)

        # Generate judgment + rationale + search_query
        prompt = format_prompt(claim, knowledge)

        judgment, rationale, search_query = '', '', ''
        for retry in range(max_retries):
            response, usage = rater.generate(prompt)
            if usage:
                total_usage['input_tokens'] += usage.get('input_tokens', 0)
                total_usage['output_tokens'] += usage.get('output_tokens', 0)

            judgment, rationale, search_query = extract_response(response)

            if judgment and search_query:
                break

            time.sleep(0.5)  # Brief pause before retry

        if not judgment:
            print(f"Warning: Failed to extract judgment for {claim_id} at step {step}")
            judgment = 'False'  # Default fallback

        if not search_query:
            search_query = claim  # Fallback to claim text

        # Compute verdict
        verdict = compute_verdict(judgment, label)

        # Save training sample (only when step >= min_steps)
        if step >= min_steps:
            sample = {
                'claim_id': claim_id,
                'step': step,
                'claim': claim,
                'knowledge': knowledge,
                'judgment': judgment,
                'rationale': rationale,
                'verdict': verdict,
                'label': label,
                'search_query': search_query,
                'trajectory_id': 0,  # Will be overwritten by run_generation
            }
            samples.append(sample)

        # Execute search to get new evidence
        try:
            search_result = searcher.run(search_query)
        except Exception as e:
            print(f"Search error for {claim_id}: {e}")
            search_result = "No search results available."

        evidence_list.append({
            'query': search_query,
            'result': search_result
        })

    return samples, total_usage


def run_generation(
    input_path: str,
    output_path: str,
    experiment_name: str,
    max_steps: int = 5,
    min_steps: int = 1,
    max_retries: int = 10,
    num_search_results: int = 3,
    start_idx: int = 0,
    end_idx: int = None,
    search_engine: str = "ddg",
    temperature: float = 0.7,
    trajectory_id: int = 0,
):
    """Run data generation for all claims.

    Args:
        input_path: Path to train claims CSV
        output_path: Path to save generated data
        experiment_name: Name for this experiment
        max_steps: Max search rounds
        min_steps: Min rounds before saving
        max_retries: Max retries per step
        num_search_results: Number of search results per query
        start_idx: Start index (for resuming)
        end_idx: End index (for partial runs)
        search_engine: "ddg" or "serper"
        temperature: LLM temperature (>0 for trajectory diversity)
        trajectory_id: Trajectory identifier (for multi-trajectory generation)
    """
    # Load claims
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} claims from {input_path}")

    # Slice if needed
    if end_idx is None:
        end_idx = len(df)
    df = df.iloc[start_idx:end_idx]
    print(f"Processing claims {start_idx} to {end_idx}")

    # Initialize models
    rater = Model(temperature=temperature)
    if search_engine == "ddg":
        searcher = DuckDuckGoSearch(k=num_search_results)
    else:
        searcher = SerperAPI(serper_api_key=serper_api_key, k=num_search_results)

    # Generate trajectories
    all_samples = []
    total_usage = {'input_tokens': 0, 'output_tokens': 0}

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating"):
        claim_id = row['claim_id']
        claim = row['claim']
        label = row['label']

        samples, usage = generate_trajectory(
            claim_id=claim_id,
            claim=claim,
            label=label,
            rater=rater,
            searcher=searcher,
            max_steps=max_steps,
            min_steps=min_steps,
            max_retries=max_retries,
        )

        # Tag trajectory_id on each sample
        for s in samples:
            s['trajectory_id'] = trajectory_id

        all_samples.extend(samples)
        total_usage['input_tokens'] += usage['input_tokens']
        total_usage['output_tokens'] += usage['output_tokens']

        # Periodic save
        if (idx + 1) % 10 == 0:
            temp_df = pd.DataFrame(all_samples)
            temp_df.to_csv(output_path, index=False)
            print(f"\nSaved {len(all_samples)} samples to {output_path}")

    # Final save
    result_df = pd.DataFrame(all_samples)
    result_df.to_csv(output_path, index=False)

    print(f"\n=== Generation Complete ===")
    print(f"Total samples: {len(all_samples)}")
    print(f"Total input tokens: {total_usage['input_tokens']}")
    print(f"Total output tokens: {total_usage['output_tokens']}")
    print(f"Saved to: {output_path}")

    # Save usage log
    log_path = f"logs/{experiment_name}_generation_log.txt"
    os.makedirs("logs", exist_ok=True)
    with open(log_path, 'w') as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Input: {input_path}\n")
        f.write(f"Output: {output_path}\n")
        f.write(f"Total samples: {len(all_samples)}\n")
        f.write(f"Total input tokens: {total_usage['input_tokens']}\n")
        f.write(f"Total output tokens: {total_usage['output_tokens']}\n")

    return result_df


def main():
    parser = argparse.ArgumentParser(description='SWIFT Phase 1: Data Generation')
    parser.add_argument('--experiment_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--input_path', type=str, default=config.TRAIN_CLAIMS_PATH, help='Input claims CSV')
    parser.add_argument('--output_path', type=str, default=None, help='Output generated data CSV')
    parser.add_argument('--max_steps', type=int, default=config.MAX_STEPS, help='Max search rounds')
    parser.add_argument('--min_steps', type=int, default=config.MIN_STEPS, help='Min rounds before saving')
    parser.add_argument('--max_retries', type=int, default=config.MAX_RETRIES, help='Max retries per step')
    parser.add_argument('--num_search_results', type=int, default=config.NUM_SEARCH_RESULTS, help='Search results per query')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index')
    parser.add_argument('--end_idx', type=int, default=None, help='End index')
    parser.add_argument('--search_engine', type=str, default='ddg', choices=['ddg', 'serper'], help='Search engine')
    parser.add_argument('--temperature', type=float, default=0.7, help='LLM temperature for diversity')
    parser.add_argument('--trajectory_id', type=int, default=0, help='Trajectory ID for multi-run generation')

    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = f"data/generated/{args.experiment_name}_generated.csv"

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    run_generation(
        input_path=args.input_path,
        output_path=args.output_path,
        experiment_name=args.experiment_name,
        max_steps=args.max_steps,
        min_steps=args.min_steps,
        max_retries=args.max_retries,
        num_search_results=args.num_search_results,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        search_engine=args.search_engine,
        temperature=args.temperature,
        trajectory_id=args.trajectory_id,
    )


if __name__ == "__main__":
    main()
