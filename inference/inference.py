"""SWIFT v5: Inference with DeBERTa Critic + all baseline modes.

Modes:
    --mode swift         SWIFT with trained DeBERTa Critic (default)
    --mode nli_baseline  Off-the-shelf NLI model as Critic (no fine-tuning)
    --mode no_search     LLM only, no evidence retrieval
    --mode fixed_k       Fixed K search rounds, no Critic
    --mode llm_critic    GPT-4o-mini as Critic (prompt-based)
"""

import os
import sys
import json
import pandas as pd
import argparse
from tqdm import tqdm
import time
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from common.modeling import Model
from common.shared_config import serper_api_key
from search.query_serper import SerperAPI
from search.query_ddg import DuckDuckGoSearch
from generation.prompts import format_prompt
from generation.generation import extract_response, format_evidence, compute_verdict


# ============================================================
# Critic Implementations
# ============================================================

class DeBERTaCritic:
    """Trained DeBERTa-v3 Critic (SWIFT v5).

    Binary classification: 0=reject, 1=accept.
    Uses two-segment NLI format (premise + hypothesis).
    """

    def __init__(self, model_path: str, threshold: float = 0.5):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        self.threshold = threshold

    def predict(self, claim: str, knowledge: str, judgment: str, rationale: str) -> str:
        premise = (
            f"Claim: {claim}\n"
            f"Evidence:\n{knowledge}\n"
            f"Rationale: {rationale}"
        )
        hypothesis = f"The judgment {judgment} is correct."

        inputs = self.tokenizer(
            premise, hypothesis,
            return_tensors="pt",
            max_length=512,
            truncation="only_first",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=0)
            p_accept = probs[1].item()  # Label 1 = accept

        return "1" if p_accept >= self.threshold else "0"


class NLICriticBaseline:
    """Off-the-shelf NLI model as Critic (no fine-tuning).

    Uses entailment probability from 3-class NLI head.
    Model label mapping: 0=entailment, 1=neutral, 2=contradiction.
    """

    def __init__(self, model_path: str, threshold: float = 0.5):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        self.threshold = threshold

    def predict(self, claim: str, knowledge: str, judgment: str, rationale: str) -> str:
        premise = (
            f"Claim: {claim}\n"
            f"Evidence:\n{knowledge}\n"
            f"Rationale: {rationale}"
        )
        hypothesis = f"The judgment {judgment} is correct."

        inputs = self.tokenizer(
            premise, hypothesis,
            return_tensors="pt",
            max_length=512,
            truncation="only_first",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=0)
            # For MoritzLaurer model: id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
            p_entailment = probs[0].item()

        return "1" if p_entailment >= self.threshold else "0"


class LLMCritic:
    """GPT-4o-mini as Critic (prompt-based, no training)."""

    LLM_CRITIC_PROMPT = """You are evaluating whether a fact-checking judgment is well-supported by the evidence.

Claim: {claim}

Evidence:
{evidence}

Judgment: {judgment}
Rationale: {rationale}

Based on the evidence provided, is this judgment reliable and well-supported?
Answer with ONLY a single digit: 1 (yes, accept) or 0 (no, need more evidence)."""

    def __init__(self, rater: Model, threshold: float = 0.5):
        self.rater = rater
        self.threshold = threshold

    def predict(self, claim: str, knowledge: str, judgment: str, rationale: str) -> str:
        prompt = self.LLM_CRITIC_PROMPT.format(
            claim=claim, evidence=knowledge,
            judgment=judgment, rationale=rationale
        )
        response, _ = self.rater.generate(prompt)
        response = response.strip()

        # Parse response: look for 0 or 1
        if response and response[0] in ('0', '1'):
            return response[0]
        if '1' in response[:10]:
            return "1"
        return "0"


# ============================================================
# Inference Functions
# ============================================================

def run_inference_single(
    claim: str,
    label: str,
    rater: Model,
    critic,  # Any Critic with .predict() method, or None
    searcher,
    max_steps: int = 5,
    min_steps: int = 1,
    max_retries: int = 10,
    mode: str = "swift",
    fixed_k: int = 1,
) -> dict:
    """Run inference for a single claim.

    Supports all modes: swift, nli_baseline, no_search, fixed_k, llm_critic.
    """
    evidence_list = []
    total_llm_calls = 0
    total_critic_calls = 0
    final_judgment = None
    final_rationale = None
    stopped_at_step = max_steps

    # === Mode: no_search ===
    if mode == "no_search":
        prompt = format_prompt(claim, "N/A")
        for retry in range(max_retries):
            response, _ = rater.generate(prompt)
            total_llm_calls += 1
            judgment, rationale, _ = extract_response(response)
            if judgment:
                break
            time.sleep(2)
        verdict = compute_verdict(judgment or 'False', label)
        return {
            'claim': claim, 'label': label,
            'prediction': judgment or 'False', 'rationale': rationale,
            'correct': verdict, 'stopped_at_step': 0,
            'total_steps': 0, 'llm_calls': total_llm_calls, 'critic_calls': 0,
        }

    # === Mode: fixed_k ===
    if mode == "fixed_k":
        search_query = claim  # Initial query
        for step in range(fixed_k):
            # Search
            try:
                search_result = searcher.run(search_query)
            except Exception:
                search_result = "No search results available."
            evidence_list.append({'query': search_query, 'result': search_result})

            # LLM judgment
            knowledge = format_evidence(evidence_list)
            prompt = format_prompt(claim, knowledge)
            for retry in range(max_retries):
                response, _ = rater.generate(prompt)
                total_llm_calls += 1
                judgment, rationale, search_query = extract_response(response)
                if judgment:
                    break
                time.sleep(2)
            if not search_query:
                search_query = claim

        # Final judgment after K rounds
        knowledge = format_evidence(evidence_list)
        prompt = format_prompt(claim, knowledge)
        for retry in range(max_retries):
            response, _ = rater.generate(prompt)
            total_llm_calls += 1
            judgment, rationale, _ = extract_response(response)
            if judgment:
                break
            time.sleep(2)

        verdict = compute_verdict(judgment or 'False', label)
        return {
            'claim': claim, 'label': label,
            'prediction': judgment or 'False', 'rationale': rationale,
            'correct': verdict, 'stopped_at_step': fixed_k,
            'total_steps': fixed_k, 'llm_calls': total_llm_calls, 'critic_calls': 0,
        }

    # === Modes with Critic: swift, nli_baseline, llm_critic ===
    for step in range(max_steps):
        # Step 1: Search (first step mandatory, or Critic said need more)
        if step == 0 or (step > 0 and evidence_needed):
            knowledge = format_evidence(evidence_list)
            prompt = format_prompt(claim, knowledge)

            judgment, rationale, search_query = '', '', ''
            for retry in range(max_retries):
                response, usage = rater.generate(prompt)
                total_llm_calls += 1
                judgment, rationale, search_query = extract_response(response)
                if judgment and search_query:
                    break
                time.sleep(2)

            if not judgment:
                judgment = 'False'
            if not search_query:
                search_query = claim

            final_judgment = judgment
            final_rationale = rationale

            # Perform search
            try:
                search_result = searcher.run(search_query)
            except Exception:
                search_result = "No search results available."
            evidence_list.append({'query': search_query, 'result': search_result})

        # Update knowledge
        knowledge = format_evidence(evidence_list)

        # Step 2: Critic decision (only when step >= min_steps)
        evidence_needed = True
        if step >= min_steps and critic is not None:
            critic_output = critic.predict(claim, knowledge, final_judgment, final_rationale)
            total_critic_calls += 1

            if critic_output == "1":
                evidence_needed = False
                stopped_at_step = step + 1
                break

    if final_judgment is None:
        knowledge = format_evidence(evidence_list)
        prompt = format_prompt(claim, knowledge)
        for retry in range(max_retries):
            response, _ = rater.generate(prompt)
            total_llm_calls += 1
            judgment, rationale, _ = extract_response(response)
            if judgment:
                final_judgment = judgment
                final_rationale = rationale
                break
            time.sleep(2)
        if not final_judgment:
            final_judgment = 'False'

    verdict = compute_verdict(final_judgment, label)

    return {
        'claim': claim, 'label': label,
        'prediction': final_judgment, 'rationale': final_rationale,
        'correct': verdict, 'stopped_at_step': stopped_at_step,
        'total_steps': len(evidence_list), 'llm_calls': total_llm_calls,
        'critic_calls': total_critic_calls,
    }


def run_inference(
    input_path: str,
    output_path: str,
    experiment_name: str,
    mode: str = "swift",
    critic_path: str = None,
    nli_model_path: str = None,
    max_steps: int = 5,
    min_steps: int = 1,
    max_retries: int = 10,
    num_search_results: int = 3,
    api_key: str = None,
    threshold: float = 0.5,
    search_engine: str = "ddg",
    model_name: str = None,
    base_url: str = None,
    fixed_k: int = 1,
):
    """Run inference on test claims."""
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} claims from {input_path}")
    print(f"Mode: {mode}, Threshold: {threshold}, Search: {search_engine}")

    # Initialize LLM
    rater = Model(model_name=model_name, api_key=api_key, base_url=base_url)

    # Initialize Critic based on mode
    critic = None
    if mode == "swift":
        assert critic_path, "Must provide --critic_path for swift mode"
        print(f"Loading DeBERTa Critic from {critic_path} (threshold={threshold})...")
        critic = DeBERTaCritic(critic_path, threshold=threshold)
    elif mode == "nli_baseline":
        nli_path = nli_model_path or "models/deberta-v3-base-mnli"
        print(f"Loading off-the-shelf NLI Critic from {nli_path} (threshold={threshold})...")
        critic = NLICriticBaseline(nli_path, threshold=threshold)
    elif mode == "llm_critic":
        print("Using LLM-as-Critic (GPT-4o-mini)...")
        critic = LLMCritic(rater, threshold=threshold)
    elif mode in ("no_search", "fixed_k"):
        print(f"Mode: {mode}" + (f", K={fixed_k}" if mode == "fixed_k" else ""))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Initialize searcher
    if mode != "no_search":
        if search_engine == "ddg":
            searcher = DuckDuckGoSearch(k=num_search_results)
        else:
            searcher = SerperAPI(serper_api_key=serper_api_key, k=num_search_results)
    else:
        searcher = None

    # Resume support
    results = []
    skip_indices = set()
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path)
            results = existing_df.to_dict('records')
            skip_indices = set(range(len(results)))
            print(f"Resuming from {len(results)} existing predictions")
        except Exception:
            pass

    # Run inference
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Inference", initial=len(skip_indices)):
        if idx in skip_indices:
            continue

        result = run_inference_single(
            claim=row['claim'],
            label=row['label'],
            rater=rater,
            critic=critic,
            searcher=searcher,
            max_steps=max_steps,
            min_steps=min_steps,
            max_retries=max_retries,
            mode=mode,
            fixed_k=fixed_k,
        )
        result['claim_id'] = row.get('claim_id', f'test_{idx}')
        results.append(result)

        # Periodic save
        if len(results) % 10 == 0:
            pd.DataFrame(results).to_csv(output_path, index=False)

    # Final save
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)

    # Print metrics
    accuracy = result_df['correct'].mean()
    avg_steps = result_df['total_steps'].mean()
    true_claims = result_df[result_df['label'].astype(str).str.lower() == 'true']
    false_claims = result_df[result_df['label'].astype(str).str.lower() == 'false']
    true_acc = true_claims['correct'].mean() if len(true_claims) > 0 else 0
    false_acc = false_claims['correct'].mean() if len(false_claims) > 0 else 0

    print(f"\n=== {mode.upper()} Inference Complete ===")
    print(f"Total: {len(result_df)}, Accuracy: {accuracy:.4f}")
    print(f"True Acc: {true_acc:.4f} ({int(true_claims['correct'].sum())}/{len(true_claims)})")
    print(f"False Acc: {false_acc:.4f} ({int(false_claims['correct'].sum())}/{len(false_claims)})")
    print(f"Avg Steps: {avg_steps:.2f}")

    # Save log
    log_path = f"logs/{experiment_name}_inference_log.txt"
    os.makedirs("logs", exist_ok=True)
    with open(log_path, 'w') as f:
        f.write(f"Experiment: {experiment_name}\nMode: {mode}\n")
        f.write(f"Accuracy: {accuracy:.4f}\nAvg Steps: {avg_steps:.2f}\n")
        f.write(f"True Acc: {true_acc:.4f}\nFalse Acc: {false_acc:.4f}\n")

    return result_df


def main():
    parser = argparse.ArgumentParser(description='SWIFT v5 Inference')
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--input_path', type=str, default=config.TEST_CLAIMS_PATH)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--mode', type=str, default='swift',
                        choices=['swift', 'nli_baseline', 'no_search', 'fixed_k', 'llm_critic'])
    parser.add_argument('--critic_path', type=str, default=None)
    parser.add_argument('--nli_model_path', type=str, default=None)
    parser.add_argument('--max_steps', type=int, default=config.INFERENCE_MAX_STEPS)
    parser.add_argument('--min_steps', type=int, default=config.MIN_STEPS)
    parser.add_argument('--max_retries', type=int, default=config.MAX_RETRIES)
    parser.add_argument('--num_search_results', type=int, default=config.NUM_SEARCH_RESULTS)
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--search_engine', type=str, default='ddg', choices=['serper', 'ddg'])
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--base_url', type=str, default=None)
    parser.add_argument('--fixed_k', type=int, default=1, help='Number of search rounds for fixed_k mode')

    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = f"predictions/{args.experiment_name}_predictions.csv"
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    run_inference(
        input_path=args.input_path,
        output_path=args.output_path,
        experiment_name=args.experiment_name,
        mode=args.mode,
        critic_path=args.critic_path,
        nli_model_path=args.nli_model_path,
        max_steps=args.max_steps,
        min_steps=args.min_steps,
        max_retries=args.max_retries,
        num_search_results=args.num_search_results,
        api_key=args.api_key,
        threshold=args.threshold,
        search_engine=args.search_engine,
        model_name=args.model,
        base_url=args.base_url,
        fixed_k=args.fixed_k,
    )


if __name__ == "__main__":
    main()
