"""
Calculate P/R/F1 from prediction CSVs and print values for results.md
Usage (on server): python utils/fill_pr.py
"""

import os
import pandas as pd
import glob

def calculate_metrics(csv_path):
    """Calculate all metrics for a prediction CSV."""
    df = pd.read_csv(csv_path)

    # Convert to binary
    y_true = df['label'].apply(lambda x: 1 if str(x).lower() in ['true', '1'] else 0)
    y_pred = df['prediction'].apply(lambda x: 1 if str(x).lower() in ['true', '1'] else 0)

    # Confusion matrix
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    # Metrics
    precision_true = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_true = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_true = 2 * precision_true * recall_true / (precision_true + recall_true) if (precision_true + recall_true) > 0 else 0

    precision_false = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_false = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_false = 2 * precision_false * recall_false / (precision_false + recall_false) if (precision_false + recall_false) > 0 else 0

    accuracy = (tp + tn) / len(df)
    avg_steps = df['total_steps'].mean() if 'total_steps' in df.columns else 0

    return {
        'precision_true': precision_true,
        'recall_true': recall_true,
        'f1_true': f1_true,
        'precision_false': precision_false,
        'recall_false': recall_false,
        'f1_false': f1_false,
        'accuracy': accuracy,
        'avg_steps': avg_steps
    }


def find_and_calculate():
    """Find all prediction files and calculate metrics."""
    predictions_dir = "predictions"

    # Define file patterns
    patterns = {
        'FCB': {
            'No-Search': 'fcb_no_search*.csv',
            'Fixed-1': 'fcb_fixed_1*.csv',
            'Fixed-3': 'fcb_fixed_3*.csv',
            'Fixed-5': 'fcb_fixed_5*.csv',
            'FIRE': 'fcb_fire*.csv',
            'NLI Baseline': 'fcb_nli_baseline*.csv',
            'LLM-Critic': 'fcb_llm_critic*.csv',
            'SWIFT': 'fcb_swift*.csv',
        },
        'FacTool': {
            'No-Search': 'factool_no_search*.csv',
            'Fixed-1': 'factool_fixed_1*.csv',
            'Fixed-3': 'factool_fixed_3*.csv',
            'Fixed-5': 'factool_fixed_5*.csv',
            'FIRE': 'factool_fire*.csv',
            'NLI Baseline': 'factool_nli_baseline*.csv',
            'LLM-Critic': 'factool_llm_critic*.csv',
            'SWIFT': 'factool_swift*.csv',
        },
        'FELM': {
            'No-Search': 'felm_no_search*.csv',
            'Fixed-1': 'felm_fixed_1*.csv',
            'Fixed-3': 'felm_fixed_3*.csv',
            'Fixed-5': 'felm_fixed_5*.csv',
            'FIRE': 'felm_fire*.csv',
            'NLI Baseline': 'felm_nli_baseline*.csv',
            'LLM-Critic': 'felm_llm_critic*.csv',
            'SWIFT': 'felm_swift*.csv',
        },
        'BingCheck': {
            'No-Search': 'bingcheck_no_search*.csv',
            'Fixed-1': 'bingcheck_fixed_1*.csv',
            'Fixed-3': 'bingcheck_fixed_3*.csv',
            'Fixed-5': 'bingcheck_fixed_5*.csv',
            'FIRE': 'bingcheck_fire*.csv',
            'NLI Baseline': 'bingcheck_nli_baseline*.csv',
            'LLM-Critic': 'bingcheck_llm_critic*.csv',
            'SWIFT': 'bingcheck_swift*.csv',
        },
        'HoVer': {
            'No-Search': 'hover_no_search*.csv',
            'Fixed-1': 'hover_fixed_1*.csv',
            'Fixed-3': 'hover_fixed_3*.csv',
            'Fixed-5': 'hover_fixed_5*.csv',
            'NLI Baseline': 'hover_nli_baseline*.csv',
            'LLM-Critic': 'hover_llm_critic*.csv',
            'SWIFT': 'hover_swift*.csv',
        }
    }

    results = {}
    for dataset, methods in patterns.items():
        results[dataset] = {}
        for method, pattern in methods.items():
            matches = glob.glob(os.path.join(predictions_dir, pattern))
            if matches:
                metrics = calculate_metrics(matches[0])
                results[dataset][method] = metrics
                print(f"[OK] {dataset:12s} {method:15s}: P_True={metrics['precision_true']:.3f}, R_True={metrics['recall_true']:.3f}, "
                      f"P_False={metrics['precision_false']:.3f}, R_False={metrics['recall_false']:.3f}")
            else:
                print(f"[MISSING] NOT FOUND: {dataset} {method}")
                results[dataset][method] = None

    return results


def print_edit_instructions(results):
    """Print formatted values for easy copying into results.md."""
    print("\n" + "="*100)
    print("COPY THESE VALUES INTO results.md")
    print("="*100)

    for dataset in ['FCB', 'FacTool', 'FELM', 'BingCheck', 'HoVer']:
        if dataset not in results:
            continue

        print(f"\n### {dataset}")
        methods_order = ['No-Search', 'Fixed-1', 'Fixed-3', 'Fixed-5', 'FIRE',
                         'NLI Baseline', 'LLM-Critic', 'SWIFT']

        for method in methods_order:
            if method not in results[dataset] or results[dataset][method] is None:
                if method == 'FIRE' and dataset == 'HoVer':
                    continue  # Skip HoVer FIRE (not available)
                print(f"# {method}: NOT FOUND")
                continue

            m = results[dataset][method]

            if method == 'FIRE' and dataset == 'HoVer':
                continue  # Skip HoVer FIRE

            # Format: | Method | True P | True R | True F1 | False P | False R | False F1 | Acc | Steps |
            print(f"| {method} | "
                  f"{m['precision_true']:.3f} | {m['recall_true']:.3f} | {m['f1_true']:.3f} | "
                  f"{m['precision_false']:.3f} | {m['recall_false']:.3f} | {m['f1_false']:.3f} | "
                  f"{m['accuracy']:.1%} | {m['avg_steps']:.2f} |")


if __name__ == "__main__":
    results = find_and_calculate()
    print_edit_instructions(results)
