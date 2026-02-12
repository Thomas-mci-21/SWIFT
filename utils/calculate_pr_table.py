"""
Calculate precision/recall for all methods and generate Table 2 data.
Run this script on the server: python utils/calculate_pr_table.py
"""

import os
import pandas as pd
from pathlib import Path

def evaluate_single_file(csv_path):
    """Calculate metrics for a single prediction CSV."""
    try:
        df = pd.read_csv(csv_path)

        # Check required columns
        if 'label' not in df.columns or 'prediction' not in df.columns:
            return None

        # Convert to binary (True=1, False=0)
        y_true = df['label'].apply(lambda x: 1 if str(x).lower() in ['true', '1'] else 0)
        y_pred = df['prediction'].apply(lambda x: 1 if str(x).lower() in ['true', '1'] else 0)

        # Confusion matrix
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()

        # Metrics for True class
        precision_true = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_true = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_true = 2 * precision_true * recall_true / (precision_true + recall_true) if (precision_true + recall_true) > 0 else 0

        # Metrics for False class
        precision_false = tn / (tn + fn) if (tn + fn) > 0 else 0
        recall_false = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_false = 2 * precision_false * recall_false / (precision_false + recall_false) if (precision_false + recall_false) > 0 else 0

        # Overall metrics
        accuracy = (tp + tn) / len(df)
        avg_steps = df['total_steps'].mean() if 'total_steps' in df.columns else 0

        return {
            'n_claims': len(df),
            'precision_true': precision_true,
            'recall_true': recall_true,
            'f1_true': f1_true,
            'precision_false': precision_false,
            'recall_false': recall_false,
            'f1_false': f1_false,
            'accuracy': accuracy,
            'avg_steps': avg_steps,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        }
    except Exception as e:
        print(f"Error evaluating {csv_path}: {e}")
        return None


def find_prediction_files(predictions_dir):
    """Find all prediction CSV files organized by dataset and method."""
    datasets = {
        'fcb': 'FactCheckBench',
        'factool': 'FacTool',
        'felm': 'FELM',
        'bingcheck': 'BingCheck',
        'hover': 'HoVer'
    }

    methods = {
        'no_search': 'No-Search',
        'fixed_1': 'Fixed-1',
        'fixed_3': 'Fixed-3',
        'fixed_5': 'Fixed-5',
        'fire': 'FIRE',
        'nli_baseline': 'NLI Baseline',
        'llm_critic': 'LLM-Critic',
        'swift': 'SWIFT'
    }

    results = {}

    for ds_key, ds_name in datasets.items():
        results[ds_name] = {}

        for method_key, method_name in methods.items():
            # Try different file naming patterns
            possible_patterns = [
                f"{predictions_dir}/{ds_key}_{method_key}_*.csv",
                f"{predictions_dir}/{ds_key}_{method_key}.csv",
                f"{predictions_dir}/{ds_name.lower()}_{method_key}_*.csv",
                f"{predictions_dir}/{ds_key}*{method_key}*.csv",
            ]

            found_file = None
            for pattern in possible_patterns:
                import glob
                matches = glob.glob(pattern)
                if matches:
                    found_file = matches[0]
                    break

            if found_file:
                print(f"Found: {ds_name} - {method_name}: {found_file}")
                metrics = evaluate_single_file(found_file)
                if metrics:
                    results[ds_name][method_name] = metrics
                else:
                    print(f"  WARNING: Could not evaluate {found_file}")
            else:
                print(f"NOT FOUND: {ds_name} - {method_name}")
                results[ds_name][method_name] = None

    return results


def generate_latex_table(results):
    """Generate LaTeX table for paper."""
    datasets_order = ['FactCheckBench', 'FacTool', 'FELM', 'BingCheck', 'HoVer']
    methods_order = ['No-Search', 'Fixed-1', 'Fixed-3', 'Fixed-5', 'FIRE',
                     'NLI Baseline', 'LLM-Critic', 'SWIFT']

    print("\n" + "="*120)
    print("LATEX TABLE FOR PAPER (Table 2)")
    print("="*120)

    for ds in datasets_order:
        if ds not in results:
            continue

        print(f"\n% {ds}")
        for method in methods_order:
            if method not in results[ds] or results[ds][method] is None:
                continue

            m = results[ds][method]
            print(f"% {method}")
            print(f"{method} & "
                  f"{m['precision_true']:.3f} & {m['recall_true']:.3f} & {m['f1_true']:.3f} & "
                  f"{m['precision_false']:.3f} & {m['recall_false']:.3f} & {m['f1_false']:.3f} & "
                  f"{m['accuracy']:.3f} & {m['avg_steps']:.2f} \\\\")


def generate_markdown_table(results):
    """Generate markdown table for easy viewing."""
    datasets_order = ['FactCheckBench', 'FacTool', 'FELM', 'BingCheck', 'HoVer']
    methods_order = ['No-Search', 'Fixed-1', 'Fixed-3', 'Fixed-5', 'FIRE',
                     'NLI Baseline', 'LLM-Critic', 'SWIFT']

    print("\n" + "="*140)
    print("MARKDOWN TABLE")
    print("="*140)

    for ds in datasets_order:
        if ds not in results:
            continue

        print(f"\n## {ds}")
        print("| Method | True P | True R | True F1 | False P | False R | False F1 | Acc | Steps |")
        print("|--------|:------:|:------:|:-------:|:-------:|:-------:|:--------:|:---:|:-----:|")

        for method in methods_order:
            if method not in results[ds] or results[ds][method] is None:
                continue

            m = results[ds][method]
            print(f"| {method} | "
                  f"{m['precision_true']:.3f} | {m['recall_true']:.3f} | {m['f1_true']:.3f} | "
                  f"{m['precision_false']:.3f} | {m['recall_false']:.3f} | {m['f1_false']:.3f} | "
                  f"{m['accuracy']:.1%} | {m['avg_steps']:.2f} |")


def generate_csv_export(results):
    """Export results to CSV for easy import to spreadsheet."""
    datasets_order = ['FactCheckBench', 'FacTool', 'FELM', 'BingCheck', 'HoVer']
    methods_order = ['No-Search', 'Fixed-1', 'Fixed-3', 'Fixed-5', 'FIRE',
                     'NLI Baseline', 'LLM-Critic', 'SWIFT']

    rows = []
    for ds in datasets_order:
        if ds not in results:
            continue
        for method in methods_order:
            if method not in results[ds] or results[ds][method] is None:
                continue

            m = results[ds][method]
            rows.append({
                'Dataset': ds,
                'Method': method,
                'True_Precision': m['precision_true'],
                'True_Recall': m['recall_true'],
                'True_F1': m['f1_true'],
                'False_Precision': m['precision_false'],
                'False_Recall': m['recall_false'],
                'False_F1': m['f1_false'],
                'Accuracy': m['accuracy'],
                'Avg_Steps': m['avg_steps']
            })

    df = pd.DataFrame(rows)
    output_path = 'docs/table2_metrics.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✓ Exported to {output_path}")


def main():
    predictions_dir = "predictions"

    if not os.path.exists(predictions_dir):
        print(f"ERROR: Predictions directory '{predictions_dir}' not found.")
        print("Please run this script from the SWIFT root directory.")
        return

    print("Scanning prediction files...")
    results = find_prediction_files(predictions_dir)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for ds_name, methods in results.items():
        print(f"\n{ds_name}:")
        for method_name, metrics in methods.items():
            if metrics:
                print(f"  {method_name:15s}: Acc={metrics['accuracy']:.1%}, "
                      f"True F1={metrics['f1_true']:.3f}, False F1={metrics['f1_false']:.3f}")

    # Generate outputs
    generate_markdown_table(results)
    generate_latex_table(results)
    generate_csv_export(results)

    print("\n" + "="*80)
    print("✓ Done! Check the output above.")
    print("="*80)


if __name__ == "__main__":
    main()
