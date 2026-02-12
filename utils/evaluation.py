"""Evaluation utilities for SWIFT."""

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import argparse


def evaluate_predictions(predictions_path: str):
    """Evaluate prediction results."""
    df = pd.read_csv(predictions_path)

    # Basic metrics
    accuracy = df['correct'].mean()
    avg_steps = df['total_steps'].mean()
    total_llm_calls = df['llm_calls'].sum()
    total_critic_calls = df['critic_calls'].sum()

    # Per-class metrics
    y_true = df['label'].apply(lambda x: 1 if str(x).strip().lower() == 'true' else 0)
    y_pred = df['prediction'].apply(lambda x: 1 if str(x).strip().lower() == 'true' else 0)

    precision_true = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall_true = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_true = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    precision_false = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    recall_false = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_false = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    results = {
        'Total Claims': len(df),
        'Accuracy': accuracy,
        'Avg. Steps': avg_steps,
        'Total LLM Calls': total_llm_calls,
        'Total Critic Calls': total_critic_calls,
        '---': '---',
        'Precision (True)': precision_true,
        'Recall (True)': recall_true,
        'F1 (True)': f1_true,
        '----': '----',
        'Precision (False)': precision_false,
        'Recall (False)': recall_false,
        'F1 (False)': f1_false,
        '-----': '-----',
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
    }

    print("\n=== Evaluation Results ===")
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate SWIFT predictions')
    parser.add_argument('--predictions', type=str, required=True, help='Path to predictions CSV')

    args = parser.parse_args()
    evaluate_predictions(args.predictions)


if __name__ == "__main__":
    main()
