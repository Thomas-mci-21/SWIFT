"""Evaluate SWIFT predictions and compare with FIRE baseline."""

import pandas as pd

def evaluate_predictions(predictions_path: str):
    """Evaluate prediction results and print detailed metrics."""
    df = pd.read_csv(predictions_path)
    
    print(f"Total claims: {len(df)}")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    print(f"\nPrediction distribution:")
    print(df['prediction'].value_counts())
    
    # Convert to binary
    y_true = df['label'].apply(lambda x: 1 if str(x).lower() == 'true' else 0)
    y_pred = df['prediction'].apply(lambda x: 1 if str(x).lower() == 'true' else 0)
    
    # Confusion matrix
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    
    print(f"\n=== Confusion Matrix ===")
    print(f"TP (True->True): {tp}")
    print(f"TN (False->False): {tn}")
    print(f"FP (False->True): {fp}")
    print(f"FN (True->False): {fn}")
    
    # Metrics for True class
    precision_true = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_true = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_true = 2 * precision_true * recall_true / (precision_true + recall_true) if (precision_true + recall_true) > 0 else 0
    
    # Metrics for False class
    precision_false = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_false = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_false = 2 * precision_false * recall_false / (precision_false + recall_false) if (precision_false + recall_false) > 0 else 0
    
    print(f"\n=== True Class (label=True) ===")
    print(f"Precision: {precision_true:.4f}")
    print(f"Recall: {recall_true:.4f}")
    print(f"F1 Score: {f1_true:.4f}")
    
    print(f"\n=== False Class (label=False) ===")
    print(f"Precision: {precision_false:.4f}")
    print(f"Recall: {recall_false:.4f}")
    print(f"F1 Score: {f1_false:.4f}")
    
    # Overall metrics
    accuracy = (tp + tn) / len(df)
    avg_steps = df['total_steps'].mean()
    total_llm_calls = df['llm_calls'].sum()
    total_critic_calls = df['critic_calls'].sum()
    
    print(f"\n=== Overall Metrics ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Avg. Steps: {avg_steps:.2f}")
    print(f"Total LLM calls: {total_llm_calls}")
    print(f"Total Critic calls: {total_critic_calls}")
    
    # Compare with FIRE baseline
    print(f"\n=== Comparison with FIRE Baseline (FactCheckBench) ===")
    print(f"{'Metric':<20} {'FIRE':<12} {'SWIFT':<12} {'Diff':<12}")
    print("-" * 56)
    print(f"{'True F1':<20} {'0.87':<12} {f1_true:<12.4f} {f1_true - 0.87:<12.4f}")
    print(f"{'False F1':<20} {'0.67':<12} {f1_false:<12.4f} {f1_false - 0.67:<12.4f}")
    
    return {
        'accuracy': accuracy,
        'avg_steps': avg_steps,
        'precision_true': precision_true,
        'recall_true': recall_true,
        'f1_true': f1_true,
        'precision_false': precision_false,
        'recall_false': recall_false,
        'f1_false': f1_false,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        evaluate_predictions(sys.argv[1])
    else:
        evaluate_predictions("predictions/swift_v2_predictions.csv")
