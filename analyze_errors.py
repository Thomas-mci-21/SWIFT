"""Analyze error patterns in SWIFT predictions."""
import pandas as pd
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "predictions/swift_v4_thresh0.7_predictions.csv"
df = pd.read_csv(path)

# Focus on False claims
false_claims = df[df["label"].astype(str).str.lower() == "false"]
correct_false = false_claims["correct"].sum()
total_false = len(false_claims)

print(f"=== False Claims Analysis ===")
print(f"Total false claims: {total_false}")
print(f"Correctly identified: {int(correct_false)}/{total_false}")

correct_steps = false_claims[false_claims["correct"] == 1]["total_steps"].mean()
wrong_steps = false_claims[false_claims["correct"] == 0]["total_steps"].mean()
print(f"Avg steps (correct): {correct_steps:.2f}")
print(f"Avg steps (wrong): {wrong_steps:.2f}")
print()

# Print misclassified false claims
misses = false_claims[false_claims["correct"] == 0]
print(f"=== Misclassified False Claims ({len(misses)}) ===")
for _, row in misses.iterrows():
    claim = str(row["claim"])[:100]
    print(f"  Claim: {claim}")
    print(f"  Pred: {row['prediction']}, Steps: {row['total_steps']}, Stopped: {row['stopped_at_step']}")
    print()

# Focus on True claims
true_claims = df[df["label"].astype(str).str.lower() == "true"]
correct_true = true_claims["correct"].sum()
total_true = len(true_claims)
print(f"=== True Claims Analysis ===")
print(f"Total true claims: {total_true}")
print(f"Correctly identified: {int(correct_true)}/{total_true}")

# Step distribution
print(f"\n=== Step Distribution ===")
print(df["total_steps"].value_counts().sort_index())
print(f"\n=== Step Distribution by correctness ===")
print("Correct:")
print(df[df["correct"] == 1]["total_steps"].value_counts().sort_index())
print("Wrong:")
print(df[df["correct"] == 0]["total_steps"].value_counts().sort_index())
