#Prepare training data for Critic - Compute verdict-based labels
import pandas as pd
import argparse
import os
from sklearn.model_selection import train_test_split
from collections import Counter


def main(input_path, train_path, test_path, log_path, split=0.2):
    # Load the CSV file
    df = pd.read_csv(input_path)

    # SWIFT uses 'verdict' instead of 'Verdict'/'Gate Verdict'
    if 'Verdict' in df.columns:
        df.rename(columns={'Verdict': 'verdict'}, inplace=True)

    # Ensure logs directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, "a") as f:
        f.write(f"Generated Data Size: {len(df)}\n")
        f.write(f"Unique Claims: {df['claim_id'].nunique()}\n")
        verdict_dist = df['verdict'].value_counts().to_string()
        f.write("Verdict Distribution:\n" + '\n'.join(verdict_dist.split('\n')[1:]) + "\n")
        f.write("====================================\n")

    print("Using verdict as label (aligned with SIM-RAG design)...")

    # Step 1: Split by claim_id (stratified by claim label) to prevent data leakage
    claim_info = df.groupby('claim_id')['label'].first()
    train_claim_ids, val_claim_ids = train_test_split(
        claim_info.index.tolist(),
        test_size=split,
        random_state=42,
        stratify=claim_info.values.tolist()
    )
    train_raw = df[df['claim_id'].isin(train_claim_ids)]
    val_raw = df[df['claim_id'].isin(val_claim_ids)]
    print(f"Claim-level split: {len(train_claim_ids)} train claims, {len(val_claim_ids)} val claims")

    # Step 2: Balance each split by verdict independently
    def balance_by_verdict(split_df):
        v0 = split_df[split_df['verdict'] == 0]
        v1 = split_df[split_df['verdict'] == 1]
        if len(v0) == 0 or len(v1) == 0:
            print(f"Warning: Imbalanced - verdict=0: {len(v0)}, verdict=1: {len(v1)}")
            return split_df
        min_count = min(len(v0), len(v1))
        v0 = v0.sample(n=min_count, random_state=42)
        v1 = v1.sample(n=min_count, random_state=42)
        return pd.concat([v0, v1]).sample(frac=1, random_state=42).reset_index(drop=True)

    train_df = balance_by_verdict(train_raw)
    dev_df = balance_by_verdict(val_raw)

    # Print final distribution
    with open(log_path, "a") as f:
        f.write(f"Train Claims: {len(train_claim_ids)}, Val Claims: {len(val_claim_ids)}\n")
        f.write(f"Train Data Size (balanced): {len(train_df)}\n")
        train_dist = train_df['verdict'].value_counts().to_string()
        f.write("Train Verdict Distribution:\n" + '\n'.join(train_dist.split('\n')[1:]) + "\n")

        f.write(f"Val Data Size (balanced): {len(dev_df)}\n")
        val_dist = dev_df['verdict'].value_counts().to_string()
        f.write("Val Verdict Distribution:\n" + '\n'.join(val_dist.split('\n')[1:]) + "\n")
        f.write(f"Train claim_ids and Val claim_ids are disjoint: {len(set(train_claim_ids) & set(val_claim_ids)) == 0}\n")
        f.write("====================================\n")

    # Ensure output directories exist
    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    # Save the train and dev sets
    train_df.to_csv(train_path, index=False)
    dev_df.to_csv(test_path, index=False)

    print(f"Train set saved to: {train_path} ({len(train_df)} samples, {train_df['claim_id'].nunique()} claims)")
    print(f"Val set saved to: {test_path} ({len(dev_df)} samples, {dev_df['claim_id'].nunique()} claims)")
    print(f"No claim overlap: {len(set(train_claim_ids) & set(val_claim_ids)) == 0}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare training data with evidence_sufficient labels.')

    parser.add_argument('--experiment_name', type=str, required=True, help='Name of the experiment')
    parser.add_argument('--input_path', type=str, required=False, help='Path to the input dataset')
    parser.add_argument('--train_path', type=str, required=False, help='Path to the training dataset')
    parser.add_argument('--test_path', type=str, required=False, help='Path to the val dataset')
    parser.add_argument('--log_path', type=str, required=False, help='Path to the log file')
    parser.add_argument('--split', type=float, default=0.2, help='Proportion for val split (default 0.2)')

    args = parser.parse_args()

    # Set defaults based on experiment_name
    if args.input_path is None:
        args.input_path = f'data/generated/{args.experiment_name}_generated.csv'
    if args.train_path is None:
        args.train_path = f'data/training/{args.experiment_name}_train.csv'
    if args.test_path is None:
        args.test_path = f'data/training/{args.experiment_name}_val.csv'
    if args.log_path is None:
        args.log_path = f'logs/{args.experiment_name}_log.txt'

    main(args.input_path, args.train_path, args.test_path, args.log_path, args.split)
