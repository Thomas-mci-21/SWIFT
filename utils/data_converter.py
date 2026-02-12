"""Data conversion utilities: JSONL <-> CSV."""

import json
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def jsonl_to_csv(input_path: str, output_path: str) -> pd.DataFrame:
    """Convert JSONL to CSV and add claim_id."""
    records = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            data = json.loads(line.strip())
            records.append({
                'claim_id': f'fc_{idx}',
                'claim': data['claim'],
                'label': data['label']
            })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Converted {len(df)} records to {output_path}")
    return df


def split_claims(input_path: str, train_path: str, test_path: str, test_size: float = 0.2):
    """Split claims into train and test sets at claim level."""
    df = pd.read_csv(input_path)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=df['label']  # Stratify by label
    )

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train set: {len(train_df)} claims -> {train_path}")
    print(f"Test set: {len(test_df)} claims -> {test_path}")
    print(f"Label distribution in train: {train_df['label'].value_counts().to_dict()}")
    print(f"Label distribution in test: {test_df['label'].value_counts().to_dict()}")

    return train_df, test_df


def main():
    parser = argparse.ArgumentParser(description='Convert and split data')
    parser.add_argument('--input', type=str, default=config.RAW_DATA_PATH,
                        help='Input JSONL file path')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set proportion')
    args = parser.parse_args()

    # Step 1: Convert JSONL to CSV
    csv_path = args.input.replace('.jsonl', '.csv')
    df = jsonl_to_csv(args.input, csv_path)

    # Step 2: Split into train/test
    split_claims(csv_path, config.TRAIN_CLAIMS_PATH, config.TEST_CLAIMS_PATH, args.test_size)


if __name__ == "__main__":
    main()
