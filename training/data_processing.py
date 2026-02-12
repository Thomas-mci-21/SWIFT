"""Data processing for DeBERTa-v3 Critic training (SWIFT v5).

Key change from v4: Uses two-segment NLI format (premise + hypothesis)
to leverage DeBERTa's NLI pre-training. Labels are integers (0/1)
instead of text strings.
"""

import pandas as pd
from datasets import Dataset, DatasetDict


def load_data(train_path, val_path, test_path=None):
    """Load train/val/test data."""
    raw_train = pd.read_csv(train_path)
    raw_val = pd.read_csv(val_path)

    if test_path:
        try:
            raw_test = pd.read_csv(test_path)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            raw_test = raw_val.copy()
    else:
        raw_test = raw_val.copy()

    print(f'Loaded: train={len(raw_train)}, val={len(raw_val)}, test={len(raw_test)}')
    return raw_train, raw_val, raw_test


def prepare_dataset(df, num_samples=None):
    """Prepare dataset for DeBERTa Critic training.

    Uses two-segment format to leverage NLI pre-training:
        Premise:    Claim + Evidence + Rationale
        Hypothesis: "The judgment {X} is correct."
        Label:      1 (accept) or 0 (reject)
    """
    if num_samples is not None:
        df = df.head(num_samples)

    premises = []
    hypotheses = []
    labels = []

    for _, row in df.iterrows():
        premise = (
            f"Claim: {row['claim']}\n"
            f"Evidence:\n{row['knowledge']}\n"
            f"Rationale: {row['rationale']}"
        )
        hypothesis = f"The judgment {row['judgment']} is correct."

        premises.append(premise)
        hypotheses.append(hypothesis)
        labels.append(int(row['verdict']))

    return Dataset.from_dict({
        "premise": premises,
        "hypothesis": hypotheses,
        "label": labels,
    })


def create_dataset_dict(train_set, val_set, test_set):
    """Create HuggingFace DatasetDict."""
    return DatasetDict({
        'train': train_set,
        'val': val_set,
        'test': test_set
    })


def tokenize_data(dataset, tokenizer, max_length):
    """Tokenize dataset with two-segment format (premise, hypothesis)."""
    def preprocess(batch):
        return tokenizer(
            batch["premise"],
            batch["hypothesis"],
            max_length=max_length,
            padding="max_length",
            truncation="only_first",  # Truncate premise (evidence) only, keep hypothesis
        )

    return dataset.map(
        preprocess,
        batched=True,
        remove_columns=["premise", "hypothesis"],
    )
