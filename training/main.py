"""SWIFT v5: Critic training with DeBERTa-v3 (binary classification + multi-head)."""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training import config
from training.data_processing import (
    load_data, prepare_dataset, prepare_dataset_v2,
    create_dataset_dict, tokenize_data, tokenize_data_v2,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeBERTa-v3 Critic for SWIFT.")
    parser.add_argument("--model_path", type=str, default=config.MODEL_PATH)
    parser.add_argument("--max_length", type=int, default=config.MAX_LENGTH)
    parser.add_argument("--experiment_name", type=str, default=config.EXPERIMENT_NAME)
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=config.PER_DEVICE_TRAIN_BATCH_SIZE)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=config.PER_DEVICE_EVAL_BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--num_train_epochs", type=int, default=config.NUM_TRAIN_EPOCHS)
    parser.add_argument("--warmup_ratio", type=float, default=config.WARMUP_RATIO)
    parser.add_argument("--fp16", action="store_true", default=config.FP16)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--weighted_training", action="store_true", default=False)
    parser.add_argument("--w_reject", type=float, default=1.5, help="Weight for reject class (0)")
    parser.add_argument("--w_accept", type=float, default=1.0, help="Weight for accept class (1)")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--model_type", type=str, default="binary", choices=["binary", "multihead"],
                        help="binary: original 2-class critic; multihead: R+Q multi-head critic")
    return parser.parse_args()


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy loss."""

    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weight = torch.tensor(self.class_weights, dtype=logits.dtype, device=logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


class MultiHeadTrainer(Trainer):
    """Trainer for multi-head R+Q critic (recoverability learning)."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        r_targets = inputs.pop("r_stop")
        q_targets = inputs.pop("q_targets")

        outputs = model(**inputs)
        logits = outputs.logits  # (batch, 4)

        r_pred = torch.sigmoid(logits[:, 0])  # stop risk in [0,1]
        q_pred = logits[:, 1:]                # (batch, 3) raw action values

        loss_r = F.binary_cross_entropy(r_pred, r_targets.float())
        loss_q = F.mse_loss(q_pred, q_targets.float())
        loss = loss_r + loss_q

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_prediction):
    predictions, labels = eval_prediction
    preds = np.argmax(predictions, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=[0, 1], zero_division=0
    )

    return {
        'accuracy': accuracy_score(labels, preds),
        'precision_reject': precision[0],
        'precision_accept': precision[1],
        'recall_reject': recall[0],
        'recall_accept': recall[1],
        'f1_reject': f1[0],
        'f1_accept': f1[1],
    }


def compute_metrics_multihead(eval_prediction):
    """Metrics for multi-head model: threshold R at 0.5 for accuracy, report MAE for Q."""
    predictions, labels = eval_prediction
    # predictions shape: (N, 4), labels shape: (N,) — but labels are r_stop only for now
    r_pred = 1.0 / (1.0 + np.exp(-predictions[:, 0]))  # sigmoid
    r_binary = (r_pred >= 0.5).astype(int)

    # labels come through as r_stop values (0 or 1)
    r_labels = labels.astype(int) if labels.ndim == 1 else labels[:, 0].astype(int)

    return {
        'r_accuracy': accuracy_score(r_labels, r_binary),
        'r_mae': float(np.mean(np.abs(r_pred - r_labels))),
    }


def main():
    args = parse_args()

    # Set default paths
    if args.train_path is None:
        args.train_path = f"data/training/{args.experiment_name}_train.csv"
    if args.val_path is None:
        args.val_path = f"data/training/{args.experiment_name}_val.csv"
    if args.output_dir is None:
        args.output_dir = f"checkpoints/{args.experiment_name}"
    if args.log_path is None:
        args.log_path = f"logs/{args.experiment_name}_log.txt"

    os.makedirs(os.path.dirname(args.output_dir) if os.path.dirname(args.output_dir) else "checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        print(f"Visible CUDA Devices: {os.environ['CUDA_VISIBLE_DEVICES']}")

    # ===== Binary mode (original) =====
    if args.model_type == "binary":
        raw_train, raw_val, raw_test = load_data(args.train_path, args.val_path)
        train_set = prepare_dataset(raw_train)
        val_set = prepare_dataset(raw_val)
        dataset = create_dataset_dict(train_set, val_set, val_set)

        print(f"Loading model from {args.model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path, num_labels=2, ignore_mismatched_sizes=True,
        )
        print(f"Model loaded: {model.config.model_type}, params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

        tokenized = tokenize_data(dataset, tokenizer, args.max_length)

        training_args = TrainingArguments(
            report_to=None, output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            learning_rate=args.learning_rate, weight_decay=args.weight_decay,
            num_train_epochs=args.num_train_epochs, warmup_ratio=args.warmup_ratio,
            logging_steps=config.LOGGING_STEPS, eval_strategy=config.EVAL_STRATEGY,
            save_strategy=config.SAVE_STRATEGY, save_total_limit=config.SAVE_TOTAL_LIMIT,
            fp16=args.fp16, load_best_model_at_end=True, metric_for_best_model="accuracy",
        )

        callbacks = [EarlyStoppingCallback(early_stopping_patience=args.patience)]

        if args.weighted_training:
            trainer = WeightedTrainer(
                class_weights=[args.w_reject, args.w_accept],
                model=model, args=training_args,
                train_dataset=tokenized["train"], eval_dataset=tokenized["val"],
                compute_metrics=compute_metrics, callbacks=callbacks,
            )
        else:
            trainer = Trainer(
                model=model, args=training_args,
                train_dataset=tokenized["train"], eval_dataset=tokenized["val"],
                compute_metrics=compute_metrics, callbacks=callbacks,
            )

        if args.eval_only:
            metrics = trainer.evaluate()
        else:
            trainer.train()
            print("Evaluating Model:")
            metrics = trainer.evaluate()

        print(f"Evaluation metrics: {metrics}")
        with open(args.log_path, 'a') as f:
            f.write(f"Critic Evaluation Metrics: {metrics}\n")
            f.write("========================================\n")

        if not args.eval_only:
            trainer.save_model(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"Model saved to {args.output_dir}")

    # ===== Multi-head mode (recoverability) =====
    elif args.model_type == "multihead":
        raw_train, raw_val, raw_test = load_data(args.train_path, args.val_path)
        train_set = prepare_dataset_v2(raw_train)
        val_set = prepare_dataset_v2(raw_val)
        dataset = create_dataset_dict(train_set, val_set, val_set)

        print(f"Loading model from {args.model_path} (multi-head, 4 outputs)...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path,
            num_labels=config.MULTIHEAD_NUM_LABELS,
            ignore_mismatched_sizes=True,
        )
        print(f"Model loaded: {model.config.model_type}, params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

        tokenized = tokenize_data_v2(dataset, tokenizer, args.max_length)

        training_args = TrainingArguments(
            report_to=None, output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            learning_rate=config.MULTIHEAD_LEARNING_RATE, weight_decay=args.weight_decay,
            num_train_epochs=config.MULTIHEAD_NUM_EPOCHS, warmup_ratio=args.warmup_ratio,
            logging_steps=config.LOGGING_STEPS, eval_strategy=config.EVAL_STRATEGY,
            save_strategy=config.SAVE_STRATEGY, save_total_limit=config.SAVE_TOTAL_LIMIT,
            fp16=args.fp16, load_best_model_at_end=True, metric_for_best_model="r_accuracy",
            remove_unused_columns=False,  # keep r_stop, q_targets for MultiHeadTrainer
        )

        callbacks = [EarlyStoppingCallback(early_stopping_patience=args.patience)]

        trainer = MultiHeadTrainer(
            model=model, args=training_args,
            train_dataset=tokenized["train"], eval_dataset=tokenized["val"],
            compute_metrics=compute_metrics_multihead, callbacks=callbacks,
        )
        trainer.label_names = ["r_stop"]  # tell Trainer which column is labels for metrics

        if args.eval_only:
            metrics = trainer.evaluate()
        else:
            trainer.train()
            print("Evaluating Model:")
            metrics = trainer.evaluate()

        print(f"Evaluation metrics: {metrics}")
        with open(args.log_path, 'a') as f:
            f.write(f"MultiHead Critic Metrics: {metrics}\n")
            f.write("========================================\n")

        if not args.eval_only:
            trainer.save_model(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            # Save a flag so inference knows this is a multi-head model
            import json
            meta = {"model_type": "multihead", "num_labels": config.MULTIHEAD_NUM_LABELS}
            with open(os.path.join(args.output_dir, "swift_meta.json"), "w") as f:
                json.dump(meta, f)
            print(f"Multi-head model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
