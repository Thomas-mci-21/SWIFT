#!/bin/bash
cd ~/SWIFT
mkdir -p models data/generated data/training logs checkpoints predictions
/home1/pzm/miniconda3/bin/python << 'PYEOF'
from transformers import AutoModelForSequenceClassification, AutoTokenizer
m = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
print("Downloading model...")
t = AutoTokenizer.from_pretrained(m)
model = AutoModelForSequenceClassification.from_pretrained(m)
t.save_pretrained("models/deberta-v3-base-mnli")
model.save_pretrained("models/deberta-v3-base-mnli")
print("DONE: saved to models/deberta-v3-base-mnli")
PYEOF
