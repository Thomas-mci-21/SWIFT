# SWIFT v5: DeBERTa-v3 Critic Configuration

# Model configuration
MODEL_PATH = "models/deberta-v3-base-mnli"  # Local path (downloaded from MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli)
MAX_LENGTH = 512  # DeBERTa max position embeddings

# Data configuration
EXPERIMENT_NAME = "swift_v5"

# Training configuration
PER_DEVICE_TRAIN_BATCH_SIZE = 32  # DeBERTa-v3-base is small (86M), A40 can handle 32
PER_DEVICE_EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5  # Standard for fine-tuning pre-trained models
WEIGHT_DECAY = 0.01
NUM_TRAIN_EPOCHS = 10  # More epochs for small dataset, use early stopping
WARMUP_RATIO = 0.1
LOGGING_STEPS = 50
EVAL_STRATEGY = "epoch"
SAVE_STRATEGY = "epoch"
SAVE_TOTAL_LIMIT = 2
FP16 = True
