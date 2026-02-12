"""Global configuration for SWIFT v5."""

# ============ Data Paths ============
# FactCheckBench dataset (main experiment)
RAW_DATA_PATH = "data/factcheckbench/raw.csv"
TRAIN_CLAIMS_PATH = "data/factcheckbench/train.csv"
TEST_CLAIMS_PATH = "data/factcheckbench/test.csv"

# ============ Generation Config ============
MAX_STEPS = 5           # Maximum search rounds
MIN_STEPS = 1           # Minimum search rounds (skip round 0 for training)
MAX_RETRIES = 10        # Max retries per step
NUM_SEARCH_RESULTS = 3  # Search results per query

# ============ LLM Config ============
LLM_MODEL = "gpt-4o-mini-ca"  # Model name for proxy API
LLM_TEMPERATURE = 0.5
LLM_MAX_TOKENS = 2048

# ============ Inference Config ============
INFERENCE_MAX_STEPS = 5
