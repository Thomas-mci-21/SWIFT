#!/bin/bash
# Setup script for lab server: download model + create shared_config
cd ~/SWIFT

# Download DeBERTa-v3-base-mnli model
mkdir -p models
/home1/pzm/miniconda3/bin/python -c "
from transformers import AutoModelForSequenceClassification, AutoTokenizer
m = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
print('Downloading DeBERTa-v3-base-mnli-fever-anli...')
t = AutoTokenizer.from_pretrained(m)
model = AutoModelForSequenceClassification.from_pretrained(m)
t.save_pretrained('models/deberta-v3-base-mnli')
model.save_pretrained('models/deberta-v3-base-mnli')
print('Saved to models/deberta-v3-base-mnli')
"

# Create shared_config.py placeholder (user fills in API key later)
if [ ! -f common/shared_config.py ]; then
    cat > common/shared_config.py << 'PYEOF'
# API configuration - fill in your keys
openai_api_key = "FILL_IN_YOUR_KEY"
openai_base_url = "https://api.chatanywhere.tech/v1"
serper_api_key = ""
PYEOF
    echo "Created common/shared_config.py (fill in API key)"
else
    echo "common/shared_config.py already exists"
fi

# Create necessary directories
mkdir -p data/generated data/training logs checkpoints predictions

echo "Setup complete!"
