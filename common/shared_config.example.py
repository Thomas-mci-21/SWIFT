"""API Keys and shared configuration.

Copy this file to shared_config.py and fill in your API keys:
    cp common/shared_config.example.py common/shared_config.py
"""

# OpenAI-compatible API
openai_api_key = "your-api-key-here"
openai_api_key_2 = "your-secondary-api-key-here"  # Optional, for parallel inference
openai_base_url = "https://api.openai.com/v1"  # Or your proxy URL

# Model names
MODEL_GPT4O_MINI = "gpt-4o-mini"
MODEL_DEEPSEEK_V3 = "deepseek-v3"

# Serper API (optional, only needed if using --search_engine serper)
serper_api_key = "your-serper-api-key-here"
