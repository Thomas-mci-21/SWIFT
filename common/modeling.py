"""LLM wrapper using OpenAI proxy API."""

import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from openai import OpenAI
from common import shared_config

SYS_PROMPT = 'You are a fact-checking agent responsible for verifying the accuracy of claims.'


class Model:
    """Class for managing language models via OpenAI-compatible API."""

    def __init__(
        self,
        model_name: str = None,
        temperature: float = 0.5,
        max_tokens: int = 2048,
        api_key: str = None,
        base_url: str = None,
    ) -> None:
        self.model_name = model_name or "gpt-4o-mini-ca"
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = OpenAI(
            api_key=api_key or shared_config.openai_api_key,
            base_url=base_url or shared_config.openai_base_url,
            http_client=httpx.Client(proxy=None),
        )

    def generate(self, context: str) -> tuple[str, dict | None]:
        """Generate a response to the provided prompt.

        Args:
            context: Input text context.

        Returns:
            tuple: (response_text, usage_dict)
        """
        for retry in range(5):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": SYS_PROMPT},
                        {"role": "user", "content": context},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                usage = {
                    'input_tokens': response.usage.prompt_tokens,
                    'output_tokens': response.usage.completion_tokens,
                }

                return response.choices[0].message.content, usage

            except Exception as e:
                err_msg = str(e)
                if '429' in err_msg or 'rate_limit' in err_msg:
                    wait = (retry + 1) * 5
                    print(f"Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"Error in generate: {err_msg[:200]}")
                    time.sleep(10)
                    return None, None

        print("Max retries reached")
        return None, None
