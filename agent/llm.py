"""
agent/llm.py - Model provider adapter and code extraction parser.
"""

import os
import re
from typing import List, Dict, Tuple, Optional

class LLMClient:
    """
    Unified LLM caller. Defaults to openai client structure (compatible with 
    OpenAI, Groq, OpenRouter, or local vLLM). Can be swapped easily.
    """
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._init_client()

    def _init_client(self):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        except ImportError:
            self.client = None

    def query(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> Tuple[str, int, int]:
        """
        Sends messages to LLM and returns (content, prompt_tokens, completion_tokens).
        """
        if not self.client:
            raise RuntimeError(
                "OpenAI client not initialized. Install openai (`pip install openai`) "
                "and set OPENAI_API_KEY environment variable."
            )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature
        )

        content = response.choices[0].message.content or ""
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0

        return content, prompt_tokens, completion_tokens

    @staticmethod
    def extract_python_code(raw_response: str) -> str:
        """
        Parses ```python ... ``` blocks. If missing, falls back to raw text.
        """
        pattern = r"```(?:python)?\s*\n(.*?)```"
        matches = re.findall(pattern, raw_response, re.DOTALL | re.IGNORECASE)
        if matches:
            # Pick the largest code block if multiple exist
            return max(matches, key=len).strip()
        
        # If no code fences are found, strip and return raw text
        return raw_response.strip()