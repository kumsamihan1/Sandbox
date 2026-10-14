"""
agent/llm.py - Model provider adapter supporting Google Gemini (Free tier).
"""

import os
import re
from typing import List, Dict, Tuple, Optional
from google import genai

class LLMClient:
    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set. Set it in your environment or pass it to LLMClient.")
        self.client = genai.Client(api_key=self.api_key)

    def query(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> Tuple[str, int, int]:
        # Convert message history into a prompt string for Gemini
        formatted_prompt = ""
        for m in messages:
            role = m["role"].upper()
            formatted_prompt += f"\n\n--- {role} ---\n{m['content']}"

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=formatted_prompt,
        )

        content = response.text or ""
        # Token metrics for your evaluation reporting
        p_tok = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
        c_tok = response.usage_metadata.candidates_token_count if response.usage_metadata else 0

        return content, p_tok, c_tok

    @staticmethod
    def extract_python_code(raw_response: str) -> str:
        pattern = r"```(?:python)?\s*\n(.*?)```"
        matches = re.findall(pattern, raw_response, re.DOTALL | re.IGNORECASE)
        if matches:
            return max(matches, key=len).strip()
        return raw_response.strip()