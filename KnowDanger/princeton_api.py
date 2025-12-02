"""
Princeton AI Sandbox / Portkey Integration

This module provides a wrapper to use Princeton's AI Sandbox (via Portkey)
as a drop-in replacement for OpenAI API in your baseline tests.

Supported models:
- gpt-4-turbo
- gpt-3.5-turbo-16k
- Gemini models

Usage:
    from princeton_api import PrincetonLLM

    llm = PrincetonLLM(api_key="your-portkey-key")
    response = llm.chat_completion("What is 2+2?")
"""

import os
from typing import Optional, List, Dict, Any


class PrincetonLLM:
    """Wrapper for Princeton AI Sandbox / Portkey API"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo-16k",
        portkey_url: Optional[str] = None
    ):
        """
        Initialize Princeton LLM client

        Args:
            api_key: Portkey API key (or set PORTKEY_API_KEY env var)
            model: Model to use (gpt-4-turbo, gpt-3.5-turbo-16k, or gemini)
            portkey_url: Portkey gateway URL (if different from default)
        """
        self.api_key = api_key or os.environ.get("PORTKEY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Portkey API key required. Set PORTKEY_API_KEY environment variable "
                "or pass api_key parameter"
            )

        self.model = model
        self.portkey_url = portkey_url or "https://api.portkey.ai/v1"

        # Import OpenAI (older SDK version that supports ChatCompletion.create)
        try:
            import openai
            self.openai = openai

            # Check if we have the old API
            if not hasattr(openai, 'ChatCompletion'):
                print("⚠️  Warning: Your OpenAI SDK version may be too new.")
                print("   Princeton AI Sandbox requires the older SDK with ChatCompletion.create")
                print("   Try: pip install openai==0.28.0")
        except ImportError:
            raise ImportError(
                "OpenAI SDK required. Install with: pip install openai==0.28.0"
            )

    def chat_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate a chat completion using Princeton AI Sandbox

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            Generated text response
        """
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Configure OpenAI to use Portkey
        self.openai.api_base = self.portkey_url
        self.openai.api_key = self.api_key

        try:
            # Use old-style ChatCompletion.create (required by Princeton AI Sandbox)
            response = self.openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )

            return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"Princeton API call failed: {e}")

    def simple_prompt(self, prompt: str, max_tokens: int = 500) -> str:
        """Simple prompt interface"""
        return self.chat_completion(prompt, max_tokens=max_tokens)


def create_llm_function(
    api_key: Optional[str] = None,
    model: str = "gpt-3.5-turbo-16k"
):
    """
    Create a simple LLM function compatible with existing code

    Usage:
        llm_func = create_llm_function()
        result = llm_func("What is 2+2?")
    """
    llm = PrincetonLLM(api_key=api_key, model=model)
    return llm.simple_prompt


def test_princeton_api():
    """Test Princeton API connection"""
    print("=" * 70)
    print("Testing Princeton AI Sandbox / Portkey Connection")
    print("=" * 70)
    print()

    # Check for API key
    api_key = os.environ.get("PORTKEY_API_KEY")
    if not api_key:
        print("✗ PORTKEY_API_KEY not set in environment")
        print()
        print("Set it with:")
        print("  export PORTKEY_API_KEY='your-portkey-key'")
        return False

    print(f"✓ API key found: {api_key[:10]}...")
    print()

    # Test each model
    models_to_test = [
        "gpt-3.5-turbo-16k",
        "gpt-4-turbo",
        # "gemini-pro",  # Uncomment to test Gemini
    ]

    test_prompt = "Say 'Hello from Princeton AI Sandbox!' and nothing else."

    for model in models_to_test:
        print(f"Testing {model}...")
        try:
            llm = PrincetonLLM(api_key=api_key, model=model)
            response = llm.chat_completion(test_prompt, max_tokens=50)
            print(f"  ✓ Success! Response: {response[:100]}")
            print()
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            print()

    return True


if __name__ == "__main__":
    # Run test when executed directly
    test_princeton_api()
