"""
Princeton LLM API Configuration

This file contains the configuration for using Princeton's LLM API.
Update the values below with your Princeton API details.
"""

import os

# Princeton API Configuration
PRINCETON_CONFIG = {
    # API endpoint - Update this with Princeton's actual endpoint
    # Common formats:
    # - "https://api.princeton.edu/llm/v1"
    # - "https://princeton-openai.azure.com/openai/deployments/YOUR-DEPLOYMENT/chat/completions"
    # - "https://llm.princeton.edu/v1"
    "api_base": os.getenv("PRINCETON_API_BASE", "REPLACE_WITH_PRINCETON_ENDPOINT"),

    # API key - Will automatically use PRINCETON_API_KEY environment variable
    "api_key": os.getenv("PRINCETON_API_KEY"),

    # API type - Usually "openai" for OpenAI-compatible endpoints
    "api_type": "princeton",

    # Model names available through Princeton
    # Update these with the actual model names Princeton provides
    "models": {
        "gpt-4": "gpt-4",  # Or Princeton's equivalent name
        "gpt-3.5-turbo": "gpt-3.5-turbo",  # Or Princeton's equivalent
    }
}


def get_princeton_config():
    """
    Get Princeton API configuration.

    Returns a dict ready to pass to Config() constructor.
    """
    if not PRINCETON_CONFIG["api_key"]:
        raise ValueError(
            "PRINCETON_API_KEY environment variable not set. "
            "Set it with: export PRINCETON_API_KEY='your-key-here'"
        )

    if PRINCETON_CONFIG["api_base"] == "REPLACE_WITH_PRINCETON_ENDPOINT":
        raise ValueError(
            "Princeton API endpoint not configured. "
            "Update PRINCETON_CONFIG['api_base'] in princeton_config.py with the actual endpoint."
        )

    return {
        "api_key": PRINCETON_CONFIG["api_key"],
        "api_base": PRINCETON_CONFIG["api_base"],
        "api_type": PRINCETON_CONFIG["api_type"],
    }


# Example usage:
if __name__ == "__main__":
    try:
        config = get_princeton_config()
        print("✓ Princeton API configuration valid:")
        print(f"  API Base: {config['api_base']}")
        print(f"  API Key: {'*' * 20} (set)")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("\nTo configure:")
        print("1. Set your API key: export PRINCETON_API_KEY='your-key-here'")
        print("2. Update api_base in princeton_config.py with Princeton's endpoint URL")
