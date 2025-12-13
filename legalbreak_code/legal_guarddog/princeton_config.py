"""
Princeton AI Sandbox / Portkey Integration Configuration

This file configures Legal Guarddog to use Princeton's AI Sandbox via Portkey.
The Portkey gateway routes requests to various LLM providers.

Supported models:
- gpt-4-turbo (recommended)
- gpt-3.5-turbo-16k
- gemini-pro
"""

import os

# Princeton API Configuration (via Portkey)
PRINCETON_CONFIG = {
    # Portkey gateway endpoint
    "api_base": os.getenv("PORTKEY_URL", "https://api.portkey.ai/v1"),

    # API key - Will automatically use PORTKEY_API_KEY environment variable
    "api_key": os.getenv("PORTKEY_API_KEY"),

    # API type
    "api_type": "portkey",

    # Model names available through Princeton AI Sandbox / Portkey
    # Note: Model names are case-sensitive and must match Portkey's routing config
    # Confirmed working models: gpt-4-turbo, gpt-4o
    "models": {
        "gpt-4": "gpt-4-turbo",  # Confirmed working
        "gpt-3.5-turbo": "gpt-4-turbo",  # Map to gpt-4-turbo (3.5 not available)
        "gpt-4o": "gpt-4o",  # Also works
    }
}


def get_princeton_config():
    """
    Get Princeton API configuration for Portkey.

    Returns a dict ready to pass to Config() constructor.
    """
    if not PRINCETON_CONFIG["api_key"]:
        raise ValueError(
            "PORTKEY_API_KEY environment variable not set. "
            "Set it with: export PORTKEY_API_KEY='your-key-here'"
        )

    return {
        "api_key": PRINCETON_CONFIG["api_key"],
        "api_base": PRINCETON_CONFIG["api_base"],
        "api_type": PRINCETON_CONFIG["api_type"],
    }


# Example usage and test
if __name__ == "__main__":
    print("=" * 70)
    print("Princeton AI Sandbox / Portkey Configuration Test")
    print("=" * 70)
    print()

    try:
        config = get_princeton_config()
        print("✓ Princeton/Portkey API configuration valid:")
        print(f"  API Base: {config['api_base']}")
        print(f"  API Key: {'*' * 20} (set)")
        print(f"\nAvailable models:")
        for openai_name, portkey_name in PRINCETON_CONFIG['models'].items():
            print(f"  {openai_name} → {portkey_name}")
        print()
        print("✓ Ready to use! Run your test script now.")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("\nTo configure:")
        print("1. Get your Portkey API key from Princeton AI Sandbox")
        print("2. Set environment variable:")
        print("   export PORTKEY_API_KEY='your-portkey-key-here'")
