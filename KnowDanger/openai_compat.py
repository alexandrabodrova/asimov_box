"""
OpenAI SDK Compatibility Shim

This module provides compatibility between old OpenAI SDK (0.28.0) and code
that expects the new SDK (1.x+). This allows RoboGuard and other components
to work with Princeton's Portkey API which requires the old SDK.

Usage:
    import openai_compat
    openai_compat.patch_roboguard()
"""

import sys
from typing import Any, Dict, List, Optional


class MockCompletion:
    """Mock completion response matching new SDK structure"""
    def __init__(self, content: str):
        self.message = MockMessage(content)


class MockMessage:
    """Mock message matching new SDK structure"""
    def __init__(self, content: str):
        self.content = content


class MockChoice:
    """Mock choice matching new SDK structure"""
    def __init__(self, content: str):
        self.message = MockMessage(content)


class MockResponse:
    """Mock response matching new SDK structure"""
    def __init__(self, content: str):
        self.choices = [MockChoice(content)]


class MockChatCompletions:
    """Mock chat.completions interface matching new SDK"""

    # Model mapping for Princeton/Portkey
    MODEL_MAP = {
        "gpt-4o": "gpt-4-turbo",  # RoboGuard uses gpt-4o, map to available model
        "gpt-4": "gpt-4-turbo",
        "gpt-3.5-turbo-16k": "gpt-3.5-turbo",  # In case the 16k variant doesn't work
    }

    def __init__(self, openai_module):
        self.openai = openai_module

    def create(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 2925,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        **kwargs
    ) -> MockResponse:
        """
        Create chat completion using old SDK

        This translates new SDK's client.chat.completions.create() to
        old SDK's openai.ChatCompletion.create()
        """
        # Map model to Princeton-supported version
        original_model = model
        model = self.MODEL_MAP.get(model, model)
        if model != original_model:
            print(f"   [Compat] Mapping {original_model} → {model}")
        # Convert new SDK message format to old SDK format
        old_messages = []
        for msg in messages:
            # Handle new SDK format with content list
            if isinstance(msg.get("content"), list):
                # Extract text from content list
                text_parts = []
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                content = "\n".join(text_parts)
            else:
                content = msg.get("content", "")

            old_messages.append({
                "role": msg.get("role", "user"),
                "content": content
            })

        # Call old SDK
        try:
            response = self.openai.ChatCompletion.create(
                model=model,
                messages=old_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )

            # Extract content from old SDK response
            content = response.choices[0].message.content if hasattr(response.choices[0], 'message') else response.choices[0]['message']['content']

            # Return in new SDK format
            return MockResponse(content)

        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}")


class MockChat:
    """Mock chat interface matching new SDK"""

    def __init__(self, openai_module):
        self.completions = MockChatCompletions(openai_module)


class MockOpenAIClient:
    """Mock OpenAI client matching new SDK structure"""

    def __init__(self, api_key: Optional[str] = None):
        # Import old SDK
        import openai
        self.openai = openai

        # Set API key if provided
        if api_key:
            openai.api_key = api_key

        # Create mock chat interface
        self.chat = MockChat(openai)


def patch_roboguard():
    """
    Patch RoboGuard to work with old OpenAI SDK

    This monkey-patches the roboguard.generator module to use our
    compatibility shim instead of the new OpenAI SDK.
    """
    try:
        # Import roboguard generator
        from roboguard import generator

        # Replace OpenAI import with our mock
        generator.OpenAI = MockOpenAIClient

        print("✓ Patched RoboGuard to use old OpenAI SDK")
        return True

    except ImportError as e:
        print(f"⚠️  Could not patch RoboGuard: {e}")
        return False
    except Exception as e:
        print(f"⚠️  Error patching RoboGuard: {e}")
        return False


def install_mock_openai():
    """
    Install mock OpenAI module in sys.modules

    This makes 'from openai import OpenAI' work with old SDK
    by providing our compatibility class.
    """
    import openai

    # Add OpenAI class to openai module
    if not hasattr(openai, 'OpenAI'):
        openai.OpenAI = MockOpenAIClient
        print("✓ Installed OpenAI compatibility class")


def setup_princeton_for_roboguard(api_key: str, model: str = "gpt-4-turbo", portkey_url: str = "https://api.portkey.ai/v1"):
    """
    Configure OpenAI SDK to use Princeton/Portkey for RoboGuard

    Args:
        api_key: Portkey API key
        model: Model to use
        portkey_url: Portkey gateway URL

    Returns:
        True if setup successful
    """
    try:
        import openai

        # Configure old SDK to use Portkey
        openai.api_base = portkey_url
        openai.api_key = api_key

        # Install our compatibility shim
        install_mock_openai()

        # Patch RoboGuard
        patch_roboguard()

        print(f"✓ Configured RoboGuard to use Princeton API with {model}")
        return True

    except Exception as e:
        print(f"✗ Failed to configure RoboGuard: {e}")
        return False


if __name__ == "__main__":
    # Test the compatibility shim
    print("Testing OpenAI SDK compatibility shim...")
    print()

    # Test mock client
    print("1. Testing MockOpenAIClient...")
    try:
        client = MockOpenAIClient()
        print("   ✓ MockOpenAIClient created")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    # Test patch
    print()
    print("2. Testing RoboGuard patch...")
    success = patch_roboguard()
    if success:
        print("   ✓ Patch successful")
    else:
        print("   ✗ Patch failed (RoboGuard may not be installed)")

    # Test install
    print()
    print("3. Testing mock OpenAI install...")
    try:
        install_mock_openai()
        from openai import OpenAI
        print("   ✓ Can import OpenAI class")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    print()
    print("Compatibility shim test complete!")
