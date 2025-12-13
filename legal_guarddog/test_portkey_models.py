#!/usr/bin/env python3
"""
Quick test to find which model names work with your Portkey setup
"""

import os
import openai

# Configure Portkey
portkey_key = os.getenv("PORTKEY_API_KEY")
if not portkey_key:
    print("❌ PORTKEY_API_KEY not set")
    exit(1)

openai.api_key = portkey_key
openai.api_base = "https://api.portkey.ai/v1"

# Models to test
models_to_test = [
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-35-turbo",  # Azure naming
    "gpt-4o",
    "gemini-pro",
]

print("Testing Portkey model names...")
print("=" * 70)

test_message = [{"role": "user", "content": "Say 'test' and nothing else"}]

for model in models_to_test:
    try:
        print(f"\nTrying: {model}...", end=" ")
        response = openai.ChatCompletion.create(
            model=model,
            messages=test_message,
            max_tokens=10,
            temperature=0
        )
        result = response.choices[0].message.content
        print(f"✅ SUCCESS! Response: {result}")
    except Exception as e:
        error_msg = str(e)
        if "unknown-model" in error_msg or "Invalid target name" in error_msg:
            print(f"❌ Model not recognized by Portkey")
        elif "404" in error_msg:
            print(f"❌ Model not found")
        else:
            print(f"❌ Error: {error_msg[:100]}")

print("\n" + "=" * 70)
print("\nIf none worked, check your Portkey dashboard for configured models:")
print("https://portkey.ai/")
