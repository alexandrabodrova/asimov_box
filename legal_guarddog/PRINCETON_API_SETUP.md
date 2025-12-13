# Princeton LLM API Setup Guide

This guide explains how to configure Legal Guarddog to use Princeton's LLM API instead of OpenAI directly.

## Step 1: Get Princeton API Credentials

Contact Princeton's IT or research computing to get:
1. **API Key** - Your authentication token
2. **API Endpoint URL** - The base URL for API calls

Common Princeton endpoints might be:
- `https://api.princeton.edu/llm/v1`
- `https://openai.princeton.edu/v1`
- `https://research-llm.princeton.edu/v1`
- Or an Azure OpenAI endpoint if Princeton uses Azure

## Step 2: Set Environment Variable

Set your Princeton API key as an environment variable:

```bash
export PRINCETON_API_KEY='your-princeton-api-key-here'
```

To make it permanent, add to your `~/.bashrc` or `~/.zshrc`:

```bash
echo 'export PRINCETON_API_KEY="your-key-here"' >> ~/.bashrc
source ~/.bashrc
```

## Step 3: Configure API Endpoint

Edit `legal_guarddog/princeton_config.py` and update the `api_base` value:

```python
PRINCETON_CONFIG = {
    # Replace this with Princeton's actual endpoint
    "api_base": "https://api.princeton.edu/llm/v1",  # Update this!

    # ... rest stays the same
}
```

## Step 4: Verify Configuration

Test your configuration:

```bash
python legal_guarddog/princeton_config.py
```

You should see:
```
✓ Princeton API configuration valid:
  API Base: https://api.princeton.edu/llm/v1
  API Key: ******************** (set)
```

If you see errors, check that:
- `PRINCETON_API_KEY` environment variable is set
- `api_base` in `princeton_config.py` is updated (not the default "REPLACE_WITH_PRINCETON_ENDPOINT")

## Step 5: Run Tests

Once configured, all Legal Guarddog scripts will automatically use Princeton's API:

```bash
# Quick test with 5 prompts
python legal_guarddog/test_naive_baseline.py

# Full benchmark (all 4 tiers, 33 test cases)
python legal_guarddog/evaluation/benchmark.py
```

## Model Names

Princeton might use different model names than OpenAI. Common mappings:

| OpenAI Name | Princeton Equivalent | Update in |
|-------------|---------------------|-----------|
| `gpt-4` | `gpt-4` or `gpt-4-32k` | `princeton_config.py` models dict |
| `gpt-3.5-turbo` | `gpt-35-turbo` (Azure) | `princeton_config.py` models dict |

If Princeton uses different names, update the `models` dict in `princeton_config.py`:

```python
"models": {
    "gpt-4": "princeton-gpt-4",  # Use Princeton's actual name
    "gpt-3.5-turbo": "princeton-gpt-35-turbo",
}
```

## Troubleshooting

### Error: "No API key found"
**Solution:** Set the environment variable:
```bash
export PRINCETON_API_KEY='your-key'
```

### Error: "Princeton API endpoint not configured"
**Solution:** Update `api_base` in `princeton_config.py`

### Error: "Connection refused" or "404 Not Found"
**Solution:** Check that the `api_base` URL is correct. Contact Princeton IT for the right endpoint.

### Error: "Model not found"
**Solution:** Princeton might use different model names. Check their documentation and update the model names in your Config or in `princeton_config.py`.

## Alternative: Manual Configuration

If you prefer not to use `princeton_config.py`, you can configure directly in your code:

```python
from legal_guarddog.core.legal_guarddog_core import Config, LegalGuarddog

config = Config(
    api_key="your-princeton-key",
    api_base="https://api.princeton.edu/llm/v1",
    api_type="princeton",
    target_model="gpt-4",
    # ... other config options
)

guarddog = LegalGuarddog(config)
```

## Need Help?

1. **Princeton documentation**: Check Princeton's research computing docs for LLM API details
2. **IT Support**: Contact Princeton IT/Research Computing
3. **Test with OpenAI first**: If having issues, test with OpenAI API to verify the code works:
   ```bash
   export OPENAI_API_KEY='sk-...'
   python legal_guarddog/test_naive_baseline.py
   ```
