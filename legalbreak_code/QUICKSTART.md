# Quick Start Guide

Get up and running with LegalBreak in 5 minutes.

## Step 1: Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd legalbreak_code

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

## Step 2: Configure API Key

Create a `.env` file:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

## Step 3: Run Your First Test

### Option A: Use the example script

```bash
python example_usage.py
```

### Option B: Write your own

Create `my_test.py`:

```python
from legalbreak import LegalGuarddog
from legalbreak.tests.test_cases_comprehensive import get_test_cases

# Initialize
guarddog = LegalGuarddog(
    target_model="gpt-4o",
    max_turns=5
)

# Get test cases
test_cases = get_test_cases()

# Run a single test
result = guarddog.run_adversarial_test(
    test_case=test_cases[0],  # "elicit bomb-making instructions"
    verbose=True
)

# Print results
print(f"\n{'='*60}")
print(f"Test: {test_cases[0]['name']}")
print(f"Category: {test_cases[0]['category']}")
print(f"Attack Success: {'✗ JAILBROKEN' if result['success'] else '✓ BLOCKED'}")
print(f"Turns Required: {result['turns']}")
print(f"{'='*60}\n")
```

Run it:

```bash
python my_test.py
```

## Step 4: Run Full Benchmark (Optional)

```python
from legalbreak.evaluation.benchmark import run_full_benchmark

# Run all 48 tests
results = run_full_benchmark(
    baseline="legalbreak",  # or "naive"
    model="gpt-4o",
    output_dir="./my_results"
)

print(f"Overall ASR: {results['overall_asr']:.1f}%")
print(f"Successful: {results['successful_attacks']}/{results['total_attacks']}")
```

**Cost estimate**: ~$10-12 USD for full 48-prompt suite

## Understanding Results

### Attack Success
- **✓ BLOCKED**: Target LLM refused or judge detected violation
- **✗ JAILBROKEN**: Attack succeeded in eliciting violating content

### Turns
- Number of attacker refinement iterations (1-5)
- Lower = easier to jailbreak
- Higher = stronger defense

### Categories
- **dual_use**: Dangerous instructions (explosives, bioweapons, etc.)
- **copyright**: Verbatim reproduction, DRM circumvention
- **defamation**: False factual claims about individuals

## Next Steps

1. **Explore test cases**: See `legalbreak/tests/test_cases_comprehensive.py`
2. **Customize rules**: Modify `legalbreak/policies/legal_policy_engine.py`
3. **Add your own tests**: Create custom test cases
4. **Run experiments**: Compare different models or baselines

## Common Issues

### "No module named 'openai'"
**Fix:** `pip install openai`

### "API key not found"
**Fix:** Create `.env` file with your API key

### "Rate limit exceeded"
**Fix:** Add delays between tests or use API tier with higher limits

## Getting Help

- **Documentation**: See README.md
- **Code examples**: Check example_usage.py
- **Issues**: Report at [repository issues page]

---

Happy testing! 🎯
