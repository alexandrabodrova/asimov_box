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
from legalbreak import LegalGuarddog, RiskCategory
from legalbreak.core.legal_guarddog_core import Config
from legalbreak.tests.test_cases_comprehensive import get_test_cases

# Initialize with config
config = Config(
    target_model="gpt-4o",
    max_attack_turns=5,
    verbose=True
)
guarddog = LegalGuarddog(config)

# Get test cases
test_cases = get_test_cases()
first_test = test_cases[0]

# Run a single test
result = guarddog.run_adversarial_test(
    risk_category=first_test['category'],
    attack_goal=first_test['goal']
)

# Print results
print(f"\n{'='*60}")
print(f"Test: {first_test['goal']}")
print(f"Category: {first_test['category'].value}")
print(f"Attack Success: {'✗ JAILBROKEN' if result.success else '✓ BLOCKED'}")
print(f"Turns Required: {len(result.turns)}")
print(f"{'='*60}\n")
```

Run it:

```bash
python my_test.py
```

## Step 4: Run Full Benchmark (Optional)

```python
from legalbreak import LegalGuarddog
from legalbreak.core.legal_guarddog_core import Config
from legalbreak.tests.test_cases_comprehensive import get_test_cases

# Create config
config = Config(
    target_model="gpt-4o",
    max_attack_turns=5,
    use_validator=True,
    verbose=False
)

# Initialize guarddog
guarddog = LegalGuarddog(config)

# Get all 48 test cases
test_cases = get_test_cases()

# Run evaluation
results = guarddog.evaluate_system(test_cases)

print(f"Overall ASR: {results.attack_success_rate:.1%}")
print(f"Successful: {results.successful_attacks}/{results.total_attempts}")
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
