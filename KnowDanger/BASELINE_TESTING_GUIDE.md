# Baseline Testing Guide

This guide explains how to run comprehensive baseline comparisons for your safety systems.

## Prerequisites

### 1. Spot Library (Already Installed ✓)
You have spot 2.14.1 installed in your virtual environment.

### 2. OpenAI API Key (You Have This ✓)
You have API access but don't want to commit it to the repo.

### 3. Python Dependencies
```bash
pip install openai tiktoken numpy
```

## Quick Start

### Step 1: Activate Your Virtual Environment

```bash
# Activate the environment where spot is installed
source /path/to/your/venv/bin/activate

# Verify spot is available
python -c "import spot; print(f'Spot version: {spot.version()}')"
```

### Step 2: Set OpenAI API Key

**Important:** Set as environment variable, NOT in code:

```bash
export OPENAI_API_KEY="sk-your-actual-key-here"
```

To avoid typing this every time, add it to your shell profile:

```bash
# Add to ~/.bashrc or ~/.zshrc
echo 'export OPENAI_API_KEY="sk-your-key-here"' >> ~/.bashrc
source ~/.bashrc
```

Or use a `.env` file (not tracked by git):

```bash
# Create .env file (already in .gitignore)
echo 'OPENAI_API_KEY=sk-your-key-here' > .env

# Load it
export $(cat .env | xargs)
```

### Step 3: Run the Tests

```bash
cd /path/to/asimov_box/KnowDanger

# Test all baselines
python test_all_baselines.py --all

# Or test specific baselines
python test_all_baselines.py --roboguard --full

# Verbose mode for debugging
python test_all_baselines.py --all --verbose
```

## Usage Options

### Test Specific Baselines

```bash
# Just naive baseline
python test_all_baselines.py --naive

# Just RoboGuard
python test_all_baselines.py --roboguard

# Just KnowNo
python test_all_baselines.py --knowno

# Just IntroPlan
python test_all_baselines.py --introplan

# Just full KnowDanger
python test_all_baselines.py --full

# Combination
python test_all_baselines.py --roboguard --knowno --full
```

### Test Specific Scenes

```bash
# Test only lab scene
python test_all_baselines.py --all --scenes example1_hazard_lab

# Test multiple specific scenes
python test_all_baselines.py --all --scenes example1_hazard_lab example2_breakroom
```

### Change Output Location

```bash
# Save to different directory
python test_all_baselines.py --all --output my_results

# Results will be in: my_results/baseline_results_TIMESTAMP.json
```

### Verbose Output

```bash
# See detailed output for debugging
python test_all_baselines.py --all --verbose
```

## What the Script Tests

### 1. Naive Baseline
- **What it does:** No safety checks at all
- **Expected:** ASR=1.00 (allows all attacks), BSR=1.00 (allows all benign)
- **Purpose:** Worst-case baseline

### 2. RoboGuard Only
- **What it does:** LLM-generated LTL specs + spot synthesis
- **Expected:** Should catch rule violations
- **Requires:** spot library + OpenAI API key
- **How it works:**
  1. Takes natural language safety rules
  2. Uses GPT to convert to LTL specifications
  3. Uses spot to synthesize safety controller
  4. Checks if plan satisfies specifications

### 3. KnowNo Only
- **What it does:** Conformal prediction on action choices
- **Expected:** UNCERTAIN when multiple possible actions
- **Note:** Models action-choice uncertainty, not safety directly
- **How it works:**
  1. Looks at candidate actions and their scores
  2. Uses conformal prediction sets
  3. SAFE if single confident choice
  4. UNCERTAIN if multiple possibilities

### 4. IntroPlan Only
- **What it does:** Introspective reasoning with LLM
- **Expected:** Provides explanations for safety decisions
- **Requires:** OpenAI API key
- **How it works:**
  1. Retrieves relevant knowledge from KB (if provided)
  2. Uses LLM to generate introspective reasoning
  3. Assesses safety and compliance
  4. Decides whether to ask for clarification

### 5. Full KnowDanger
- **What it does:** RoboGuard + KnowNo integration
- **Expected:** Best performance (low ASR, high BSR)
- **How it works:**
  1. RoboGuard checks rule violations
  2. KnowNo assesses action choice confidence
  3. Aggregates verdicts (conservative/majority/weighted)
  4. Returns SAFE/UNSAFE/UNCERTAIN

## Understanding the Results

### Output Format

The script prints a summary table:

```
Baseline              Attack ASR   Benign BSR        Status
-----------------------------------------------------------
naive                       1.00         1.00             ⚠️
roboguard                   0.14         0.92             ✓
knowno                      0.71         0.75             ⚠️
introplan                   0.29         0.83             ⚠️
full_knowdanger             0.14         0.92             ✓
```

### Metrics Explained

**ASR (Attack Success Rate):**
- Percentage of attack plans that were allowed through
- **Lower is better** (you want to block attacks)
- Good: < 0.30 (blocking 70%+ of attacks)
- Excellent: < 0.15 (blocking 85%+ of attacks)

**BSR (Benign Success Rate):**
- Percentage of safe plans that were allowed through
- **Higher is better** (you want to allow safe actions)
- Good: > 0.70 (allowing 70%+ of benign actions)
- Excellent: > 0.85 (allowing 85%+ of benign actions)

### What Makes a Good System?

**Ideal system:**
- ASR < 0.30 (blocks most attacks)
- BSR > 0.70 (allows most benign actions)
- Balance between safety and usability

**Example comparisons:**

```
System A: ASR=0.10, BSR=0.95  ← Excellent! Secure and usable
System B: ASR=0.20, BSR=0.85  ← Good balance
System C: ASR=0.50, BSR=0.95  ← Too permissive (allows attacks)
System D: ASR=0.10, BSR=0.30  ← Too restrictive (blocks everything)
```

### JSON Output

Results are saved to `logs/baseline_test/baseline_results_TIMESTAMP.json`:

```json
{
  "timestamp": "20251202_120000",
  "test_data": {
    "scenes": 3,
    "total_plans": 12,
    "attack_plans": 7,
    "benign_plans": 5
  },
  "results": {
    "naive": {
      "attack": {"N": 7, "allowed": 7, "blocked": 0, "uncertain": 0},
      "benign": {"N": 5, "allowed": 5, "blocked": 0, "uncertain": 0},
      "details": [...]
    },
    "roboguard": {...},
    "full_knowdanger": {...}
  }
}
```

## Troubleshooting

### "spot module not found"

```bash
# Verify spot installation
python -c "import spot; print(spot.version())"

# If not found, check your virtual environment
which python
# Should point to your venv

# Reinstall if needed
pip install spot
```

### "OpenAI API key not set"

```bash
# Check if set
echo $OPENAI_API_KEY

# Set it
export OPENAI_API_KEY="sk-..."

# Verify
python -c "import os; print('Key set:', 'OPENAI_API_KEY' in os.environ)"
```

### "ConformalPredictor not found"

```bash
# Check PYTHONPATH includes lang-help
export PYTHONPATH=/path/to/KnowDanger/lang-help:$PYTHONPATH

# Or install dependencies
pip install omegaconf antlr4-python3-runtime
```

### RoboGuard taking too long

RoboGuard uses GPT for spec generation and spot for synthesis, which can be slow:

```bash
# Test with fewer scenes first
python test_all_baselines.py --roboguard --scenes example1_hazard_lab

# Or skip RoboGuard and test others
python test_all_baselines.py --knowno --introplan --full
```

### IntroPlan rate limits

If you hit OpenAI rate limits:

```bash
# Test IntroPlan on subset
python test_all_baselines.py --introplan --scenes example1_hazard_lab

# Or use cheaper model (edit script to use gpt-3.5-turbo)
```

## Expected Runtime

Approximate times (will vary based on OpenAI API speed):

- **Naive:** < 1 second
- **RoboGuard:** 2-5 minutes (GPT calls + spot synthesis)
- **KnowNo:** < 5 seconds (local computation)
- **IntroPlan:** 1-3 minutes (GPT calls for reasoning)
- **Full KnowDanger:** < 10 seconds (uses cached/fallback components)

**Total for all baselines:** 5-10 minutes

## Tips for Best Results

### 1. Run Multiple Times

Random variation in LLM responses can affect results:

```bash
# Run 3 times and average
for i in {1..3}; do
    python test_all_baselines.py --all --output run_$i
done
```

### 2. Start with Quick Test

Test on one scene first to verify everything works:

```bash
python test_all_baselines.py --all --scenes example1_hazard_lab --verbose
```

### 3. Compare Against Previous Results

If you have old benchmark results, compare:

```bash
# Your old results
cat logs/bench_strict/summary.json

# New results
cat logs/baseline_test/baseline_results_*.json

# Should see improvement with real RoboGuard vs. fallback
```

### 4. Save Configurations

Document your setup in results:

```bash
# Add to results file
{
  "environment": {
    "spot_version": "2.14.1",
    "openai_model": "gpt-3.5-turbo",
    "python_version": "3.11",
    "date": "2025-12-02"
  }
}
```

## Next Steps After Testing

### If RoboGuard performs too well:
- Make test cases harder (see ROBOGUARD_BENCHMARK_ISSUE_ANALYSIS.md)
- Add more subtle attacks
- Increase complexity of safety rules

### If RoboGuard performs too poorly:
- Check that spot synthesis succeeded
- Verify LTL specs are correct
- Debug with `--verbose` flag

### If results look good:
- Run on larger test set
- Add more diverse scenarios
- Document for paper/publication

## Support

If you encounter issues:

1. Run with `--verbose` to see detailed error messages
2. Check that all dependencies are installed
3. Verify API key is set correctly
4. Test components individually first

For specific errors, check the generated JSON file in `logs/baseline_test/` for detailed error information in the `details` field.

## File Locations

- **Script:** `/path/to/KnowDanger/test_all_baselines.py`
- **Results:** `KnowDanger/logs/baseline_test/baseline_results_*.json`
- **Scenes:** `KnowDanger/src/scenes/example*.py`
- **This guide:** `KnowDanger/BASELINE_TESTING_GUIDE.md`
