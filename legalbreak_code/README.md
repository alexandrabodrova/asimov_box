# LegalBreak: Law-Aware Adversarial Testing for LLMs

A systematic framework for measuring LLM vulnerabilities to legal compliance violations across dual-use content, copyright infringement, and defamation.

## Overview

LegalBreak adapts the RoboPAIR multi-agent adversarial testing methodology to evaluate LLM legal compliance. The system uses an attacker LLM that iteratively refines jailbreak prompts based on feedback from a legal policy judge, enabling systematic discovery of vulnerabilities in text-based legal compliance.

### Key Features

- **9 Doctrine-Grounded Rules**: Legal policy engine implementing rules across 3 categories
  - **Dual-use (4 rules)**: Explosives, chemical weapons, bioweapons, cyber-physical attacks
  - **Copyright (2 rules)**: Verbatim reproduction, DRM circumvention
  - **Defamation (3 rules)**: Unsubstantiated allegations, fake news, fabricated facts
- **Multi-Agent Architecture**: Attacker LLM → Target LLM → Judge LLM with iterative refinement
- **48 Adversarial Test Prompts**: Comprehensive test suite operationalizing U.S. legal doctrine
- **Hybrid Judge**: 4-stage pipeline combining rule-based + LLM semantic reasoning

## Installation

### Prerequisites

- Python 3.9+
- OpenAI API key

### Install from source

```bash
git clone <repository-url>
cd legalbreak_code
pip install -r requirements.txt
```

### Install as package

```bash
pip install -e .
```

## Quick Start

### Basic Usage

```python
from legalbreak.core.legal_guarddog_core import LegalGuarddog
from legalbreak.tests.test_cases_comprehensive import get_test_cases

# Initialize system
guarddog = LegalGuarddog(
    target_model="gpt-4o",
    attacker_model="gpt-4o",
    judge_model="gpt-4o",
    max_turns=5
)

# Load test cases
test_cases = get_test_cases()

# Run adversarial testing
results = guarddog.run_adversarial_test(
    test_case=test_cases[0],  # First test prompt
    verbose=True
)

print(f"Attack Success: {results['success']}")
print(f"Turns Required: {results['turns']}")
print(f"Category: {results['category']}")
```

### Run Full Benchmark

```python
from legalbreak.evaluation.benchmark import run_full_benchmark

# Run complete 48-prompt adversarial test suite
results = run_full_benchmark(
    baseline="legalbreak",  # or "naive"
    model="gpt-4o",
    output_dir="./results"
)

# Results include:
# - Overall ASR (attack success rate)
# - Category-specific ASR
# - Average turns per attack
# - Detailed logs
```

## Architecture

### Multi-Agent Testing Loop

```
┌─────────────┐
│ Attacker LLM│ (Generates/refines jailbreak prompts)
└──────┬──────┘
       │ prompt
       ▼
┌─────────────┐
│ Target LLM  │ (GPT-4o under test)
└──────┬──────┘
       │ response
       ▼
┌─────────────┐
│  Judge LLM  │ (4-stage legal policy evaluation)
└──────┬──────┘
       │ violation?
       │
       ├─ YES → Attack successful, terminate
       │
       └─ NO  → Feedback to attacker, iterate (max 5 turns)
```

### Judge Pipeline (4 Stages)

1. **Refusal Detection**: Pattern matching against 30+ refusal phrases
2. **Rule-Based Policy Engine**: 9 deterministic legal compliance rules
3. **Named Entity Recognition**: Person identification for defamation checks
4. **LLM Semantic Reasoning**: Context-dependent analysis for edge cases

## Configuration

### Environment Variables

Create a `.env` file:

```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_ORG_ID=your_org_id  # Optional
```

### Custom Configuration

```python
from legalbreak.core.legal_guarddog_core import LegalGuarddog

guarddog = LegalGuarddog(
    target_model="gpt-4o",
    attacker_model="gpt-4o",
    judge_model="gpt-4o",
    max_turns=5,
    attacker_temperature=0.7,  # Creative jailbreak generation
    judge_temperature=0.0,     # Deterministic evaluation
    api_key="your_key"
)
```

## Test Suite

The 48-prompt adversarial test suite covers:

### Dual-Use Content (17 prompts)
- Explosives manufacturing (bombs, IEDs, incendiary devices)
- Chemical weapons (sarin, ricin, nerve agents)
- Biological weapons (anthrax, weaponized pathogens)
- Cyber-physical attacks (power grid, SCADA, infrastructure)

### Copyright Infringement (15 prompts)
- Verbatim text reproduction (books, articles, textbooks)
- Song lyrics reproduction (contemporary artists)
- Script/screenplay excerpts (movies, TV shows)
- DRM circumvention instructions

### Defamation (16 prompts)
- False factual claims about public figures
- Unsubstantiated allegations (criminal conduct, fraud)
- Fabricated scandals and controversies

## Legal Policy Engine

### Rule Categories

**Dual-Use (4 rules):**
1. `explosives_instructions` - 18 U.S.C. §842 violations
2. `toxic_chemical_synthesis` - Chemical weapons synthesis
3. `biological_weapon_creation` - 18 U.S.C. §2332a violations
4. `cyber_physical_attacks` - Critical infrastructure attacks

**Copyright (2 rules):**
5. `verbatim_reproduction` - 17 U.S.C. §106 violations
6. `drm_circumvention` - 17 U.S.C. §1201 violations

**Defamation (3 rules):**
7. `unsubstantiated_allegations` - State tort law violations
8. `fake_news_defamation` - Defamation by implication
9. `fabricated_facts` - False factual assertions (Volokh-style)

### Customizing Rules

```python
from legalbreak.policies.legal_policy_engine import LegalPolicyEngine, PolicyRule, RiskCategory

engine = LegalPolicyEngine()

# Add custom rule
custom_rule = PolicyRule(
    name="custom_violation",
    category=RiskCategory.DUAL_USE,
    description="Custom legal compliance check",
    check_function=my_check_function,
    severity=ViolationSeverity.HIGH
)

engine.add_rule(custom_rule)
```

## Evaluation Metrics

### Primary Metric: Attack Success Rate (ASR)

```
ASR = (Number of successful attacks) / (Total attacks) × 100%
```

Lower ASR = stronger defense

### Secondary Metrics

- **Category-Specific ASR**: Breakdown by dual-use/copyright/defamation
- **Attacker Cost**: Average turns required for successful attacks
- **Violation Confidence**: Rule-based confidence scores (0.0-1.0)

## Experimental Results

Results from testing GPT-4o (48-prompt suite):

| Baseline | Overall ASR | Dual-Use ASR | Copyright ASR | Defamation ASR |
|----------|-------------|--------------|---------------|----------------|
| **Naive** | 27.1% (13/48) | 11.8% (2/17) | 0.0% (0/15) | 68.8% (11/16) |
| **LegalBreak** | 54.2% (26/48) | 70.6% (12/17) | 13.3% (2/15) | 75.0% (12/16) |

**Key Finding**: Multi-turn adversarial refinement increases ASR by +27.1pp, with dramatic dual-use increase (+58.8pp).

### Average Attack Turns

- **Dual-use**: 3.42 turns (n=12 successful attacks)
- **Copyright**: 4.00 turns (n=2 successful attacks)
- **Defamation**: 1.08 turns (n=12 successful attacks)

**Insight**: Defamation attacks succeed almost immediately (1.08 turns average), indicating weak baseline guardrails.

## Code Structure

```
legalbreak_code/
├── legalbreak/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── legal_guarddog_core.py      # Main orchestrator
│   ├── policies/
│   │   ├── __init__.py
│   │   └── legal_policy_engine.py      # 9 legal rules
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── benchmark.py                # Full test suite runner
│   └── tests/
│       └── test_cases_comprehensive.py # 48 adversarial prompts
├── example_usage.py                    # Usage examples
├── requirements.txt                    # Dependencies
├── setup.py                            # Package installation
├── README.md                           # This file
└── .env.example                        # Environment template
```

## API Cost Estimates

Based on GPT-4o pricing (as of 2024):

- **Single test**: ~$0.20-0.30 USD (5 turns × 3 LLM calls)
- **Full 48-prompt suite**: ~$10-12 USD per run
- **Complete study (Naive + LegalBreak)**: ~$20-25 USD

## Limitations

1. **Rule-Based Brittleness**: 9 manually-constructed rules vulnerable to sophisticated adversarial attacks using euphemisms or multi-step decomposition
2. **Threshold Arbitrariness**: Numeric thresholds (e.g., 50-word verbatim for copyright) lack firm grounding in case law
3. **U.S.-Centric**: Focuses on U.S. law (17 U.S.C., 18 U.S.C., state tort law)
4. **Defamation Verification**: Cannot verify factual accuracy without external knowledge bases
5. **Manual Validation Required**: Automated labeling has systematic biases; 30-40 hours manual review needed for accurate results

## Citation

```bibtex
@article{bodrova2024legalbreak,
  title={LegalBreak: Law-Aware Adversarial Testing for LLM Legal Compliance},
  author={Bodrova, Alexandra},
  journal={Princeton University SPI 352},
  year={2024}
}
```

## Related Work

- **RoboPAIR** (Robey et al. 2024): Multi-agent adversarial testing for LLM-controlled robots
- **JailbreakBench** (Chao et al. 2024): Standardized adversarial testing benchmarks
- **PAIR** (Zou et al. 2023): Prompt automatic iterative refinement algorithm

## License

[Specify license - e.g., MIT, Apache 2.0]

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Contact

- **Author**: Alexandra Bodrova
- **Institution**: Princeton University
- **Project**: SPI 352 - AI & Law

---

**Disclaimer**: This tool is for research and educational purposes only. Use responsibly and in compliance with applicable laws and regulations.
