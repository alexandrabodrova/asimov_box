# Legal Guarddog Benchmark Results

**Date:** 2025-11-24T17:43:45.040469

**Test Cases:** 24 adversarial, 9 benign

## Summary Table

| Configuration | Target Model | ASR ↓ | Refusal Rate | Rewrite Rate | FP Rate ↓ |
|--------------|--------------|-------|--------------|--------------|----------|
| 1_naive_baseline | gpt-3.5-turbo | 100.0% | 0.0% | 0.0% | 0.0% |
| 2_adversarial_baseline | gpt-3.5-turbo | 100.0% | 0.0% | 0.0% | 0.0% |
| 3_generic_safety | gpt-3.5-turbo | 0.0% | 0.0% | 0.0% | 0.0% |
| 4_full_legal_guarddog | gpt-3.5-turbo | 0.0% | 0.0% | 0.0% | 0.0% |

## By Category

### 1_naive_baseline

*Naive: Direct prompts → GPT (tests built-in guardrails)*

**Target: gpt-3.5-turbo**

| Category | Total | Successful | ASR |
|----------|-------|------------|-----|

### 2_adversarial_baseline

*Adversarial: RoboPAIR → GPT (tests sophisticated jailbreaks)*

**Target: gpt-3.5-turbo**

| Category | Total | Successful | ASR |
|----------|-------|------------|-----|

### 3_generic_safety

*Generic Safety: RoboPAIR → GPT → Generic Validator*

**Target: gpt-3.5-turbo**

| Category | Total | Successful | ASR |
|----------|-------|------------|-----|

### 4_full_legal_guarddog

*Full System: RoboPAIR → GPT → Legal Judge → Legal Validator*

**Target: gpt-3.5-turbo**

| Category | Total | Successful | ASR |
|----------|-------|------------|-----|

## Key Findings

- **Lower ASR is better** (fewer successful attacks)
- **Lower FP rate is better** (fewer false alarms on benign requests)
- **Balanced scoring** considers both safety and utility

