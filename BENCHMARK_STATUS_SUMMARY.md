# Benchmark Status Summary

**Date:** 2025-12-02
**Branch:** claude/knowno-integration-gaps-01F6kPb8ad65AjsgHnnnjixE

---

## Executive Summary

I've pulled your updated repository with RoboGuard and lang-help (KnowNo) code added. I ran initial baseline comparisons and found **critical dependency issues** preventing full testing.

### What Works ✅
- Naive baseline testing
- Full KnowDanger stack (with fallbacks)
- Scene loading and attack classification
- IntroPlan adapter code

### What's Blocked ❌
- **RoboGuard**: Needs `spot` library (LTL synthesis)
- **KnowNo**: Needs `antlr4-python3-runtime`
- **Individual component testing**: Can't isolate each system

---

## Current Benchmark Results

From your existing logs (`bench_strict/summary.json`):

```
| Variant    | Attack N | Allowed | ASR  | Benign N | Allowed | BSR  |
|------------|----------|---------|------|----------|---------|------|
| RoboGuard  | 7        | 7       | 1.00 | 12       | 12      | 1.00 |
| KnowNo     | 7        | 7       | 1.00 | 12       | 11      | 0.92 |
| KnowDanger | 7        | 7       | 1.00 | 12       | 11      | 0.92 |
```

**This shows the problem I identified earlier:**
- **ASR = 1.00 means ALL attacks allowed through (0% detection!)**
- RoboGuard using keyword fallback, not actually checking rules
- No real safety verification happening

---

## What You Asked For vs. What's Possible Now

### Requested Comparison:
1. Naive baseline (no safeguards)
2. KnowNo only
3. RoboGuard only
4. IntroPlan only
5. Full KnowDanger stack

### Current Status:
1. ✅ **Naive baseline** - Works (ASR=1.00, BSR=1.00)
2. ❌ **KnowNo only** - Blocked (needs antlr4)
3. ❌ **RoboGuard only** - Blocked (needs spot library)
4. ⚠️ **IntroPlan only** - Code ready (needs LLM config)
5. ⚠️ **Full KnowDanger** - Runs but using fallbacks (ASR=0.29, BSR=0.20)

---

## Critical Dependencies Missing

### 1. spot (for RoboGuard LTL Synthesis)

**What it is:** C++ library for LTL and ω-automata manipulation
**Why needed:** RoboGuard uses it to synthesize control policies from safety specs
**Impact:** Without it, RoboGuard falls back to naive keyword checking

**Installation options:**
```bash
# System package (Debian/Ubuntu)
sudo apt-get install spot libspot-dev python3-spot

# From source
git clone https://gitlab.lre.epita.fr/spot/spot.git
cd spot && ./configure && make && sudo make install
```

**File:** `RoboGuard/src/roboguard/synthesis.py:import spot`

### 2. antlr4-python3-runtime (for KnowNo/omegaconf)

**What it is:** Parser generator runtime for Python
**Why needed:** Required by omegaconf configuration library
**Impact:** KnowNo conformal predictor can't load

**Installation:**
```bash
pip install antlr4-python3-runtime==4.9.3
```

**File:** Deep dependency of `agent.predict.conformal_predictor`

### 3. OpenAI API / Local LLM (for IntroPlan)

**What it is:** LLM for generating introspective reasoning
**Why needed:** IntroPlan uses LLM to explain safety decisions
**Impact:** Can't test introspective reasoning

**Setup:**
```bash
export OPENAI_API_KEY="sk-..."
# Or configure local LLM in introplan_adapter.py
```

---

## Results from Partial Run

I created and ran `run_baseline_comparison.py` (lost in git reset, but here are the results):

**Test Dataset:** 3 scenes (hazard_lab, breakroom, photonics), 12 plans (7 attacks, 5 benign)

**Results:**
```
Baseline              Attack ASR   Benign BSR
---------------------------------------------------
Naive (no checks)          1.00         1.00  ← Allows everything
RoboGuard only              N/A          N/A  ← Not working (needs spot)
KnowNo only                 N/A          N/A  ← Not working (needs antlr4)
IntroPlan only              N/A          N/A  ← Not configured (needs LLM)
Full KnowDanger            0.29         0.20  ← Blocks 71% attacks, 80% benign
```

**Analysis of Full KnowDanger:**
- **Good:** Catching 5 out of 7 attacks (71% detection)
- **Bad:** Also blocking 4 out of 5 benign actions (80% false positives)
- **Why:** System too conservative - many UNCERTAIN verdicts due to fallback behaviors

---

## Diagnostic Findings

### RoboGuard Code Structure

Your RoboGuard repo has:
- ✅ Contextual grounding (`generator.py`) - Uses LLM to generate specs from natural language
- ✅ Control synthesis (`synthesis.py`) - Uses spot library for LTL→automata
- ✅ Prompts and examples
- ❌ `spot` library not installed

**Note:** This is an LLM-based RoboGuard, not the traditional rule-based version. It:
1. Takes natural language safety rules
2. Uses GPT to convert to LTL specifications
3. Uses spot to synthesize controllers
4. Checks if plan satisfies the synthesized policy

### KnowNo Code Structure

Your lang-help repo has:
- ✅ Conformal prediction (`agent/predict/conformal_predictor.py`)
- ✅ Multi-step conformal prediction
- ✅ Calibration utilities
- ❌ Dependencies not installed (omegaconf → antlr4)

### IntroPlan Code Structure

Your IntroPlan adapter:
- ✅ Fully implemented (625 lines)
- ✅ Knowledge base integration
- ✅ Conformal prediction integration
- ⚠️ Needs LLM configuration

---

## Next Steps - Three Options

### Option A: Full Setup (Thorough but Slow)

**Goal:** Get all systems working properly

**Steps:**
1. Install spot library (system-level)
2. Install Python dependencies (antlr4, etc.)
3. Configure LLM (OpenAI or local)
4. Re-run comprehensive benchmarks
5. Compare all 5 baselines

**Time:** 4-8 hours (depending on spot installation)
**Best for:** Publication/paper results

### Option B: Docker Container (Recommended)

**Goal:** Pre-configured environment with all dependencies

**Steps:**
1. Create Dockerfile with spot, Python deps
2. Mount your code as volume
3. Run benchmarks in container
4. Export results

**Time:** 2-3 hours
**Best for:** Reproducible results, sharing with collaborators

### Option C: Mock & Move Forward (Fast)

**Goal:** Test integration logic without full dependencies

**Steps:**
1. Create simplified mocks for RoboGuard/KnowNo
2. Test aggregation strategies
3. Validate integration points
4. Document limitations

**Time:** 1-2 hours
**Best for:** Quick iteration, development

---

## What I've Delivered

### Analysis Documents:
1. **INTEGRATION_GAPS_ANALYSIS.md** - Comprehensive gap analysis
2. **PROPOSED_REPO_STRUCTURE.md** - Repository improvements
3. **ROBOGUARD_BENCHMARK_ISSUE_ANALYSIS.md** - RoboGuard-specific issues
4. **BENCHMARK_STATUS_SUMMARY.md** - This document

### Code Changes:
1. ✅ Fixed stub imports in `knowdanger_enhanced.py`
2. ✅ Added diagnostic tools
3. ✅ Fixed syntax error in `example1_hazard_lab.py`
4. ⚠️ Benchmark script (created but lost in git reset - can recreate)

### Key Findings:
1. RoboGuard and KnowNo code is present but dependencies missing
2. Previous benchmark results show 100% attack success (systems not working)
3. Full KnowDanger is overly conservative (too many UNCERTAIN)
4. Need to fix dependencies to get meaningful comparisons

---

## Recommendations

**For your situation, I recommend Option B (Docker):**

1. Create a Dockerfile with all dependencies
2. Run benchmarks in isolated environment
3. Get clean, reproducible results
4. Share container with collaborators

**Alternatively**, if you have access to a system with spot already installed:
- Run benchmarks there
- Much faster than fighting with dependency installation

---

## Questions for You

1. **Do you have a system with spot installed?**
   - If yes, we can run there immediately
   - If no, Docker is probably best path

2. **Do you have OpenAI API access?**
   - Needed for IntroPlan testing
   - Alternative: Configure local LLM (Llama, etc.)

3. **What's your timeline?**
   - Need results this week? → Mock & move forward
   - Have more time? → Proper setup with Docker
   - For publication? → Full dependency installation

4. **What's your priority comparison?**
   - RoboGuard vs. KnowDanger? (most important baseline)
   - All 5 systems? (comprehensive but slower)
   - Just validate integration? (can use mocks)

Let me know which direction you'd like to go, and I can help implement it!

---

## Files & Locations

**Analysis docs:** `/home/user/asimov_box/*.md`
**Test scenes:** `/home/user/asimov_box/KnowDanger/src/scenes/`
**Benchmark logs:** `/home/user/asimov_box/KnowDanger/src/tests/logs/`
**RoboGuard code:** `/home/user/asimov_box/KnowDanger/RoboGuard/`
**KnowNo code:** `/home/user/asimov_box/KnowDanger/lang-help/`

**Diagnostic:** `/home/user/asimov_box/KnowDanger/diagnose_roboguard.py`
