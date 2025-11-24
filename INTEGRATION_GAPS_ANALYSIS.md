# Integration Gaps Analysis: KnowNo + IntroPlan + RoboGUARD

**Generated:** 2025-11-18
**Repository:** https://github.com/alexandrabodrova/asimov_box
**Branch:** claude/knowno-integration-gaps-01F6kPb8ad65AjsgHnnnjixE

---

## Executive Summary

This document identifies **critical gaps** in the integration of KnowNo, IntrospectivePlan (IntroPlan), and RoboGUARD within the Asimov Box repository, and proposes an improved repository structure.

### Key Findings

1. **RoboGUARD is MISSING** - Empty directory, no implementation
2. **Stub vs. Full Implementation** - Main orchestrator uses stubs instead of full adapters
3. **No End-to-End Integration Tests** - Systems not tested together
4. **Missing LLM Configuration** - IntroPlan requires LLM but has no default setup
5. **No Knowledge Base** - Required for IntroPlan but not provided
6. **Calibration Pipeline Incomplete** - KnowNo needs calibration data
7. **Poor Repository Structure** - Mixed research code, examples, and libraries

---

## Critical Gaps

### 1. RoboGUARD Implementation Gap ⚠️ CRITICAL

**Status:** **MISSING**

**Location:** `/home/user/asimov_box/KnowDanger/RoboGuard/`

**Issue:**
```bash
$ ls -la KnowDanger/RoboGuard/
total 8
drwxr-xr-x  2 root root 4096 Nov 18 00:58 .
drwxr-xr-x 10 root root 4096 Nov 18 00:58 ..
```

The RoboGuard directory is **completely empty**. This is a critical gap because:

- `roboguard_adapter.py` expects to import `roboguard` module
- All integration code assumes RoboGuard is available
- Documentation claims "full integration" but RG is missing
- The adapter will fail at runtime: `importlib.import_module("roboguard")` → ModuleNotFoundError

**Impact:** HIGH - System cannot function without RoboGuard

**Required Action:**
- Add RoboGuard implementation to `/RoboGuard/src/roboguard/`
- OR document that RoboGuard must be installed separately via pip
- OR provide fallback/mock RoboGuard for testing

---

### 2. Stub vs. Full Implementation Gap ⚠️ HIGH

**Status:** **INCOMPLETE**

**Location:** `src/knowdanger/core/knowdanger_enhanced.py:263-287`

**Issue:**

The main orchestrator (`knowdanger_enhanced.py`) contains **stub implementations** instead of using the full adapters:

```python
class IntroPlanAdapter:
    """
    Minimal stub - use full version from introplan_adapter.py

    In production, import as:
    from knowdanger.adapters.introplan_adapter import IntroPlanAdapter
    """
    def __init__(self, knowledge_base_path=None, use_conformal=True, retrieval_k=3):
        # Stub implementation
        pass

    def generate_introspective_reasoning(self, task, scene_context, candidate_actions, llm_func=None):
        # Stub - returns minimal reasoning
        return IntrospectiveReasoning(
            explanation="Introspection not fully implemented (using stub)",
            confidence_scores={action: score for action, score in candidate_actions},
            safety_assessment="Unknown",
            compliance_assessment="Unknown",
            should_ask_clarification=True
        )
```

**Full implementation exists at:** `src/knowdanger/adapters/introplan_adapter.py` (625 lines)

**Impact:** MEDIUM-HIGH - IntroPlan functionality is not actually working in the main orchestrator

**Why This Matters:**
- Users following quickstart guide will get stub behavior, not real introspection
- Verdicts will always be UNCERTAIN because stub always sets `should_ask_clarification=True`
- No actual LLM reasoning happens
- Knowledge base retrieval is not used

**Required Action:**
- Update `knowdanger_enhanced.py` to import full adapter:
  ```python
  from knowdanger.adapters.introplan_adapter import IntroPlanAdapter
  ```
- Remove stub implementations
- Add proper import handling with fallback

---

### 3. Missing LLM Integration ⚠️ HIGH

**Status:** **INCOMPLETE**

**Location:** `src/knowdanger/adapters/introplan_adapter.py:290-304`

**Issue:**

IntroPlan requires an LLM to generate introspective reasoning, but:

1. **No default LLM configured** - Users must provide `llm_func` parameter
2. **IntroPlan modules not guaranteed** - Dynamically loaded, may not exist
3. **Fallback is too simple** - Just picks highest-score action without real reasoning

```python
def generate_introspective_reasoning(self, ...):
    # Generate reasoning using LLM or IntroPlan module
    if llm_func is not None:
        reasoning_text = llm_func(prompt)
    elif "llm" in self.modules:
        # Use IntroPlan's LLM module
        llm_module = self.modules["llm"]
        if hasattr(llm_module, "query_llm"):
            reasoning_text = llm_module.query_llm(prompt)
        else:
            reasoning_text = self._fallback_reasoning(candidate_actions)  # ← Too simple
    else:
        reasoning_text = self._fallback_reasoning(candidate_actions)
```

**IntroPlan LLM module exists:** `IntroPlan/llm.py`

But it's not automatically configured:
```python
# IntroPlan/llm.py
import openai
# Requires OPENAI_API_KEY environment variable
```

**Impact:** HIGH - IntroPlan cannot provide meaningful introspective reasoning without LLM

**Required Action:**
- Document LLM requirements clearly (OpenAI API key, model selection)
- Provide example LLM configuration
- Add support for local LLMs (Llama, etc.) as mentioned in IntroPlan notebooks
- Consider providing better heuristic fallback

---

### 4. Missing Knowledge Base ⚠️ MEDIUM

**Status:** **INCOMPLETE**

**Location:** Knowledge base files referenced but not provided as defaults

**Issue:**

IntroPlan relies on knowledge base for retrieval, but:

1. **No default KB provided** - Users must build their own
2. **Example KBs in IntroPlan/data/** are domain-specific (mobile manipulation)
3. **No KB for lab safety, breakroom, photonics** (the example scenes)
4. **No KB construction guide** - Just the code in `construct_knowledge_entry()`

```python
# IntroPlan/data/ contents:
- mobile_manipulation.txt          # Mobile manipulation domain
- mobile_manipulation_knowledge.txt
- safe_mobile_knowledge.txt
- safe_mobile_test.txt

# But example scenes need:
- example1_hazard_lab.py    # Lab safety - NO KB
- example2_breakroom.py     # Breakroom - NO KB
- example3_photonics.py     # Photonics - NO KB
```

**Impact:** MEDIUM - IntroPlan works without KB but provides degraded reasoning

**Required Action:**
- Create starter knowledge bases for each example domain
- Provide KB construction tutorial
- Add automated KB building from human feedback
- Document KB format clearly (JSON vs TXT)

---

### 5. Incomplete Calibration Pipeline ⚠️ MEDIUM

**Status:** **INCOMPLETE**

**Location:** `src/scripts/calibration_knowno/`

**Issue:**

KnowNo requires calibration data to compute tau threshold, but:

1. **No sample calibration data** provided
2. **Calibration scripts exist** but require domain-specific data collection
3. **No clear pipeline** from data → calibration → deployment
4. **Fallback quantile** used if no calibration (may be suboptimal)

```python
# From lang_help/knowno/api.py:49-59
def calibrate(alpha, score_sets):
    """Compute calibration threshold"""
    # Try to use upstream CP class
    try:
        import CP  # External dependency
        cp = CP.ChoiceBaseline()
        return cp.calibrate(score_sets, alpha=alpha)
    except Exception:
        # FALLBACK: Just use quantile
        scores = [score for lst in score_sets for score in lst]
        return np.quantile(scores, 1-alpha)  # May be suboptimal
```

**Calibration scripts available:**
- `src/scripts/calibration_knowno/calibrate_knowno.py`
- `src/scripts/calibration_knowno/compute_calibration.py`
- `src/scripts/calibration_knowno/runtime.py`

But **no example calibration datasets** or **step-by-step guide**.

**Impact:** MEDIUM - System works with fallback but may have suboptimal confidence bounds

**Required Action:**
- Provide sample calibration datasets
- Create calibration tutorial
- Document when to recalibrate
- Add calibration validation metrics

---

### 6. No End-to-End Integration Tests ⚠️ MEDIUM

**Status:** **MISSING**

**Location:** Tests directory lacks three-way integration tests

**Issue:**

Available tests:
```
src/tests/
├── benchmark_knowno_roboguard.py     # Only RG + KN (two-way)
├── benchmark_true_baselines.py       # Baselines
├── demo_spot_roboguard.py           # Only RG
├── knowdanger_vs_baselines.py       # Comparisons
└── roboguard_paper_bench.py         # Only RG
```

**Missing:**
- ❌ Test for RG + KN + IP (three-way integration)
- ❌ Test for `knowdanger_enhanced.py` specifically
- ❌ Test for each aggregation strategy
- ❌ Test for plan rewriting/refinement
- ❌ Test for IntroPlan KB retrieval
- ❌ Integration test across all example scenes

**Impact:** MEDIUM - No confidence that three-way integration actually works

**Required Action:**
- Create `test_enhanced_integration.py` for three-way tests
- Add tests for each aggregation strategy
- Add tests for plan refinement
- Add CI/CD pipeline

---

### 7. IntroPlan Module Discovery Issues ⚠️ LOW-MEDIUM

**Status:** **FRAGILE**

**Location:** `src/knowdanger/adapters/introplan_adapter.py:95-136`

**Issue:**

The adapter tries to find IntroPlan modules dynamically:

```python
def _find_introplan_root(self, provided_path: Optional[str] = None) -> Optional[Path]:
    """Find IntroPlan repository root directory"""
    # Check common locations
    candidates = [
        Path.cwd() / "IntroPlan",
        Path.cwd().parent / "IntroPlan",
        Path(__file__).parent.parent / "IntroPlan",
    ]
```

**Problems:**
- Depends on directory structure
- Fails if run from different working directory
- No clear error message if IntroPlan not found
- `sys.path` manipulation can cause conflicts

**Impact:** LOW-MEDIUM - Works in expected setup, fails unpredictably otherwise

**Required Action:**
- Use package-relative imports
- Make IntroPlan a proper Python package
- Add clear error messages
- Document IntroPlan installation

---

### 8. Documentation vs. Reality Gap ⚠️ LOW

**Status:** **MISLEADING**

**Issue:**

Documentation claims in `CODEBASE_ANALYSIS.md:555-567`:

```markdown
### What's Integrated ✓

1. **RoboGuard + KnowNo** (Original):
   - Both systems fully functional  ✓

2. **RoboGuard + KnowNo + IntroPlan** (Enhanced):
   - Full three-way integration  ✗ (Stubs used)
   - New `knowdanger_enhanced.py` orchestrator  ✓
   - IntroPlanAdapter with KB support  ✓ (but not used in enhanced)
   - Three aggregation strategies  ✓
   - Iterative plan refinement  ✓
```

**Reality:**
- RoboGuard: MISSING (empty directory)
- KnowNo: WORKS (with fallback calibration)
- IntroPlan: PARTIAL (adapter exists, but enhanced.py uses stubs)
- Three-way integration: NOT TESTED

**Impact:** LOW - Confuses users about system capabilities

**Required Action:**
- Update documentation to reflect actual status
- Add "Known Limitations" section
- Create setup checklist
- Add status badges (implemented/partial/missing)

---

## Integration Completeness Matrix

| Component | Implementation | Integration | Testing | Documentation | Overall |
|-----------|---------------|-------------|---------|---------------|---------|
| **RoboGuard** | ❌ MISSING | ⚠️ Adapter exists | ❌ No tests | ⚠️ Assumed present | 🔴 **0%** |
| **KnowNo** | ✅ Complete | ✅ Working | ⚠️ Partial | ✅ Good | 🟢 **85%** |
| **IntroPlan** | ✅ Complete | ⚠️ Stub used | ❌ No tests | ⚠️ Partial | 🟡 **60%** |
| **Enhanced Orchestrator** | ⚠️ Uses stubs | ⚠️ Partial | ❌ No tests | ✅ Good | 🟡 **50%** |
| **Knowledge Base** | ✅ Code exists | ⚠️ No defaults | ❌ No tests | ⚠️ Minimal | 🟡 **40%** |
| **Calibration** | ✅ Code exists | ⚠️ Fallback used | ❌ No tests | ⚠️ Minimal | 🟡 **50%** |
| **End-to-End** | ⚠️ Partial | ⚠️ Untested | ❌ MISSING | ⚠️ Aspirational | 🔴 **30%** |

**Legend:**
- ✅ Complete / Working
- ⚠️ Partial / Issues
- ❌ Missing / Not Working
- 🟢 >75% | 🟡 50-75% | 🔴 <50%

---

## Required Fixes (Priority Order)

### Priority 1: Critical Gaps

1. **Fix RoboGuard**
   - Add RoboGuard implementation OR
   - Document external installation OR
   - Provide mock for testing

2. **Fix Stub Usage in Enhanced Orchestrator**
   - Replace stubs with real adapters
   - Proper import handling
   - Fallback behavior for missing components

3. **Add End-to-End Tests**
   - Test three-way integration
   - Test all example scenes
   - Test aggregation strategies

### Priority 2: High-Impact Gaps

4. **Configure LLM Integration**
   - Default LLM setup guide
   - Support for multiple LLM backends
   - Better fallback heuristics

5. **Provide Starter Knowledge Bases**
   - KB for each example scene
   - KB construction tutorial
   - Automated KB building

6. **Calibration Pipeline**
   - Sample calibration datasets
   - Step-by-step calibration guide
   - Validation metrics

### Priority 3: Quality Improvements

7. **Fix IntroPlan Module Discovery**
   - Proper package structure
   - Clear error messages
   - Installation documentation

8. **Update Documentation**
   - Reflect actual status
   - Add known limitations
   - Setup checklist

---

## Recommended Actions Checklist

### Immediate (Week 1)

- [ ] **Add RoboGuard** - Get working implementation
- [ ] **Fix knowdanger_enhanced.py** - Use real adapters, not stubs
- [ ] **Create one end-to-end test** - Verify basic three-way integration
- [ ] **Document LLM requirements** - Clear setup guide

### Short-term (Weeks 2-3)

- [ ] **Create starter KBs** - One for each example scene
- [ ] **Add calibration guide** - With sample data
- [ ] **Write integration tests** - Cover all aggregation strategies
- [ ] **Update documentation** - Reflect actual capabilities

### Medium-term (Month 2)

- [ ] **Package IntroPlan properly** - Make it pip-installable
- [ ] **Add CI/CD pipeline** - Automated testing
- [ ] **Create tutorials** - End-to-end usage examples
- [ ] **Performance benchmarks** - Validate three-way approach

---

## Gap Summary

**Total Identified Gaps:** 8 critical issues

**Breakdown:**
- 🔴 Critical: 2 (RoboGuard missing, Stubs used)
- 🟡 High: 3 (LLM config, KB missing, Calibration incomplete)
- 🟠 Medium: 2 (No integration tests, Module discovery)
- 🔵 Low: 1 (Documentation mismatch)

**System Readiness:**
- **Research/Demo:** 60% ready (can show concept)
- **Development:** 40% ready (needs significant work)
- **Production:** 20% ready (too many gaps)

---

## Next Steps

1. Review this gap analysis with team
2. Prioritize gaps based on use case (research vs. production)
3. Create implementation plan
4. Consider repository restructuring (see next section)
