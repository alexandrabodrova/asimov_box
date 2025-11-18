# Migration Guide: From Current KnowDanger to Enhanced Version

This guide helps you migrate your existing KnowDanger code to use the new enhanced version with full RoboGuard, KnowNo, and IntroPlan integration.

## Option 1: Side-by-Side (Recommended)

Keep both versions and gradually migrate. This is the **safest approach** for ongoing research.

### Setup

```bash
# Your current structure
asimov_box/
├── src/knowdanger/core/
│   ├── knowdanger_core.py          # Keep existing
│   └── knowdanger_enhanced.py      # Add new ← NEW FILE
├── src/knowdanger/adapters/
│   ├── roboguard_adapter.py        # Keep existing
│   └── introplan_adapter.py        # Add new ← NEW FILE
└── src/knowdanger/utils/
    └── integration_utils.py         # Add new ← NEW FILE
```

### Usage

```python
# Old code continues to work
from knowdanger.core.knowdanger_core import KnowDanger
kd_old = KnowDanger(config)

# New code uses enhanced version
from knowdanger.core.knowdanger_enhanced import EnhancedKnowDanger
kd_new = EnhancedKnowDanger(config)
```

## Option 2: Direct Replacement

Replace your existing `knowdanger_core.py` with enhanced version.

### Backup First!

```bash
cd src/knowdanger/core/
cp knowdanger_core.py knowdanger_core_backup.py
```

### Update Imports

The enhanced version is **backwards compatible**. Just update class name:

```python
# OLD
from knowdanger.core.knowdanger_core import KnowDanger
kd = KnowDanger(config)

# NEW
from knowdanger.core.knowdanger_core import EnhancedKnowDanger as KnowDanger
kd = KnowDanger(config)
```

Or globally replace:

```bash
# Find all files using KnowDanger
grep -r "from knowdanger.core.knowdanger_core import KnowDanger" .

# Replace in all Python files
find . -name "*.py" -exec sed -i 's/from knowdanger.core.knowdanger_core import KnowDanger/from knowdanger.core.knowdanger_enhanced import EnhancedKnowDanger as KnowDanger/g' {} +
```

## Migration Path by Feature

### 1. Basic Evaluation (No Changes Needed)

Your existing code works as-is:

```python
# This code works with BOTH versions
scene = Scene(name="lab", semantic_graph={...}, rules=[...])
plan = PlanCandidate(name="test", steps=[...], user_prompt="...")
assessment = kd.run(scene, plan)
print(assessment.overall.label)
```

**Action**: ✅ No changes needed

### 2. Adding IntroPlan

To enable introspection:

```python
# OLD: Only RG + KN
config = Config(alpha=0.1, ask_threshold_confidence=0.7)

# NEW: Add IntroPlan
config = Config(
    alpha=0.1,
    ask_threshold_confidence=0.7,
    use_introspection=True,  # ← NEW
    introplan_kb_path="path/to/kb.json"  # ← NEW
)
```

**Action**: 
1. Add `use_introspection=True` to config
2. Provide knowledge base path
3. Optionally set `introplan_retrieval_k=3`

### 3. Accessing IntroPlan Results

New information available in assessments:

```python
assessment = kd.run(scene, plan)

# OLD: Only RG and KN verdicts
for step_assess in assessment.steps:
    print(step_assess.roboguard.label)
    print(step_assess.knowno.label)

# NEW: Also IntroPlan verdict
for step_assess in assessment.steps:
    print(step_assess.roboguard.label)
    print(step_assess.knowno.label)
    print(step_assess.introplan.label)  # ← NEW
    
    # Access introspective reasoning
    if step_assess.introplan:
        details = step_assess.introplan.details
        print(f"Reasoning: {details.get('reasoning', 'N/A')}")
        print(f"Safety: {details.get('safety', 'N/A')}")
```

**Action**: Update code that processes assessments to handle `introplan` field

### 4. Using Plan Refinement

New feature for iterative improvement:

```python
# OLD: Single evaluation
assessment = kd.run(scene, plan)

# NEW: With refinement
assessment = kd.run_with_rewriting(
    scene, 
    plan, 
    max_iterations=3  # Try up to 3 refinements
)

# Check if refinement occurred
if "refinement_iterations" in assessment.meta:
    print(f"Refined {assessment.meta['refinement_iterations']} times")
```

**Action**: Use `run_with_rewriting()` for plans that might benefit from refinement

### 5. Changing Aggregation Strategy

New aggregation options:

```python
# OLD: Fixed conservative aggregation
config = Config(alpha=0.1)

# NEW: Choose strategy
config = Config(
    alpha=0.1,
    aggregation_strategy="conservative"  # or "majority" or "weighted"
)

# Or custom
def my_aggregator(rg, kn, ip):
    # Your logic
    return Verdict("SAFE", "custom logic")

config.aggregator = my_aggregator
```

**Action**: Choose aggregation strategy based on your safety requirements

### 6. Adding Logging

Use the new logging utilities:

```python
# OLD: Manual logging
for plan in plans:
    assessment = kd.run(scene, plan)
    # Manual CSV writing...

# NEW: Automatic logging
from integration_utils import evaluate_with_logging

assessments, metrics = evaluate_with_logging(
    kd, scene, plans, log_dir="logs/experiment_1"
)
# Logs automatically saved to:
# - logs/experiment_1/step_log.csv
# - logs/experiment_1/plan_log.csv
# - logs/experiment_1/metrics.json
```

**Action**: Replace manual logging with `evaluate_with_logging()`

### 7. Knowledge Base Management

New KB features:

```python
# Add entries from human feedback
kd.add_knowledge_entry(
    task="move chemical safely",
    scene_context="lab with hot plate",
    correct_option="place_in_fume_hood",
    human_feedback="Fume hood provides ventilation and isolation"
)

# Save KB
kd.save_knowledge_base("knowledge_base.json")

# Or use manager for batch operations
from integration_utils import KnowledgeBaseManager

kb_manager = KnowledgeBaseManager("kb.json")
kb_manager.add_from_human_feedback(assessment, corrections)
kb_manager.save()
```

**Action**: Set up KB collection pipeline for your domain

### 8. Metrics Collection

Track performance systematically:

```python
# OLD: Manual tracking
safe_count = 0
unsafe_count = 0
for plan in plans:
    assessment = kd.run(scene, plan)
    if assessment.overall.label == "SAFE":
        safe_count += 1
    # ... more manual counting

# NEW: Automatic metrics
from integration_utils import MetricsCollector

collector = MetricsCollector()
for plan in plans:
    assessment = kd.run(scene, plan)
    collector.update_from_assessment(assessment)

summary = collector.get_summary()
print(f"Success rate: {summary['rates']['success_rate']:.2%}")
print(f"Help rate: {summary['rates']['help_rate']:.2%}")
```

**Action**: Use `MetricsCollector` for comprehensive metrics

## Updating Test Files

### Your Current Tests

Example of updating tests in `tests/benchmark_knowno_roboguard.py`:

```python
# OLD
from knowdanger.core.knowdanger_core import KnowDanger, Config

def test_basic():
    config = Config(alpha=0.1)
    kd = KnowDanger(config)
    assessment = kd.run(scene, plan)
    assert assessment.overall.label == "SAFE"

# NEW - Option 1: Minimal change
from knowdanger.core.knowdanger_enhanced import EnhancedKnowDanger as KnowDanger

def test_basic():
    config = Config(alpha=0.1)
    kd = KnowDanger(config)
    assessment = kd.run(scene, plan)
    assert assessment.overall.label == "SAFE"

# NEW - Option 2: Test IntroPlan too
from knowdanger.core.knowdanger_enhanced import EnhancedKnowDanger

def test_basic():
    config = Config(alpha=0.1, use_introspection=False)  # Disable IP for fair comparison
    kd = EnhancedKnowDanger(config)
    assessment = kd.run(scene, plan)
    assert assessment.overall.label == "SAFE"

def test_with_introspection():
    config = Config(alpha=0.1, use_introspection=True)
    kd = EnhancedKnowDanger(config)
    assessment = kd.run(scene, plan)
    assert assessment.overall.label == "SAFE"
    # Check IntroPlan ran
    assert any(s.introplan is not None for s in assessment.steps)
```

## Updating Your Benchmarks

### Example: `tests/benchmark_true_baselines.py`

```python
# Add new baseline: IntroPlan
def run_benchmark():
    results = {
        "vanilla": run_vanilla(),
        "roboguard": run_roboguard(),
        "knowno": run_knowno(),
        "roboguard+knowno": run_rg_kn(),
        "full_integration": run_full()  # ← NEW
    }
    return results

def run_full():
    """Benchmark full RG+KN+IP integration"""
    config = Config(
        alpha=0.1,
        use_introspection=True,
        aggregation_strategy="conservative"
    )
    kd = EnhancedKnowDanger(config)
    
    results = []
    for scene, plan in test_cases:
        assessment = kd.run(scene, plan)
        results.append({
            "verdict": assessment.overall.label,
            "rg_triggered": any(s.roboguard.label == "UNSAFE" for s in assessment.steps),
            "kn_triggered": any(s.knowno and s.knowno.label == "UNCERTAIN" for s in assessment.steps),
            "ip_triggered": any(s.introplan and s.introplan.label == "UNCERTAIN" for s in assessment.steps)
        })
    
    return results
```

## Configuration Migration

### Config Class Changes

Your old configs work, but new options available:

```python
# OLD Config options (still work)
Config(
    alpha=0.1,
    ask_threshold_confidence=0.7,
    KNOWNO_ENV="KNOWNO_ROOT"
)

# NEW Config options (all optional)
Config(
    # Old options
    alpha=0.1,
    ask_threshold_confidence=0.7,
    
    # New IntroPlan options
    use_introspection=True,
    introplan_kb_path="kb.json",
    introplan_retrieval_k=3,
    introspection_weight=0.5,
    
    # New aggregation options
    aggregation_strategy="conservative",
    
    # New LLM option
    llm_func=my_llm_function
)
```

## Data Structure Changes

### StepAssessment

```python
# OLD structure (still present)
StepAssessment(
    step=...,
    roboguard=...,
    knowno=...,
    final=...
)

# NEW structure (additional field)
StepAssessment(
    step=...,
    roboguard=...,
    knowno=...,
    introplan=...,  # ← NEW (Optional[Verdict])
    final=...
)
```

**Action**: Code that creates `StepAssessment` objects can omit `introplan` (it's optional)

### PlanAssessment

```python
# OLD structure (still present)
PlanAssessment(
    plan=...,
    steps=...,
    overall=...
)

# NEW structure (additional field)
PlanAssessment(
    plan=...,
    steps=...,
    overall=...,
    meta={  # ← NEW (Dict[str, Any])
        "introspective_explanations": [...],
        "systems_used": ["RoboGuard", "KnowNo", "IntroPlan"],
        "refinement_iterations": 2  # If refinement was used
    }
)
```

**Action**: Code can ignore `meta` if not needed

## Common Migration Patterns

### Pattern 1: Conservative Migration

Start with IntroPlan disabled:

```python
# Week 1: Test with existing setup
config = Config(alpha=0.1, use_introspection=False)
kd = EnhancedKnowDanger(config)
# Behaves identically to old version

# Week 2: Enable introspection
config = Config(alpha=0.1, use_introspection=True)
kd = EnhancedKnowDanger(config)
# Now using all three systems

# Week 3: Add refinement
assessment = kd.run_with_rewriting(scene, plan)
# Now using iterative refinement
```

### Pattern 2: Feature-by-Feature

Enable features one at a time:

```python
# Phase 1: Basic introspection
config = Config(
    alpha=0.1,
    use_introspection=True,
    introplan_kb_path=None  # No KB yet
)

# Phase 2: Add knowledge base
config = Config(
    alpha=0.1,
    use_introspection=True,
    introplan_kb_path="kb.json"
)

# Phase 3: Add refinement
assessment = kd.run_with_rewriting(scene, plan, max_iterations=1)

# Phase 4: Full features
config = Config(
    alpha=0.1,
    use_introspection=True,
    introplan_kb_path="kb.json",
    aggregation_strategy="weighted"
)
assessment = kd.run_with_rewriting(scene, plan, max_iterations=3)
```

### Pattern 3: A/B Testing

Compare old vs new:

```python
from knowdanger.core.knowdanger_core import KnowDanger as OldKD
from knowdanger.core.knowdanger_enhanced import EnhancedKnowDanger as NewKD

# Test both versions
old_config = Config(alpha=0.1)
new_config = Config(alpha=0.1, use_introspection=True)

old_kd = OldKD(old_config)
new_kd = NewKD(new_config)

old_assessment = old_kd.run(scene, plan)
new_assessment = new_kd.run(scene, plan)

# Compare
print(f"Old verdict: {old_assessment.overall.label}")
print(f"New verdict: {new_assessment.overall.label}")
```

## File-by-File Migration Checklist

Go through each file in your repo:

### `tests/` Directory

- [ ] `benchmark_knowno_roboguard.py` - Update imports, add IP baseline
- [ ] `benchmark_true_baselines.py` - Add full integration baseline
- [ ] `knowdanger_vs_baselines.py` - Compare with/without IP
- [ ] Update test assertions to handle optional `introplan` field

### `scripts_api/` Directory

- [ ] `smoke_test.py` - Add IP to smoke tests
- [ ] `run_scene.py` - Update to use enhanced version
- [ ] Add logging with `LoggingHelper`

### `src/knowdanger/core/` Directory

- [ ] Keep `knowdanger_core.py` as backup
- [ ] Add `knowdanger_enhanced.py`
- [ ] Update `__init__.py` to export both

### `src/scenes/` Directory

- [ ] Update scene runners to use enhanced version
- [ ] Add knowledge base paths to scene configs
- [ ] Enable introspection for safety-critical scenes

## Rollback Plan

If you need to rollback:

```python
# Quick rollback: just change import
from knowdanger.core.knowdanger_core import KnowDanger  # Old version
# from knowdanger.core.knowdanger_enhanced import EnhancedKnowDanger  # New version

kd = KnowDanger(config)
```

Or keep a backup:

```bash
# Before migration
git checkout -b backup-before-enhancement

# After migration, if issues
git checkout backup-before-enhancement
```

## Testing Checklist

After migration, verify:

- [ ] Old tests still pass
- [ ] New tests with IntroPlan pass
- [ ] Smoke tests run successfully
- [ ] Benchmarks run with comparable performance
- [ ] Logging works correctly
- [ ] Knowledge base loads properly
- [ ] Refinement produces better results
- [ ] Metrics collection works

## Performance Considerations

The enhanced version is slightly slower due to IntroPlan:

```
Old (RG + KN):    ~50ms per step
New (RG + KN):    ~50ms per step (no change if IP disabled)
New (RG+KN+IP):   ~200ms per step (with introspection)
```

**Optimization tips**:

1. Disable introspection for non-critical plans
2. Use smaller knowledge bases
3. Cache introspective reasoning
4. Run systems in parallel (future enhancement)

## Questions?

If you encounter issues:

1. Check the INTEGRATION_GUIDE.md
2. Review example_usage.py
3. Look at inline code comments
4. Open an issue with:
   - What you tried
   - Error messages
   - Your config

## Next Steps After Migration

1. **Collect Data**: Build IntroPlan KB for your domain
2. **Calibrate**: Run calibration with your test set
3. **Evaluate**: Compare old vs new performance
4. **Iterate**: Refine based on results
5. **Deploy**: Use in your robot system

---

**Migration complete?** Run the full test suite:

```bash
pytest tests/
python examples/example_usage.py
python tests/benchmark_knowno_roboguard.py
```

If all pass, you're ready to go! 🎉