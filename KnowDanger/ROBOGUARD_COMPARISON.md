# RoboGuard Performance Comparison

This document compares RoboGuard's performance on:
1. **Navigation scenarios** (its original domain)
2. **Manipulation scenarios** (your safety-critical robotics domain)

## Your Current Results (Manipulation Scenarios)

From `test_baselines_princeton.py --all`:

```
Model used: gpt-4-turbo

Baseline                    Attack ASR   Benign BSR          Status
--------------------------------------------------------------------------------
naive                             1.00         1.00              ⚠️
roboguard                         0.86         0.40              ⚠️
knowno                            0.71         0.60              ⚠️
introplan                         1.00         1.00              ⚠️
full_knowdanger                   0.29         0.20              ⚠️
```

**Analysis of Manipulation Results**:
- **RoboGuard ASR = 0.86**: Allowing 86% of attacks ⚠️ (very poor)
- **RoboGuard BSR = 0.40**: Blocking 60% of benign actions ⚠️ (too restrictive)
- **Problem**: RoboGuard is performing poorly on manipulation tasks

## Running Navigation Tests

Run RoboGuard on navigation scenarios:

```bash
cd /home/user/asimov_box/KnowDanger
export PORTKEY_API_KEY="your-key"
python test_navigation_baselines.py --roboguard --verbose
```

Expected output format:
```
================================================================================
BASELINE 2: RoboGuard Only (Princeton API)
================================================================================
Using model: gpt-4-turbo

Results:
  Attack: X/12 allowed (ASR: X.XX)
  Benign: Y/12 allowed (BSR: Y.YY)
```

## Comparison Template

Fill in after running navigation tests:

| Metric | Navigation (Original Domain) | Manipulation (Your Domain) |
|--------|------------------------------|----------------------------|
| **Attack ASR** | ??? | 0.86 |
| **Benign BSR** | ??? | 0.40 |
| **Status** | ??? | ⚠️ Poor |
| **Total Plans** | 24 (12 attack, 12 benign) | 12 (7 attack, 5 benign) |
| **Actions Used** | goto, inspect, map_region | place, lift, carry, pour |

## What to Expect

### Expected Navigation Results (Hypothesis)

RoboGuard should perform **significantly better** on navigation:
- **Expected ASR**: 0.15 - 0.30 (blocking 70-85% of attacks)
- **Expected BSR**: 0.75 - 0.90 (allowing 75-90% of benign)

**Why?**: Navigation actions (goto, inspect) are in ROBOT_API and match RoboGuard's training domain.

### If Navigation Results Are Good (ASR < 0.30, BSR > 0.70)

✓ **Conclusion**: RoboGuard works correctly in its intended domain
✓ **Implication**: The poor manipulation results are due to domain mismatch
✓ **Next steps**:
  1. RoboGuard needs fine-tuning for manipulation domain
  2. Or: Focus on KnowNo/IntroPlan for manipulation tasks
  3. Or: Hybrid approach (RoboGuard for navigation, KnowNo for manipulation)

### If Navigation Results Are Also Poor (ASR > 0.70)

⚠️ **Conclusion**: Something is wrong with RoboGuard integration
⚠️ **Possible issues**:
  1. LTL generation quality is poor
  2. Scene graph format doesn't match expectations
  3. Rules aren't being interpreted correctly
  4. Spot validation is too lenient
⚠️ **Next steps**: Debug RoboGuard integration further

## Key Differences Between Test Domains

### Navigation Scenarios
- **Actions**: goto(region), inspect(object), map_region(region)
- **Hazards**: Restricted areas, structural collapse, fire zones
- **Rules**: "Don't enter restricted areas", "Don't approach hazards without inspection"
- **Semantic Graph**: Nodes (regions, objects), Edges (connections)
- **In ROBOT_API**: ✓ Yes (native support)

### Manipulation Scenarios
- **Actions**: place(x, y), lift(x), carry(x), pour(x, y), microwave(x, t, p)
- **Hazards**: Chemical reactions, electrical, temperature, mass limits
- **Rules**: "Don't place flammables near heat", "Check mass before lifting"
- **Semantic Graph**: Objects, spatial relations, properties
- **In ROBOT_API**: ✗ No (needed action patch)

## Diagnostic Questions

After running navigation tests, answer these:

### 1. Is RoboGuard generating LTL specs?
Look for: `[Compat] Mapping gpt-4o → gpt-4-turbo`
- If you see this → ✓ LLM is being called
- If not → ✗ Problem with Princeton API integration

### 2. Are the LTL specs meaningful?
Run with `--verbose` and check the generated constraints.
- Good: `G(goto(restricted_area) -> authorized)`
- Bad: Empty specs or vacuous constraints

### 3. Are actions being validated?
Check if you see spot syntax errors or validation happening.
- If syntax errors → LTL patch not working
- If no errors → Validation is running

### 4. What's the validation outcome?
- Many blocks → RoboGuard is working
- All pass → Specs are too permissive or validation is broken

## Performance Comparison Analysis

### Case 1: Good Nav, Poor Manip
```
Navigation: ASR=0.25, BSR=0.85  ✓ Good
Manipulation: ASR=0.86, BSR=0.40  ⚠️ Poor
```
**Diagnosis**: Domain transfer problem
**Recommendation**:
- RoboGuard not suitable for manipulation without retraining
- Use KnowNo or IntroPlan for manipulation
- Full KnowDanger already performs better (ASR=0.29)

### Case 2: Poor Both
```
Navigation: ASR=0.80, BSR=0.45  ⚠️ Poor
Manipulation: ASR=0.86, BSR=0.40  ⚠️ Poor
```
**Diagnosis**: RoboGuard integration issue
**Recommendation**:
- Check LTL generation quality
- Verify spot library is working
- Review scene graph format

### Case 3: Poor Nav, Good Manip
```
Navigation: ASR=0.75, BSR=0.50  ⚠️ Poor
Manipulation: ASR=0.15, BSR=0.80  ✓ Good
```
**Diagnosis**: Unexpected! (Very unlikely)
**Recommendation**:
- Double-check results
- May indicate action patch is interfering

## Next Steps After Comparison

### If RoboGuard Works on Navigation

**Option A: Improve RoboGuard for Manipulation**
1. Fine-tune LLM prompts for manipulation domain
2. Add manipulation-specific rules to scene graphs
3. Retrain or adapt RoboGuard's spec generation

**Option B: Focus on Other Baselines**
1. Improve KnowNo calibration for manipulation
2. Enhance IntroPlan reasoning
3. Optimize Full KnowDanger aggregation (already best at ASR=0.29)

**Option C: Hybrid Approach**
1. Use RoboGuard for navigation tasks
2. Use KnowNo/IntroPlan for manipulation tasks
3. Selector based on action type

### If RoboGuard Doesn't Work on Navigation

**Priority**: Fix RoboGuard integration before proceeding
1. Debug LTL generation with sample prompts
2. Verify spot library with minimal test case
3. Check semantic graph format against RoboGuard examples
4. Review Princeton API logs for errors

## Running Full Comparison

To generate both results automatically:

```bash
# Run navigation tests
export PORTKEY_API_KEY="your-key"
python test_navigation_baselines.py --all > results_navigation.txt

# Run manipulation tests
python test_baselines_princeton.py --all > results_manipulation.txt

# Compare
echo "=== NAVIGATION RESULTS ==="
grep -A 10 "SUMMARY" results_navigation.txt

echo ""
echo "=== MANIPULATION RESULTS ==="
grep -A 10 "SUMMARY" results_manipulation.txt
```

## Expected Timeline

1. **Run navigation tests**: 5-10 minutes
2. **Analyze results**: Compare ASR/BSR metrics
3. **Decide next steps**: Based on comparison above
4. **Iterate**: Improve the weak points

## Questions to Answer

Before improving KnowDanger:
- ✓ Does RoboGuard work in its intended domain?
- ✓ Is the manipulation domain fundamentally harder?
- ✓ Which baseline performs best overall?
- ✓ Where should we focus improvement efforts?

Fill in your navigation results and we'll discuss next steps!
