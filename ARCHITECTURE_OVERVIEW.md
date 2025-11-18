# Asimov Box Architecture Overview

## Three-System Integration Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         USER COMMAND & LLM PLANNER                          │
│                                                                             │
│  "Move the solvent bottle to a safe location"                              │
│                                                                             │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│               LLM GENERATES PLAN WITH ACTION CANDIDATES                     │
│                                                                             │
│  Plan:                                                                      │
│    Step 1: pick(solvent_bottle)                                            │
│    Step 2: place(solvent_bottle, bench1)                                   │
│             Candidates: [("place_on_bench1", 0.6),                         │
│                          ("place_on_bench2", 0.25),                        │
│                          ("place_in_hood", 0.15)]                          │
│                                                                             │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────────────────────┐
        │                                                        │
        │      ASIMOV BOX: ENHANCED KNOWDANGER SYSTEM           │
        │     (Three-Way Safety Verification System)           │
        │                                                        │
        └────────────────────────────────────────────────────────┘
                             │
        ┌────────────┬───────┴───────┬────────────┐
        │            │               │            │
        ▼            ▼               ▼            ▼
   ┌─────────┐  ┌─────────┐    ┌──────────┐  ┌────────┐
   │ROBOGUARD│  │ KNOWNO  │    │INTROPLAN │  │AGGREG. │
   └────┬────┘  └────┬────┘    └────┬─────┘  └───┬────┘
        │            │             │              │
        │            │             │              │
        ▼            ▼             ▼              │
   
   Rules Check │ Uncertainty │ Introspection   │
   (Binary)    │ Quantif.    │ (Reasoning)    │
                │             │                 │
        │      │ SAFE/     │ SAFE/UNSAFE/   │
        │      │ UNCERTAIN │ UNCERTAIN      │
        │      │           │                │
        └──────┴───────────┴────────────────┘
                     │
                     ▼
            ┌─────────────────────┐
            │   AGGREGATOR        │
            │ (Conservative/Major)│
            └────────┬────────────┘
                     │
                     ▼
            ┌─────────────────────┐
            │  FINAL VERDICT      │
            │ SAFE/UNSAFE/UNCERT. │
            └────────┬────────────┘
                     │
         ┌───────────┼───────────┐
         │           │           │
         ▼           ▼           ▼
      EXECUTE    ASK HUMAN    BLOCK
      ACTION     FOR HELP     ACTION
```

---

## Component Details

### 1. RoboGuard (Rule-Based Safety)

```
┌─────────────────────────────────────────┐
│           ROBOGUARD SYSTEM              │
├─────────────────────────────────────────┤
│                                         │
│  INPUT:  - Safety Rules (temporal      │
│            logic)                       │
│          - Scene Graph                 │
│          - Action Parameters           │
│                                         │
│  PROCESS:                               │
│    1. Compile rules with scene context │
│    2. Check each action against rules  │
│    3. Identify rule violations         │
│                                         │
│  OUTPUT: SAFE | UNSAFE                 │
│  NOTES:  - Binary verdict              │
│          - Rule violation info         │
│          - Fast (symbolic)             │
│                                         │
└─────────────────────────────────────────┘

EXAMPLE RULES:
- G( place(x, s) & flammable(x) -> !near(s, hot_surface) )
- G( carry(o) & fragile(o) -> avoid_shocks & low_speed )
```

### 2. KnowNo (Uncertainty Quantification)

```
┌─────────────────────────────────────────┐
│            KNOWNO SYSTEM                │
├─────────────────────────────────────────┤
│                                         │
│  INPUT:  - LLM Scores/Logits           │
│          - Calibration Threshold (tau) │
│          - Alpha (confidence level)    │
│                                         │
│  PROCESS:                               │
│    1. Compute prediction set via CP    │
│    2. If |pred_set| == 1: action clear │
│    3. If |pred_set| > 1: ambiguous     │
│    4. Apply ask-for-help threshold     │
│                                         │
│  OUTPUT: SAFE | UNCERTAIN              │
│  NOTES:  - Conformal prediction        │
│          - Handles uncertainty         │
│          - Ask-for-help mechanism      │
│                                         │
└─────────────────────────────────────────┘

EXAMPLE:
Candidates: [("pick_safe", 0.8),
             ("pick_unsafe", 0.2)]
Pred_set: {pick_safe}  → SAFE
Pred_set: {both}       → UNCERTAIN
```

### 3. IntroPlan (Introspective Reasoning)

```
┌─────────────────────────────────────────┐
│         INTROPLAN SYSTEM                │
├─────────────────────────────────────────┤
│                                         │
│  INPUT:  - Task Description            │
│          - Scene Context               │
│          - Action Candidates           │
│          - Knowledge Base               │
│                                         │
│  PROCESS:                               │
│    1. Retrieve similar examples from KB│
│    2. Generate introspective reasoning │
│    3. Explain safety decision          │
│    4. Detect ambiguity/ask clarify     │
│    5. Suggest safer alternative        │
│                                         │
│  OUTPUT: SAFE | UNSAFE | UNCERTAIN     │
│  REASONING: Natural language explain.  │
│  NOTES:  - LLM-based                   │
│          - Explainable                 │
│          - Learning from feedback      │
│                                         │
└─────────────────────────────────────────┘

EXAMPLE KNOWLEDGE ENTRY:
{
  "task": "place flammable item",
  "scene": "lab with heat sources",
  "correct": "place_in_hood",
  "reasoning": "Hood provides isolation..."
}
```

### 4. Aggregator (Verdict Fusion)

```
┌─────────────────────────────────────────┐
│        AGGREGATOR STRATEGIES            │
├─────────────────────────────────────────┤
│                                         │
│  CONSERVATIVE (Default):                │
│    - ANY UNSAFE → UNSAFE                │
│    - ALL SAFE → SAFE                    │
│    - Otherwise → UNCERTAIN              │
│                                         │
│  MAJORITY:                              │
│    - Vote among systems                │
│    - Majority wins                      │
│                                         │
│  WEIGHTED:                              │
│    - Confidence-weighted combination   │
│    - System-specific weights            │
│                                         │
└─────────────────────────────────────────┘

VOTING EXAMPLE:
RoboGuard:  UNSAFE
KnowNo:     UNCERTAIN
IntroPlan:  SAFE

Conservative → UNSAFE (ANY)
Majority    → UNCERTAIN
Weighted    → Depends on weights
```

---

## Data Flow for a Single Step

```
STEP INPUT:
  action="place"
  params={"object": "solvent", "location": "bench1"}
  candidates=[("place_on_bench1", 0.6), 
              ("place_on_bench2", 0.25),
              ("place_in_hood", 0.15)]

┌──────────────────────────────────────────────────┐
│                RoboGuard Check                   │
├──────────────────────────────────────────────────┤
│ Query: place(solvent, bench1) ∧ flammable(solv.)│
│ Rule: !near(flammable, hot_surface)             │
│ Context: bench1 has hotplate nearby             │
│ Result: VIOLATES RULE                           │
│ Verdict: UNSAFE                                 │
└─────┬───────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────┐
│               KnowNo Check                       │
├──────────────────────────────────────────────────┤
│ Scores: [0.6, 0.25, 0.15]                       │
│ Tau (calibrated threshold): 0.5                 │
│ Candidates above tau: [bench1]                  │
│ Pred_set size: 1 (confident)                    │
│ Verdict: SAFE (high confidence)                 │
└─────┬───────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────┐
│              IntroPlan Check                     │
├──────────────────────────────────────────────────┤
│ Retrieve similar: "place flammable in hood"     │
│ Generate reasoning via LLM:                      │
│  "Placing flammable item near heat source is    │
│   dangerous. The hood provides better isolation.│
│   Recommend place_in_hood instead."             │
│ Verdict: UNSAFE                                 │
│ Recommendation: place_in_hood (0.15)            │
└─────┬───────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────┐
│           AGGREGATION (Conservative)            │
├──────────────────────────────────────────────────┤
│ RoboGuard: UNSAFE                               │
│ KnowNo:    SAFE                                 │
│ IntroPlan: UNSAFE                               │
│                                                 │
│ Decision Rule: ANY UNSAFE → UNSAFE              │
│ Final Verdict: UNSAFE                           │
│                                                 │
│ Explanation:                                    │
│   "This action violates safety rules:          │
│    - Placing flammable item near heat          │
│    - IntroPlan recommends place_in_hood        │
│    Action must be refined or blocked."         │
└──────────────────────────────────────────────────┘

OUTPUT:
  StepAssessment(
    step=step,
    roboguard=Verdict("UNSAFE", "violates rule"),
    knowno=Verdict("SAFE", "high confidence"),
    introplan=Verdict("UNSAFE", "heat hazard"),
    final=Verdict("UNSAFE", "Any system blocks")
  )
```

---

## File Organization & Data Flow

```
INPUT (Scene + Plan)
    │
    ├─ Scene: semantic_graph, rules, env_params
    └─ Plan: steps with candidates
    
    │
    ▼
┌──────────────────────────────────────────┐
│    EnhancedKnowDanger.run()              │
└──────────────────────┬───────────────────┘
    │
    ├─ For each step:
    │   │
    │   ├─→ RoboGuardBridge.evaluate_step()
    │   │   └─ /knowdanger/adapters/roboguard_adapter.py
    │   │
    │   ├─→ KnowNoAdapter.evaluate_step()
    │   │   └─ /lang_help/knowno/api.py + /adapters/paper_knowno.py
    │   │
    │   ├─→ IntroPlanAdapter.evaluate_step()
    │   │   └─ /knowdanger/adapters/introplan_adapter.py
    │   │   └─ Uses: /IntroPlan/llm.py, utils.py, etc.
    │   │
    │   └─→ Config.aggregator()
    │       └─ Combines three verdicts
    │
    ▼
┌──────────────────────────────────────────┐
│    PlanAssessment                        │
│    ├─ steps: List[StepAssessment]        │
│    │  ├─ roboguard verdict               │
│    │  ├─ knowno verdict                  │
│    │  ├─ introplan verdict               │
│    │  └─ final verdict                   │
│    └─ overall verdict                    │
└──────────────────────────────────────────┘
    │
    ▼
OUTPUT (Assessment with all verdicts)
```

---

## Integration Layers

### Layer 1: Adapters
```
┌─────────────────────────────────────────────────────────┐
│                  ADAPTER LAYER                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  RoboGuardAdapter                                       │
│  ├─ Imports: roboguard module dynamically              │
│  ├─ Interface: fit(scene_graph, rules) →               │
│  └─ evaluate_step(step) → Verdict                      │
│                                                         │
│  KnowNoAdapter  (paper_knowno.py)                       │
│  ├─ Imports: lang_help.knowno.api                      │
│  ├─ Interface: calibrate(scores, alpha) → tau          │
│  └─ evaluate_step(candidates) → Verdict                │
│                                                         │
│  IntroPlanAdapter                                       │
│  ├─ Imports: IntroPlan/llm.py, utils.py                │
│  ├─ Interface: generate_reasoning(...) →               │
│  └─ evaluate_step(...) → Verdict                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Layer 2: Core Orchestrator
```
┌─────────────────────────────────────────────────────────┐
│              ORCHESTRATOR LAYER                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  EnhancedKnowDanger                                     │
│  ├─ __init__: Initialize all adapters                  │
│  ├─ run(scene, plan): Main evaluation method           │
│  ├─ run_with_rewriting(): Iterative refinement         │
│  └─ calibrate_knowno(): Calibration                    │
│                                                         │
│  Config                                                 │
│  ├─ alpha, ask_threshold_confidence                    │
│  ├─ use_introspection, aggregation_strategy            │
│  └─ aggregator(): Verdict fusion logic                 │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Layer 3: Utilities
```
┌─────────────────────────────────────────────────────────┐
│              UTILITIES LAYER                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  integration_utils.py                                   │
│  ├─ FormatConverter: Between-system format conversion   │
│  ├─ CalibrationHelper: KnowNo calibration              │
│  ├─ MetricsCollector: Performance metrics              │
│  ├─ KnowledgeBaseManager: IntroPlan KB lifecycle       │
│  └─ LoggingHelper: Comprehensive logging               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Knowledge Base Structure (IntroPlan)

```
Knowledge Base (kb.json or kb.txt)
│
└─ List of KnowledgeEntry:
   │
   ├─ Entry 1: "place flammable item"
   │  ├─ task_description: "..."
   │  ├─ scene_context: "lab with heat"
   │  ├─ correct_option: "place_in_hood"
   │  ├─ introspective_reasoning: "..."
   │  └─ safety_considerations: [...]
   │
   ├─ Entry 2: "carry fragile item"
   │  └─ ...
   │
   └─ Entry N: ...

RETRIEVAL:
  Query: (task, scene, candidates)
    ↓
  Similarity matching (cosine/semantic)
    ↓
  Top-k entries retrieved
    ↓
  Used in LLM context for reasoning
```

---

## Configuration Hierarchy

```
┌─────────────────────────────────────────┐
│         DEFAULT CONFIG                  │
│  (Config.__init__ defaults)             │
│                                         │
│  alpha = 0.1                            │
│  ask_threshold_confidence = 0.7         │
│  use_introspection = True               │
│  aggregation_strategy = "conservative"  │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  USER-PROVIDED CONFIG                   │
│  (Overrides defaults)                   │
│                                         │
│  config = Config(                       │
│      alpha=0.05,                        │
│      use_introspection=True,            │
│      introplan_kb_path="my_kb.json"     │
│  )                                      │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  RUNTIME CONFIG                         │
│  (Used by EnhancedKnowDanger)           │
│                                         │
│  Passed to:                             │
│  - RoboGuardBridge.fit(rules)           │
│  - KnowNoAdapter.calibrate()            │
│  - IntroPlanAdapter(kb_path)            │
│  - aggregator() for verdict fusion      │
└─────────────────────────────────────────┘
```

---

## Error Handling & Fallbacks

```
EnhancedKnowDanger.run()
│
├─ RoboGuardBridge:
│  ├─ IF roboguard import fails
│  │  └─ Return SAFE (default)
│  └─ IF rule compilation fails
│     └─ Return SAFE (no rules)
│
├─ KnowNoAdapter:
│  ├─ IF no calibration data
│  │  └─ Use fallback quantile
│  └─ IF predict_set fails
│     └─ Return top-1 as SAFE
│
├─ IntroPlanAdapter:
│  ├─ IF KB not found
│  │  └─ Continue without retrieval
│  ├─ IF LLM call fails
│  │  └─ Use heuristic reasoning
│  └─ IF IntroPlan modules not found
│     └─ Skip introspection
│
└─ Aggregator:
   └─ Always produces a verdict
      (never fails)
```

---

## Testing & Evaluation Flow

```
Test Suite:
│
├─ benchmark_knowno_roboguard.py
│  └─ Compare systems on example scenes
│
├─ benchmark_true_baselines.py
│  └─ Baseline comparisons
│
├─ Example scenes:
│  ├─ example1_hazard_lab.py
│  ├─ example2_breakroom.py
│  └─ example3_photonics.py
│
└─ example_usage.py
   └─ 6 worked examples

EVALUATION PIPELINE:
  Scene + Plans
    ↓
  EnhancedKnowDanger.run() × N
    ↓
  MetricsCollector.collect()
    ↓
  Generate Report
    ├─ success_rate
    ├─ help_rate
    └─ safety_violation_rate
```

---

## Key Interface Summary

```python
# MAIN ENTRY POINT
from knowdanger.core.knowdanger_enhanced import EnhancedKnowDanger, Config

config = Config(
    alpha=0.1,
    use_introspection=True,
    introplan_kb_path="kb.json",
    aggregation_strategy="conservative"
)
kd = EnhancedKnowDanger(config)
assessment = kd.run(scene, plan)

# ASSESSMENT STRUCTURE
assessment: PlanAssessment
├─ plan: PlanCandidate
├─ steps: List[StepAssessment]
│  └─ StepAssessment:
│     ├─ step: Step
│     ├─ roboguard: Verdict (SAFE/UNSAFE)
│     ├─ knowno: Verdict (SAFE/UNCERTAIN)
│     ├─ introplan: Verdict (SAFE/UNSAFE/UNCERTAIN)
│     └─ final: Verdict (aggregated)
└─ overall: Verdict (plan-level)

# EXTENDED USAGE
assessment = kd.run_with_rewriting(scene, plan, max_iterations=3)
kd.calibrate_knowno(calibration_data)
kb_mgr = KnowledgeBaseManager("kb.json")
```

---

## Deployment Path

```
DEVELOPMENT:
  ├─ Single system (RoboGuard only)
  ├─ Two systems (RG + KN)
  └─ Three systems (RG + KN + IP)

PRODUCTION:
  ├─ API wrapper
  ├─ Real-time latency monitoring
  ├─ Graceful degradation
  └─ Continuous knowledge base updates
```

