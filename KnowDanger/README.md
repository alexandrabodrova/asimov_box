
### 2. Install (if needed)

```bash
pip install -e .
```

### 3. Test It

```python
from knowdanger_enhanced import EnhancedKnowDanger, create_default_config
from knowdanger_enhanced import Scene, PlanCandidate, Step

# Configure
config = create_default_config(alpha=0.1, use_introspection=True)
kd = EnhancedKnowDanger(config)

# Create a simple test
scene = Scene(
    name="test_lab",
    semantic_graph={"locations": ["bench"], "objects": ["chemical"]},
    rules=["!near(flammable, heat)"]
)

plan = PlanCandidate(
    name="test_plan",
    user_prompt="Test safety verification",
    steps=[
        Step(
            action="pick",
            params={"object": "chemical"},
            candidates=[("pick_safe", 0.8), ("pick_unsafe", 0.2)]
        )
    ]
)

# Evaluate
assessment = kd.run(scene, plan)
print(f"Verdict: {assessment.overall.label}")
```

### 4. Run Full Examples

```bash
python examples/example_usage.py
```

## 📖 Documentation Structure

### Start Here
1. **IMPLEMENTATION_SUMMARY.md** - Read this first! Quick overview of what you got
2. **INTEGRATION_GUIDE.md** - Complete documentation with architecture, API, troubleshooting
3. **MIGRATION_GUIDE.md** - Step-by-step guide to update your existing code

### Code Examples
4. **example_usage.py** - 6 working examples demonstrating all features

### Source Code
5. **introplan_adapter.py** - IntroPlan integration (fully documented)
6. **knowdanger_enhanced.py** - Enhanced core orchestrator (fully documented)
7. **integration_utils.py** - Utilities and helpers (fully documented)

## 🎯 Key Features

### 1. Three-Way Safety Verification

Every action is checked by three independent systems:

```
User → LLM → Plan → ┌─ RoboGuard (rules)
                    ├─ KnowNo (uncertainty)
                    └─ IntroPlan (reasoning)
                         ↓
                    Aggregator
                         ↓
                    SAFE/UNSAFE/UNCERTAIN
```

### 2. Flexible Integration

Use what you need:

```python
# Just RoboGuard
config = Config(use_introspection=False)

# RoboGuard + KnowNo
config = Config(use_introspection=False)

# All three systems
config = Config(use_introspection=True)
```

### 3. Iterative Refinement

Plans can be automatically improved:

```python
# System detects issues and suggests fixes
assessment = kd.run_with_rewriting(scene, plan, max_iterations=3)
```

### 4. Explainable Decisions

Every verdict comes with explanations:

```python
for step_assess in assessment.steps:
    if step_assess.introplan:
        print(f"Reasoning: {step_assess.introplan.details['reasoning']}")
        print(f"Safety: {step_assess.introplan.details['safety']}")
```

### 5. Learning from Feedback

Build knowledge base from experience:

```python
kd.add_knowledge_entry(
    task="move chemical",
    scene_context="lab with heat",
    correct_option="place_in_hood",
    human_feedback="Hood provides isolation"
)
```

## 🏗️ Architecture

### System Flow

```
┌─────────────────────────────────────────┐
│  EnhancedKnowDanger (Orchestrator)      │
│                                         │
│  ┌────────────────────────────────────┐│
│  │ 1. RoboGuardBridge                 ││
│  │    - Compile safety rules          ││
│  │    - Verify rule violations        ││
│  │    - Return: SAFE/UNSAFE           ││
│  └────────────────────────────────────┘│
│                  ↓                      │
│  ┌────────────────────────────────────┐│
│  │ 2. KnowNoAdapter                   ││
│  │    - Conformal prediction          ││
│  │    - Compute prediction sets       ││
│  │    - Return: SAFE/UNCERTAIN        ││
│  └────────────────────────────────────┘│
│                  ↓                      │
│  ┌────────────────────────────────────┐│
│  │ 3. IntroPlanAdapter                ││
│  │    - Retrieve similar examples     ││
│  │    - Generate introspective reason.││
│  │    - Refine confidence bounds      ││
│  │    - Return: SAFE/UNSAFE/UNCERTAIN ││
│  └────────────────────────────────────┘│
│                  ↓                      │
│  ┌────────────────────────────────────┐│
│  │ 4. Aggregator                      ││
│  │    - Combine verdicts              ││
│  │    - Conservative/Majority/Weighted││
│  │    - Return: Final verdict         ││
│  └────────────────────────────────────┘│
└─────────────────────────────────────────┘
```

### Aggregation Strategies

**Conservative** (default):
- ANY system says UNSAFE → block
- ALL systems say SAFE → execute
- Otherwise → ask for help

**Majority**:
- Democratic vote among systems
- Majority wins

**Weighted**:
- Confidence-weighted combination
- Adjustable system weights

## 📊 What Makes This Unique

### 1. First Complete Integration

This is the **first implementation** that fully integrates:
- RoboPAIR's adversarial robustness
- KnowNo's uncertainty quantification  
- IntroPlan's introspective reasoning

### 2. Production-Ready

- ✅ Comprehensive error handling
- ✅ Full type hints
- ✅ Extensive documentation
- ✅ Logging and metrics
- ✅ Format conversion utilities
- ✅ Testing utilities

### 3. Research-Friendly

- 📊 Automatic metrics collection
- 📝 Detailed logging for analysis
- 🔍 Explainable decisions
- 🧪 Easy to run experiments
- 📈 Built-in benchmarking utilities

### 4. Backward Compatible

- Works with your existing `knowdanger_core.py`
- No breaking changes
- Gradual migration path
- Can run old and new versions side-by-side

## 🎓 Academic Context

This integration directly implements concepts from three key papers:

### 1. RoboPAIR (2024)
- **Paper**: "Jailbreaking LLM-Controlled Robots"
- **Implementation**: `RoboGuardBridge` with buddy system verification
- **Website**: https://robopair.org

### 2. KnowNo (2023)
- **Paper**: "Robots That Ask For Help"
- **Implementation**: `KnowNoAdapter` with conformal prediction
- **GitHub**: google-research/language_model_uncertainty

### 3. IntroPlan (2024, NeurIPS)
- **Paper**: "Introspective Planning: Aligning Robots' Uncertainty with Inherent Task Ambiguity"
- **Implementation**: `IntroPlanAdapter` with KB retrieval and introspection
- **Website**: https://introplan.github.io

## 💡 Usage Scenarios

### Scenario 1: Basic Safety Verification

```python
# For standard plans with known safety rules
config = Config(alpha=0.1, use_introspection=False)
kd = EnhancedKnowDanger(config)
assessment = kd.run(scene, plan)
```

### Scenario 2: High-Stakes Operations

```python
# For safety-critical operations requiring maximum verification
config = Config(
    alpha=0.05,  # 95% confidence
    use_introspection=True,
    aggregation_strategy="conservative"
)
kd = EnhancedKnowDanger(config)
assessment = kd.run(scene, plan)
```

### Scenario 3: Ambiguous Commands

```python
# When user commands are ambiguous
config = Config(
    alpha=0.1,
    use_introspection=True,
    ask_threshold_confidence=0.6  # Ask more often
)
kd = EnhancedKnowDanger(config)
# System will identify ambiguity and request clarification
```

### Scenario 4: Plan Refinement

```python
# When initial plans might be unsafe
config = Config(use_introspection=True)
kd = EnhancedKnowDanger(config)
assessment = kd.run_with_rewriting(scene, plan, max_iterations=5)
# System iteratively refines until safe
```

### Scenario 5: Research Benchmarking

```python
# For comparing different configurations
from integration_utils import MetricsCollector, evaluate_with_logging

configs = [
    Config(use_introspection=False),  # RG+KN baseline
    Config(use_introspection=True),   # Full system
]

for i, config in enumerate(configs):
    kd = EnhancedKnowDanger(config)
    assessments, metrics = evaluate_with_logging(
        kd, scene, plans, f"logs/config_{i}"
    )
    print(f"Config {i}: Success={metrics.success_rate():.2%}")
```

## 🔧 Integration Points

```python
# Your existing RoboGuard setup works automatically
from roboguard import compile_specs  # Your installation
# EnhancedKnowDanger finds and uses it automatically

# Your existing KnowNo calibration data works
calibration_data = load_your_data()  # Your format
kd.calibrate_knowno(calibration_data)

# Your IntroPlan knowledge base is used
config = Config(introplan_kb_path="IntroPlan/data/kb.txt")
```

### With Test Suite

```python
# Your existing tests continue to work
from knowdanger_enhanced import EnhancedKnowDanger as KnowDanger

# Add new tests for IntroPlan
def test_introspection():
    config = Config(use_introspection=True)
    kd = KnowDanger(config)
    # Test introspective features
```

## 📈 Performance

### Optimization Tips

1. Disable introspection for non-critical plans
2. Use smaller knowledge bases (top-100 examples)
3. Cache frequently-used introspective reasoning
4. Run calibration offline

## 🧪 Testing

### Run All Examples

```bash
python examples/example_usage.py
```

### Run Your Existing Tests

```bash
pytest tests/
```

### Benchmark Performance

```bash
python tests/benchmark_knowno_roboguard.py
```

## 🐛 Troubleshooting

### Common Issues

**Q: "ModuleNotFoundError: No module named 'roboguard'"**

A: Add RoboGuard to Python path:
```python
import sys
sys.path.insert(0, "RoboGuard/src")
```

**Q: "IntroPlan knowledge base not found"**

A: Check path and use absolute path:
```python
from pathlib import Path
kb_path = Path("IntroPlan/data/kb.txt").absolute()
config = Config(introplan_kb_path=str(kb_path))
```

**Q: "Calibration not converging"**

A: Ensure sufficient calibration data:
```python
cal_data = CalibrationHelper.load_calibration_data("data.csv")
print(f"Loaded {len(cal_data)} examples")  # Need 100+
```

### Getting Help

1. Check INTEGRATION_GUIDE.md troubleshooting section
2. Review inline code comments  
3. Look at example_usage.py for working examples
4. Open GitHub issue with minimal reproducible example


## 📄 Citation

If you use this integration in your research:

```bibtex
@phdthesis{bodrova2024asimovbox,
  title={Asimov Box: Robust Safety Verification for LLM-Controlled Robots},
  author={Bodrova, Alexandra},
  year={2025},
  school={Princeton University}
}
```

## 🤝 Contributing

Future improvements could include:

- Parallel execution of three systems
- Better IntroPlan confidence calibration
- Automated knowledge base construction
- Additional safety metrics
- Integration with more robotics frameworks


---

*Integration of RoboPAIR (2024) + KnowNo (2023) + IntroPlan (2024, NeurIPS)*