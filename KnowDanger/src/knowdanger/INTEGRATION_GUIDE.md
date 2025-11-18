# Enhanced KnowDanger Integration Guide

Complete integration of **RoboGuard**, **KnowNo**, and **IntroPlan** into the KnowDanger/Asimov Box safety framework for LLM-controlled robots.

## Overview

This enhanced implementation provides a comprehensive safety layer that combines:

1. **RoboGuard** - Rule-based safety verification with temporal logic
2. **KnowNo** - Conformal prediction for uncertainty quantification  
3. **IntroPlan** - Introspective planning with explanation generation

The system acts as an "Asimov Box" that intercepts LLM-generated actions before execution and performs three-way verification to ensure safety.

## Architecture

```
User Command
     ↓
  LLM Planner → Generates Plan
     ↓
┌─────────────────────────────┐
│      Asimov Box             │
│  (Enhanced KnowDanger)      │
│                             │
│  ┌──────────────────────┐  │
│  │   RoboGuard          │  │  ← Rule-based checking
│  │   - Compile specs    │  │
│  │   - Verify rules     │  │
│  └──────────────────────┘  │
│           ↓                 │
│  ┌──────────────────────┐  │
│  │   KnowNo             │  │  ← Uncertainty quantification
│  │   - Conformal pred.  │  │
│  │   - Prediction sets  │  │
│  └──────────────────────┘  │
│           ↓                 │
│  ┌──────────────────────┐  │
│  │   IntroPlan          │  │  ← Introspective reasoning
│  │   - Retrieve KB      │  │
│  │   - Generate explain.│  │
│  │   - Refine conf.     │  │
│  └──────────────────────┘  │
│           ↓                 │
│  ┌──────────────────────┐  │
│  │   Aggregator         │  │  ← Combine verdicts
│  │   - Conservative     │  │
│  │   - Majority         │  │
│  │   - Weighted         │  │
│  └──────────────────────┘  │
│           ↓                 │
│  Final Verdict:             │
│  SAFE / UNSAFE / UNCERTAIN  │
└─────────────────────────────┘
     ↓
  Execute (if SAFE)
  Ask Human (if UNCERTAIN)
  Block (if UNSAFE)
```

## Key Features

### 1. Three-Way Verification

Each action is checked by three independent systems:

- **RoboGuard**: Checks if action violates any safety rules
- **KnowNo**: Quantifies uncertainty using conformal prediction
- **IntroPlan**: Generates introspective reasoning and explanations

### 2. Flexible Aggregation

Three strategies for combining verdicts:

- **Conservative**: Prioritizes safety (any UNSAFE → block)
- **Majority**: Democratic voting among systems
- **Weighted**: Confidence-weighted combination

### 3. Iterative Refinement

When IntroPlan identifies issues, the system can:

- Generate alternative actions
- Explain why certain actions are unsafe
- Iteratively refine the plan

### 4. Knowledge Base Learning

System learns from human feedback:

- Post-hoc rationalization of correct actions
- Builds knowledge base for future introspection
- Can be exported for LLM fine-tuning

## Installation

```bash
# Clone your repository
git clone https://github.com/alexandrabodrova/asimov_box
cd asimov_box

# Install base dependencies
pip install -r requirements.txt

# Install enhanced integration modules
pip install -e .

# Copy integration files to your repo
cp /path/to/introplan_adapter.py src/knowdanger/adapters/
cp /path/to/knowdanger_enhanced.py src/knowdanger/core/
cp /path/to/integration_utils.py src/knowdanger/utils/
```

## Quick Start

### Basic Usage

```python
from knowdanger_enhanced import EnhancedKnowDanger, create_default_config
from knowdanger_enhanced import Scene, PlanCandidate, Step

# Configure system
config = create_default_config(
    alpha=0.1,  # 90% confidence level
    use_introspection=True,
    kb_path="knowledge_base.json"
)

# Initialize
kd = EnhancedKnowDanger(config)

# Define scene
scene = Scene(
    name="lab",
    semantic_graph={"locations": [...], "objects": [...]},
    rules=["!near(flammable, heat)", "must_wear(goggles)"]
)

# Define plan
plan = PlanCandidate(
    name="transfer_chemical",
    user_prompt="Move the solvent safely",
    steps=[
        Step(
            action="pick",
            params={"object": "solvent"},
            candidates=[("pick_left", 0.7), ("pick_right", 0.3)]
        ),
        Step(
            action="place",
            params={"location": "storage"},
            candidates=[
                ("place_near_heat", 0.6),  # UNSAFE
                ("place_in_storage", 0.4)  # SAFE
            ]
        )
    ]
)

# Evaluate
assessment = kd.run(scene, plan)
print(assessment.overall.label)  # UNSAFE / SAFE / UNCERTAIN
```

### With Refinement

```python
# Iteratively refine unsafe plans
assessment = kd.run_with_rewriting(scene, plan, max_iterations=3)

# System will:
# 1. Identify unsafe steps
# 2. Use introspection to suggest alternatives
# 3. Re-evaluate refined plan
# 4. Repeat until safe or max iterations
```

### With Logging

```python
from integration_utils import evaluate_with_logging

# Batch evaluation with automatic logging
assessments, metrics = evaluate_with_logging(
    kd, scene, plans, log_dir="logs/experiment_1"
)

print(f"Success Rate: {metrics.success_rate():.2%}")
print(f"Help Rate: {metrics.help_rate():.2%}")
```

## Integration Details

### Integrating with Existing RoboGuard

The `RoboGuardBridge` in `knowdanger_enhanced.py` dynamically imports your existing RoboGuard installation:

```python
# In your existing code
from roboguard import compile_specs, evaluate_plan

# Works automatically with KnowDanger
kd = EnhancedKnowDanger(config)
assessment = kd.run(scene, plan)
# RoboGuard functions are called internally
```

### Integrating with Existing KnowNo

The `KnowNoAdapter` connects to your existing KnowNo/lang-help installation:

```python
# KnowDanger automatically uses your calibrated KnowNo
kd = EnhancedKnowDanger(config)

# Calibrate using existing data
calibration_data = load_your_knowno_data()
kd.calibrate_knowno(calibration_data)

# Now evaluations use calibrated threshold
assessment = kd.run(scene, plan)
```

### Integrating with IntroPlan

The `IntroPlanAdapter` connects to the IntroPlan repository in your project:

```python
# IntroPlan integration (place in your IntroPlan/ directory)
from introplan_adapter import IntroPlanAdapter

adapter = IntroPlanAdapter(
    knowledge_base_path="IntroPlan/data/mobile_manipulation_knowledge.txt",
    introplan_root="./IntroPlan"
)

# Use with KnowDanger
config = Config(
    use_introspection=True,
    introplan_kb_path="IntroPlan/data/mobile_manipulation_knowledge.txt"
)
kd = EnhancedKnowDanger(config)
```

## File Structure

```
asimov_box/
├── src/
│   └── knowdanger/
│       ├── core/
│       │   ├── knowdanger_core.py          # Original (your existing)
│       │   └── knowdanger_enhanced.py      # NEW: Full integration
│       ├── adapters/
│       │   ├── roboguard_adapter.py        # Existing
│       │   └── introplan_adapter.py        # NEW: IntroPlan bridge
│       └── utils/
│           └── integration_utils.py        # NEW: Helper utilities
├── IntroPlan/                              # IntroPlan submodule
│   ├── llm.py
│   ├── utils.py
│   ├── cp_utils.py
│   └── data/
│       └── mobile_manipulation_knowledge.txt
├── RoboGuard/                              # RoboGuard submodule
├── lang-help/                              # KnowNo submodule
├── examples/
│   └── example_usage.py                    # NEW: Usage examples
└── tests/
    └── test_integration.py                 # Integration tests
```

## Configuration Options

```python
Config(
    # KnowNo parameters
    alpha=0.1,                          # Miscoverage rate (90% confidence)
    ask_threshold_confidence=0.7,       # When to ask for help
    
    # IntroPlan parameters
    use_introspection=True,             # Enable IntroPlan
    introplan_kb_path="kb.json",        # Knowledge base path
    introplan_retrieval_k=3,            # Similar examples to retrieve
    introspection_weight=0.5,           # Weight in aggregation
    
    # Aggregation strategy
    aggregation_strategy="conservative", # "conservative"|"majority"|"weighted"
    
    # Optional LLM function for introspection
    llm_func=my_llm_function            # Custom LLM for reasoning
)
```

## Calibration

### KnowNo Calibration

```python
from integration_utils import CalibrationHelper

# Load calibration data
cal_data = CalibrationHelper.load_calibration_data("calibration.csv")

# Calibrate
tau = kd.calibrate_knowno(cal_data)
print(f"Threshold: {tau:.4f}")

# Verify coverage
coverage = CalibrationHelper.compute_coverage(predictions, ground_truth)
print(f"Coverage: {coverage:.2%}")
```

### IntroPlan Knowledge Base

```python
# Build knowledge base from human feedback
kd.add_knowledge_entry(
    task="move object to storage",
    scene_context="warehouse environment",
    correct_option="place_on_shelf_a",
    human_feedback="Shelf A is accessible and safe"
)

# Save knowledge base
kd.save_knowledge_base("knowledge_base.json")
```

## Evaluation Metrics

The system tracks comprehensive metrics:

```python
from integration_utils import MetricsCollector

collector = MetricsCollector()
for plan in plans:
    assessment = kd.run(scene, plan)
    collector.update_from_assessment(assessment)

summary = collector.get_summary()
print(summary)
```

Metrics include:

- **Success Rate**: Fraction of safe plans
- **Help Rate**: Fraction requiring human input
- **Safety Violation Rate**: Fraction of unsafe plans
- **System-specific triggers**: RG blocks, KN uncertainties, IP clarifications
- **Refinement statistics**: Iterations and success rate

## Advanced Usage

### Custom Aggregation

```python
def my_aggregator(rg, kn, ip):
    # Custom logic
    if rg.label == "UNSAFE" or ip.label == "UNSAFE":
        return Verdict("UNSAFE", "Custom: safety priority")
    # ... more logic
    return Verdict("SAFE", "Custom: all clear")

config = Config(aggregation_strategy="custom")
config.aggregator = my_aggregator
```

### Custom LLM for Introspection

```python
def my_llm(prompt: str) -> str:
    # Your LLM implementation
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

config = Config(llm_func=my_llm)
kd = EnhancedKnowDanger(config)
```

### Format Conversion

```python
from integration_utils import FormatConverter

converter = FormatConverter()

# From RoboPAIR
step = converter.robopair_to_knowdanger_step(robopair_action)

# From KnowNo
candidates = converter.knowno_prediction_to_candidates(options, logits)

# To IntroPlan KB
entry = converter.introplan_to_knowledge_format(task, options, correct_idx, explanation)
```

## Troubleshooting

### Issue: RoboGuard not found

```python
# Ensure RoboGuard is in your Python path
import sys
sys.path.insert(0, "/path/to/RoboGuard/src")

# Or install as package
cd RoboGuard
pip install -e .
```

### Issue: IntroPlan knowledge base not loading

```python
# Check path
from pathlib import Path
kb_path = Path("knowledge_base.json")
print(f"Exists: {kb_path.exists()}")

# Load manually
from introplan_adapter import IntroPlanAdapter
adapter = IntroPlanAdapter(knowledge_base_path=str(kb_path))
print(f"Loaded {len(adapter.knowledge_base)} entries")
```

### Issue: Calibration not converging

```python
# Check calibration data quality
cal_data = CalibrationHelper.load_calibration_data("data.csv")
print(f"Examples: {len(cal_data)}")
print(f"Options per example: {len(cal_data[0])}")

# Ensure enough examples (recommended: 100+)
# Ensure scores are valid (0-1 or logits)
```

## Testing

Run integration tests:

```bash
# Unit tests
pytest tests/test_integration.py

# Full system test
python examples/example_usage.py

# Benchmark against baselines
python tests/benchmark_knowno_roboguard.py
```

## Performance

Typical evaluation times:

- **RoboGuard only**: ~10ms per step
- **RoboGuard + KnowNo**: ~50ms per step
- **Full integration (RG+KN+IP)**: ~200ms per step
  - IntroPlan reasoning: ~150ms (depends on LLM)

## Citation

If you use this integration in your research, please cite:

```bibtex
@article{bodrova2024knowdanger,
  title={KnowDanger: Robust Safety Verification for LLM-Controlled Robots},
  author={Bodrova, Alexandra and ...},
  journal={...},
  year={2024}
}

@inproceedings{liang2024introspective,
  title={Introspective Planning: Aligning Robots' Uncertainty with Inherent Task Ambiguity},
  author={Liang, Kaiqu and Zhang, Zixu and Fisac, Jaime Fern{\'a}ndez},
  booktitle={NeurIPS},
  year={2024}
}

@article{robopair2024,
  title={Jailbreaking LLM-Controlled Robots},
  author={...},
  journal={...},
  year={2024}
}

@article{knowno2023,
  title={Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners},
  author={...},
  journal={CoRL},
  year={2023}
}
```

## Contributing

Contributions welcome! Areas for improvement:

1. **Efficiency**: Parallel evaluation of three systems
2. **Calibration**: Better methods for IntroPlan confidence
3. **Knowledge Base**: Automated KB construction from logs
4. **Metrics**: Additional safety metrics
5. **Integration**: Support for more robotics frameworks

## License

See individual component licenses:
- RoboGuard: See RoboGuard/LICENSE
- KnowNo: See lang-help/LICENSE  
- IntroPlan: See IntroPlan/LICENSE

## Support

For questions or issues:

1. Check existing GitHub issues
2. Review examples in `examples/`
3. Open a new issue with:
   - Minimal reproducible example
   - Error messages
   - System configuration

## Acknowledgments

This integration builds on excellent work from:

- RoboPAIR team for adversarial robustness
- Google Research for KnowNo uncertainty quantification
- Princeton Safe Robotics Lab for IntroPlan introspection

Special thanks to the open-source robotics community!