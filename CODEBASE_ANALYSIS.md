# Asimov Box (KnowDanger) - Comprehensive Codebase Analysis

**Generated:** 2025-11-18  
**Repository:** https://github.com/alexandrabodrova/asimov_box

---

## Executive Summary

This repository implements **Asimov Box**, an integrated safety verification system for LLM-controlled robots that combines three state-of-the-art safety systems:

1. **RoboGuard** - Rule-based safety verification using temporal logic
2. **KnowNo** - Uncertainty quantification using conformal prediction  
3. **IntrospectivePlan (IntroPlan)** - Introspective reasoning with explanation generation

The system acts as a "guard dog" that intercepts LLM-generated plans and performs three-way verification before execution.

---

## System 1: RoboGuard

### Purpose & Functionality
**RoboGuard** is a rule-based safety verification system that:
- Checks if planned actions violate explicitly defined safety rules
- Uses temporal logic to express complex spatial and temporal constraints
- Provides binary SAFE/UNSAFE verdicts based on rule compliance
- Prevents adversarial or harmful actions through syntactic rule verification

### Key Concepts
- **Safety Rules**: Expressed in temporal logic (e.g., `G( place(x, s) & flammable(x) -> !near(s, hot_surface) )`)
- **Scene Graphs**: Semantic representation of the environment with objects, locations, and properties
- **Action Syntax**: Structured format for robot actions with parameters

### File Locations

**Main Implementation:**
```
/home/user/asimov_box/KnowDanger/RoboGuard/
├── src/roboguard/
│   ├── __init__.py
│   ├── roboguard.py          # Core RoboGuard implementation
│   ├── generator.py           # Rule/spec generation from natural language
│   ├── synthesis.py           # Control synthesis from specs
│   └── eval/
│       └── eval.py            # Evaluation utilities
└── setup.py
```

**Adapter for Integration:**
```
/home/user/asimov_box/KnowDanger/src/knowdanger/adapters/
├── roboguard_adapter.py       # Main adapter (100+ lines)
├── paper_roboguard.py         # Paper-faithful implementation
└── rg_adapter.py              # Alternative adapter
```

### Main Interface Classes

**`RoboGuardAdapter`** (`roboguard_adapter.py`):
```python
class RoboGuardAdapter:
    def fit(self, scene_graph: Dict[str, Any], rules: Sequence[str]) -> int
        # Compile rules against scene graph
        
    def evaluate_plan(self, plan: List[Action]) -> List[RGResult]
        # Verify each action against compiled specs
        
    def evaluate_step(self, step: Step) -> Verdict
        # Single step verification
```

**Data Structure:**
```python
@dataclass
class RGResult:
    ok: bool                              # SAFE=True, UNSAFE=False
    per_step: List[Tuple[str, bool]]     # Per-step verdicts
    raw: Any                             # Raw RoboGuard output
```

### Integration Points
- Dynamically imports upstream `roboguard` module
- Accepts `Scene.semantic_graph` and `Scene.rules` as input
- Returns verdicts compatible with KnowDanger aggregator
- Used as first check in three-way verification pipeline

---

## System 2: KnowNo

### Purpose & Functionality
**KnowNo** is an uncertainty quantification system that:
- Quantifies confidence in LLM predictions using conformal prediction
- Determines when the system should ask for human help vs. proceeding
- Handles ambiguous or low-confidence situations gracefully
- Provides prediction sets (multiple plausible options) instead of single choices

### Key Concepts
- **Conformal Prediction**: Distribution-free uncertainty quantification (maintains 1-α coverage)
- **Prediction Sets**: Set of candidate actions with quantified uncertainty
- **Calibration**: Learns threshold `tau` from calibration data
- **Ask-for-Help**: Transitions to UNCERTAIN when confidence is below threshold

### File Locations

**Main Implementation:**
```
/home/user/asimov_box/KnowDanger/src/lang_help/
├── knowno/
│   ├── __init__.py
│   └── api.py                # Core KnowNo API (98 lines)
└── __init__.py

/home/user/asimov_box/KnowDanger/src/known/
├── knowno/
│   ├── __init__.py
│   └── api.py                # Alternative KnowNo implementation
└── __init__.py
```

**Adapter & Integration:**
```
/home/user/asimov_box/KnowDanger/src/knowdanger/adapters/
├── paper_knowno.py           # Paper-faithful adapter (150+ lines)
└── (integrated in knowdanger_core.py)
```

**Calibration Scripts:**
```
/home/user/asimov_box/KnowDanger/src/scripts/calibration_knowno/
├── calibrate_knowno.py       # Main calibration pipeline
├── compute_calibration.py    # Compute calibration metrics
├── runtime.py                # Runtime integration
└── sandbox/                  # Data collection utilities
```

### Main Interface Classes

**`paper_knowno.py` - ChoiceBaseline:**
```python
class ChoiceBaseline:
    def calibrate(self, score_sets: List[List[float]], alpha: float) -> float
        # Returns tau threshold for conformal prediction
        
    def predict_set(self, scores: List[float]) -> List[int]
        # Returns indices of prediction set
```

**`lang_help/knowno/api.py`:**
```python
def calibrate(alpha: float, score_sets: List[List[float]]) -> float:
    # Compute tau using upstream CP class or fallback quantile

def predict_set(scores: List[float], tau: float, alpha: float) -> List[int]:
    # Return prediction set indices
```

**Data Flow:**
```python
# Step candidates: List[Tuple[str, float]]
candidates = [
    ("pick_safe", 0.8),
    ("pick_unsafe", 0.2)
]
scores = [0.8, 0.2]

# Compute prediction set
pred_set = predict_set(scores, tau=0.7)  # Returns [0] if |set|=1, else UNCERTAIN
```

### Integration Points
- Requires calibration data from environment
- Uses LLM logits/probabilities from `Step.candidates`
- Integrates with conformal prediction for uncertainty bounds
- In three-way system: checks if action is in high-confidence prediction set

---

## System 3: IntrospectivePlan (IntroPlan)

### Purpose & Functionality
**IntroPlan** is an introspective planning system that:
- Uses LLMs to reflect on their own uncertainty and reasoning
- Retrieves similar examples from knowledge base for context
- Generates explanations for safety decisions
- Combines introspection with conformal prediction for refined confidence bounds
- Handles task ambiguity by asking for clarification when needed

### Key Concepts
- **Introspective Reasoning**: LLM reflects on uncertainty in its own planning
- **Knowledge Base Retrieval**: Find similar past examples to inform decisions
- **Explanation Generation**: Natural language justifications for actions
- **Ambiguity Detection**: Identifies when user intent is unclear
- **Refinement**: Iteratively improves unsafe plans

### File Locations

**Main Implementation:**
```
/home/user/asimov_box/KnowDanger/IntroPlan/
├── IntroPlan_Mobile.ipynb                  # Direct prediction notebook
├── IntroPlan_CP_Mobile.ipynb               # Conformal prediction version
├── IntroPlan_CP_Safe_Mobile.ipynb          # Safety-aware version
├── IntroPlan_Safe_Mobile.ipynb             # Safe navigation version
├── Llama_IntroPlan_Mobile.ipynb            # Llama-3 variant
│
├── llm.py                     # LLM API calls (OpenAI/Llama interface) ~100 lines
├── utils.py                   # Utility functions ~300 lines
├── prompt_init.py             # Prompt initialization ~150 lines
├── cp_utils.py                # Conformal prediction utilities ~150 lines
├── metrics.py                 # Evaluation metrics ~300 lines
├── process_results.py         # Results processing
│
├── data/                      # Knowledge base data
│   ├── mobile_manipulation.txt
│   ├── mobile_manipulation_knowledge.txt
│   ├── safe_mobile_knowledge.txt
│   └── safe_mobile_test.txt
│
├── LICENSE
├── README.md
└── requirements.txt
```

**Adapter for Integration:**
```
/home/user/asimov_box/KnowDanger/src/knowdanger/adapters/
└── introplan_adapter.py       # Main adapter (400+ lines)
```

### Main Interface Classes

**`IntroPlanAdapter`** (`introplan_adapter.py`):
```python
@dataclass
class IntrospectiveReasoning:
    explanation: str                       # Natural language explanation
    confidence_scores: Dict[str, float]   # Per-action confidence
    safety_assessment: str                # Safety evaluation
    compliance_assessment: str            # Rule compliance check
    recommended_action: Optional[str]     # Suggested safe action
    should_ask_clarification: bool        # Whether to ask user
    reasoning_chain: List[str]           # Step-by-step reasoning
    meta: Dict[str, Any]                 # Additional metadata

class IntroPlanAdapter:
    def __init__(
        self,
        knowledge_base_path: Optional[str] = None,
        introplan_root: Optional[str] = None,
        use_conformal: bool = True,
        retrieval_k: int = 3
    ):
        # Initialize with optional KB and IntroPlan modules
    
    def load_knowledge_base(self, path: str) -> int:
        # Load KB from JSON or TXT file
    
    def generate_introspective_reasoning(
        self,
        task: str,
        scene: Scene,
        candidates: List[Tuple[str, float]]
    ) -> IntrospectiveReasoning:
        # Generate introspective reasoning via LLM
    
    def integrate_with_conformal_prediction(
        self,
        reasoning: IntrospectiveReasoning,
        cp_set: List[int],
        candidates: List[Tuple[str, float]],
        alpha: float
    ) -> Tuple[List[int], Dict[str, Any]]:
        # Combine introspection with CP for refined bounds
    
    def retrieve_similar_examples(
        self,
        task: str,
        scene: Scene,
        k: int = 3
    ) -> List[KnowledgeEntry]:
        # Retrieve top-k similar examples from KB
    
    def add_knowledge_entry(self, entry: KnowledgeEntry):
        # Add new entry to knowledge base
    
    def save_knowledge_base(self, path: str):
        # Export KB to file
```

**Data Structures:**
```python
@dataclass
class KnowledgeEntry:
    task_description: str           # Task description
    scene_context: str             # Environment context
    correct_option: str            # Correct/safe action
    introspective_reasoning: str   # Explanation
    safety_considerations: List[str]
    meta: Dict[str, Any]
```

### Integration Points
- Dynamically finds IntroPlan repository (checks multiple locations)
- Loads knowledge base from files (JSON or TXT format)
- Calls external LLM for introspective reasoning
- Integrates with conformal prediction for uncertainty quantification
- In three-way system: checks safety reasoning and provides explanations

---

## System Integration: The Orchestrator

### Enhanced KnowDanger (`knowdanger_enhanced.py`)

The orchestrator that coordinates all three systems:

```
User Command
    ↓
LLM Planner → Generates Plan
    ↓
┌────────────────────────────────┐
│   EnhancedKnowDanger           │
│   (Asimov Box Orchestrator)    │
│                                │
│  Step 1: RoboGuard Check       │  ← Rules-based (SAFE/UNSAFE)
│  Step 2: KnowNo Check          │  ← Uncertainty (SAFE/UNCERTAIN)
│  Step 3: IntroPlan Check       │  ← Reasoning (SAFE/UNSAFE/UNCERTAIN)
│  Step 4: Aggregation           │  ← Combine verdicts
│                                │
└────────────────────────────────┘
    ↓
Final Verdict: SAFE / UNSAFE / UNCERTAIN
    ↓
Execute (if SAFE) / Ask Human (if UNCERTAIN) / Block (if UNSAFE)
```

**Main Class:**
```python
class EnhancedKnowDanger:
    def __init__(self, config: Config):
        # Initialize all three system adapters
        self.rg = RoboGuardBridge()      # Rules-based
        self.kn = KnowNoAdapter()         # Uncertainty
        self.ip = IntroPlanAdapter()      # Reasoning
    
    def run(self, scene: Scene, plan: PlanCandidate) -> PlanAssessment:
        # Evaluate plan with all three systems
        
    def run_with_rewriting(
        self,
        scene: Scene,
        plan: PlanCandidate,
        max_iterations: int = 3
    ) -> PlanAssessment:
        # Iteratively refine unsafe plans
```

### Aggregation Strategies

Three verdict aggregation methods:

1. **Conservative** (Default):
   - ANY UNSAFE → UNSAFE
   - ALL SAFE → SAFE  
   - Otherwise → UNCERTAIN

2. **Majority**:
   - Democratic voting among systems
   - Majority label wins

3. **Weighted**:
   - Confidence-weighted combination
   - Adjustable per-system weights

### Configuration

```python
@dataclass
class Config:
    alpha: float = 0.1                      # CP confidence level
    ask_threshold_confidence: float = 0.7   # KnowNo threshold
    
    use_introspection: bool = True          # Enable IntroPlan
    introplan_kb_path: Optional[str] = None # Knowledge base path
    introplan_retrieval_k: int = 3          # Similar examples to retrieve
    
    aggregation_strategy: str = "conservative"  # Aggregation method
    llm_func: Optional[Callable] = None     # Custom LLM function
```

---

## Data Structures & Interfaces

### Common Data Classes

All systems use compatible data structures:

```python
@dataclass
class Step:
    action: str                              # Action name
    params: Dict[str, Any]                  # Parameters
    candidates: Optional[List[Tuple[str, float]]]  # LLM alternatives
    meta: Dict[str, Any] = {}              # Additional metadata

@dataclass
class PlanCandidate:
    name: str
    steps: List[Step]
    user_prompt: str
    meta: Dict[str, Any] = {}

@dataclass
class Scene:
    name: str
    semantic_graph: Dict[str, Any]         # Objects, locations, properties
    rules: List[str]                       # Safety rules
    env_params: Dict[str, Any] = {}       # Environment parameters
    helpers: Dict[str, Callable] = {}      # Helper functions

@dataclass
class Verdict:
    label: Label                           # "SAFE" | "UNSAFE" | "UNCERTAIN"
    why: str                               # Explanation
    details: Dict[str, Any] = {}          # System-specific details

@dataclass
class StepAssessment:
    step: Step
    roboguard: Verdict                     # RoboGuard verdict
    knowno: Optional[Verdict]              # KnowNo verdict
    introplan: Optional[Verdict]           # IntroPlan verdict (NEW)
    final: Verdict                         # Aggregated verdict

@dataclass
class PlanAssessment:
    plan: PlanCandidate
    steps: List[StepAssessment]
    overall: Verdict
    meta: Dict[str, Any] = {}
```

---

## Directory Structure & Organization

```
/home/user/asimov_box/
└── KnowDanger/                          # Main project root
    ├── README.md                        # Project overview
    ├── pyproject.toml                   # Python package config
    ├── environment.yml                  # Conda environment
    ├── pyproject.toml                   # Build config
    │
    ├── RoboGuard/                       # Rule-based safety (empty placeholder in this repo)
    │   ├── src/roboguard/               # Full RoboGuard implementation
    │   ├── setup.py
    │   └── README.md
    │
    ├── IntroPlan/                       # Introspective planning system
    │   ├── *.ipynb                      # Jupyter notebooks (7 variants)
    │   ├── llm.py                       # LLM interface
    │   ├── utils.py                     # Utilities
    │   ├── cp_utils.py                  # Conformal prediction utilities
    │   ├── metrics.py                   # Evaluation metrics
    │   ├── prompt_init.py              # Prompt templates
    │   ├── data/                        # Knowledge base files
    │   └── README.md
    │
    ├── SPINE/                           # Mapping and navigation system (separate)
    │   ├── src/spine/                   # SPINE implementation
    │   ├── ros/                         # ROS integration
    │   └── setup.py
    │
    ├── src/                             # Main source code
    │   ├── knowdanger/                  # Core KnowDanger system
    │   │   ├── IMPLEMENTATION_SUMMARY.md
    │   │   ├── INTEGRATION_GUIDE.md
    │   │   ├── MIGRATION_GUIDE.md
    │   │   │
    │   │   ├── core/                    # Core implementations
    │   │   │   ├── knowdanger_core.py   # Original core (400+ lines)
    │   │   │   ├── knowdanger_enhanced.py   # Enhanced with IntroPlan (400+ lines)
    │   │   │   ├── integration_utils.py     # Utilities (400+ lines)
    │   │   │   ├── example_usage.py        # 6 usage examples
    │   │   │   ├── runtime_safety.py
    │   │   │   ├── knowledge_base.json
    │   │   │   └── *.py                 # Various implementations
    │   │   │
    │   │   ├── adapters/                # System adapters
    │   │   │   ├── roboguard_adapter.py    # RoboGuard bridge (300+ lines)
    │   │   │   ├── introplan_adapter.py    # IntroPlan bridge (400+ lines)
    │   │   │   ├── paper_knowno.py         # KnowNo bridge (150+ lines)
    │   │   │   ├── paper_roboguard.py
    │   │   │   └── rg_adapter.py
    │   │   │
    │   │   ├── calibration/             # KnowNo calibration
    │   │   │   └── mcqa/
    │   │   │       ├── compute_calibration_kd.py
    │   │   │       ├── kd_calibrate.py
    │   │   │       ├── scorer_knowno.py
    │   │   │       ├── query_lm.py
    │   │   │       └── README.txt
    │   │   │
    │   │   ├── compat_knowno.py        # KnowNo compatibility
    │   │   ├── config.py               # Configuration utilities
    │   │   └── knowno.yaml             # KnowNo config
    │   │
    │   ├── lang_help/                   # KnowNo interface
    │   │   └── knowno/
    │   │       ├── __init__.py
    │   │       └── api.py               # Main API (100 lines)
    │   │
    │   ├── known/                       # Alternative KnowNo
    │   │   └── knowno/
    │   │       └── api.py
    │   │
    │   ├── scripts/                     # Utility scripts
    │   │   ├── calibration_knowno/      # Calibration pipeline
    │   │   │   ├── calibrate_knowno.py
    │   │   │   ├── compute_calibration.py
    │   │   │   ├── runtime.py
    │   │   │   ├── examples_loader.py
    │   │   │   └── sandbox/             # Data collection
    │   │   └── sanity_check_knowno.py
    │   │
    │   ├── scenes/                      # Example scenarios
    │   │   ├── example1_hazard_lab.py   # Lab safety scene
    │   │   ├── example2_breakroom.py    # Breakroom scene
    │   │   ├── example3_photonics.py    # Photonics scene
    │   │   └── results_logger.py
    │   │
    │   └── tests/                       # Evaluation scripts
    │       ├── benchmark_knowno_roboguard.py   # Main benchmark
    │       ├── benchmark_report.py
    │       ├── benchmark_true_baselines.py
    │       ├── demo_spot_roboguard.py
    │       ├── knowdanger_vs_baselines.py
    │       ├── roboguard_paper_bench.py
    │       └── logs/                    # Test results
    │
    ├── configs/                         # Configuration files
    │   └── knowno_cfg.json
    │
    ├── lang-help/                       # Utilities
    └── scripts_api/                     # API scripts
```

---

## Current Integration Status

### What's Integrated ✓

1. **RoboGuard + KnowNo** (Original):
   - Both systems fully functional
   - Bidirectional verdicts with aggregation
   - Used in `knowdanger_core.py`

2. **RoboGuard + KnowNo + IntroPlan** (Enhanced):
   - Full three-way integration
   - New `knowdanger_enhanced.py` orchestrator
   - IntroPlanAdapter with KB support
   - Three aggregation strategies
   - Iterative plan refinement

### Integration Points

```python
# Entry point for all three systems
from knowdanger.core.knowdanger_enhanced import EnhancedKnowDanger, Config

# Create unified configuration
config = Config(
    alpha=0.1,                                    # KnowNo CP level
    use_introspection=True,                       # Enable IntroPlan
    introplan_kb_path="IntroPlan/data/kb.txt",  # Knowledge base
    aggregation_strategy="conservative"           # Verdict combination
)

# Run integrated system
kd = EnhancedKnowDanger(config)
assessment = kd.run(scene, plan)

# Access per-system verdicts
for step_assess in assessment.steps:
    print(f"RoboGuard: {step_assess.roboguard.label}")
    print(f"KnowNo: {step_assess.knowno.label}")
    print(f"IntroPlan: {step_assess.introplan.label}")
    print(f"Final: {step_assess.final.label}")
```

### Key Files for Integration

| File | Purpose | Lines |
|------|---------|-------|
| `knowdanger_enhanced.py` | Main orchestrator | 400+ |
| `introplan_adapter.py` | IntroPlan bridge | 400+ |
| `integration_utils.py` | Utilities & helpers | 400+ |
| `example_usage.py` | 6 working examples | 300+ |
| `roboguard_adapter.py` | RoboGuard bridge | 300+ |
| `paper_knowno.py` | KnowNo bridge | 150+ |
| `lang_help/knowno/api.py` | KnowNo API | 100+ |

---

## Main Entry Points & Interfaces

### For Users

**Option 1: Original System (RoboGuard + KnowNo)**
```python
from knowdanger.core.knowdanger_core import KnowDanger, Config

config = Config(alpha=0.1)
kd = KnowDanger(config)
assessment = kd.run(scene, plan)
```

**Option 2: Enhanced System (All Three)**
```python
from knowdanger.core.knowdanger_enhanced import EnhancedKnowDanger, Config

config = Config(alpha=0.1, use_introspection=True)
kd = EnhancedKnowDanger(config)
assessment = kd.run(scene, plan)

# With refinement
assessment = kd.run_with_rewriting(scene, plan, max_iterations=3)
```

### For Developers

**Extending RoboGuard:**
```python
from knowdanger.adapters.roboguard_adapter import RoboGuardAdapter

rg = RoboGuardAdapter(rules=scene.rules)
rg.fit(scene.semantic_graph, scene.rules)
verdict = rg.evaluate_step(step)
```

**Extending KnowNo:**
```python
from lang_help.knowno import api as knowno_api

tau = knowno_api.calibrate(alpha=0.1, score_sets=calibration_data)
pred_set = knowno_api.predict_set(scores, tau)
```

**Extending IntroPlan:**
```python
from knowdanger.adapters.introplan_adapter import IntroPlanAdapter

adapter = IntroPlanAdapter(knowledge_base_path="kb.json")
reasoning = adapter.generate_introspective_reasoning(task, scene, candidates)
refined_set = adapter.integrate_with_conformal_prediction(
    reasoning, cp_set, candidates, alpha
)
```

---

## Testing & Benchmarking

### Test Files
```
src/tests/
├── benchmark_knowno_roboguard.py     # Compare RoboGuard vs KnowNo
├── benchmark_true_baselines.py       # Baseline comparisons
├── benchmark_report.py               # Report generation
├── demo_spot_roboguard.py           # Spot robot demo
├── roboguard_paper_bench.py         # Paper benchmarks
└── logs/                            # Results storage
```

### Example Scenes
```
src/scenes/
├── example1_hazard_lab.py    # Lab with chemical hazards
├── example2_breakroom.py     # Breakroom navigation
└── example3_photonics.py     # Photonics equipment
```

---

## Key Features & Capabilities

### Per-System Features

**RoboGuard:**
- Compile temporal logic rules
- Scene-aware safety checking
- Binary safe/unsafe verdicts
- Rule violation explanations

**KnowNo:**
- Conformal prediction calibration
- Prediction set computation
- Confidence-based filtering
- Ask-for-help thresholding

**IntroPlan:**
- Introspective reasoning
- Knowledge base retrieval
- Explanation generation
- Ambiguity detection
- Plan refinement suggestions

### Integrated System Features

- **Three-Way Verification**: Every action checked by all three systems
- **Aggregation Strategies**: Conservative, majority, or weighted voting
- **Iterative Refinement**: Automatically improve unsafe plans
- **Explainability**: Natural language explanations for all decisions
- **Learning from Feedback**: Build knowledge base from experience
- **Backward Compatibility**: Works with existing KnowDanger code
- **Format Conversion**: Utilities to convert between system formats
- **Comprehensive Logging**: Automatic metrics and result tracking

---

## Configuration & Customization

### Configuration Options

```python
# Full configuration
config = Config(
    # KnowNo parameters
    alpha=0.1,                          # Confidence level (higher = more confident)
    ask_threshold_confidence=0.7,       # When to ask for help
    
    # IntroPlan parameters
    use_introspection=True,             # Enable/disable
    introplan_kb_path="path/to/kb",    # Knowledge base location
    introplan_retrieval_k=3,           # Similar examples to retrieve
    introspection_weight=0.5,          # Weight in aggregation
    
    # Integration
    aggregation_strategy="conservative", # conservative|majority|weighted
    llm_func=None,                     # Custom LLM function
)
```

### Environment Variables

```bash
# KnowNo upstream root (if using upstream package)
export KNOWNO_ROOT=/path/to/knowno

# OpenAI API key (for IntroPlan LLM calls)
export OPENAI_API_KEY=sk-...
```

---

## Documentation Files

### In Repository
- **README.md** (main) - Project overview and quick start
- **IMPLEMENTATION_SUMMARY.md** - Summary of what was built
- **INTEGRATION_GUIDE.md** - Complete integration guide (15 KB)
- **MIGRATION_GUIDE.md** - How to migrate from old to new code
- **IntroPlan/README.md** - IntroPlan-specific documentation
- **RoboGuard/README.md** - RoboGuard information

### External References
- **RoboGuard Paper**: https://robopair.org
- **KnowNo Paper**: Google Research language_model_uncertainty
- **IntroPlan Paper**: https://introplan.github.io (NeurIPS 2024)

---

## Key Metrics & Evaluation

### Available Metrics

```python
from knowdanger.core.integration_utils import MetricsCollector

collector = MetricsCollector()
for plan in plans:
    assessment = kd.run(scene, plan)
    collector.update_from_assessment(assessment)

summary = collector.get_summary()
# {
#   "success_rate": 0.85,           # SAFE / total
#   "help_rate": 0.10,              # UNCERTAIN / total
#   "safety_violation_rate": 0.05,  # UNSAFE / total
#   "roboguard_blocks": 3,
#   "knowno_uncertainties": 7,
#   "introplan_clarifications": 5
# }
```

---

## Known Limitations & Future Improvements

### Current Limitations
- IntroPlan requires OpenAI API (GPT-3.5/GPT-4)
- Knowledge base must be manually curated initially
- RoboGuard empty in this repo (uses external package)
- Limited Llama-3 support (see IntroPlan README)

### Planned Improvements
- Parallel execution of three systems
- Better IntroPlan confidence calibration
- Automated knowledge base construction
- Additional safety metrics
- Integration with more robotics frameworks
- Support for additional LLM providers

---

## Getting Started

### Quick Installation

```bash
# Clone and setup
git clone https://github.com/alexandrabodrova/asimov_box
cd asimov_box/KnowDanger

# Create environment
conda env create -f environment.yml
conda activate knowdanger_venv-311

# Install
pip install -e .

# Run examples
python src/knowdanger/core/example_usage.py
```

### First Integration Step

```python
# 1. Basic setup
from knowdanger.core.knowdanger_enhanced import EnhancedKnowDanger, Config
config = Config(alpha=0.1, use_introspection=False)  # Start simple
kd = EnhancedKnowDanger(config)

# 2. Define your scene
from knowdanger.core.knowdanger_core import Scene
scene = Scene(
    name="my_lab",
    semantic_graph={"objects": [...], "locations": [...]},
    rules=["safety rule 1", "safety rule 2"]
)

# 3. Create a plan
plan = PlanCandidate(name="test", steps=[...], user_prompt="...")

# 4. Evaluate
assessment = kd.run(scene, plan)

# 5. Add IntroPlan later
config = Config(alpha=0.1, use_introspection=True, 
                introplan_kb_path="my_kb.json")
# ... and repeat
```

---

## Summary Table

| Aspect | RoboGuard | KnowNo | IntroPlan |
|--------|-----------|--------|-----------|
| **Type** | Rule-based | Uncertainty | Reasoning-based |
| **Input** | Rules + Scene | LLM scores | Task + Context |
| **Output** | SAFE/UNSAFE | SAFE/UNCERTAIN | SAFE/UNSAFE/UNCERTAIN |
| **Verification** | Temporal logic | Conformal pred. | Introspection |
| **Calibration** | None | Required | Optional |
| **Explanation** | Rule violations | Confidence | Natural language |
| **Delay** | Low | Low | Medium |
| **Location** | `adapters/roboguard_adapter.py` | `lang_help/knowno/api.py` | `adapters/introplan_adapter.py` |
| **Lines of Code** | 300+ | 100+ | 400+ |

---

## Contact & Citation

**Repository**: https://github.com/alexandrabodrova/asimov_box  
**Author**: Alexandra Bodrova (Princeton University)

If you use this integration in research, cite:

```bibtex
@phdthesis{bodrova2024asimovbox,
  title={Asimov Box: Robust Safety Verification for LLM-Controlled Robots},
  author={Bodrova, Alexandra},
  year={2025},
  school={Princeton University}
}
```

