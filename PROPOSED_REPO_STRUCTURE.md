# Proposed Repository Structure for Asimov Box

**Generated:** 2025-11-18
**Current Structure:** Mixed research code, examples, and libraries
**Proposed Structure:** Modular, maintainable, production-ready

---

## Current Structure Problems

### Issues with Current Layout

```
/home/user/asimov_box/
└── KnowDanger/                    # ❌ Everything nested under one project
    ├── RoboGuard/                 # ❌ Empty placeholder
    ├── IntroPlan/                 # ❌ Embedded subproject with own structure
    ├── SPINE/                     # ❌ Unrelated project mixed in
    ├── src/
    │   ├── knowdanger/            # ❌ Mixed with lang_help, known, scripts
    │   ├── lang_help/
    │   ├── known/
    │   ├── scenes/                # ❌ Examples mixed with source
    │   ├── scripts/               # ❌ Scripts mixed with source
    │   └── tests/                 # ❌ Tests mixed with source
    ├── configs/                   # ❌ Top-level and scattered configs
    └── [15+ documentation files]  # ❌ Docs scattered everywhere
```

**Problems:**

1. **Unclear Separation:** Research code, library code, examples all mixed
2. **Not Pip-Installable:** Can't easily `pip install asimov-box`
3. **Subproject Confusion:** IntroPlan is both embedded and standalone
4. **SPINE Pollution:** Unrelated mapping project in same repo
5. **Testing Issues:** Tests don't follow package structure
6. **Import Hell:** Relative imports, sys.path manipulation, brittle discovery
7. **Documentation Sprawl:** 15+ docs at different levels
8. **No Clear Entry Point:** Multiple ways to use the system

---

## Proposed Structure: Option 1 (Recommended)

### Monorepo with Clear Separation

```
asimov-box/                                  # Repository root
│
├── README.md                                # Main project README
├── LICENSE
├── .gitignore
├── pyproject.toml                           # Main project config
├── setup.py                                 # Installation script
│
├── docs/                                    # Centralized documentation
│   ├── index.md                            # Documentation hub
│   ├── getting-started.md                  # Quickstart guide
│   ├── architecture/
│   │   ├── overview.md                     # System architecture
│   │   ├── roboguard.md                    # RoboGuard details
│   │   ├── knowno.md                       # KnowNo details
│   │   └── introplan.md                    # IntroPlan details
│   ├── integration/
│   │   ├── setup.md                        # Integration setup
│   │   ├── configuration.md                # Config options
│   │   └── api-reference.md                # API docs
│   ├── tutorials/
│   │   ├── basic-usage.md
│   │   ├── building-knowledge-base.md
│   │   ├── calibration.md
│   │   └── custom-llm.md
│   └── research/
│       ├── gaps-analysis.md                # This document
│       └── proposed-structure.md           # This document
│
├── asimov/                                  # Main package (pip installable)
│   ├── __init__.py                         # Package init
│   ├── version.py                          # Version info
│   │
│   ├── core/                               # Core orchestration
│   │   ├── __init__.py
│   │   ├── config.py                       # Configuration classes
│   │   ├── types.py                        # Common types (Step, Scene, etc.)
│   │   ├── orchestrator.py                 # Main EnhancedKnowDanger class
│   │   └── aggregation.py                  # Verdict aggregation strategies
│   │
│   ├── adapters/                           # System adapters
│   │   ├── __init__.py
│   │   ├── base.py                         # Base adapter interface
│   │   ├── roboguard.py                    # RoboGuard adapter
│   │   ├── knowno.py                       # KnowNo adapter
│   │   └── introplan.py                    # IntroPlan adapter
│   │
│   ├── systems/                            # Actual system implementations
│   │   ├── __init__.py
│   │   │
│   │   ├── roboguard/                      # RoboGuard implementation
│   │   │   ├── __init__.py
│   │   │   ├── core.py                     # Core RG logic
│   │   │   ├── generator.py                # Spec generation
│   │   │   ├── synthesis.py                # Control synthesis
│   │   │   └── eval.py                     # Evaluation
│   │   │
│   │   ├── knowno/                         # KnowNo implementation
│   │   │   ├── __init__.py
│   │   │   ├── api.py                      # Main API
│   │   │   ├── conformal.py                # Conformal prediction
│   │   │   └── calibration.py              # Calibration utilities
│   │   │
│   │   └── introplan/                      # IntroPlan implementation
│   │       ├── __init__.py
│   │       ├── reasoning.py                # Introspective reasoning
│   │       ├── knowledge_base.py           # KB management
│   │       ├── llm.py                      # LLM interface
│   │       ├── prompts.py                  # Prompt templates
│   │       ├── conformal.py                # CP integration
│   │       └── utils.py                    # Utilities
│   │
│   ├── utils/                              # Shared utilities
│   │   ├── __init__.py
│   │   ├── format_converter.py
│   │   ├── metrics.py
│   │   ├── logging.py
│   │   └── validation.py
│   │
│   └── cli/                                # Command-line interface
│       ├── __init__.py
│       ├── main.py                         # Main CLI entry point
│       ├── calibrate.py                    # Calibration CLI
│       └── evaluate.py                     # Evaluation CLI
│
├── tests/                                   # Test suite (mirrors package structure)
│   ├── __init__.py
│   ├── conftest.py                         # Pytest config
│   ├── fixtures/                           # Test fixtures
│   │   ├── scenes.py
│   │   ├── plans.py
│   │   └── knowledge_bases.py
│   │
│   ├── unit/                               # Unit tests
│   │   ├── test_roboguard.py
│   │   ├── test_knowno.py
│   │   ├── test_introplan.py
│   │   └── test_aggregation.py
│   │
│   ├── integration/                        # Integration tests
│   │   ├── test_two_way.py                # RG + KN
│   │   ├── test_three_way.py              # RG + KN + IP
│   │   ├── test_orchestrator.py
│   │   └── test_plan_refinement.py
│   │
│   └── benchmarks/                         # Performance benchmarks
│       ├── benchmark_systems.py
│       └── benchmark_aggregation.py
│
├── examples/                                # Example usage
│   ├── README.md                           # Examples overview
│   ├── quickstart.py                       # Simple example
│   ├── custom_llm.py                       # Custom LLM integration
│   │
│   ├── scenes/                             # Example scenes
│   │   ├── __init__.py
│   │   ├── hazard_lab.py
│   │   ├── breakroom.py
│   │   └── photonics.py
│   │
│   ├── notebooks/                          # Jupyter notebooks
│   │   ├── 01_basic_usage.ipynb
│   │   ├── 02_knowledge_base.ipynb
│   │   ├── 03_calibration.ipynb
│   │   └── 04_custom_aggregation.ipynb
│   │
│   └── configs/                            # Example configurations
│       ├── conservative.yaml
│       ├── majority_vote.yaml
│       └── weighted.yaml
│
├── data/                                    # Data files
│   ├── knowledge_bases/                    # Starter KBs
│   │   ├── general_safety.json
│   │   ├── lab_safety.json
│   │   ├── navigation.json
│   │   └── mobile_manipulation.json
│   │
│   ├── calibration/                        # Sample calibration data
│   │   ├── lab_scenarios.json
│   │   └── navigation_scenarios.json
│   │
│   └── rules/                              # Common safety rules
│       ├── general.txt
│       └── lab_specific.txt
│
├── scripts/                                 # Standalone scripts
│   ├── calibrate_knowno.py                 # Calibration script
│   ├── build_knowledge_base.py             # KB construction
│   ├── run_benchmark.py                    # Benchmarking
│   └── validate_installation.py            # Post-install validation
│
├── research/                                # Research-specific code
│   ├── README.md                           # Research code overview
│   ├── introplan_notebooks/                # Original IntroPlan notebooks
│   │   ├── IntroPlan_CP_Mobile.ipynb
│   │   ├── IntroPlan_Safe_Mobile.ipynb
│   │   └── Llama_IntroPlan_Mobile.ipynb
│   │
│   └── experiments/                        # Experiment scripts
│       ├── compare_baselines.py
│       └── ablation_studies.py
│
└── .github/                                 # GitHub-specific files
    ├── workflows/
    │   ├── tests.yml                       # CI testing
    │   ├── docs.yml                        # Doc generation
    │   └── publish.yml                     # PyPI publishing
    └── ISSUE_TEMPLATE/
        ├── bug_report.md
        └── feature_request.md
```

---

## Proposed Structure: Option 2 (Multi-Repo)

### Separate Repositories for Each System

If you want maximum modularity:

```
asimov-box/                                  # Main orchestrator
├── asimov/
│   ├── core/
│   ├── adapters/
│   └── utils/
└── [standard structure]

roboguard/                                   # Separate repo
├── roboguard/
│   ├── core/
│   ├── synthesis/
│   └── eval/
└── [installable via pip install roboguard]

knowno/                                      # Separate repo
├── knowno/
│   ├── conformal/
│   └── calibration/
└── [installable via pip install knowno]

introplan/                                   # Separate repo
├── introplan/
│   ├── reasoning/
│   ├── knowledge_base/
│   └── llm/
└── [installable via pip install introplan]
```

**Then in asimov-box:**
```python
# setup.py
install_requires=[
    'roboguard>=1.0.0',
    'knowno>=1.0.0',
    'introplan>=1.0.0'
]
```

**Pros:**
- Each system independently versioned
- Cleaner separation of concerns
- Can use systems individually

**Cons:**
- More complex dependency management
- Harder to coordinate changes
- More repos to maintain

**Recommendation:** Start with **Option 1 (Monorepo)**, migrate to Option 2 when systems stabilize

---

## Migration Plan

### Phase 1: Restructure (Week 1)

1. **Create new directory structure**
   ```bash
   mkdir -p asimov/{core,adapters,systems/{roboguard,knowno,introplan},utils,cli}
   mkdir -p tests/{unit,integration,benchmarks}
   mkdir -p examples/{scenes,notebooks,configs}
   mkdir -p data/{knowledge_bases,calibration,rules}
   mkdir -p docs/{architecture,integration,tutorials}
   ```

2. **Move files to new locations**
   ```bash
   # Core files
   mv src/knowdanger/core/knowdanger_enhanced.py → asimov/core/orchestrator.py
   mv src/knowdanger/core/knowdanger_core.py → asimov/core/legacy.py

   # Adapters
   mv src/knowdanger/adapters/roboguard_adapter.py → asimov/adapters/roboguard.py
   mv src/knowdanger/adapters/introplan_adapter.py → asimov/adapters/introplan.py
   mv src/knowdanger/adapters/paper_knowno.py → asimov/adapters/knowno.py

   # Systems
   mv RoboGuard/src/roboguard/* → asimov/systems/roboguard/
   mv src/lang_help/knowno/* → asimov/systems/knowno/
   mv IntroPlan/*.py → asimov/systems/introplan/

   # Examples
   mv src/scenes/* → examples/scenes/
   mv IntroPlan/*.ipynb → examples/notebooks/ OR research/introplan_notebooks/

   # Tests
   mv src/tests/* → tests/benchmarks/
   ```

3. **Update imports**
   ```python
   # Old
   from knowdanger.core.knowdanger_enhanced import EnhancedKnowDanger
   from knowdanger.adapters.introplan_adapter import IntroPlanAdapter

   # New
   from asimov.core import EnhancedKnowDanger
   from asimov.adapters import IntroPlanAdapter
   ```

4. **Create proper package structure**
   ```bash
   # Add __init__.py files
   touch asimov/__init__.py
   touch asimov/core/__init__.py
   touch asimov/adapters/__init__.py
   # ... etc
   ```

### Phase 2: Cleanup (Week 2)

1. **Remove duplicates**
   - Consolidate lang_help/knowno and known/knowno
   - Merge paper_*.py with main implementations
   - Remove stub implementations

2. **Fix imports**
   - Use absolute imports everywhere
   - Remove sys.path manipulation
   - Update all example files

3. **Update documentation**
   - Consolidate scattered docs
   - Update paths in README
   - Create docs/ index

### Phase 3: Testing (Week 3)

1. **Create test suite**
   ```bash
   # Unit tests
   tests/unit/test_roboguard.py
   tests/unit/test_knowno.py
   tests/unit/test_introplan.py

   # Integration tests
   tests/integration/test_three_way.py
   ```

2. **Run validation**
   ```bash
   pytest tests/
   python scripts/validate_installation.py
   ```

### Phase 4: Packaging (Week 4)

1. **Create proper pyproject.toml**
   ```toml
   [build-system]
   requires = ["setuptools>=45", "wheel"]
   build-backend = "setuptools.build_meta"

   [project]
   name = "asimov-box"
   version = "0.1.0"
   description = "Integrated safety verification for LLM-controlled robots"
   authors = [{name = "Alexandra Bodrova", email = "your@email.com"}]
   dependencies = [
       "numpy>=1.20.0",
       "torch>=1.9.0",
       "openai>=0.27.0",
   ]

   [project.optional-dependencies]
   dev = ["pytest", "black", "flake8", "mypy"]
   notebooks = ["jupyter", "matplotlib"]

   [project.scripts]
   asimov = "asimov.cli.main:main"
   ```

2. **Test installation**
   ```bash
   pip install -e .
   asimov --version
   python -c "from asimov import EnhancedKnowDanger"
   ```

---

## Key Improvements

### Before → After

| Aspect | Current | Proposed |
|--------|---------|----------|
| **Installation** | Manual setup, path hacks | `pip install asimov-box` |
| **Imports** | Brittle relative imports | Clean absolute imports |
| **Testing** | Scattered, incomplete | Comprehensive test suite |
| **Documentation** | 15+ files scattered | Centralized docs/ |
| **Examples** | Mixed with source | Separate examples/ |
| **Research** | Mixed everywhere | Isolated in research/ |
| **Versioning** | No versioning | Proper semantic versioning |
| **CI/CD** | None | GitHub Actions |
| **Entry Point** | Multiple files | Single CLI + API |

---

## Package API Design

### Simplified User-Facing API

```python
# Option 1: Simple usage
from asimov import AsimovBox

box = AsimovBox.from_config("config.yaml")
result = box.verify(scene, plan)
print(result.overall_verdict)  # SAFE, UNSAFE, or UNCERTAIN

# Option 2: Advanced usage
from asimov import AsimovBox, Config, AggregationStrategy

config = Config(
    alpha=0.1,
    aggregation=AggregationStrategy.CONSERVATIVE,
    use_introspection=True
)

box = AsimovBox(config)
box.calibrate_knowno(calibration_data)
box.load_knowledge_base("data/knowledge_bases/lab_safety.json")

result = box.verify(scene, plan)
for step_result in result.steps:
    print(f"Step: {step_result.action}")
    print(f"  RoboGuard: {step_result.roboguard}")
    print(f"  KnowNo: {step_result.knowno}")
    print(f"  IntroPlan: {step_result.introplan}")
    print(f"  Final: {step_result.final}")

# Option 3: Individual systems
from asimov.systems import RoboGuard, KnowNo, IntroPlan

rg = RoboGuard(rules=scene.rules)
rg.compile(scene.graph)
verdict = rg.check(plan)
```

### CLI Design

```bash
# Verify a plan
asimov verify --scene scenes/lab.yaml --plan plans/experiment.yaml

# Calibrate KnowNo
asimov calibrate --data calibration/lab_data.json --output tau.pkl

# Build knowledge base
asimov kb build --from-feedback feedback.json --output kb.json

# Run benchmark
asimov benchmark --suite all --output results/

# Validate installation
asimov validate
```

---

## Implementation Checklist

### Immediate (This Week)

- [ ] Create proposed directory structure
- [ ] Move core files to new locations
- [ ] Update all imports
- [ ] Add __init__.py files
- [ ] Create basic pyproject.toml

### Short-term (Next 2 Weeks)

- [ ] Consolidate duplicate code
- [ ] Remove stubs, use real implementations
- [ ] Create unified test suite
- [ ] Consolidate documentation
- [ ] Add starter knowledge bases
- [ ] Add sample calibration data

### Medium-term (Next Month)

- [ ] Set up CI/CD pipeline
- [ ] Create comprehensive tutorials
- [ ] Add CLI interface
- [ ] Write API reference
- [ ] Performance benchmarks
- [ ] Publish to PyPI (if public)

---

## Recommended Structure: Final

```
asimov-box/
├── asimov/              # Main package (pip installable)
│   ├── core/
│   ├── adapters/
│   ├── systems/
│   ├── utils/
│   └── cli/
├── tests/               # Comprehensive test suite
├── examples/            # Usage examples
├── data/                # Starter data
├── docs/                # Centralized docs
├── scripts/             # Utility scripts
├── research/            # Research code
└── .github/             # CI/CD

REMOVE:
- KnowDanger/ wrapper (flatten)
- SPINE/ (move to separate repo)
- Scattered docs (consolidate)
```

This structure provides:
- ✅ Clear separation of concerns
- ✅ Easy installation and testing
- ✅ Maintainable and scalable
- ✅ Research-friendly
- ✅ Production-ready
