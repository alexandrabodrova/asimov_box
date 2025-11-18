# Asimov Box Codebase Exploration - Complete Report

This directory now contains comprehensive documentation of the Asimov Box (KnowDanger) codebase, created through thorough exploration of the repository.

## Generated Documentation Files

### 1. **CODEBASE_ANALYSIS.md** (30 KB, 902 lines)
The most comprehensive reference document covering:
- **Executive Summary** - What Asimov Box is and how it works
- **System 1: RoboGuard** - Purpose, functionality, file locations, interfaces
- **System 2: KnowNo** - Uncertainty quantification system details
- **System 3: IntrospectivePlan** - Introspective reasoning system
- **System Integration** - How all three systems work together
- **Data Structures & Interfaces** - Common types used across systems
- **Directory Structure** - Complete file organization
- **Current Integration Status** - What's working and how
- **Main Entry Points** - How to use the system
- **Testing & Benchmarking** - Available tests and examples
- **Key Features** - Capabilities of the integrated system
- **Configuration Options** - All settable parameters
- **Documentation Files** - Where to find more info
- **Key Metrics** - How to measure performance
- **Known Limitations & Future Work**

**Start here for:** Deep understanding of all components, file locations, interfaces, and how they integrate.

---

### 2. **ARCHITECTURE_OVERVIEW.md** (26 KB, 575 lines)
Visual architecture guide with diagrams showing:
- **Three-System Integration Diagram** - Visual flow of the system
- **Component Details** - How RoboGuard, KnowNo, and IntroPlan work
- **Data Flow** - Step-by-step how a single action is verified
- **File Organization** - How data flows through the system
- **Integration Layers** - Adapters, orchestrator, and utilities
- **Knowledge Base Structure** - IntroPlan KB organization
- **Configuration Hierarchy** - How config flows through system
- **Error Handling & Fallbacks** - Robustness mechanisms
- **Testing & Evaluation** - How to run tests
- **Key Interface Summary** - Quick code reference
- **Deployment Path** - How to take from development to production

**Start here for:** Visual understanding of system architecture, data flows, and ASCII diagrams.

---

### 3. **EXPLORATION_SUMMARY.txt** (15 KB, 406 lines)
Quick reference guide containing:
- **Three Safety Systems Overview** - Quick description of each system
- **Key Findings** - Important discoveries about the codebase
- **Directory Structure** - Key component locations
- **Main Entry Points** - How to use the system
- **Current Integration Status** - What's working
- **Data Structures** - Core types used
- **Configuration Reference** - Key parameters
- **Testing & Examples** - Available tests
- **Quick Start Guide** - Installation and first steps
- **Key Metrics** - Performance metrics to track
- **Troubleshooting** - Common issues and solutions
- **Recommended Next Steps** - How to proceed

**Start here for:** Quick navigation, key facts, and getting started.

---

## Quick Navigation

### I want to...

**Understand what Asimov Box is:**
→ Start with EXPLORATION_SUMMARY.txt (overview section)

**Know all the files and their purposes:**
→ See CODEBASE_ANALYSIS.md (directory structure & file locations)

**See how the systems work together:**
→ Look at ARCHITECTURE_OVERVIEW.md (integration diagrams)

**Get started using the system:**
→ Follow EXPLORATION_SUMMARY.txt (quick start guide)

**Understand the data structures:**
→ Check CODEBASE_ANALYSIS.md (data structures & interfaces)

**See code examples:**
→ Look at CODEBASE_ANALYSIS.md (usage examples throughout)

**Understand configuration:**
→ See CODEBASE_ANALYSIS.md (configuration & customization section)

**Run tests:**
→ See EXPLORATION_SUMMARY.txt (testing & examples section)

---

## The Three Safety Systems

### RoboGuard (Rule-Based)
- **What:** Checks actions against explicit safety rules
- **How:** Temporal logic verification
- **Where:** `RoboGuard/src/roboguard/` and `src/knowdanger/adapters/roboguard_adapter.py`
- **Output:** SAFE or UNSAFE

### KnowNo (Uncertainty Quantification)
- **What:** Quantifies confidence in LLM predictions
- **How:** Conformal prediction
- **Where:** `src/lang_help/knowno/api.py` and `src/knowdanger/adapters/paper_knowno.py`
- **Output:** SAFE or UNCERTAIN

### IntrospectivePlan (Introspective Reasoning)
- **What:** Generates introspective reasoning for safety decisions
- **How:** LLM introspection + knowledge base retrieval
- **Where:** `IntroPlan/` and `src/knowdanger/adapters/introplan_adapter.py`
- **Output:** SAFE, UNSAFE, or UNCERTAIN (with explanation)

---

## Getting Started in 5 Minutes

1. **Read Quick Overview:** Open EXPLORATION_SUMMARY.txt

2. **Understand the Architecture:** Look at ARCHITECTURE_OVERVIEW.md (first diagram)

3. **Try an Example:**
   ```bash
   cd /home/user/asimov_box/KnowDanger
   python src/knowdanger/core/example_usage.py
   ```

4. **Explore Code:**
   - Main orchestrator: `src/knowdanger/core/knowdanger_enhanced.py`
   - System adapters: `src/knowdanger/adapters/`
   - Example scenes: `src/scenes/`

5. **Read Full Details:** Dive into CODEBASE_ANALYSIS.md

---

## Document Statistics

| Document | Size | Lines | Coverage |
|----------|------|-------|----------|
| CODEBASE_ANALYSIS.md | 30 KB | 902 | Comprehensive technical reference |
| ARCHITECTURE_OVERVIEW.md | 26 KB | 575 | Visual diagrams and architecture |
| EXPLORATION_SUMMARY.txt | 15 KB | 406 | Quick reference and navigation |
| **Total** | **71 KB** | **1,883** | Complete system documentation |

---

## Key Insights from Exploration

### Complete Integration ✓
All three systems are fully integrated in `knowdanger_enhanced.py`. You can use:
- Two systems (RoboGuard + KnowNo) via `knowdanger_core.py`
- Three systems (RG + KN + IntroPlan) via `knowdanger_enhanced.py`

### Modular Design ✓
Each system has its own adapter, making it easy to:
- Extend individual systems
- Replace systems with alternatives
- Use subset of functionality

### Flexible Aggregation ✓
Three strategies to combine verdicts:
- **Conservative:** Any UNSAFE → UNSAFE (most safe)
- **Majority:** Vote among systems
- **Weighted:** Confidence-weighted combination

### Well-Documented ✓
- Implementation guides in repository
- 6 working examples
- 3 example scenarios
- Comprehensive inline code comments
- This exploration documentation

---

## File Structure Highlights

```
KnowDanger/
├── RoboGuard/              → Rule-based safety system
├── IntroPlan/              → Introspective planning system
├── SPINE/                  → Mapping and navigation (separate)
└── src/
    ├── knowdanger/
    │   ├── core/
    │   │   ├── knowdanger_core.py      → Original (RG+KN)
    │   │   └── knowdanger_enhanced.py  → Enhanced (RG+KN+IP)
    │   ├── adapters/                   → System bridges
    │   ├── calibration/                → KnowNo calibration
    │   └── [documentation files]
    ├── lang_help/knowno/               → KnowNo API
    ├── scenes/                         → Example scenarios
    └── tests/                          → Benchmarks
```

---

## Usage Quick Start

```python
# Setup
from knowdanger.core.knowdanger_enhanced import EnhancedKnowDanger, Config

# Configure (all three systems)
config = Config(
    alpha=0.1,                                    # Confidence level
    use_introspection=True,                       # Enable IntroPlan
    introplan_kb_path="IntroPlan/data/kb.txt",  # Knowledge base
    aggregation_strategy="conservative"           # Verdict combination
)

# Create system
kd = EnhancedKnowDanger(config)

# Evaluate plan
assessment = kd.run(scene, plan)

# Results
for step in assessment.steps:
    print(f"Step: {step.step.action}")
    print(f"  RoboGuard: {step.roboguard.label}")
    print(f"  KnowNo: {step.knowno.label}")
    print(f"  IntroPlan: {step.introplan.label}")
    print(f"  Final: {step.final.label}")
```

---

## Next Steps

1. **For Understanding:**
   - Read CODEBASE_ANALYSIS.md for comprehensive technical details
   - Review ARCHITECTURE_OVERVIEW.md for visual understanding
   - Check EXPLORATION_SUMMARY.txt for quick facts

2. **For Implementation:**
   - Follow the Quick Start Guide in EXPLORATION_SUMMARY.txt
   - Run example_usage.py to see working code
   - Modify example scenarios for your use case

3. **For Integration:**
   - Review INTEGRATION_GUIDE.md in the repository
   - Calibrate KnowNo with your data
   - Build IntroPlan knowledge base for your domain

4. **For Deployment:**
   - Use MetricsCollector for performance tracking
   - Implement error handling and fallbacks
   - Deploy as API endpoint or service

---

## External Resources

- **RoboGuard Paper:** https://robopair.org
- **KnowNo Paper:** Google Research language_model_uncertainty
- **IntroPlan Paper:** https://introplan.github.io (NeurIPS 2024)
- **Repository:** https://github.com/alexandrabodrova/asimov_box

---

## Citation

```bibtex
@phdthesis{bodrova2024asimovbox,
  title={Asimov Box: Robust Safety Verification for LLM-Controlled Robots},
  author={Bodrova, Alexandra},
  year={2025},
  school={Princeton University}
}
```

---

**Exploration Completed:** 2025-11-18  
**Total Documentation Generated:** 71 KB (1,883 lines)  
**Time Invested:** Comprehensive codebase analysis

---

## How to Use These Documents

- **Print-friendly:** All documents are in markdown/text format
- **Searchable:** Use your editor's search function to find topics
- **Cross-referenced:** Documents reference each other for deeper dives
- **Standalone:** Each document can be read independently
- **Complete:** No external documentation needed for most questions

Start with **EXPLORATION_SUMMARY.txt** for orientation, then dive into the specific document that matches your need.

