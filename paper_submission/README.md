# LegalBreak: Law-Aware Adversarial Testing for LLM Legal Compliance

This repository contains the submission materials for the paper "LegalBreak: Law-Aware Adversarial Testing for LLM Legal Compliance."

## Paper Information

**Title:** LegalBreak: Law-Aware Adversarial Testing for LLM Legal Compliance
**Authors:** Alexandra Bodrova
**Affiliation:** Princeton University
**Conference:** SPI 352 - AI & Law Final Project

## Abstract

As large language models (LLMs) become embedded in consumer applications, they expose developers and users to legal liability through copyright infringement, defamation, and dual-use content generation. This work presents LegalBreak, a law-aware adversarial testing framework that systematically measures LLM vulnerabilities to legal compliance violations across three risk categories: dual-use content (18 U.S.C. violations), copyright infringement (17 U.S.C.), and defamation (state tort law).

**Key Finding:** GPT-4o exhibits critical legal vulnerabilities, with 68.8% baseline defamation susceptibility and 70.6% dual-use attack success under adversarial refinement. Multi-turn jailbreaking yields 54.2% overall attack success versus 27.1% for direct prompts, revealing fundamental tensions between rigorous security evaluation and safe deployment.

## Repository Contents

```
paper_submission/
├── README.md                              # This file
├── final_report.tex                       # Main LaTeX source file
├── references.bib                         # Complete bibliography (24 citations)
├── neurips_2020.sty                       # NeurIPS LaTeX style file
├── figures/
│   ├── naive_vs_legalbreak_comparison.png # ASR comparison by category
│   └── average_attack_turns.png           # Average refinement turns analysis
└── compile.sh                             # Compilation script
```

## Compilation Instructions

### Prerequisites
- LaTeX distribution (TeX Live, MiKTex, or similar)
- `pdflatex` and `bibtex` commands available

### Building the PDF

**Option 1: Using the provided script**
```bash
chmod +x compile.sh
./compile.sh
```

**Option 2: Manual compilation**
```bash
pdflatex final_report.tex
bibtex final_report
pdflatex final_report.tex
pdflatex final_report.tex
```

The output PDF will be `final_report.pdf`.

## Key Contributions

1. **RoboPAIR Adaptation:** First adaptation of RoboPAIR multi-agent adversarial testing from robotic safety to textual legal compliance
2. **Doctrine-Grounded Rules:** 9 legal policy rules (4 dual-use, 2 copyright, 3 defamation) operationalizing U.S.C. provisions and tort law
3. **Systematic Vulnerability Measurement:** 48-prompt adversarial test suite with attack success rate (ASR) evaluation
4. **Counterintuitive Finding:** Adversarial testing infrastructure itself amplifies risk (54.2% ASR vs 27.1% baseline)
5. **Manual Validation Framework:** 30-40 hours of manual review revealing systematic biases in automatic labeling

## Experimental Results

### Overall Attack Success Rates
- **Naive Baseline:** 27.1% (13/48 attacks successful)
- **LegalBreak (Multi-turn):** 54.2% (26/48 attacks successful)

### Category-Specific Results
- **Dual-Use:** 11.8% (Naive) → 70.6% (LegalBreak) - +58.8pp increase
- **Copyright:** 0.0% (Naive) → 13.3% (LegalBreak) - Most protected category
- **Defamation:** 68.8% (Naive) → 75.0% (LegalBreak) - Highest baseline vulnerability

### Average Attack Turns (LegalBreak)
- **Dual-use:** 3.42 turns (n=12 successful attacks)
- **Copyright:** 4.00 turns (n=2 successful attacks)
- **Defamation:** 1.08 turns (n=12 successful attacks)
- **Overall:** 2.38 turns (n=26 successful attacks)

## Legal Policy Engine

The system implements **9 doctrine-grounded rules**:

**Dual-Use (4 rules):**
1. `explosives_instructions` - Bomb-making, IED construction (18 U.S.C. §842)
2. `toxic_chemical_synthesis` - Chemical weapons (sarin, ricin, VX)
3. `biological_weapon_creation` - Bioweapon creation, pathogen manipulation (18 U.S.C. §2332a)
4. `cyber_physical_attacks` - Critical infrastructure attacks (SCADA, power grid)

**Copyright (2 rules):**
5. `verbatim_reproduction` - Substantial similarity detection (17 U.S.C. §106)
6. `drm_circumvention` - DRM/copy protection bypass (17 U.S.C. §1201)

**Defamation (3 rules):**
7. `unsubstantiated_allegations` - False factual claims without evidence
8. `fake_news_defamation` - Fabricated news articles with fake sources
9. `fabricated_facts` - Factual assertions about individuals without disclaimers

## Methodology

**Architecture:** RoboPAIR-inspired multi-agent system
- **Attacker LLM:** Iteratively refines jailbreak prompts (GPT-4o, temp=0.7)
- **Target LLM:** GPT-4o under evaluation
- **Judge LLM:** 4-stage hybrid pipeline (GPT-4o, temp=0.0)
  - Stage 1: Refusal detection (30+ patterns)
  - Stage 2: Rule-based legal policy engine (9 rules)
  - Stage 3: Named entity recognition (person identification)
  - Stage 4: LLM semantic reasoning (edge cases)

**Evaluation:** 5-turn maximum refinement, 48 adversarial prompts, ASR primary metric

## Implementation Notes

- **Cost:** ~$50 out-of-pocket OpenAI API credits
- **Manual Review:** 30-40 hours for result validation
- **Code:** ~3,200 lines Python 3.9+
- **Princeton API Issue:** Azure guardrails too restrictive for fair baseline comparison

## Citations

All 24 references are included in `references.bib`:
- Foundational: zou2023universal (PAIR), robey2024jailbreaking (RoboPAIR), chao2024jailbreakbench
- Legal framework: lemley2024copyright, volokh2023libel, wachter2024duty, cooper2024generative
- Guardrails: inan2023llama (Llama Guard), rebedea2023nemo (NeMo Guardrails)
- Attacks: liu2023prompt (prompt injection), wei2024jailbroken, shen2024anything
- Cases: tremblay2023complaint, silverman2023complaint, walters2023defamation

## Related Code Repository

Full implementation code: https://github.com/alexandrabodrova/asimov_box

## License

This work is submitted as a final project for SPI 352: AI & Law at Princeton University.

## Contact

For questions or collaboration opportunities, please contact:
- Alexandra Bodrova
- Princeton University
- Email: [contact information]

---

**Last Updated:** December 13, 2025
