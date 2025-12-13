# Paper Submission Checklist

Use this checklist before submitting your paper to ensure everything is ready.

## ✅ Pre-Submission Checklist

### Document Quality
- [ ] Run spell-check on the entire document
- [ ] Verify all citations are correctly formatted
- [ ] Check all figures are referenced in text
- [ ] Ensure all sections are complete
- [ ] Verify abstract is under word limit (if applicable)
- [ ] Check page limit compliance

### Technical Verification
- [ ] Compile the paper successfully (run `./compile.sh`)
- [ ] Verify PDF renders correctly on different PDF viewers
- [ ] Check all figures display properly in the PDF
- [ ] Ensure all cross-references work (sections, figures, tables)
- [ ] Verify bibliography entries are complete

### Content Checks
- [ ] All author information is correct
- [ ] Affiliation and contact details are accurate
- [ ] Abstract accurately summarizes the work
- [ ] Introduction clearly states the problem
- [ ] Methods section is reproducible
- [ ] Results are clearly presented
- [ ] Discussion addresses limitations
- [ ] Conclusion summarizes contributions

### Specific to This Paper
- [ ] 9 legal policy rules correctly described (not 24)
- [ ] RoboPAIR methodology clearly explained (not RoboGuard)
- [ ] All 24 citations in references.bib are used
- [ ] Attack success rates match results (26/48 = 54.2%)
- [ ] Manual validation challenges documented
- [ ] API cost limitations mentioned (~$50)

### Figures and Tables
- [ ] Figure 1: naive_vs_legalbreak_comparison.png displays correctly
- [ ] Figure 2: average_attack_turns.png displays correctly
- [ ] Table 1: Overall ASR (Naive 27.1%, LegalBreak 54.2%)
- [ ] Table 2: Category breakdown (Dual-use, Copyright, Defamation)
- [ ] All captions are descriptive and self-contained

### Supplementary Materials (if required)
- [ ] Code repository link is working (GitHub)
- [ ] Data availability statement (if applicable)
- [ ] Ethics statement (if required)
- [ ] Conflict of interest disclosure

### Final Steps
- [ ] Remove all TODO comments and draft notes
- [ ] Anonymize if required for blind review
- [ ] Generate final PDF with correct filename
- [ ] Check PDF file size (usually <10MB for submissions)
- [ ] Verify PDF is not password-protected
- [ ] Create backup of all submission materials

## 📊 Quick Stats (for reference)

- **Total pages:** ~15-20 (depends on compilation)
- **Word count (approx):** ~8,000-10,000 words
- **Figures:** 2
- **Tables:** 2
- **References:** 24
- **Sections:** 6 main sections

## 🔧 Common Issues and Fixes

### Issue: "File not found" for figures
**Fix:** Ensure figures are in `figures/` subdirectory and paths in .tex use `figures/filename.png`

### Issue: Bibliography not showing
**Fix:** Run the full compilation sequence: pdflatex → bibtex → pdflatex → pdflatex

### Issue: Citations showing as [?]
**Fix:** Run bibtex and compile twice more with pdflatex

### Issue: Figure placement issues
**Fix:** LaTeX float placement is automatic. Consider using `[h]` or `[htbp]` options if needed

## 📝 Notes for Repository Upload

When creating a new clean repository:

1. **Initialize repository:**
   ```bash
   git init
   git add .
   git commit -m "Initial submission: LegalBreak paper"
   ```

2. **Add remote:**
   ```bash
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

3. **Recommended repository structure:**
   ```
   legalbreak-paper/
   ├── README.md
   ├── final_report.tex
   ├── references.bib
   ├── neurips_2020.sty
   ├── compile.sh
   ├── figures/
   │   ├── naive_vs_legalbreak_comparison.png
   │   └── average_attack_turns.png
   └── .gitignore
   ```

4. **Repository settings:**
   - Add a license (MIT, Apache 2.0, or Academic Free License)
   - Add topics/tags: `nlp`, `ai-safety`, `legal-tech`, `adversarial-attacks`, `llm`
   - Set visibility (public after publication, private during review)

## 🎯 Submission Platforms

Common conference submission systems:
- **OpenReview** - NeurIPS, ICLR, ICML
- **CMT** - ACL, EMNLP, NAACL
- **EasyChair** - Various workshops
- **Softconf START** - ACL Rolling Review

## 📧 Final Submission Email Template (if needed)

```
Subject: Paper Submission - LegalBreak: Law-Aware Adversarial Testing

Dear [Conference/Journal Name],

Please find attached our submission titled "LegalBreak: Law-Aware Adversarial Testing for LLM Legal Compliance."

Paper details:
- Title: LegalBreak: Law-Aware Adversarial Testing for LLM Legal Compliance
- Authors: Alexandra Bodrova (Princeton University)
- Track: [Conference Track]
- Supplementary materials: Code repository at https://github.com/alexandrabodrova/asimov_box

Best regards,
Alexandra Bodrova
```

---

**Last Updated:** December 13, 2025
