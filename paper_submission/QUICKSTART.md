# Quick Start Guide

Get your paper compiled and ready for submission in 3 simple steps.

## Step 1: Verify Files

Check that all required files are present:

```bash
ls -l
```

You should see:
- ✓ `final_report.tex` (main paper)
- ✓ `references.bib` (24 citations)
- ✓ `neurips_2020.sty` (LaTeX style)
- ✓ `figures/` directory with 2 PNG files
- ✓ `compile.sh` (compilation script)

## Step 2: Compile the Paper

Simply run:

```bash
./compile.sh
```

This will:
1. Run pdflatex (first pass)
2. Run bibtex (process citations)
3. Run pdflatex (second pass - resolve references)
4. Run pdflatex (third pass - finalize)

**Output:** `final_report.pdf`

## Step 3: Review the PDF

Open `final_report.pdf` and verify:
- [ ] All text renders correctly
- [ ] Both figures appear
- [ ] References are numbered correctly
- [ ] Page numbers are present
- [ ] No error messages or warnings

## Troubleshooting

### Problem: "./compile.sh: Permission denied"
**Solution:**
```bash
chmod +x compile.sh
./compile.sh
```

### Problem: "pdflatex: command not found"
**Solution:** Install LaTeX:
- **macOS:** `brew install --cask mactex`
- **Ubuntu/Debian:** `sudo apt-get install texlive-full`
- **Windows:** Download MiKTeX from https://miktex.org/

### Problem: Figures not appearing
**Solution:** Ensure PNG files are in `figures/` subdirectory with correct names

### Problem: Bibliography empty
**Solution:** Make sure `references.bib` is in the same directory and run the full compilation sequence

## Manual Compilation (if needed)

If the script doesn't work, compile manually:

```bash
pdflatex final_report.tex
bibtex final_report
pdflatex final_report.tex
pdflatex final_report.tex
```

## Clean Build

To start fresh and remove auxiliary files:

```bash
rm -f *.aux *.log *.bbl *.blg *.out *.fls *.fdb_latexmk
./compile.sh
```

## Upload to Overleaf (Alternative)

1. Go to https://www.overleaf.com
2. Click "New Project" → "Upload Project"
3. Upload ZIP of this entire folder
4. Click "Recompile" in Overleaf

## Next Steps

1. ✅ Compiled successfully? → Review SUBMISSION_CHECKLIST.md
2. ✅ Ready to submit? → See README.md for submission guidelines
3. ✅ Need to make changes? → Edit final_report.tex and recompile

---

**Estimated time:** 2-5 minutes for first compilation
