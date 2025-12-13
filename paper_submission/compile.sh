#!/bin/bash

# Compilation script for LegalBreak paper
# Compiles final_report.tex to PDF with proper bibliography handling

echo "=========================================="
echo "LegalBreak Paper Compilation Script"
echo "=========================================="
echo ""

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "ERROR: pdflatex not found. Please install a LaTeX distribution (TeX Live, MiKTeX, etc.)"
    exit 1
fi

# Check if bibtex is available
if ! command -v bibtex &> /dev/null; then
    echo "ERROR: bibtex not found. Please install a LaTeX distribution with BibTeX support."
    exit 1
fi

echo "Step 1/4: Running pdflatex (first pass)..."
pdflatex -interaction=nonstopmode final_report.tex > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: First pdflatex pass failed. Check final_report.log for details."
    pdflatex final_report.tex
    exit 1
fi

echo "Step 2/4: Running bibtex..."
bibtex final_report > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "WARNING: BibTeX encountered issues. Continuing anyway..."
fi

echo "Step 3/4: Running pdflatex (second pass)..."
pdflatex -interaction=nonstopmode final_report.tex > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: Second pdflatex pass failed. Check final_report.log for details."
    pdflatex final_report.tex
    exit 1
fi

echo "Step 4/4: Running pdflatex (final pass)..."
pdflatex -interaction=nonstopmode final_report.tex > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: Final pdflatex pass failed. Check final_report.log for details."
    pdflatex final_report.tex
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ Compilation successful!"
echo "=========================================="
echo ""
echo "Output file: final_report.pdf"
echo ""

# Show file size
if [ -f final_report.pdf ]; then
    file_size=$(du -h final_report.pdf | cut -f1)
    echo "PDF size: $file_size"
    echo ""
fi

# Optional: Clean up auxiliary files
read -p "Clean up auxiliary files (.aux, .log, .bbl, .blg)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -f final_report.aux final_report.log final_report.bbl final_report.blg final_report.out final_report.fls final_report.fdb_latexmk
    echo "✓ Auxiliary files cleaned up."
fi

echo ""
echo "Done!"
