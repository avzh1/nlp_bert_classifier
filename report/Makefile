# Makefile for LaTeX document with BibTeX bibliography

MAIN = my_paper

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex 
	pdflatex -interaction nonstopmode $(MAIN).tex || true
	bibtex $(MAIN) || true
	pdflatex -interaction nonstopmode $(MAIN).tex || true
	pdflatex -interaction nonstopmode $(MAIN).tex || true

clean:
	rm -f $(MAIN).pdf $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out

.PHONY: all clean
