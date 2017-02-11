## Put this Makefile in your project directory---i.e., the directory
## containing the paper you are writing. Assuming you are using the
## rest of the toolchain here, you can use it to create .html, .tex,
## and .pdf output files (complete with bibliography, if present) from
## your markdown file.
## -    Install the `pandoc-citeproc`, `pandoc-crossref` (and `pandoc-citeproc-preamble`)
##      filters for `pandoc`. See the README for links.
## -	Change the paths at the top of the file as needed.
## -	Using `make` without arguments will generate html, tex, and pdf
## 	output files from all of the files with the designated markdown
##	extension. The default is `.md` but you can change this.
## -	You can specify an output format with `make tex`, `make pdf` or
## - 	`make html`.
## -	Doing `make clean` will remove all the .tex, .html, and .pdf files
## 	in your working directory. Make sure you do not have files in these
##	formats that you want to keep!
##
## git clone https://github.com/kjhealy/pandoc-templates.git
## mkdir -p ~/.pandoc
## mv pandoc-templates/* ~/.pandoc
## git clone https://github.com/kjhealy/latex-custom-kjh.git
## mkdir ~/Library/texmf/tex/latex/org-preamble
## cp latex-custom-kjh/needs-org-preamble/* ~/Library/texmf/tex/latex/org-preamble
## mkdir ~/Library/texmf/tex/latex/memoir
## cp latex-custom-kjh/needs-memoir/* ~/Library/texmf/tex/latex/memoir
## more details at https://kieranhealy.org/blog/archives/2014/01/23/plain-text/

## Markdown extension (e.g. md, markdown, mdown).
MEXT = md

## All markdown files in the working directory
SRC = $(wildcard *.$(MEXT))

## Location of Pandoc support files.
PREFIX = $(HOME)/.pandoc

## Location of your working bibliography file
# BIB = /Users/anaderi/Documents/bibs/socbib-pandoc.bib
BIB = bibliography.bib

## CSL stylesheet (located in the csl folder of the PREFIX directory).
CSL = apsa


PDFS=$(SRC:.md=.pdf)
HTML=$(SRC:.md=.html)
TEX=$(SRC:.md=.tex)

help:
	echo $(HOME)

html:	clean $(HTML)

view: $(HTML)
	open $(HTML)

pdf:	clean $(PDFS)
tex:	clean $(TEX)
all:	$(PDFS) $(HTML) $(TEX)


%.html:	%.md
	pandoc -r markdown+simple_tables+table_captions+yaml_metadata_block -w html \
	 -S --template=$(PREFIX)/templates/html.template \
 	 --filter pandoc-crossref --filter pandoc-citeproc \
 	 --css=$(PREFIX)/marked/kultiad-serif.css\
	 --csl=$(PREFIX)/csl/$(CSL).csl \
	 --bibliography=$(BIB) -o $@ $<
     # --filter-pandoc-citeproc-preamble \

%.tex:	%.md
	pandoc -r markdown+simple_tables+table_captions+yaml_metadata_block -w latex -s -S \
	--latex-engine=xelatex --template=$(PREFIX)/templates/latex.template \
	--filter pandoc-crossref --filter pandoc-citeproc \
	--csl=$(PREFIX)/csl/ajps.csl --bibliography=$(BIB) -o $@ $<
#	--filter pandoc-citeproc-preamble \


%.pdf:	%.md
	pandoc -r markdown+simple_tables+table_captions+yaml_metadata_block -s -S \
	--latex-engine=xelatex --template=$(PREFIX)/templates/latex.template \
	--filter pandoc-crossref --filter pandoc-citeproc \
	--bibliography=$(BIB)	\
	--csl=$(PREFIX)/csl/$(CSL).csl -o $@ $<
#	--filter pandoc-citeproc-preamble \

clean:
	rm -f *.html *.pdf *.tex
