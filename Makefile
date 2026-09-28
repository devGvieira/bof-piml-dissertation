FILE=modelo_dissertacao

export TEXINPUTS=.:./styles//:
export BIBINPUTS=.:./styles//:./bibtex//:
export BSTINPUTS=.:./styles//:

all: clean
	rm -f $(FILE).bbl
	pdflatex $(FILE)
	bibtex $(FILE) && pdflatex $(FILE)
	pdflatex $(FILE)
	sort $(FILE).lsg > $(FILE).lsg.tmp
	mv $(FILE).lsg.tmp $(FILE).lsg
	pdflatex $(FILE)
	xdg-open $(FILE).pdf &

clean:
	rm -f $(FILE).aux
	rm -f $(FILE).dvi
	rm -f $(FILE).lof
	rm -f $(FILE).log
	rm -f $(FILE).lot
	rm -f $(FILE).lsb
	rm -f $(FILE).lsg
	rm -f $(FILE).loquadro
	rm -f $(FILE).pdf
	rm -f $(FILE).toc
	rm -f $(FILE).bbl
	rm -f $(FILE).blg
	rm -f $(FILE).out
	rm -rf $(FILE).auto

# Run manually after `make` if symbols/siglas lists need regeneration
index:
	makeindex -s styles/tabela-siglas.ist -o $(FILE).sigla $(FILE).siglax
	makeindex -s styles/tabela-simbolos.ist -o $(FILE).symbols $(FILE).symbolsx
