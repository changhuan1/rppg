#!/usr/bin/env bash
set -e

pdflatex -interaction=nonstopmode rt_mrcpnet_en.tex
bibtex rt_mrcpnet_en
pdflatex -interaction=nonstopmode rt_mrcpnet_en.tex
pdflatex -interaction=nonstopmode rt_mrcpnet_en.tex

xelatex -interaction=nonstopmode rt_mrcpnet_zh.tex
bibtex rt_mrcpnet_zh
xelatex -interaction=nonstopmode rt_mrcpnet_zh.tex
xelatex -interaction=nonstopmode rt_mrcpnet_zh.tex
