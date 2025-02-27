#!/bin/bash

echo "Download Vandermonde.txt if it is not present in the repo..."
if [ ! -e Vandermonde.txt ]; then
  wget https://home.strw.leidenuniv.nl/~daalen/Handin_files/Vandermonde.txt 
fi

echo "Clearing/creating the plotting directory"
if [ ! -d "plots" ]; then
  mkdir plots
fi
rm -rf plots/*

echo "Run the script of Ex. 1..."
python3 NUR_Handin_1_1.py > NUR_Handin_1_1_output.txt

echo "Run the script of Ex. 2..."
python3 NUR_Handin_1_2.py > NUR_Handin_1_2_output.txt

echo "Generating the pdf"
pdflatex beckers.tex
pdflatex beckers.tex
pdflatex beckers.tex


