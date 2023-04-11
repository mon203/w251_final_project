# w251 Spring 2023 Final Project

## Project code name: Rover
#### Team Casey Hsiung, Daniele Grandi, Evan Fjeld, Mon Young, Preethi Raju

### Links
- Our demo video
- Our paper
- our presentation slides

## Introduction
(from the paper and the presentation)

This repository contains information and codes we use to build our Rover.

### The Architecture Pipeline

- [diagrams here]

### Folder structure

- [dataset](dataset) - dataset creation step-by-step. More detail in this folder's README.md file
- [casapose](casapose) - our training, validation, and eval environment. More detail in this folder's README.md file
- [csv_outputs](csv_outputs) - our training, validation, and eval output files. It is used to generate plots and tables in the [251_final_project_plots_tables.ipynb](251_final_project_plots_tables.ipynb) file.
- [edge](edge) - our docker and codes to place on the hover edge device. More detail in this folder's README.md file
- [img](img) - our training, validation, and eval environment. More detail in this folder's README.md file

Details are in our paper and presentation file. Noticeable highlights
- We synthetically created our new headphone object 
- We created our dataset containing 15 Linemod objects and our headphones object. Our dataset contains 5,000 synthetic images with associated JSON and meshes files.
- Our synthetic dataset outperforms the PBR dataset
- Our synthetic test dataset outperforms the LMO test dataset

Photo comparisons and graphs