# w251 Spring 2023 Final Project

## 6DoF Multi-Object Pose Estimation
#### Team Casey Hsiung, Daniele Grandi, Evan Fjeld, Mon Young, Preethi Raju

### Links
- [Example demo video](https://drive.google.com/file/d/1k8y4MAEKrb0HFcJcRr9AbG-AbUXjzYdz/view?usp=sharing)
- [White paper](https://docs.google.com/document/d/1zfUYY-vTt2yYEcxVGPECIPDVON8W9Ux_a7JyBCBgyzg/edit?usp=sharing)
- [Presentation slides](https://docs.google.com/presentation/d/1IoGQCWiyFQC74rt1_hlF1GrydBEhGGNRcdf96U5-y20/edit?usp=sharing)

## Purpose

This repository contains information and code used for our CASAPose 6D pose estimation project.

### Edge Pipeline

-  EC2 edge implmentation.
![edge](img/edge.png)

-  Theoretical application of our CASAPose model implementation packaged and deployed on a robot.
![edgeapp](img/edge_app.png)


### Folder Structure

- [dataset](dataset) - Dataset creation step-by-step. More detail in this folder's README.md file
- [casapose](casapose) - Our training, validation, and eval environment. More detail in this folder's README.md file
- [csv_outputs](csv_outputs) - Our training, validation, and eval output files. It is used to generate plots and tables in the [251_final_project_plots_tables.ipynb](251_final_project_plots_tables.ipynb) file.
- [edge_final](edge_final) - Our docker and codes to place on the edge device. More detail in this folder's README.md file
- [workings](workings) - Our notes, drafts, and test code for our project.

Details are in our paper and presentation file. Noticeable highlights
- We synthetically created our new headphone object 
- We created our dataset containing 15 Linemod objects and our headphones object. Our dataset contains 5,000 synthetic images with associated JSON and meshes files.
- Our synthetic dataset outperforms the PBR dataset
- Our synthetic eval test dataset outperforms the LMO eval test dataset (as expected)

Photo comparisons and graphs

![comparison1](img/8_vs_obj16_lmo_compare.png)
-  baseline vs training2 evaluated with LM-O dataset. Our training2 detects glue, has better accuracy on cat, but mis-detect ape.

### CASAPose Citation

Gard, Niklas, Anna Hilsmann, and Peter Eisert. "CASAPose: Class-Adaptive and Semantic-Aware Multi-Object Pose Estimation." arXiv preprint arXiv:2210.05318 (2022).