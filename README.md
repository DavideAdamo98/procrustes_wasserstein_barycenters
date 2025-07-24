# PW-bary

This repository contains code for:  
[An in depth look at the Procrustes-Wasserstein distance: properties and barycenters](https://arxiv.org/abs/2507.00894)
by Davide Adamo, Marco Corneli, Manon Vuillien, and Emmanuelle Vila.

## Overview & structure

This work introduces a formal framework for the Procrustes-Wasserstein (PW) distance, an Optimal Transport (OT) metric invariant to rigid transformations such as rotations and reflections.

- `Example1.ipynb`: Comparisons between different OT barycenters.
- `Example2_2D.ipynb`: Evaluation of different initialization strategies for PW matching on 2D point clouds.
- `Example2_3D.ipynb`: The same with 3D point clouds.
- `Example3.ipynb`: Clustering 2D MNIST point clouds using OT metrics.

For OT computation we make use of POT toolbox (Python Optimal Transport library).
