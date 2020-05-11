# MVPA_simulations
WARNING: Current setup is for 1000 iterations of 15 participants across a spread of noise levels. This is very computationally intensive. Recommended to reduce niter to 100 or below if running locally. 

Simulations comparing sensitivity of various MVPA methods (LDC, Euclidean, SVM discriminant and SVM classification). Results of 1000 simulations are shown in the folder graphs.

This work is the updated version of the code used in this paper (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6175330/). Only two types of noise were simulated, thermal and physiological noise. LDC is shown to be more sensitive to changes in physiological noise.

Parameters of the simulation can be modified in the code itself and the add_noise function can be modified to add only thermal noise (using add_tnoise instead of add_noise) or only physiological noise (using params.thermal_noise = 0).

MVPA_compare generates the dataset of measurements and ttest_plot generates the graphs in the folder using the output of MVPA_compare. 
