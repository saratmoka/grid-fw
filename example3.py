#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Sarat Moka
Email: s.moka@unsw.edu.au
Website: saratmoka.com
Affiliation: The University of New South Wales
Date: 15 May 2025
"""

"""
Example 2:

This python code provides Example 2 results in Table 2 in the neurips submission titled:
"A Scalable Gradient-Based Optimization Framework for Sparse Minimum-Variance Portfolio Selection".
by Moka, Quiroz, Asimit, and Muller

Note that the third synthetic dataset with p = 3000 used in this example is NOT included in 
the github repository due to over size issue. 
"""
#
"""
Uncomment the next five lines if you want to specify the number of treads used.
"""
# import os
# os.environ["OMP_NUM_THREADS"]     = "1"
# os.environ["MKL_NUM_THREADS"]     = "1"
# os.environ["OPENBLAS_NUM_THREADS"]= "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"


import portfolio as pf # Loads our portfolio code
import pandas as pd
import numpy as np
import time


#%%
"""
Specify the path to the dataset: uncomment only one of three options
"""
file = "./data/example3/1000assets.csv"
# file = "./data/example3/2000assets.csv"


"""
Load the datasets as a numpy array covaraince matrices
"""
df = pd.read_csv(file, header=None)
Sigma = df.to_numpy()
p = Sigma.shape[0]

"""
Specify the paramaters of Grid-FW method
n : number of epcohs
m : number of FW steps in each epoch
"""
n = 500
m = 10

"""
Specify the list of k values
"""
k_list = [int(z*p) for z in (0.10, 0.25, 0.50)]

#%%
"""
Compute the largest and smallest eigenvalues of Sigma using the power method, 
as well as the largest and smallest varainces. 
Note that diagonal of Sigma provides the varainces of the asserts.
"""
eta_1, _ = pf.power_method(Sigma)
eta_p, _ = pf.power_method(eta_1*np.identity(p) - Sigma) 
eta_p = -eta_p + eta_1
v  = np.diag(Sigma)

text = f"""
Dataset path:         {file}
Largest eigenvalue:   {eta_1}
Smallest eigenvalue:  {eta_p}
Max variance:         {np.max(v)}
Min variance:         {np.min(v)}
"""
pf.print_in_box(text)


"""
Run our method Grid-FW for the above list of k values
"""
model_var_list = []
time_list = []

for k in k_list:
    tic = time.process_time()
    result = pf.grid_fw(Sigma, k, maxepoch=n, maxiter=m) 
    toc = time.process_time()  
    
    model_var_list.append((result['model_best'], result['var_best'])) 
    time_list.append(toc-tic)

print("\n---------------------------------------------")
print("     Results for the dataset with p = ", p)
print("---------------------------------------------")

for i in range(len(k_list)):
    print("\nSize k   :", k_list[i])
    print("Variance :", model_var_list[i][1])
    print("Time     :", round(time_list[i], 4), "sec")
    #print("Model    :\n", model_var_list[i][0])


    
    
