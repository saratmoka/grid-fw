#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file provides a general instrusction on how to run the Grid-FW method in python. 
The Grid-FW method is proposed in the NeurIPS submission titled:
"A Scalable Gradient-Based Optimization Framework for Sparse Minimum-Variance Portfolio Selection".
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



#%%

import portfolio as pf # Loads our portfolio code
import pandas as pd
import numpy as np
import time


#%%
"""
Specify the path to the dataset. Uncomment only one of three options
"""
file = "./data/example1/31assets.csv"
# file = "./data/example1/85assets.csv"
# file = "./data/example1/89assets.csv"
# file = "./data/example1/98assets.csv"
# file = "./data/example1/225assets.csv"
# file = "./data/example2/2000-2007.csv"
# file = "./data/example2/2008-2015.csv"
# file = "./data/example2/2016-2023.csv"
# file = "./data/example3/1000assets.csv"
# file = "./data/example3/2000assets.csv"


#%%
"""
Load the dataset as a numpy array
"""
df = pd.read_csv(file, header=None)
Sigma = df.to_numpy()

#%%
"""
(optional) 

Compute the largest and smallest eigenvalues of Sigma using the power method. 
Also, print the largest and smallest varainces in Sigma. 
Note that diagonal of Sigma provides the varainces of the assets.
"""
eta_1, _ = pf.power_method(Sigma)
p = Sigma.shape[0]
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


#%%
"""
Specify the paramaters of Grid-FW method
n : number of epcohs
m : number of FW steps in each epoch
"""

n = 500
m = 10

#%%
"""
Run our method, Grid-FW, for a specific value of k, the number of assets for selction.
"""

k = 5

tic = time.process_time()
result = pf.grid_fw(Sigma, k, maxepoch=n, maxiter=m) 
toc = time.process_time()  

print('\n----------- k =', k, '------------')
print("Best model :", result['model_best'])
print("Final model:", result['model_final'])
print("Best var   :", pf.sci(result['var_best']))
print("Final var  :", pf.sci(result['var_final']))
print("# FW steps :", result['nepoch']*m)
print("Time       :", round(toc-tic, 2), " sec")  

    
