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
Table 1:
    
This python code provides results in Table 1 in the paper titled:
"A Scalable Gradient-Based Optimization Framework for Sparse Minimum-Variance Portfolio Selection".
by Moka, Quiroz, Asimit, and Muller
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
import time


#%%
"""
Specify the paths to the two datasets
"""
file31 = "./data/example1/31assets.csv"
file85 = "./data/example1/85assets.csv"


#%%
"""
Load the datasets as a numpy array covaraince matrices
"""
df = pd.read_csv(file31, header=None)
Sigma31 = df.to_numpy()

df = pd.read_csv(file85, header=None)
Sigma85 = df.to_numpy()

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
k_list = range(1, 11)


"""
Run our method Grid-FW for the above list of k values
"""

print("\n---------------------------------------------")
print("     Results for the dataset with p = 31")
print("\n---------------------------------------------")

for k in k_list:
    tic = time.process_time()
    result = pf.grid_fw(Sigma31, k, maxepoch=n, maxiter=m) 
    toc = time.process_time()  
    
    print('\n----------- k =', k, '------------')
    print("Best model: ", result['model_best'])
    print("Final model:", result['model_final'])
    print("Best var:  ", pf.sci(result['var_best']))
    print("Final var:  ", pf.sci(result['var_final']))
    print("No of FW steps: ", result['nepoch']*m)
    print("Time:", round(toc-tic, 2), " sec")  
    
    
print("\n---------------------------------------------")
print("     Results for the dataset with p = 85")
print("\n---------------------------------------------")
for k in k_list:
    tic = time.process_time()
    result = pf.grid_fw(Sigma85, k, maxepoch=n, maxiter=m) 
    toc = time.process_time()  
    
    print('\n----------- k =', k, '------------')
    print("Best model: ", result['model_best'])
    print("Final model:", result['model_final'])
    print("Best var:  ", pf.sci(result['var_best']))
    print("Final var:  ", pf.sci(result['var_final']))
    print("No of FW steps: ", result['nepoch']*m)
    print("Time:", round(toc-tic, 2), " sec") 
