# Grid-FW
A Scalable Gradient-Based Optimization Framework for Sparse Minimum-Variance Portfolio Selection

<img src="media/path_animation.gif" width="600" />


This repository provides a Python implementation of our method Grid-FW proposed in <arXiv paper link>. It also provides Julia code used for executing the Big-M approach using CPLEX to compare our method. 

# Python Dependencies (Grid-FW)
```
Python >= 3.12.2
Numpy >= 2.2.4
Pandas >= 2.2.3
SciPy >= 1.14.1
```
# Julia Dependencies (CPLEX)
```
JuMP >= 0.22.3
CPLEX >= 0.8.1
Julia >= 1.2
```

# Download Instructions
Download this repository and extract into a folder, which will contain:

```
data            : contains all the datasets used in our applications
results.xlsx    : Excel file with all experiment results (Table 2 of the paper)
portfolio.py    : functions required to execute the Grid-FW algorithm
main.py         : imports portfolio.py and runs the algorithm
table1.py       : run this to obtain the results of Table 1 in the paper
example1.py     : run this to get Grid-FW results for Example 1 in Table 2
example2.py     : run this to get Grid-FW results for Example 2 in Table 2
example3.py     : run this to get Grid-FW results for Example 3 in Table 2
```
```
main.ipynb      : Jupyter notebook combining all the above experiments
```
```
big-M-cplex.jl  : Julia code for sparse portfolio via CPLEX Big-M approach
```



# Execution Instructions
There are two ways you can execute Grid-FW:
  1. [Open main.ipynb](./main.ipynb) in Jupyter and follow the instructions given on the notebook.
  2. Alternatively, run each Python file according to the instructions provided in the comments.

# Important Notes
  1. A license is needed to run CPLEX.
  2. The third synthetic dataset with p = 3000 used in Example 3 in Table 2 is NOT included in here due to its size being more than 350MB. 
 
