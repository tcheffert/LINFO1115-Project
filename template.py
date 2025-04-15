import numpy as np
import pandas as pd
import sys 
from template_utils import *
from scipy.stats.t import sf as p_value

sys.setrecursionlimit(6000)

# Undirected graph
# Task 1: Basic graph properties
def Q1(dataframe):
    # Your code here
    return [0.0, 0, 0, 0.0] # [average degree, nb bridges, nb local bridges, p-value of the t-test]

# Directed graph
# Task 2: Best score node
def Q2(dataframe):
    # Your code here
    return [0, 0.0] # the id of the node with the highest score and its score

# Undirected graph
# Task 3: Paths lengths analysis
def Q3(dataframe):
    # Your code here
    return [0, 0, 0, 0, 0 ...] # at index 0, the diameter of the largest connected component, at index 1 the total number of shortest paths of length 1 accross all components,
    # at index the 2 the number of shortest paths of length 2...
    

# Directed graph
# Task 4: PageRank
def Q4(dataframe):
    # Your code here
    return [0, 0.0] # the id of the node with the highest pagerank score, the associated pagerank value.
    # Note that we consider that we reached convergence when the sum of the updates on all nodes after one iteration of PageRank is smaller than 10^(-6)

# Undirected graph
# Task 5: Relationship analysis 
def Q5(dataframe):
    # Your code here
    return [0, 0, 0, 0.0] # number of triangles, number of balanced triangles, number of unbalanced triangles and the GCC.

# you can write additionnal functions that can be used in Q1-Q5 functions in the file "template_utils.py", a specific place is available to copy them at the end of the Inginious task.

df = pd.read_csv('epinion.txt', header=None,sep="    ", engine="python")
print("Q1", Q1(df))
print("Q2", Q2(df))
print("Q3", Q3(df))
print("Q4", Q4(df))
print("Q5", Q5(df))