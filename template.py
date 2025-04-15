import numpy as np
import pandas as pd
import sys 
from template_utils import *
from scipy.stats import t
p_value=t.sf
#import networkx as nx

sys.setrecursionlimit(6000)
#A line of the dataset (u, v, w) means that node u has opinion w on node v.
# Undirected graph
# Task 1: Basic graph properties
def Q1(dataframe):
    u_df=undirect(dataframe).copy()
    
    #Q1.1
    unique_values=sorted(u_df[0].unique())
    full_range = list(range(min(unique_values), max(unique_values) + 1))
    missing_values = list(set(full_range) - set(unique_values))

    
    current=u_df[0][0]
    occ=0
    degree_list=[]
    for i in u_df[0]:
        if i==current:
            occ+=1
        else:
            degree_list+=[occ]
            current=i
            occ=1
    degree_list+=[0]*len(missing_values)
    degree_mean=np.mean(degree_list)
    #Q1.2    
    hist = [0] * 21
    for i in degree_list:
        if i<21:
            hist[i]+=1
    hist[0]+=len(missing_values)
    # Q1.3
    adj = {}
    for row in u_df.values:
        u, v = row[0], row[1]
        if u not in adj:
            adj[u] = set()
        if v not in adj:
            adj[v] = set()
        adj[u].add(v)
        adj[v].add(u)
        
    bridges = find_bridges(adj)

    # Q1.4
    local_bridges = 0
    for u in adj:
        for v in adj[u]:
            if u < v and (u, v) not in bridges and (v, u) not in bridges:
                if is_local_bridge(adj, u, v):
                    local_bridges += 1

    # Q1.5
    degrees = []
    for u in adj:
        for v in adj[u]:
            if u < v and (u, v) not in bridges and (v, u) not in bridges:
                if is_local_bridge(adj, u, v):
                    degrees.append(len(adj[u]))
                    degrees.append(len(adj[v]))

    if degrees:
        mean_lb = sum(degrees) / len(degrees)
        std_lb = (sum((x - mean_lb) ** 2 for x in degrees) / len(degrees)) ** 0.5
        n = len(degrees)
        t_stat = (mean_lb - degree_mean) / (std_lb / n**0.5) if std_lb != 0 else 0
        p_v = 2 * t.sf(abs(t_stat), n-1)

    else:
        p_v = 1.0

    return [degree_mean, hist, len(bridges), local_bridges, p_v]
# Directed graph
# Task 2: Best score node
def Q2(dataframe):
     # the id of the node with the highest score and its score
    df=dataframe.copy()
    p=[0]*(df[1].max()+1)
    for i in df[1]:
        p[i]+=1
    index=0
    max=0
    for i in range(len(p)):
        if p[i]>max:
            max=p[i]
            index=i
    return [index, max] # the id of the node with the highest score and its score
# Undirected graph
# Task 3: Paths lengths analysis
def Q3(dataframe):
    u_df = undirect(dataframe).copy()

    # Build the undirected graph using adjacency list
    adj = {}
    for row in u_df.values:
        u, v = row[0], row[1]
        if u not in adj:
            adj[u] = set()
        if v not in adj:
            adj[v] = set()
        adj[u].add(v)
        adj[v].add(u)

    # Compute all shortest paths
    result = shortest_paths(adj)

    if not result:
        return [0]  # If graph is empty

    max_len = max(result)
    path_counts = [0] * max_len

    for dist in result:
        path_counts[dist - 1] += 1

    return [max_len] + path_counts # at index 0, the diameter of the largest connected component, at index 1 the total number of shortest paths of length 1 accross all components,
    # at index the 2 the number of shortest paths of length 2...
    

# Directed graph
# Task 4: PageRank
def Q4(dataframe):
    # Build adjacency lists from the dataframe
    df=dataframe.copy()
    edges = df[[0, 1]].values
    outgoing = {}
    incoming = {}

    nodes = set()
    for src, dst in edges:
        nodes.update([src, dst])
        outgoing.setdefault(src, set()).add(dst)
        incoming.setdefault(dst, set()).add(src)
        outgoing.setdefault(dst, set())  
        incoming.setdefault(src, set())

    nodes = list(nodes)
    N = len(nodes)
    d = 0.85
    pr = {node: 1.0 / N for node in nodes}
    convergence_threshold = N * 1e-6
    delta = 1

    while delta > convergence_threshold:
        new_pr = {}
        delta = 0

        for node in nodes:
            sum_contrib = 0.0
            for src in incoming[node]:
                out_degree = len(outgoing[src])
                if out_degree > 0:
                    sum_contrib += pr[src] / out_degree

            new_score = (1 - d) / N + d * sum_contrib
            new_pr[node] = new_score
            delta += abs(new_score - pr[node])

        pr = new_pr

    total_pr = sum(pr.values())
    for node in pr:
        pr[node] /= total_pr

    max_node = max(pr, key=pr.get)
    max_score = pr[max_node]

    return [max_node, max_score] # the id of the node with the highest pagerank score, the associated pagerank value.
    # Note that we consider that we reached convergence when the sum of the updates on all nodes after one iteration of PageRank is smaller than 10^(-6)

# Undirected graph
# Task 5: Relationship analysis 
def Q5(dataframe):
    # Convert to undirected graph (with signed edges)
    u_df = undirect(dataframe).copy()

    # Build adjacency list and edge sign dictionary
    adj = {}          
    edge_sign = {}   

    for row in u_df.values:
        u, v, w = row
        if u not in adj:
            adj[u] = set()
        if v not in adj:
            adj[v] = set()
        adj[u].add(v)
        adj[v].add(u)
        edge_sign[frozenset([u, v])] = w

    # Find triangles
    triangles = set()
    for u in adj:
        neighbors = list(adj[u])
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                v = neighbors[i]
                w = neighbors[j]
                if v in adj and w in adj[v]:
                    triangle = tuple(sorted([u, v, w]))
                    triangles.add(triangle)

    # Classify triangles
    balanced = 0
    unbalanced = 0
    for t in triangles:
        u, v, w = t
        s1 = edge_sign[frozenset([u, v])]
        s2 = edge_sign[frozenset([v, w])]
        s3 = edge_sign[frozenset([w, u])]
        if s1 * s2 * s3 > 0:
            balanced += 1
        else:
            unbalanced += 1

    # Count total triplets (open + closed)
    def count_triplets(adj):
        count = 0
        for node in adj:
            deg = len(adj[node])
            if deg >= 2:
                count += deg * (deg - 1) // 2
        return count

    num_closed_triplets = 3 * len(triangles)
    total_triplets = count_triplets(adj)
    num_open_triplets = total_triplets - num_closed_triplets
    gcc = num_closed_triplets / num_open_triplets if num_open_triplets > 0 else 0

    return [len(triangles), balanced, unbalanced, gcc] # number of triangles, number of balanced triangles, number of unbalanced triangles and the GCC.

# you can write additionnal functions that can be used in Q1-Q5 functions in the file "template_utils.py", a specific place is available to copy them at the end of the Inginious task.

print("Reading epinion.txt ...")
df = pd.read_csv('epinion.txt', header=None,sep="    ", engine="python")
print("Reading done.")
#print("Q1", Q1(df))
print("Q2", Q2(df))
print("Q3", Q3(df))
print("Q4", Q4(df))
print("Q5", Q5(df))

