import numpy as np
import pandas as pd
import sys
from template_utils import *
from scipy.stats import t
p_value = t.sf

sys.setrecursionlimit(6000)

# A line of the dataset (u, v, w) means that node u has opinion w on node v.
# Undirected graph

#---- Task 1: Basic graph properties ----#
def Q1(dataframe):
    # Undirected adjacency list WITH self-loops
    adj = build_adjacency_list(dataframe)

    # Q1.1: Degree mean
    degree_count = {}
    for node, neighbors in adj.items():
        degree = len(neighbors)
        if node in neighbors:
            degree += 1  # Self-loop counts twice
        degree_count[node] = degree

    degree_mean = sum(degree_count.values()) / len(degree_count)

    # Q1.2: Histogram of degrees (up to 20)
    hist = [0] * 21
    for deg in degree_count.values():
        if deg < 21:
            hist[deg] += 1

    # Q1.3: Bridges
    bridges = find_bridges(adj)
    bridges_set = set((min(u, v), max(u, v)) for u, v in bridges) # Store bridges as sorted tuples

    # Q1.4 & Q1.5: Local Bridges and degree stats combined
    local_bridges = 0
    degrees = []

    for u in adj:
        for v in adj[u]:
            if u < v:
                # Vérifier s'il existe un voisin commun ≠ u et ≠ v
                # (u, v) est un local bridge uniquement s’il n’a aucun voisin commun autre que u ou v
                common_neighbors = adj[u].intersection(adj[v]) - {u, v} 
                if not common_neighbors:
                    local_bridges += 1
                    # Seulement si ce n'est pas un pont global
                    if (u, v) not in bridges_set and (v, u) not in bridges_set:
                        deg_u = len(adj[u]) + (1 if u in adj[u] else 0)
                        deg_v = len(adj[v]) + (1 if v in adj[v] else 0)
                        degrees.extend([deg_u, deg_v])

    # Q1.5: Statistical test
    if degrees:
        mean_lb = sum(degrees) / len(degrees)
        std_lb = (sum((x - mean_lb) ** 2 for x in degrees) / len(degrees)) ** 0.5   # Standard deviation for local bridges
        n = len(degrees)
        t_stat = (mean_lb - degree_mean) / (std_lb / n**0.5) if std_lb != 0 else 0
        p_v = 2 * t.sf(abs(t_stat), n - 1)
    else:
        p_v = 1.0

    return [float(degree_mean), hist, len(bridges), local_bridges, p_v]



# Directed graph
#---- Task 2: Best score node ----#
def Q2(dataframe):
    df = dataframe.copy()
    scores = {}  # Dico pour stocker les scores des nodes: {node_i:score_i}

    # Parcours de chaque ligne de la df
    for _, row in df.iterrows():
        node_v = row[1]   #Colonne 1
        weight_w = row[2] #Colonne 2

        # Si node_v pas dans le dico => init score à 0
        if node_v not in scores:
            scores[node_v] = 0

        #Add poids w au score du node v
        scores[node_v] += weight_w

    # Parcours les scores pour trouver le noeud avec le plus grd score
    best_node = None
    max_score = float('-inf') 
    for node, score in scores.items():
        if score > max_score:
            max_score = score
            best_node = node

    # Return l'id du node avec le plus grd score et celui-ci
    return [int(best_node), int(max_score)]

# Undirected graph
#---- Task 3: Paths lengths analysis ----#
def Q3(dataframe):
    # Undirected adjacency list WITH self-loops
    adj = build_adjacency_list(dataframe)

    # Compute all shortest paths
    result = shortest_paths(adj)

    if not result:
        return [0]  # If graph is empty

    max_len = max(result)
    path_counts = [0] * max_len

    for dist in result:
        path_counts[dist - 1] += 2  # Each path is counted twice (u, v) and (v, u) -> https://moodle.uclouvain.be/mod/forum/discuss.php?d=174502

    diameter = max_len
    # at index 0, the diameter of the largest connected component, at index 1 the total number of shortest paths of length 1 accross all components,
    return [diameter] + path_counts
    # at index the 2 the number of shortest paths of length 2...


# Directed graph
# Task 4: PageRank
def Q4(dataframe):
    # Build adjacency lists from the dataframe
    df = dataframe.copy()
    edges = df[[0, 1]].values # Extraction des arêtes (paires source-destination) du dataframe

    # Dictionnaires pour stocker les listes d'adjacence sortantes et entrantes
    outgoing = {}
    incoming = {}

    nodes = set() # Set pour stocker les noeuds uniques

    # Parcours des arêtes pour construire les listes d'adjacence
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

    # PageRank algorithm
    while delta > convergence_threshold:
        new_pr = {}
        delta = 0

        # Calculate new PageRank scores
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

    # Normalize PageRank scores
    total_pr = sum(pr.values())
    for node in pr:
        pr[node] /= total_pr

    # Find the node with the highest PageRank score
    max_node = max(pr, key=pr.get)
    max_score = pr[max_node]

    # the id of the node with the highest pagerank score, the associated pagerank value.
    return [int(max_node), float(max_score)]
    # Note that we consider that we reached convergence when the sum of the updates on all nodes after one iteration of PageRank is smaller than 10^(-6)

# Undirected graph
# Task 5: Relationship analysis


def Q5(dataframe):
    # Build adjacency list and edge sign dictionary
    adj = build_adjacency_list_without_self_loops(dataframe)  #On retire les self-loops de la liste d'adjacence pour les triangles ! (definition)
    edge_sign = {}
    for row in dataframe.values:
        u, v, w = row
        edge_sign[frozenset([u, v])] = w

    # Find triangles
    triangles = set()
    for u in adj:
        neighbors = list(adj[u])
        # Check all pairs of neighbors
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
    # Check the sign of each triangle
    for t in triangles:
        u, v, w = t
        s1 = edge_sign[frozenset([u, v])]
        s2 = edge_sign[frozenset([v, w])]
        s3 = edge_sign[frozenset([w, u])]
        if s1 * s2 * s3 > 0:
            balanced += 1
        else:
            unbalanced += 1

    # Calculate GCC
    num_closed_triplets = 3 * len(triangles)
    total_triplets = count_triplets(adj)
    num_open_triplets = total_triplets - num_closed_triplets
    gcc = num_closed_triplets / num_open_triplets if num_open_triplets > 0 else 0

    # number of triangles, number of balanced triangles, number of unbalanced triangles and the GCC.
    return [len(triangles), balanced, unbalanced, gcc]

# you can write additionnal functions that can be used in Q1-Q5 functions in the file "template_utils.py", a specific place is available to copy them at the end of the Inginious task.

# ---- Running ---- #
print("Reading epinion.txt ...")
df = pd.read_csv('epinion.txt', header=None, sep="    ", engine="python")
print("Reading done.")
print("Q1 ▶", Q1(df))  #OK
print("Q2 ▶", Q2(df))  #OK
print("Q3 ▶", Q3(df))  #OK
print("Q4 ▶", Q4(df))  #OK
print("Q5 ▶", Q5(df))  #OK
