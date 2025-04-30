# First, import the libraries needed for your helper functions
import numpy as np
import pandas as pd
import networkx as nx

# Then write the classes and/or functions you wish to use in the exercises

def build_adjacency_list(dataframe):
    """
    Build an adjacency list from a dataframe.
    The dataframe contains three columns: source, target, and weight.
    """
    # Build the graph using adjacency list
    adj = {}

    # Iterate through the dataframe rows
    for row in dataframe.values:
        u, v = row[0], row[1]
        if u not in adj:
            adj[u] = set()
        if v not in adj:
            adj[v] = set()
            
        # Add edges to the adjacency list
        adj[u].add(v)
        adj[v].add(u)

    return adj

# Pour la Q5
def build_adjacency_list_without_self_loops(dataframe):
    # Build the graph using adjacency list without self-loops
    adj = {}
    for row in dataframe.values:
        u, v = row[0], row[1]
        # Check for self-loops
        if u == v:
            continue
        # Add edges to the adjacency list
        if u not in adj:
            adj[u] = set()
        if v not in adj:
            adj[v] = set()
        adj[u].add(v)
        adj[v].add(u)

    return adj
    


def dfs(u, graph, visited, disc, low, parent, bridges, time):
    """
    Perform DFS to find bridges in the graph.
    """
    # Initialize variables
    visited[u] = True
    disc[u] = low[u] = time[0]
    time[0] += 1

    # Visit all adjacent vertices
    for v in graph[u]:
        # If v is not visited, recurse on it
        if v not in visited:
            parent[v] = u
            dfs(v, graph, visited, disc, low, parent, bridges, time) # DFS recursion
            low[u] = min(low[u], low[v])
            if low[v] > disc[u]:
                bridges.append((u, v))
        # If v is already visited and is not the parent of u, update low value of u
        elif v != parent.get(u):
            low[u] = min(low[u], disc[v])

def find_bridges(graph):
    """
    Find all bridges in an undirected graph using DFS.
    """
    # Initialize variables
    time = [0]
    visited = {}
    low = {}
    disc = {}
    parent = {}
    bridges = []

    for node in graph:
        if node not in visited:
            dfs(node, graph, visited, disc, low, parent, bridges, time) # DFS

    return bridges

def is_local_bridge(graph, u, v):
    """
    Determine if edge (u, v) is a local bridge.
    """
    graph[u].remove(v)
    graph[v].remove(u)

    visited = {u}
    queue = [(u, 0)]
    found = False
    while queue:
        current, dist = queue.pop(0)
        if current == v:
            found = True
            break
        for neighbor in graph.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    graph[u].add(v)
    graph[v].add(u)

    return not found or dist > 2

def shortest_paths(adj):
    """
    Compute shortest path lengths between all node pairs across all connected components.
    Avoids double counting by only counting (u, v) where u < v.
    Uses BFS for each node.
    """
    visited_pairs = set()
    path_lengths = []

    for start in adj:
        visited = {start: 0}
        queue = [start]

        while queue:
            current = queue.pop(0)
            for neighbor in adj[current]:
                if neighbor not in visited:
                    visited[neighbor] = visited[current] + 1
                    queue.append(neighbor)

        for target, dist in visited.items():
            # Eviter de compter 2 fois les paires (u, v) et (v, u)
            if start < target:  
                path_lengths.append(dist)

    return path_lengths

def count_triplets(adj):
    """
    Count the number of connected triplets in the graph.
    A connected triplet consists of a node connected to two others (i.e., degree ≥ 2).
    """
    total = 0
    for node in adj:
        deg = len(adj[node])
        if deg >= 2:
            total += deg * (deg - 1) // 2 
    return total
