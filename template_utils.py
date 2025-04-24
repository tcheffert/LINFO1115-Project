# First, import the libraries needed for your helper functions
import numpy as np
import pandas as pd
import networkx as nx

# Then write the classes and/or functions you wish to use in the exercises

def undirect(dataframe):
    """
    Create an undirected version of the graph with only one edge per node pair.
    Keeps the first encountered weight (if needed).
    """
    seen = set()
    edges = []

    for row in dataframe.itertuples(index=False):
        u, v, w = row
        edge = tuple(sorted([u, v]))  # ensures (u, v) == (v, u)
        if edge not in seen:
            seen.add(edge)
            edges.append([edge[0], edge[1], w])  # keep u < v convention

    undirected_df = pd.DataFrame(edges, columns=[0, 1, 2])
    return undirected_df.sort_values(by=0).reset_index(drop=True)

def old_undirect(dataframe):
    """
    Make the graph undirected.
    Return an ordered dataframe with each edge duplicated in both directions (u, v) and (v, u).
    """
    df = dataframe.copy()
    reversed_edges = dataframe[[1, 0, 2]].copy()  
    reversed_edges.columns = [0, 1, 2]
    result_df = pd.concat([df, reversed_edges], ignore_index=True)
    return result_df.sort_values(by=0).reset_index(drop=True)

def find_bridges(graph):
    """
    Find all bridges in an undirected graph using DFS.
    """
    time = [0]
    visited = {}
    low = {}
    disc = {}
    parent = {}
    bridges = []

    def dfs(u):
        visited[u] = True
        disc[u] = low[u] = time[0]
        time[0] += 1

        for v in graph[u]:
            if v not in visited:
                parent[v] = u
                dfs(v)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    bridges.append((u, v))
            elif v != parent.get(u):
                low[u] = min(low[u], disc[v])

    for node in graph:
        if node not in visited:
            dfs(node)

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

def bfs(start, adj, visited: set):
        queue = [start]
        component = set()
        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                component.add(node)
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)
        return component

def largest_connected_component(adj):
    visited = set()
    largest_component = set()

    for node in adj:
        if node not in visited:
            component = bfs(node, adj, visited)
            if len(component) > len(largest_component):
                largest_component = component

    return largest_component

def diameter(adj, component):
    max_distance = 0

    for start in component:
        distances = {start: 0}
        queue = [start]
        while queue:
            current = queue.pop(0)
            for neighbor in adj[current]:
                if neighbor in component and neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        max_distance = max(max_distance, max(distances.values()))

    return max_distance


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
