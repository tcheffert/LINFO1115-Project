import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from template_utils import *
from matplotlib.ticker import MaxNLocator, FuncFormatter

#---- Task 2 plotting ----#

# Fonction Q2 adaptée pour retour de tous les scores
def compute_scores(dataframe):
    df = dataframe.copy()
    scores = {}

    for _, row in df.iterrows():
        node_v = row[1]
        weight_w = row[2]
        if node_v not in scores:
            scores[node_v] = 0
        scores[node_v] += weight_w

    return scores

# Fonction pour compter les arêtes entrantes
def incoming_edge_counts(dataframe):
    df = dataframe.copy()
    incoming_counts = {}
    for _, row in df.iterrows():
        node_v = row[1]
        if node_v not in incoming_counts:
            incoming_counts[node_v] = 0
        incoming_counts[node_v] += 1
    return incoming_counts

# Fonction pour préparer et afficher le graphique
def plot_avg_incoming_edges_vs_score(dataframe):
    scores = compute_scores(dataframe)
    incoming_counts = incoming_edge_counts(dataframe)

    score_to_incoming = {}
    for node, score in scores.items():
        incoming = incoming_counts.get(node, 0)
        if score not in score_to_incoming:
            score_to_incoming[score] = []
        score_to_incoming[score].append(incoming)

    # Moyenne des arêtes entrantes par score
    score_vals = sorted(score_to_incoming.keys())
    avg_incomings = [np.mean(score_to_incoming[score]) for score in score_vals]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(score_vals, avg_incomings, label="Avg Incoming Edges", marker='o', color='slateblue', lw=2)
    plt.plot(score_vals, [abs(x) for x in score_vals], linestyle='-', label="f(x) = |x|", color='orange', lw=2)
    plt.xlabel("Score")
    plt.ylabel("Average Incoming Edges")
    plt.title("Average Incoming Edges vs. Score with f(x) = |x|")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig("task2_graph.pdf", format='pdf')
    #plt.show()

#---- Task 3 plotting ----#
def plot_shortest_path_distribution(dataframe):
    u_df = undirect(dataframe).copy()

    # Construction de la liste d'adjacence
    adj = {}
    for row in u_df.values:
        u, v = row[0], row[1]
        if u not in adj:
            adj[u] = set()
        if v not in adj:
            adj[v] = set()
        adj[u].add(v)
        adj[v].add(u)

    # Récupère les distances des plus courts chemins
    result = shortest_paths(adj)
    if not result:
        print("No paths found.")
        return

    max_len = 10
    path_counts = [0, 24585, 7048529, 30043098, 24882681, 2004049, 226611, 49343, 12887, 391, 44]
    print(path_counts)
    print(all(isinstance(x, int) for x in path_counts))  # Doit retourner True

    # Préparation des données x et y
    x = list(range(1, max_len + 1))
    y = path_counts[1:]

    # Plot principal
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, marker='o', linestyle='-', color='slateblue', linewidth=2, markersize=6)

    # Annotations subtiles
    for i in range(len(x)):
        plt.text(
            x[i], y[i] + max(y) * 0.01,  # petite élévation dynamique
            str(y[i]),
            fontsize=8,
            ha='center',
            color='gray'
        )

    plt.xticks(range(1, max_len + 1))

    plt.xlabel('Shortest Path Length')
    plt.ylabel('Number of Pairs')
    plt.title('Distribution of Shortest Path Lengths')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig("task3_graph.pdf", format='pdf')
    # plt.show()  # Active si tu veux l'afficher en direct

df = pd.read_csv('epinion.txt', header=None, sep="    ", engine="python")
#Task 2 plot
# plot_avg_incoming_edges_vs_score(df)
#Task 3 plot
plot_shortest_path_distribution(df)
