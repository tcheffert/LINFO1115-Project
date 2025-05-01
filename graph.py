import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from template_utils import *
from matplotlib.ticker import MaxNLocator, FuncFormatter

#---- Task 1 plotting ----#
def plot_degree_distribution_hist(degree_hist):
    """
    Affiche l'histogramme de la distribution des degrés.
    Param:
        degree_hist: liste contenant le nombre de nœuds pour chaque degré (index = degré)
    """
    degrees = list(range(len(degree_hist)))
    counts = degree_hist

    plt.figure(figsize=(10, 5))
    bars = plt.bar(degrees, counts, color='darkseagreen', edgecolor='black')

    # Ajouter les valeurs au-dessus de chaque barre
    for bar, count in zip(bars, counts):
        if count > 0:
            plt.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(counts)*0.01,
                     str(count),
                     ha='center',
                     fontsize=8,
                     alpha=0.75)

    plt.xlabel('Node Degree')
    plt.ylabel('Number of Nodes')
    plt.title('Histogram of Node Degree Distribution')
    plt.xticks(degrees)  # Affiche tous les degrés sur l'axe x
    plt.grid(axis='y', alpha=0.4, linestyle='--')
    plt.tight_layout()
    plt.savefig("task1_graph.pdf", format='pdf')
    # plt.show()

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
    # Construction de la liste d'adjacence
    adj = build_adjacency_list(dataframe)

    # Récupère les distances des plus courts chemins
    result = shortest_paths(adj)
    if not result:
        print("No paths found.")
        return

    max_len = 10
    path_counts = [0, 49170, 14097058, 60086196, 49765362, 4008098, 453222, 98686, 25774, 782, 88]
    # print(path_counts)
    # print(all(isinstance(x, int) for x in path_counts))  # Doit retourner True

    # Préparation des données x et y
    x = list(range(1, max_len + 1))
    y = path_counts[1:]

    # Plot principal
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, marker='o', linestyle='-', color='firebrick', linewidth=2, markersize=6)

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
#Task 1 plot
# degree_hist = [0, 7601, 1576, 638, 375, 263, 171, 119, 87, 91, 66, 50, 41, 26, 26, 28, 15, 18, 11, 14, 17]
# plot_degree_distribution_hist(degree_hist)
#Task 2 plot
# plot_avg_incoming_edges_vs_score(df)
#Task 3 plot
# plot_shortest_path_distribution(df)
