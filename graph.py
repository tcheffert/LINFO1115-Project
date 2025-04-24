import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("task2_graph.pdf", format='pdf')
    #plt.show()

#Task 2 plot
plot_avg_incoming_edges_vs_score(pd.read_csv('epinion.txt', header=None, sep="    ", engine="python"))