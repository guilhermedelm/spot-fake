import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Carregar dataset usando raw string para o separador e definir os nomes das colunas
df = pd.read_csv('twitter_combined.txt', sep=r'\s+', header=None)
df.columns = ['user1', 'user2']
df.to_csv('arquivo.csv', index=False)

# Criar grafo direcionado
G = nx.DiGraph()
edges = df[['user1', 'user2']].values.tolist()
G.add_edges_from(edges)

# PageRank
pagerank = nx.pagerank(G, alpha=0.85)

# Detectar comunidades (convertendo para não direcionado)
from networkx.algorithms.community import greedy_modularity_communities
communities = greedy_modularity_communities(G.to_undirected())

# Desenhar o grafo (atenção: grafos grandes podem travar)
plt.figure(figsize=(10, 7))
nx.draw(G, with_labels=False, node_size=20)
plt.savefig("grafo.png")

print("Grafo salvo como grafo.png!")