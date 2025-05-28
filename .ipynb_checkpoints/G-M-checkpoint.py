import igraph as ig
import pandas as pd
import matplotlib.pyplot as plt

centralidadels=()
def desglobal(graph,metric_values,filename="graph_global.png",layout_type="fr",min_size=20,max_size=80,use_communities=True,show_labels=True)
    

    #calculando tamanho dos nodos
    def scale(values, min_size, max_size):
        min_val, max_val = min(values), max(values)
        return [
            min_size + (v - min_val) / (max_val - min_val + 1e-6) * (max_size - min_size)
            for v in values
        ]
    node_sizes = scale(metric_values, min_size, max_size)

    #cor das comunidades
    if use_communities and "community" in graph.vs.attributes():
        colors = [plt.cm.tab20(c % 20) for c in graph.vs["community"]]
    else:
        colors = "lightblue"

    # Layout e plot
    layout = graph.layout(layout_type)
    ig.plot(
        graph,
        layout=layout,
        vertex_size=node_sizes,
        vertex_color=colors,
        vertex_label=graph.vs["name"] if show_labels else None,
        bbox=(1000, 800),
        target=filename
    )
    print(f"✅ Grafo salvo como {filename}")


    


# 1. Carregar dados
df = pd.read_csv("twitter_combined.txt", sep=r"\s+", header=None, names=["user1", "user2"])

# 2. Criar grafo dirigido
g = ig.Graph.TupleList(df.itertuples(index=False), directed=True)
print(f"Nós: {g.vcount()}, Arestas: {g.ecount()}")

# 3. PageRank
pagerank = g.pagerank(damping=0.85)
g.vs["pagerank"] = pagerank

# 4. Comunidades (Louvain)
communities = g.as_undirected().community_multilevel()
g.vs["community"] = communities.membership
print(f"Detectadas {len(communities)} comunidades")

# 5. centralidade de informação
betweenness=g.betweenness()
g.vs["betweenness"]=betweenness

top_bet

# 5. Selecionar top 100 nós por PageRank
top_100_ids = sorted(range(len(pagerank)), key=lambda i: pagerank[i], reverse=True)[:100]
subgraph = g.subgraph(top_100_ids)

# 6. Layout e visualização
layout = subgraph.layout("fr")  # Fruchterman-Reingold layout
colors = [plt.cm.tab20(c % 20) for c in subgraph.vs["community"]]

# Converter cores para formato hexadecimal que igraph entende
colors_hex = [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}' for r, g, b, _ in colors]

ig.plot(
    subgraph,
    layout=layout,
    vertex_size=20,
    vertex_color=colors_hex,
    vertex_label=subgraph.vs["name"],
    bbox=(1000, 800),
    target="grafo_igraph.png"
)

print("✅ Subgrafo salvo como grafo_igraph.png!")


#ordenar por pagerank
df_pr = pd.DataFrame({
    "user": g.vs["name"],
    "pagerank": pagerank
})

# Ordenar do maior pro menor
df_pr = df_pr.sort_values(by="pagerank", ascending=False)

# Mostrar os 10 primeiros
print(df_pr.head(10))


