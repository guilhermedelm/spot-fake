import igraph as ig
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

"""função que cria gráfico 3d contendo data1 , data 2 , data 3 . Utiliza plt."""
def plot_3dgraph(data1,data2,data3,communities,scale_size=True):
    
    #função normalizadora do tamanho com base na métrica pagerank
    def scale(values, min_size=5, max_size=100):
        min_val = min(values)
        max_val = max(values)
        return [min_size + (v - min_val) / (max_val - min_val + 1e-6) * (max_size - min_size) for v in values]
    
    fig=plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')

    betweenness = data2 if "betweenness" in subgraph.vs.attributes() else g.betweenness()

    if scale_size:
        sizes = scale(data1)
    else:
        sizes = 50  # tamanho fixo

    sc = ax.scatter(
        data1,
        data2,
        data3,
        c=communities,  
        cmap='tab20',
        alpha=0.8,
        s = sizes
    )
    ax.set_xlabel("PageRank")
    ax.set_ylabel("Betweenness")
    ax.set_zlabel("Communities Connected")
    plt.title("Correlação entre métricas dos nós")

    plt.colorbar(sc, label="Comunidade")
    plt.tight_layout()
    plt.show()



            
        
"""
    Plota um grafo iGraph com os tamanhos dos nós baseados em uma métrica (pagerank, betweenness, etc.)
"""

def plot_graph_with_metric(graph, metric_values, filename="grafo_metric.png", layout_type="fr", 
                           min_size=20, max_size=80, use_communities=True, show_labels=True):
    #função normalizadora do tamanho com base na métrica pagerank
    def scale(values, min_size, max_size):
        min_val, max_val = min(values), max(values)
        return [
            min_size + (v - min_val) / (max_val - min_val + 1e-6) * (max_size - min_size)
            for v in values
        ]

    node_sizes = scale(metric_values, min_size, max_size)
    

    # Cores por comunidade, se existir
    if use_communities and "community" in graph.vs.attributes():
        colors = [plt.cm.tab20(c % 20) for c in graph.vs["community"]]
    else:
        colors = "lightblue"

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


def plot_graph_with_metric(graph, metric_values, filename="grafo_local.png", layout_type="fr", 
                           min_size=0, max_size=80, use_communities=True, show_labels=True, filter=0):
    
    def scale(values, min_size, max_size):
        min_val, max_val = min(values), max(values)
        return [
            min_size + (v - min_val) / (max_val - min_val + 1e-6) * (max_size - min_size)
            for v in values
        ]

    # Filtrar vértices com base no atributo "communities_connected"
    vertices_conexoes = [v.index for v in graph.vs if v["communities_connected"] > filter]
    subgraph = graph.subgraph(vertices_conexoes)

    # Filtrar os valores da métrica com base nos vértices do subgrafo
    metric_values_sub = [metric_values[v.index] for v in subgraph.vs]

    node_sizes = scale(metric_values_sub, min_size, max_size)

    # Cores por comunidade
    if use_communities and "community" in subgraph.vs.attributes():
        colors = [plt.cm.tab20(c % 20) for c in subgraph.vs["community"]]
    else:
        colors = "lightblue"

    layout = subgraph.layout(layout_type)

    ig.plot(
        subgraph,
        layout=layout,
        vertex_size=node_sizes,
        vertex_color=colors,
        vertex_label=subgraph.vs["communities_connected"] if show_labels and "communities_connected" in subgraph.vs.attributes() else None,
        bbox=(1000, 800),
        target=filename
    )
    print(f"✅ Grafo salvo como {filename}")

# 1. Carregar dados
df = pd.read_csv("twitter_combined.txt", sep=r"\s+", header=None, names=["user1", "user2"])
print(f"📄 Dados carregados: {len(df)} interações")

# 2. Criar grafo direcionado
g = ig.Graph.TupleList(df.itertuples(index=False), directed=True)
print(f"🔧 Grafo criado: {g.vcount()} nós, {g.ecount()} arestas")

# 3. PageRank
pagerank = g.pagerank(damping=0.85)
g.vs["pagerank"] = pagerank
print("📈 PageRank calculado")

# 4. Detecção de comunidades (Louvain requer grafo não direcionado)
g_undirected = g.as_undirected()
communities = g_undirected.community_multilevel()
g.vs["community"] = communities.membership
print(f"🧠 {len(communities)} comunidades detectadas")

# calcular betwenness

"""betweenness=g.betweenness()
g.vs["betweennes"]=betweenness
print(f"{len(betweenness)} comunidades detectadas")"""

#calcular conexão entre comunidades
communities_connected=[]

for v in g.vs:
    vizinhos=g.neighbors(v.index,mode="OUT")
    viz_communities=set([g.vs[n]["community"] for n in vizinhos])
    num_communities=len(viz_communities)
    communities_connected.append(num_communities)

g.vs["communities_connected"]=communities_connected

print(f"{len(communities_connected)}comunidades conectadas")

# 5. Criar subgrafo com top 100 por PageRank
top_100_ids = sorted(range(len(pagerank)), key=lambda i: pagerank[i], reverse=True)[:1000]
subgraph = g.subgraph(top_100_ids)
print(f"🔍 Subgrafo com top 100 nós criado: {subgraph.vcount()} nós, {subgraph.ecount()} arestas")

# Subrafo com top 100 betwennes em cada comunidade
"""top_100_comunidades= sorted(range(len(betweenness)),key=lambda i: betweenness[i], reverse=True)[:1000]
betweenness_subgraph=g.subgraph(top_100_comunidades)
print(f"🔍 Subgrafo com top 100 nós criado: {betweenness_subgraph.vcount()} nós, {betweenness_subgraph.ecount()} arestas")"""

""""betweenness1=g.betweenness()
g.vs["betweenness"]=betweenness1"""

betweenness=subgraph.betweenness()
subgraph.vs["betweenness"]=betweenness

for v in subgraph.vs:
    vizinhos=subgraph.neighbors(v.index,mode="OUT")
    viz_communities=set([subgraph.vs[n]["community"] for n in vizinhos])
    num_communities=len(viz_communities)
    communities_connected.append(num_communities)


#Grafo com pagerank
plot_graph_with_metric(
    subgraph,
    subgraph.vs["pagerank"],
    filename="grafo_pagerank.png",
    layout_type="fr",
    min_size=20,
    max_size=80,
    use_communities=True,
    show_labels=True
)

#grafo com betweenness
plot_graph_with_metric(
    subgraph,
    subgraph.vs["betweenness"],
    filename="grafo_betweenness.png",
    layout_type="fr",
    min_size=0,
    max_size=80,
    use_communities=True,
    show_labels=False
)

plot_graph_with_metric(
    g,
    g.vs["communities_connected"],
    filename="grafo_filtrado.png",
    layout_type="fr",
    min_size=20,
    max_size=80,
    use_communities=True,
    show_labels=True,
    filter=20
)
plot_3dgraph(
    subgraph.vs["pagerank"],
    subgraph.vs["betweenness"],
    subgraph.vs["communities_connected"],
    subgraph.vs["community"]
)