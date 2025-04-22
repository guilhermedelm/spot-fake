import igraph as ig
import pandas as pd
import matplotlib.pyplot as plt


# Função para plotar grafo com métricas
def plot_graph_pagerank(graph,metric_values,filename="grafo_local.png",layout_type="fr", min_size=0,max_size=80,use_communities=True,show_labels=True):
    
    def scale(values,min_size,max_size):
        min_val,max_val=min(values),max(values)
        if (v<=(max_val - min_val)/2):
            return[0]
        
        else:
            return [
                min_size + (v - min_val) / (max_val - min_val + 1e-6) * (max_size - min_size)
                for v in values
            ]    
        
    node_sizes = scale(metric_values,min_size,max_size)

    if use_communities and "community" in graph.vs.attributes():
        colors = [plt.cm.tab20(c % 20) for c in graph.vs["community"]]
    else:
        colors = "lightblue"

    layout = graph.layout(layout_type)

    ig.plot(
        graph[100:],
        layout=layout,
        vertex_size=node_sizes,
        vertex_color=colors,
        vertex_label=graph.vs["name"] if show_labels else None,
        bbox=(1000, 800),
        target=filename
    )
    print(f"✅ Grafo salvo como {filename}")


            
        

def plot_graph_with_metric(graph, metric_values, filename="grafo_metric.png", layout_type="fr", 
                           min_size=20, max_size=80, use_communities=True, show_labels=True):
    """
    Plota um grafo iGraph com os tamanhos dos nós baseados em uma métrica (pagerank, betweenness, etc.)
    """
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

# 5. Criar subgrafo com top 100 por PageRank
top_100_ids = sorted(range(len(pagerank)), key=lambda i: pagerank[i], reverse=True)[:100]
subgraph = g.subgraph(top_100_ids)
print(f"🔍 Subgrafo com top 100 nós criado: {subgraph.vcount()} nós, {subgraph.ecount()} arestas")

# 6. Plotar subgrafo com nós escalonados por PageRank
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