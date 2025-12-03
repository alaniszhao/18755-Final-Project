import networkx as nx
import matplotlib.pyplot as plt
from scipy.io import mmread

# process graph and print n and M
A = mmread("soc-twitter-follows/soc-twitter-follows.mtx").tocsr()
G = nx.from_scipy_sparse_array(A, create_using=nx.DiGraph)
print(f"number of nodes: {G.number_of_nodes()}")
print(f"number of edges: {G.number_of_edges()}")

# calculate min/max/avg degree and plot degree histogram
degrees = [d for _, d in G.degree()]
print(f"min degree: {min(degrees)}")
print(f"max degree: {max(degrees)}")
print(f"avg degree: {sum(degrees)/len(degrees):.2f}")

plt.figure()
plt.hist(degrees, bins=50)
plt.xscale("log")
plt.yscale("log")
plt.title("Degree Distribution")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.show()

# calculate average clustering coefficient
avg_clustering = nx.average_clustering(G)
print(f"avg clustering coefficient: {avg_clustering:.4f}")

# convert graph to undirected & calculate GC size and approx diameter
print("undirected:")
components = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
giant = G.subgraph(components[0])
print(f"giant component size: {giant.number_of_nodes()} nodes, {giant.number_of_edges()} edges")

giant = giant.to_undirected()
est_diameter = nx.algorithms.approximation.diameter(giant)
print("approximate diameter:", est_diameter)

# calculate community sizes
communities = nx.community.louvain_communities(G.to_undirected(), seed=18755)
for i, c in enumerate(sorted(communities, key=len, reverse=True)[:5]):
    print(f"community {i+1}: {len(c)} nodes")