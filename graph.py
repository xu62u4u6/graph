import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class Graph(dict):
    def __init__(self):
        super().__init__()

    @property
    def nodes(self):
        nodes = set(self.keys())
        for targets in self.values():
            for target in targets:
                nodes.add(target)
        return sorted(list(nodes))

    @property
    def edges(self):
        edges = []
        for src in self:
            for dst in self[src]:
                edges.append((src, dst))
        return sorted(edges)
    
    def add_edge(self, src, dst):
        if src in self.keys():
            self[src].append(dst)
        else:
            self[src] = [dst]

        if dst not in self.keys():
            self[dst] = []

    def nx_plot(self, layout_hunc=nx.spring_layout, is_direct=True):
        if is_direct:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        G.add_edges_from(self.edges)
        pos = layout_hunc(G)
        nx.draw(G, pos, with_labels=True, node_size=300, node_color='skyblue', font_size=12, font_color='black', font_weight='bold')
        plt.show()
    
    def copy(self):
        new_graph = Graph()
        new_graph.update(self)
        return new_graph
    
    def to_undirected_graph(self):
        undirected_graph = Graph()
        for src in self:
            for dst in self[src]:
                undirected_graph.add_edge(src, dst)
                if src == dst:
                    continue
                undirected_graph.add_edge(dst, src)
        return undirected_graph
    
    def load_graph_from_edges_file(self, path):
        with open(path, "r") as f:
            lines = f.readlines()

        for line in lines:
            src, dst = line.strip().split("\t\t")
            self.add_edge(src, dst)