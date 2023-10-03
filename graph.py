import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class Graph:
    def __init__(self, name, is_directed=False):
        self.graph = {}
        self.name = name
        self.is_directed = is_directed

    def __str__(self):
        return str(self.graph)

    def add_edge(self, node, target):
        node = ord(node) - ord("a")
        target = ord(target) - ord("a")

        # 當前節點->目標節點
        if node in self.graph:
            self.graph[node].append(target)
        else:
            self.graph[node] = [target]

        if target not in self.graph:
            self.graph[target] = []

        # 若有向則結束
        if self.is_directed or node == target:
            return

        # 紀錄目標節點->當前節點
        if target in self.graph:
            self.graph[target].append(node)
        else:
            self.graph[target] = [node]

    def load_edges_from_edge_file(self, path):
        with open(path) as f:
            lines = f.readlines()

        for i in range(len(lines)):
            node1, node2 = lines[i].strip().split("\t")
            self.add_edge(node1, node2)

    def find_self_loop_nodes(self):
        adjacency_matrix = self.adjacency_matrix
        return np.where(adjacency_matrix*np.eye(self.node_num)==1)

    def find_multi_edges_nodes(self):
        adjacency_matrix = self.adjacency_matrix
        if self.is_directed:
            return np.where(adjacency_matrix>1)
        else:
            return np.where(np.triu(adjacency_matrix)>1)

    def degree_sequence(self):
        degree_sequence = []
        for node in self.graph.keys():
            degree_sequence.append(len(self.graph[node]))

    def to_edges(self, show_node_name=False):
        adjacency_matrix = self.adjacency_matrix
        x, y = np.where(n2.adjacency_matrix>0)
        if show_node_name:
            return [(chr(node1+ord("a")), chr(node2+ord("a"))) for node1, node2 in zip(x, y)]
        return [(node1, node2) for node1, node2 in zip(x, y)]

    @property
    def adjacency_matrix(self):
        adjacency_matrix = np.zeros((max(self.graph.keys())+1, max(self.graph.keys())+1), dtype=int)

        # 填充相鄰矩陣
        for node1 in self.graph:
            for node2 in self.graph[node1]:
                adjacency_matrix[node1, node2] += 1
        return adjacency_matrix

    def adjacency_heatmap(self):
        adjacency_matrix = self.adjacency_matrix
        sns.set(font_scale=1)
        plt.figure(figsize=(8, 6))
        labels = sorted(self.graph.keys())
        sns.heatmap(adjacency_matrix, cmap="summer", xticklabels=labels, yticklabels=labels)
        plt.title(f'{self.name} Adjacency Matrix Heatmap')
        plt.show()

    @property
    def node_num(self):
        return len(self.graph.keys())

    @property
    def edge_num(self):
        adjacency_matrix = self.adjacency_matrix
        if self.is_directed:
            return adjacency_matrix.sum()
        else:
            return np.triu(self.adjacency_matrix).sum()

    def is_simple(self):
        adjacency_matrix = self.adjacency_matrix

        if self.find_self_loop_nodes()[0].sum() == 0 and self.find_multi_edges_nodes()[0].sum() == 0:
            return True
        return False

    def networkx_plot(self):
        G = nx.Graph()
        G.add_edges_from(self.to_edges(show_node_name=True))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=300, node_color='skyblue', font_size=12, font_color='black', font_weight='bold')
        plt.show()

    def dfs(self, start):
        stack = []
        visited = set()
        stack.append(start)

        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                print(f"Visited node {chr(node + ord('a'))}")
                for neighbor in self.graph[node]:
                    if neighbor not in visited:
                        stack.append(neighbor)

    def bfs(self, start):
        queue = []
        visited = set()
        queue.append(start)

        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                print(f"Visited node {chr(node + ord('a'))}") 
                for neighbor in self.graph[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)


def find_duplicate_elements(l):
    unique_elements = set()
    duplicate_elements = set()

    for element in l:
        if element in unique_elements:
            duplicate_elements.add(element)
        else:
            unique_elements.add(element)
    return list(duplicate_elements)
