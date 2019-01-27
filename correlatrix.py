import networkx as nx


class Correlation:
    def __init__(self):
        self.G = nx.Graph()
        return None

    def annotate_correlations(self, linked_graph):
        groups = [c for c in sorted(nx.connected_components(linked_graph), key=len, reverse=True)]
        print (groups)
        return self.G