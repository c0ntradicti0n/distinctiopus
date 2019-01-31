import networkx as nx

from nested_list_tools import type_spec
from pairix import Pairix
from simmix import Simmix
from predicatrix2 import Predication

from grammarannotator import nlp


class Arguments(Pairix):
    def __init__(self, corpus):
        super().__init__()

        self.P = Predication (corpus)

        global nlp
        standard_ex = nlp("The thing is round from the right side.")
        self.standard_predicate              = self.P.collect_all_predicates(standard_ex)
        self.standard_entity_exs             = self.standard_predicate[0]['arguments'][0:1]
        self.standard_differential_layer_exs = self.standard_predicate[0]['arguments'][1:2]


    def annotate(self, *, original_pair=None, graph_coro=None):
        pass


    def draw_correlations (self, G, source, target):
        import pylab as P
        path = './img/contradiction -- new correlation'+ source +" -- "+ target +".svg"

        G.graph['graph'] = {'rankdir': 'LR', 'splines':'line'}
        G.graph['edges'] = {'arrowsize': '4.0'}

        A = nx.drawing.nx_agraph.to_agraph(G)
        nbunch_pn =[n for n, d in G.nodes(data='kind') if d == 'poss new']
        A.add_subgraph(nbunch=nbunch_pn,
                        name="cluster1",
                        style='filled',
                        color='lightgrey',
                        label='possible new correlations')
        nbunch_cr = [n for n, d in G.nodes(data='kind') if d == 'contra']

        A.add_subgraph (nbunch=nbunch_cr,
                        name="cluster2",
                        style='filled',
                        color='lightgrey',
                        label='found contradictions')

        A.layout('dot')

        A.draw(path)
        P.clf()

import unittest

class TestArgumentatrix(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestArgumentatrix, self).__init__(*args, **kwargs)
        self.A = Arguments()


    def test_modifix(self):
        G = nx.Graph()
        self.A.annotate_example_nodes(G)



if __name__ == '__main__':
    unittest.main()


