import networkx as nx

from pairix import Pairix
from simmix import Simmix
from predicatrix2 import Predication

from grammarannotator import nlp


class Arguments(Pairix):
    def __init__(self, corpus):
        super().__init__()

        self.P = Predication (corpus)
        self.fit_mix_subject = \
             Simmix ( [(1, Simmix.elmo_layer_sim(layer=[0,1]
                        ), 0.2,1)
                       ],
                      n=100)

        global nlp
        standard_ex = nlp("The thing is round from the right side.")
        self.standard_predicate              = self.P.collect_all_predicates(standard_ex)
        self.standard_entity_exs             = self.standard_predicate[0]['arguments'][0:1]
        self.standard_differential_layer_exs = self.standard_predicate[0]['arguments'][1:2]


    def annotate (self, predications=None, graph_coro_ent=None, graph_coro_asp = None):
        pass
        # Collect arguments for nodes
        # Iterate through from the right sideattributes of nodes
        arguments1 = predications[0][0]['arguments']
        arguments2 = predications[1][0]['arguments']

        subject1            = self.fit_mix_subject.choose((arguments1, self.standard_entity_exs), out='ex', layout='n', n=1)
        subject2            = self.fit_mix_subject.choose((arguments2, self.standard_entity_exs), out='ex', layout='n', n=1)
        aspect1            = self.fit_mix_subject.choose((arguments1, self.standard_differential_layer_exs), out='ex', layout='n', n=1)
        aspect2            = self.fit_mix_subject.choose((arguments2, self.standard_differential_layer_exs), out='ex', layout='n', n=1)

        graph_coro_ent.send( (subject1[0][0],predications[0][0]) + ('subject',))
        graph_coro_ent.send( (subject2[0][0],predications[1][0]) + ('subject',))
        graph_coro_asp.send( (aspect1[0][0],predications[0][0]) + ('aspect',))
        graph_coro_asp.send( (aspect2[0][0],predications[1][0]) + ('aspect',))


    def draw_correlations (self, G, source, target):
        import pylab as P
        import textwrap

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


