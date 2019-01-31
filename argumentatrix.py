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
        '''  Look in the arguments of the pair and group them as pairs and then find a pair that fits most and above a
             limit the pattern of the standard example.

        '''
        pred1, pred2 = original_pair

        arguments1 = pred1[0] ['arguments']
        arguments2 = pred2[0] ['arguments']

        poss_argument = self.similar.choose(      # What correlates
            (arguments1,
             arguments2),
            layout='1:1')

        if not poss_argument:
            return []

        assert type_spec (poss_argument) == type_spec (self.pattern_pair)

        argument = self.filter.choose(            # next to a standard subject
            (poss_argument,
             self.pattern_pair),
            n=1,
            minimize=True,
            layout='n',
            out='lx')

        if not argument:
            return []

        # Put it in the graph
        p1 = pred1[0]
        p2 = pred2[0]
        s1 = argument[0][0][0]
        s2 = argument[0][1][0]

        lr1_edge = (p1, s1)
        lr2_edge = (p2, s2)
        l_edge   = (s1, s2)

        graph_coro.send(lr1_edge + (self.bind_predicates,))
        graph_coro.send(lr2_edge + (self.bind_predicates,))
        graph_coro.send(l_edge   + (self.bind_subjects,))
        return argument


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


