import networkx as nx
import logging

from nested_list_tools import check_for_tuple_in_list, flatten_reduce, flatten_list
from simmix import Simmix
from predicatrix2 import Predication
from generator_tools import count_up

from grammarannotator import nlp

class Arguments:
    def __init__(self, corpus_path):
        self.P = Predication (corpus_path)
        self.fit_mix_subject = \
             Simmix ( [(1, Simmix.elmo_layer_sim(layer=[0,1]
                        ), 0.2,1)
                       ],
                      n=100)
        global nlp
        standard_ex = nlp("The thing is round from the right side.")
        self.standard_predicate = self.P.collect_all_predicates(standard_ex)
        self.standard_entity_exs             = self.standard_predicate[0]['arguments'][0:1]
        self.standard_differential_layer_exs = self.standard_predicate[0]['arguments'][1:2]


    def annotate_common_concepts (self, *,predications=None, graph_coro_ent=None, graph_coro_asp = None):
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


    def outsource_as_node (self, linked_graph, things, kind, label, no_tuple=False, **kwargs):
        for node, val in things.items(): # Simmix findings for node in dict
            if not len(val) > 0:
                logging.warning ("No subject in %s?" % (str(linked_graph.node[node]['predicate']['full_ex'])))
                continue

            if isinstance(node, tuple):
                egde = node
                self.add_this_as_egde_if_new(kind, kwargs, label, linked_graph, val)
                continue


            if no_tuple:
                l2 = val
            else:
                (l1, l2) = val[0]

            thing = flatten_reduce (l2)
            if isinstance(thing, dict):
                thing = [thing]
            for th in thing:
                self.add_this_as_node_if_new(kind, kwargs, label, linked_graph, node, th)
        return None

    def try_to_find_yet_set_node(self, node, this_node, G):
        node_exists_with_same_key = this_node in G
        edge_to_same_key    = G.has_edge(node, this_node)
        return node_exists_with_same_key and edge_to_same_key

    def try_to_find_yet_set_edge(self, node1, node2, G):
        edge = G.has_edge(node1, node2)
        return edge

    def add_this_as_node_if_new(self, kind, kwargs, label, linked_graph, node, thing):
        this_node = kind + thing['key']
        if not self.try_to_find_yet_set_node(node, this_node, linked_graph):
            linked_graph.add_node(this_node, **{'kind': kind, kind: thing, 'label': " ".join(thing['text'])})
            linked_graph.add_edge(node, this_node, triggering_text_labels=label, kind=kind, **kwargs)

    def add_this_as_egde_if_new(self, kind, kwargs, label, linked_graph, things):
        for thing in things:
            thing1, thing2 = thing
            if (isinstance(thing1, list)):
                this_edge_node1 = kind + thing1[0]['key']
            else:
                this_edge_node1 = kind + thing1['key']

            if (isinstance(thing2, list)):
                this_edge_node2 = kind + thing2[0]['key']
            else:
                this_edge_node2 = kind + thing2['key']

            if not self.try_to_find_yet_set_edge(this_edge_node1, this_edge_node2, linked_graph):
                linked_graph.add_edge(this_edge_node1, this_edge_node2, **{'kind': kind, 'computing':True})
            else:
                logging.warning("edge between %s yet set!" % (kind))



    def find_str_patterns(self, node_texts, example_dict):
        node_names = []
        for node_name, node_attrs in node_texts.items():
            if any([isinstance(t, str) and t in node_attrs['lemma_']
                    or isinstance(t, tuple) and Arguments.match(t, node_attrs['lemma_'])
                    for t in example_dict]):
                node_names.append (node_name)
        return node_names

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


