import networkx as nx
import logging

from nested_list_tools import check_for_tuple_in_list, flatten_reduce, flatten_list
from simmix import Simmix
from predicatrix2 import Predication
from generator_tools import count_up

from grammarannotator import nlp

class Arguments:
    def __init__(self, corpus_path):
        self.id_counter = count_up()
        self.P = Predication (corpus_path)

        self.fit_mix_subject = \
             Simmix ( [(1, Simmix.elmo_layer_sim(layer=[0,1]
                        ), 0.2,1)
                       ],
                      n=100)
        self.fit_mix_diff_layer = \
             Simmix ( [(1, Simmix.elmo_layer_sim(layer=[0,1]
                        ), 0.2,1)
                       ],
                      n=100)
        self.fit_mix_example_expr = \
             Simmix ( [(1, Simmix.elmo_layer_sim(layer=[2]
                        ), 0.8,1)
                       ],
                      n=100)
        return None

    def new_id (self):
        return str(next(self.id_counter))

    def annotate_common_arguments(self, linked_graph):
        global nlp
        # subordinated predicates subgraph
        G = nx.classes.function.induced_subgraph(
                linked_graph,
                [n for n, data in linked_graph.nodes(data='kind') if data not in ['example', 'correl']])

        groups = nx.connected_components(G)

        # Collect arguments for nodes
        # Iterate through from the right sideattributes of nodes
        arguments = {}
        for n, attrs in G.nodes(data=True):
            if 'predicate' in attrs:
                arguments.update({n: attrs['predicate']['arguments']})

        standard_ex = nlp("The thing is round from the right side.")
        standard_predicate = self.P.collect_all_predicates(standard_ex)
        standard_entity_exs             = standard_predicate[0]['arguments'][0:1]
        standard_differential_layer_exs = standard_predicate[0]['arguments'][1:2]

        subjects            = self.find_idealized_expr(self.fit_mix_subject,groups, arguments, standard_entity_exs)
        differential_layers = self.find_idealized_expr(self.fit_mix_diff_layer, groups, arguments, standard_differential_layer_exs)

        self.outsource_as_node(linked_graph, subjects, kind='subject', label='entity')
        self.outsource_as_node(linked_graph, differential_layers, kind='differential_layer', label="differential layer")
        return linked_graph


    def annotate_example_nodes (self, G):
        example_dict = ['thus', ('for', 'instance'), ('for', 'example'), 'instance', ('for', 'if'),
                        ('by', '*', '-PRON-', '*', 'mean'), 'inasmuch'] # ('such', 'as')
        example_noun_dict = ['TwoFooted', 'White', 'account']
        return self.annotate_nodes_by_markers_whole_p(G, example_dict + example_noun_dict, 'example')

    def annotate_explanations_nodes (self, G):
        explanation_markers = [('sketch', '*', 'meaning'), ('by', '*', '-PRON-', '*', 'mean')]
        return self.annotate_nodes_by_markers_whole_p(G, explanation_markers, 'explanation')

    def annotate_nodes_by_markers (self, G, marker_dict, kind):
        node_texts = {}
        marked = {}
        def pred_len(x):
            return len (x['i'])

        nodes
        for n, attrs in [(node,attributes) for node, attributes in G.nodes(data=True) if 'predicate' in attributes]:
            node_texts.update({i: p for i, p in  enumerate(G.nodes[n]['predicate']['part_predications'])})
            marker_containing_i = self.find_str_patterns (node_texts, marker_dict)
            marker_containing   = [node_texts[i] for i in marker_containing_i]
            if not marker_containing:
                continue
            min_pred          = min(marker_containing, key=pred_len)
            marked.update({n: min_pred})

        whole_marked = [n for n in marked.keys() if 'ROOT' in G.nodes[n]['predicate']['dep_']]
        sub_marked   = {n: [d] for n, d in marked.items() if 'ROOT' not in G.nodes[n]['predicate']['dep_']}

        node_attr = {n: {'kind': kind} for n in whole_marked}
        edge_attr = {ed:{'kind': kind} for ed in G.edges(nbunch=whole_marked)}
        nx.set_node_attributes(G, node_attr)
        nx.set_edge_attributes(G, edge_attr)

        self.outsource_as_node(G, sub_marked, kind=kind, label=kind, no_tuple=True)
        return G

    def annotate_nodes_by_markers_whole_p (self, G, marker_dict, kind):
        node_texts = {}
        for n, attrs in [(node,attributes) for node, attributes in G.nodes(data=True) if 'predicate' in attributes]:
            node_texts.update({n: G.nodes[n]['predicate']})
        marked = self.find_str_patterns (node_texts, marker_dict)
        node_attr = {n: {'kind': kind} for n in marked}
        edge_attr = {ed:{'kind': kind} for ed in G.edges(nbunch=marked)}
        nx.set_node_attributes(G, node_attr)
        nx.set_edge_attributes(G, edge_attr)
        return G

    def find_idealized_expr(self, find_sim, groups, arguments, standard_exs):
        return {node:
            find_sim.choose(
                (standard_exs,
                 arguments[node]),
                n=10,
                out='ex',
                layout="1:1",
                output=True)
            for group in groups for node in group if node in arguments}

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


