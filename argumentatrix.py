import networkx as nx
import logging

from nested_list_tools import check_for_tuple_in_list, flatten_reduce
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

    def annotate_correlations(self, *, graph_coro=None, contradiction, possible_to_correlate=[]):
        """Annotates the correlations, that means expressions that are similar to each other and are distinct from the
        pair, that was found as excluding each other. For instance 'from the one side' and 'from the other side'.

        In part the graph is cleaned, because also exmaples can be marked as seemingly contradictions.
        On the other side the same operation is done for additional (sub)predications in context, following
        coreferential relations
        """

        correlative =    \
             Simmix ( [(18, Simmix.common_words_sim, 0.2,1),
                       (1, Simmix.dep_sim, 0.2,1),
                       (1, Simmix.pos_sim, 0.2,1),
                       #(1,Simmix.elmo_sim(), 0.5,1),
                       #(1,Simmix.fuzzystr_sim, 0.5,1),
                       (-100, Simmix.boolean_subsame_sim, 0,0.1)
                       ],
                      )
        # That's a distinctive criterium, that the correlative keys mustn't be very similar to the contradicting key
        distinct =    \
             Simmix ( [(1, Simmix.multi_sim(fun=Simmix.common_words_sim, n=7), 0,0.7),
                       #(2,Simmix.elmo_multi_sim(), 0,0.4),
                       #(1, Simmix.multi_sim(fun=Simmix.sub_i, n=4), 0,0.3),
                       #(1, Simmix.multi_sim(fun=Simmix.longer_sim, n=4), 0, 0.7) \
                       ],
                      n=None)

        poss_correlations = correlative.choose((contradiction, possible_to_correlate), layout='1:1')
        correlation       = distinct.choose   ((contradiction, poss_correlations),
                                                n=1, minimize=True, out='i')
        G.send(correlation)


        def get_poss_correlation_predicates (node):
            sub_pred_graph = node['predicate']['part_predications']
            sub_pred_coref = []
            for cr in node['predicate']['coref']:
                sub_pred_coref += corpus.coref_lookup (cr, what='sub_pred')
            return sub_pred_graph + sub_pred_coref

        for source, target, attributes in not_annotated_G.edges(data=True):
            sub_pred1 = get_poss_correlation_predicates(not_annotated_G.nodes[source])
            sub_pred2 = get_poss_correlation_predicates(not_annotated_G.nodes[target])

            poss_correlations = correlative.choose((sub_pred1,sub_pred2), layout='1:1')



            if not correlation:
                print ("no correlative found within the sentence")
                continue

            if not source in correlations:
                correlations[source] = []
                contradictions[source] = []

            if not target in correlations:
                correlations[target] = []
                contradictions[target] = []

            if not (source,target) in correlations:
                correlations[(source,target)] = []
                contradictions[(source,target)] = []

            for corr_i, trigg_i in correlation:
                correlations[source].append(poss_correlations[corr_i[0]][0])
                correlations[target].append(poss_correlations[corr_i[0]][1])
                correlations[(source,target)].append(((poss_correlations[corr_i[0]][0]),poss_correlations[corr_i[0]][1]))

                contradictions[source].append(trigger[trigg_i[0]][0])
                contradictions[target].append(trigger[trigg_i[0]][1])
                contradictions[(source,target)].append((trigger[trigg_i[0]][0],trigger[trigg_i[0]][1]))


            co_graph = self.correl_to_graph (poss_correlations, trigger, correlation)
            self.draw_correlations(co_graph, source, target)

        self.outsource_as_node(G, correlations, kind='correl', label='correlation', no_tuple=True)
        self.outsource_as_node(G, contradictions, kind='contra', label='contradiction', no_tuple=True)

        return  G

    def correl_to_graph (self, possible_correlations, triggers, correlation):
        dig = nx.DiGraph()
        import textwrap
        def wrap (strs):
            return textwrap.fill(strs, 20)


        def add_possible_correlation_node (correlated, kind=None):
            key_co1 = kind + correlated['key']
            label = wrap(" ".join(correlated['text']))
            dig.add_node (key_co1, label=label, kind=kind)

        def add_possible_correlation_edge (correlated1, correlated2, label=None, kind = None):
            key_co1 = kind + correlated1['key']
            key_co2 = kind + correlated2['key']
            dig.add_edge(key_co1, key_co2, label=label)

        for ex1, ex2 in possible_correlations:
            add_possible_correlation_node(ex1[0], kind = 'poss new')
            add_possible_correlation_node(ex2[0], kind = 'poss new')
            add_possible_correlation_edge(ex1[0], ex2[0], label="possible new", kind="poss new")

        for ex1, ex2 in triggers:
            add_possible_correlation_node(ex1, kind = 'contra')
            add_possible_correlation_node(ex2, kind = 'contra')
            add_possible_correlation_edge(ex1, ex2, label="contradiction", kind="contra")

        def add_edge_between (key_corr_i, key_trigg_i):
            key_corr  = "poss new" + possible_correlations[key_corr_i[0]][0][0]['key']
            key_trigg = "contra" + triggers[key_trigg_i[0]][0]['key']
            dig.add_edge (key_corr, key_trigg, label = "correl")
            key_corr  = "poss new" + possible_correlations[key_corr_i[0]][1][0]['key']
            key_trigg = "contra" + triggers[key_trigg_i[0]][1]['key']
            dig.add_edge (key_corr, key_trigg, label = "correl")

        for corr_i, trigg_i in correlation:
           add_edge_between(corr_i, trigg_i)
        return dig

    def match(needle, stack):
        val = (needle in stack or
               (isinstance(needle, tuple) and check_for_tuple_in_list(stack, needle)))
        return val

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


