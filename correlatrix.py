import networkx as nx

from pairix import Pairix
from simmix import Simmix

class Correlation(Pairix):
    def __init__(self):
        ''' This module looks for pairwise modifying expressions in the horizon of a predicate
        '''
        # https://en.wikipedia.org/wiki/Grammatical_modifier

        # This looks for relativly similar phrases to get some pairs that are possible modifiers to the contradiction
        self.correlative = \
            Simmix([(18, Simmix.common_words_sim, 0.25, 1),
                    (1, Simmix.dep_sim, 0.25, 1),
                    (1, Simmix.pos_sim, 0.25, 1),
                    # (1,Simmix.elmo_sim(), 0.5,1),
                    # (1,Simmix.fuzzystr_sim, 0.5,1),
                    (-100, Simmix.boolean_subsame_sim, 0, 0.1)
                    ],
                   )

        # That's a distinctive criterium, that the correlative keys can't be too similar to the contradicting pair
        self.distinct = \
            Simmix([(1, Simmix.multi_sim(fun=Simmix.common_words_sim, n=7), 0, 0.6),
                    # (2,Simmix.elmo_multi_sim(), 0,0.4),
                    # (1, Simmix.multi_sim(fun=Simmix.sub_i, n=4), 0,0.3),
                    # (1, Simmix.multi_sim(fun=Simmix.longer_sim, n=4), 0, 0.7) \
                    ],
                   n=None)

        # TODO define a stringer contraint to detect the actually modifying bindings
        return None


    def annotate_correlations(self, *, contradiction=None, possible_to_correlate=None, graph_coro=None, save_graph=True):
        '''Annotates the correlations, that means expressions that are similar to each other and are distinct from the
           pair, that was found as excluding each other. For instance 'from the one side' and 'from the other side'.

           In part the graph is cleaned, because also exmaples can be marked as seemingly contradictions.
           On the other side the same operation is done for additional (sub)predications in context, following
           coreferential relations
        '''

        poss_correlations = self.correlative.choose(  # What correlates
            possible_to_correlate,
            layout='1:1')

        if not poss_correlations:
            return []

        correlation = self.distinct.choose(           # not too much
            ([contradiction],
             poss_correlations),
            n=1,
            minimize=True,
            layout='n',
            out='ex',
            type=("opposed", "opposed", "correlated"),
            graph_coro=graph_coro)                             # Put it in the graph


        if save_graph:
            G = self.correl_to_nxdigraph(contradiction, poss_correlations, correlation)
            self.draw_correlations(G=G, source=contradiction[0][0]['id'], target=contradiction[1][0]['id'])

        return correlation


    def correl_to_nxdigraph (self, contradiction, possible_correlations, correlation):
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

        for ex1, ex2 in [contradiction]:
            add_possible_correlation_node(ex1[0], kind = 'contra')
            add_possible_correlation_node(ex2[0], kind = 'contra')
            add_possible_correlation_edge(ex1[0], ex2[0], label="contradiction", kind="contra")

        def add_edge_between (key_corr_i, key_trigg_i):
            key_corr  = "poss new" + key_corr_i[0][1][0]['key']
            key_trigg = "contra"   + contradiction[0][0]['key']
            dig.add_edge (key_corr, key_trigg, label = "correl")
            key_corr  = "poss new" + key_corr_i[0][1][0]['key']
            key_trigg = "contra"   + contradiction[1][0]['key']
            dig.add_edge (key_corr, key_trigg, label = "correl")

        for corr_i, trigg_i in correlation:
           add_edge_between(corr_i, trigg_i)
        return dig


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
