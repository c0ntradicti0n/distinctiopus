import networkx as nx

from pairix import Pairix
from simmix import Simmix

from hardcore_annotated_expression import HEAL, HEAT

class Correlation(Pairix):
    def __init__(self):
        ''' This module looks for pairwise modifying expressions in the horizon of a predicate
        '''
        # https://en.wikipedia.org/wiki/Grammatical_modifier

        # This looks for relativly similar phrases to get some pairs that are possible modifiers to the contradiction
        self.correlative = \
            Simmix([(3, Simmix.common_words_sim (invert=True), 0.35, 1),
                    (1, Simmix.dep_sim, 0.65, 1),
                    (1, Simmix.pos_sim, 0.65, 1),
                    (1, Simmix.elmo_sim(), 0.35,1),
                    (1,Simmix.fuzzystr_sim, 0.3,1),
                    (-1000, Simmix.boolean_subsame_sim, 0, 0.1)
                    ],
                   )

        # That's a distinctive criterium, predicates cant be correlated with a part of itself on the opposite direction at all
        self.distinct_from_the_opposite = \
            Simmix([(1, Simmix.multi_cross2tup_sim(fun=Simmix.boolean_subsame_sim,n=2), 0, 0.1)
                    ],
                   n=None)

        # That's a distinctive criterium, that the correlative keys can't be too similar to the contradicting pair
        self.distinct = \
            Simmix([(1, Simmix.multi_sim(fun=Simmix.common_words_sim(), n=7), 0, 0.95),
                    (1, Simmix.multi_sim(fun=Simmix.elmo_sim(), n=7), 0, 0.95),
                    (1, Simmix.multi_sim(fun=Simmix.dep_sim, n=7), 0.0, 1),
                    (1, Simmix.multi_sim(fun=Simmix.pos_sim,n=7), 0.0, 1)
                    ],
                   n=None)

        # TODO define a stringer contraint to detect the actually modifying bindings
        return None


    def annotate_correlations(self, *, contradiction=None, possible_to_correlate=None, graph_coro=None, save_graph=True):
        ''' Annotates the correlations, that means expressions that are similar to each other and are distinct from the
            pair, that was found as excluding each other. For instance 'from the one side' and 'from the other side'.

            In part the graph is cleaned, because also exmaples can be marked as seemingly contradictions.
            On the other side the same operation is done for additional (sub)predications in context, following
            coreferential relations

            :param contradiction: pair of predicates
            :param possible_to_correlate: list of predicate pairs, that should be evaluated for being as distinct as possible
            :param graph_coro: coroutine for graph wirting
            :param save_graph: set to True, if results should be written to the graph db
            :return correlating-correlated predicate pair list

        '''
        poss_correlations = self.correlative.choose(           # What correlates
            possible_to_correlate,
            layout='1:1')

        if not poss_correlations:
            return []

        poss_correlations_no_opps = self.distinct_from_the_opposite.choose(                    # not too much
            HEAT((HEAL([contradiction]),
             poss_correlations)),
            n=len(poss_correlations),
            minimize=True,
            layout='n',
            out='ex')

        correlation = self.distinct.choose(                    # not too much
            poss_correlations_no_opps[0],
            n=int(1),
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
            key_co1 = kind + correlated['id']
            label = wrap(" ".join(correlated['text']))
            dig.add_node (key_co1, label=label, kind=kind)

        def add_possible_correlation_edge (correlated1, correlated2, label=None, kind = None):
            key_co1 = kind + correlated1['id']
            key_co2 = kind + correlated2['id']
            dig.add_edge(key_co1, key_co2, label=label)

        for ex1, ex2 in [contradiction]:
            add_possible_correlation_node(ex1[0], kind = 'contra')
            add_possible_correlation_node(ex2[0], kind = 'contra')
            add_possible_correlation_edge(ex1[0], ex2[0], label="contradicting", kind="contra")

        for ex1, ex2 in possible_correlations:
            add_possible_correlation_node(ex1[0], kind = 'poss_new')
            add_possible_correlation_node(ex2[0], kind = 'poss_new')
            add_possible_correlation_edge(ex1[0], ex2[0], label="possibly correlated", kind="poss_new")

        def add_edge_between (contradicting_preds, correlated_preds):
            for contradicting_pred in contradicting_preds:
                for correlated_pred in correlated_preds:
                    key_corr1  = "poss_new" + correlated_pred    [0][0]['id']
                    key_corr2  = "poss_new" + correlated_pred    [1][0]['id']

                    key_trigg2 = "contra"   + contradicting_pred [1][0]['id']
                    key_trigg1 = "contra"   + contradicting_pred [0][0]['id']

                    dig.add_edge (key_trigg1, key_corr1, label = "correlated")
                    dig.add_edge (key_trigg2, key_corr2, label = "correlated")

                    dig.add_edge (key_corr1, key_corr2, label = "opposed")
                    dig.add_edge (key_corr2, key_corr1, label = "opposed")

        for contra, correl in correlation:
           add_edge_between(contra, correl)
        return dig


    def draw_correlations (self, G, source, target):
        import pylab as P

        path = './img/correlation'+ source +" -- "+ target +".svg"

        G.graph['graph'] = {'rankdir': 'LR', 'splines':'line'}
        G.graph['edges'] = {'arrowsize': '4.0'}

        A = nx.drawing.nx_agraph.to_agraph(G)

        # Add contradictions to graph as cluster
        nbunch_cr = [n for n, d in G.nodes(data='kind') if d == 'contra']
        A.add_subgraph (nbunch=nbunch_cr,
                        name="cluster1",
                        style='filled',
                        color='lightgrey',
                        label='Found Contradictions')

        # Add correlations to graph as cluster
        nbunch_pn =[n for n, d in G.nodes(data='kind') if d == 'poss_new']
        A.add_subgraph(nbunch=nbunch_pn,
                        name="cluster2",
                        style='filled',
                        color='lightgrey',
                        label='Possible New Correlations')

        A.layout('dot')

        A.draw(path)
        P.clf()
