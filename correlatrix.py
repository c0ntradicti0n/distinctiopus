import networkx as nx

from pairix import Pairix
from similaritymixer import SimilarityMixer

from hardcore_annotated_expression import eL, eT

class Correlation(Pairix):
    def __init__(self):
        ''' This module looks for pairwise modifying expressions in the horizon of a predicate
        '''
        # https://en.wikipedia.org/wiki/Grammatical_modifier

        # This looks for relativly similar phrases to get some pairs that are possible modifiers to the contradiction
        self.CorrelationFilter = \
            SimilarityMixer([
                    (1, SimilarityMixer.dep_sim, 0.0, 1),
                    (1, SimilarityMixer.pos_sim, 0.0, 1),
                    (20, SimilarityMixer.elmo_complex_sim(), 0.0, 1),
                    (0, SimilarityMixer.same_expression_sim, 0, 0.1)
                    ],
                   )

        # This distinctive criterium expresses, that predicates cannot be correlated with a part of itself on the opposite side
        self.DistinctFromCrossOppositeFilter = \
            SimilarityMixer([
                    (1, SimilarityMixer.multi_cross2tup_sim(fun=SimilarityMixer.same_expression_sim, n=2), 0, 0.1)
                     ],
                    n=None)

        # This distinctive criterium expresses, that predicates should be distinct from their coining phrases
        self.DistinctFilter = \
            SimilarityMixer([(-1, SimilarityMixer.multi_paral2tup_sim(fun=SimilarityMixer.common_words_sim(), n=7), 0, 1),
                    (-1, SimilarityMixer.multi_paral2tup_sim(fun=SimilarityMixer.elmo_complex_sim(), n=7), 0, 1),
                    (-1, SimilarityMixer.multi_paral2tup_sim(fun=SimilarityMixer.dep_sim, n=7), 0.0, 1),
                    (-1, SimilarityMixer.multi_paral2tup_sim(fun=SimilarityMixer.pos_sim, n=7), 0.0, 1),
                    (-1000, SimilarityMixer.multi_paral2tup_sim(SimilarityMixer.same_expression_sim), 0, 0.1)
                    ],
                   n=None)

        # TODO define a stringer contraint to detect the actually modifying bindings
        return None


    def annotate_correlations(self, *, contradiction=None, possible_to_correlate=None, graph_coro=None, save_graph=False, types=None):
        ''' Annotates the correlations, that means expressions that are similar to each other and are DistinctFilter from the
            pair, that was found as excluding each other. For instance 'from the one side' and 'from the other side'.

            In part the graph is cleaned, because also exmaples can be marked as seemingly contradictions.
            On the other side the same operation is done for additional (sub)predications in context, following
            coreferential relations

            :param contradiction: pair of predicates
            :param possible_to_correlate: list of predicate pairs, that should be evaluated for being as DistinctFilter as possible
            :param graph_coro: coroutine for graph wirting
            :param save_graph: set to True, if results should be written to the graph db
            :return correlating-correlated predicate pair list

        '''
        poss_correl_l, poss_correl_r = possible_to_correlate
        #poss_correl_r = [p for p in poss_correl_r if not p in poss_correl_l]
        #poss_correl_l = [p for p in poss_correl_l if not p in poss_correl_r]
        possible_to_correlate = (poss_correl_l, poss_correl_r)

        if not poss_correl_r or not poss_correl_l:
            return []

        possible_correlations = self.CorrelationFilter.choose(           # What correlates
            possible_to_correlate,
            layout='1:1',
            )

        if not possible_correlations:
            return []

        poss_correlations_no_opps = self.DistinctFromCrossOppositeFilter.choose(                    # not too much
            eT((eL([contradiction]),
                possible_correlations)),
            n=len(possible_correlations),
            layout='n',
            out='ex')

        if not poss_correlations_no_opps:
            return []

        correlation = self.DistinctFilter.choose(                    # not too much
            poss_correlations_no_opps[0],
            n=2,
            layout='n',
            out='ex',
            type=types,
            graph_coro=graph_coro)                             # Put it in the graph

        if not correlation:
            return []

        if save_graph:
            G = self.correl_to_nxdigraph(contradiction, possible_correlations, correlation)
            self.draw_correlations(G=G, source=contradiction[0][0]['id'], target=contradiction[1][0]['id'])

        return correlation


    def correl_to_nxdigraph (self, contradiction, possible_correlations, correlation):
        dig = nx.DiGraph()
        import textwrap

        def wrap (strs):
            return textwrap.fill(strs, 20)


        def add_possible_correlation_node (correlated, kind=None):
            try:
                if isinstance(correlated['id'], int):
                    print  ('something became an int!')
                    correlated['id'] = str(correlated['id'])
                key_co1 = kind + correlated['id']
                label = wrap(" ".join(correlated['text']))
                dig.add_node (key_co1, label=label, kind=kind)
            except:
                pass

        def add_possible_correlation_edge (correlated1, correlated2, label=None, kind = None):
            if isinstance(correlated1['id'], int):
                print('something became an int!')
            if isinstance(correlated2['id'], int):
                print  ('something became an int!')
            key_co1 = kind + correlated1['id']
            key_co2 = kind + correlated2['id']
            dig.add_edge(key_co1, key_co2, label=label)

        for ex1, ex2 in [contradiction]:
            add_possible_correlation_node(ex1[0], kind = 'contra')
            add_possible_correlation_node(ex2[0], kind = 'contra')
            add_possible_correlation_edge(ex1[0], ex2[0], label="constrasts to", kind="contra")

        for ex1, ex2 in possible_correlations:
            add_possible_correlation_node(ex1[0], kind = 'poss_new')
            add_possible_correlation_node(ex2[0], kind = 'poss_new')
            add_possible_correlation_edge(ex1[0], ex2[0], label="similar and distinct enough to", kind="poss_new")

        def add_edge_between (contradicting_preds, correlated_preds):
            for contradicting_pred in contradicting_preds:
                for correlated_pred in correlated_preds:
                    key_corr1  = "poss_new" + correlated_pred    [0][0]['id']
                    key_corr2  = "poss_new" + correlated_pred    [1][0]['id']

                    key_trigg2 = "contra"   + contradicting_pred [1][0]['id']
                    key_trigg1 = "contra"   + contradicting_pred [0][0]['id']

                    dig.add_edge (key_trigg1, key_corr1, label = "ASSIGNED TO")
                    dig.add_edge (key_trigg2, key_corr2, label = "ASSIGNED TO")

                    dig.add_edge (key_corr1, key_corr2, label = "opposed")
                    dig.add_edge (key_corr2, key_corr1, label = "opposed")

        for contra, correl in correlation:
           add_edge_between(contra, correl)
        return dig


    def draw_correlations (self, G, source, target):
        path = './img/correlation'+ source +" -- "+ target +".svg"

        G.graph['graph'] = {'rankdir': 'LR', 'splines':'line'}
        G.graph['edges'] = {'arrowsize': '4.0'}

        A = nx.drawing.nx_agraph.to_agraph(G)

        # Add contradictions to graph as cluster
        nbunch_cr = [n for n, d in G.nodes(data='kind') if d == 'contra']
        A.add_subgraph (nbunch=nbunch_cr,
                        name="cluster1",
                        node="[style=filled]",
                        color='red',
                        label='Contrasting explanations')

        # Add correlations to graph as cluster
        nbunch_pn = [n for n, d in G.nodes(data='kind') if d == 'poss_new']
        A.add_subgraph(nbunch=nbunch_pn,
                        name="cluster2",
                        node="[style=filled]",
                        color='blue',
                        label='assignable as term coinages')
        A.layout('dot')
        A.draw(path)
