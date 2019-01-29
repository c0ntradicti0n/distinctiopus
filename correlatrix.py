import networkx as nx

from simmix import Simmix

class Correlation:
    def __init__(self):
        self.G = nx.Graph()
        return None

    def annotate_correlations(self, *, contradiction=None, possible_to_correlate=None, graph_coro=None, ):
        """Annotates the correlations, that means expressions that are similar to each other and are distinct from the
        pair, that was found as excluding each other. For instance 'from the one side' and 'from the other side'.

        In part the graph is cleaned, because also exmaples can be marked as seemingly contradictions.
        On the other side the same operation is done for additional (sub)predications in context, following
        coreferential relations
        """
        correlative = \
            Simmix([(18, Simmix.common_words_sim, 0.2, 1),
                    (1, Simmix.dep_sim, 0.2, 1),
                    (1, Simmix.pos_sim, 0.2, 1),
                    # (1,Simmix.elmo_sim(), 0.5,1),
                    # (1,Simmix.fuzzystr_sim, 0.5,1),
                    (-100, Simmix.boolean_subsame_sim, 0, 0.1)
                    ],
                   )

        # That's the distinctive criterium, that the correlative keys can't be too similar to the contradicting key
        distinct = \
            Simmix([(1, Simmix.multi_sim(fun=Simmix.common_words_sim, n=7), 0, 0.7),
                    # (2,Simmix.elmo_multi_sim(), 0,0.4),
                    # (1, Simmix.multi_sim(fun=Simmix.sub_i, n=4), 0,0.3),
                    # (1, Simmix.multi_sim(fun=Simmix.longer_sim, n=4), 0, 0.7) \
                    ],
                   n=None)

        poss_correlations = correlative.choose(  # What correlates
            (possible_to_correlate,
             possible_to_correlate),
            layout='1:1')
        correlation = distinct.choose(  # not too much
            ([contradiction],
             poss_correlations),
            n=1,
            minimize=True,
            layout='n',
            out='ex',
            type='modifyer',
            type2='opposed',
            G=graph_coro)  # Put it in the graph