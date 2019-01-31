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


    def annotate_correlations(self, *, contradiction=None, possible_to_correlate=None, graph_coro=None, ):
        '''Annotates the correlations, that means expressions that are similar to each other and are distinct from the
           pair, that was found as excluding each other. For instance 'from the one side' and 'from the other side'.

           In part the graph is cleaned, because also exmaples can be marked as seemingly contradictions.
           On the other side the same operation is done for additional (sub)predications in context, following
           coreferential relations
        '''

        poss_correlations = self.correlative.choose(  # What correlates
            (possible_to_correlate,
             possible_to_correlate),
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
            G=graph_coro)                             # Put it in the graph

        return correlation
