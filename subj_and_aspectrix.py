from argumentatrix import Arguments
from simmix import Simmix


class Subjects_and_aspects(Arguments):
    ''' This module finds pairs of arguments

    '''
    def __init__(self, corpus):
        super().__init__(corpus)

        self.similar = \
            Simmix([(18, Simmix.common_words_sim, 0.1, 1),
                    (1, Simmix.dep_sim, 0.1, 1),
                    (1, Simmix.pos_sim, 0.1, 1),
                    # (1,Simmix.elmo_sim(), 0.5,1),
                    # (1,Simmix.fuzzystr_sim, 0.5,1),
                    #(-100, Simmix.boolean_subsame_sim, 0, 0.1)
                    ],
                   )

        self.filter = \
            Simmix([(1, Simmix.multi_sim(Simmix.elmo_layer_sim(layer=[0, 1])
                                         ), 0.2, 1)
                    ],
                   n=100)

        self.concrete_abstract = \
             Simmix ([(1, Simmix.multi_sim(Simmix.abtract_conrete_sim()), 0.0,1)],
                      n=1)

        self.pattern_pair_ent = [(self.standard_entity_exs, self.standard_entity_exs)]
        self.pattern_pair_asp = [(self.standard_aspect_exs, self.standard_aspect_exs)]


    def get_arguments(self, pred):
        ''' Gets the arguments of the predicate

            :param pred: predicate-dict
            :return: argument-dict

        '''
        return pred['arguments']


    def get_correlated (self, oppo):
        ''' Returns pairs of similar arguments

            :param oppo: opposed pair of predicate-dict-2tuples
            :return: correlated argument-dict-2tuples

        '''
        arguments = self.get_arguments(oppo[0][0]), self.get_arguments(oppo[1][0])
        return self.similar.choose(                                     # What correlates
            arguments,
            layout='1:1',
            out='ex')


    def annotate(self, opposed_pair_pair=None, graph_coro_subj_asp=None, graph_coro_arg_binding=None):
        ''' Annotates the correlations, that means expressions that are similar to each other and are distinct from the
            pair, that was found as excluding each other. For instance 'from the one side' and 'from the other side'.

            In part the graph is cleaned, because also exmaples can be marked as seemingly contradictions.
            On the other side the same operation is done for additional (sub)predications in context, following
            coreferential relations

            What is a subject and what a subject is decided on the sequence in respect to the theme-rheme-distinction.
            What you speak about, comes first, the subject. With what you want to divide this thing up, is the aspect.
            If you change the direction of explanation, this also changes. E.g. If you are in the first sentence talking
            about microsoft warranties, and secondly you explain different cases, you speak about warranty. If you
            start with these cases and then tell, that in one case you have in the other not warranty, you speak about
            these cases.

            :param opposed_pair_pair: 2tuple-2tuple-list-predicate-dicts, so 4 predicates in contradicting/corralating
                constellation
            :param graph_coro: coroutine to connect the subjects and aspects to debug their bindings
            :param graph_coro_arg_binding: coroutine to connect subject and aspects to their predicates

        '''
        oppo1, oppo2 = opposed_pair_pair
        poss_correlates1 = self.get_correlated(oppo1)
        poss_correlates2 = self.get_correlated(oppo2)

        if not poss_correlates1 or not poss_correlates2:
            return []

        def filter_possibilities (poss_correlates, pattern):
            return self.filter.choose(                    # not too much
                (poss_correlates,
                 (self.pattern_pair_ent)),
                n=2,
                out = 'lx',
                layout='n',
            )

        poss_subjects1 = filter_possibilities(poss_correlates1, self.pattern_pair_ent)
        poss_aspects1  = filter_possibilities(poss_correlates1, self.pattern_pair_ent)
        poss_subjects2 = filter_possibilities(poss_correlates2, self.pattern_pair_asp)
        poss_aspects2  = filter_possibilities(poss_correlates2, self.pattern_pair_asp)

        subjects_aspects = self.concrete_abstract.choose(
            (poss_subjects1 + poss_subjects2,poss_aspects2 + poss_aspects1),
            n=1,
            minimize=True,
            layout='n',
            out='ex',
            type=("subjects", "aspects", "compared"),
            graph_coro=graph_coro_subj_asp)                             # Put it in the graph

        subject1 = subjects_aspects[0][0][0][0][0]
        subject2 = subjects_aspects[0][0][0][1][0]
        aspect1  = subjects_aspects[0][1][0][0][0]
        aspect2  = subjects_aspects[0][1][0][1][0]

        pred1    = oppo1[0][0]
        pred2    = oppo1[1][0]
        pred3    = oppo2[0][0]
        pred4    = oppo2[1][0]

        graph_coro_arg_binding.send ((pred1, subject1) + ('subject',))
        graph_coro_arg_binding.send ((pred2, subject2) + ('subject',))

        graph_coro_arg_binding.send ((pred3, aspect1) + ('aspect',))
        graph_coro_arg_binding.send ((pred4, aspect2) + ('aspect',))

        return subjects_aspects