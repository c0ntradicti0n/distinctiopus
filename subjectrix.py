from argumentatrix import Arguments
from simmix import Simmix


class Subjects(Arguments):
    ''' This module finds actual common subjects for predications.

        For Example, if there are two sentences like this (from a microsoft licences aggreement):

        THIS LIMITED WARRANTY GIVES YOU SPECIFIC LEGAL RIGHTS.
                     --------
        YOU MAY HAVE OTHERS, WHICH VARY FROM STATE/JURISDICTION TO STATE/JURISDICTION.
                     ------

        You have two times, where the text speaks about the subject/theme of warranties. Thats, what this module finds,
        if such things occur pairwise in sentences.

    '''
    def __init__(self, corpus, type):
        super().__init__(corpus)

        self.bind_predicates, self.bind_subjects = type

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
             Simmix ( [(1, Simmix.multi_sim(Simmix.elmo_layer_sim(layer=[0,1])
                        ), 0.2,1)
                       ],
                      n=100)

        self.pattern_pair = [(self.standard_entity_exs, self.standard_entity_exs)]


