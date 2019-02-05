from argumentatrix import Arguments
from corutine_utils import coroutine
from generator_tools import count_up
from simmix import Simmix

import networkx as nx
import textwrap

import logging
logging.captureWarnings(True)
logging.getLogger().setLevel(logging.INFO)


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

        self.filter1 = \
            Simmix([(    1, Simmix.multi_sim(Simmix.elmo_layer_sim(layer=[0, 1]) ), 0.2, 1),
                    (-1000, Simmix.multi_sim(Simmix.boolean_subsame_sim, n=2),      0.0, 0.1)],
                   n=100)

        self.concrete_abstract = \
             Simmix ([(1, Simmix.multi_sim(Simmix.abtract_conrete_sim(), n=2), 0, 1)],
                      n=1)

        self.theme_rheme = \
             Simmix ([(1, Simmix.multi_sim(Simmix.left_sim, n=2), 0.2, 1),
                     (-1000, Simmix.multi_sim(Simmix.boolean_subsame_sim, n=2), 0.0, 0.1)],
                     n=1)

        self.pattern_pair_ent = [(self.standard_entity_exs, self.standard_entity_exs)]
        self.pattern_pair_asp = [(self.standard_aspect_exs, self.standard_aspect_exs)]
        self.subjasp_counter = count_up()


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


    def annotate(self, opposed_pair_pair=None, graph_coro_subj_asp=None, graph_coro_arg_binding=None, paint_graph=True):
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
        if paint_graph:
            G = nx.DiGraph()
            put_into_nx = self.put_into_nx(general_kind='contradiction', G=G)
            graph_coro_subj_asp = [graph_coro_subj_asp, put_into_nx ]

        oppo1, oppo2 = opposed_pair_pair
        poss_correlates1 = self.get_correlated(oppo1)
        poss_correlates2 = self.get_correlated(oppo2)

        if not poss_correlates1 or not poss_correlates2:
            return []



        def filter_possibilities (poss_correlates, pattern):
            return self.filter1.choose(                    # not too much
                (poss_correlates,
                 pattern),
                n=2,
                out = 'lx',
                layout='n',
            )

        poss_subjects1 = filter_possibilities(poss_correlates1, self.pattern_pair_ent)
        poss_aspects1  = filter_possibilities(poss_correlates1, self.pattern_pair_ent)
        poss_subjects2 = filter_possibilities(poss_correlates2, self.pattern_pair_asp)
        poss_aspects2  = filter_possibilities(poss_correlates2, self.pattern_pair_asp)

        subjects_aspects = self.theme_rheme.choose(
            (poss_subjects1 + poss_subjects2, poss_aspects2 + poss_aspects1),
            n=1,
            minimize=False,
            layout='n',
            out='ex',
            type=("subjects", "aspects", "compared"),
            graph_coro=graph_coro_subj_asp)                             # Put it in the graph

        if not subjects_aspects:
            return []

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

        if paint_graph:
            put_into_nx.send('draw')

        return subjects_aspects


    def wrap (self, line):
        return textwrap.fill(line, 20)

    def add_possible_correlation_node (self, G, predicate_dict, kind=None):
        key_co1 = kind + predicate_dict['key']
        label = self.wrap(" ".join(predicate_dict['text']))
        G.add_node (key_co1, label=label, kind=kind)

    def add_correlation_edge (self, G, predicate_dict1, predicate_dict2, label=None, kind = None):
        key_co1 = kind + predicate_dict1['key']
        key_co2 = kind + predicate_dict2['key']
        G.add_edge(key_co1, key_co2, label=label)

    def add_determined_expression_nx(self, G, general_kind, special_kind, n1, n2):
        ''' Throws node data into neo4j by expanding data as dictionary.

            :param general_kind: Property "GeneralKind" in the GDB
            :param special_kind: Property "SpecialKind" in the GDB ('anonym'/'negation')
            :param n1: predicate dict 1
            :param n2: predicate dict 2
            :return:
        '''
        if isinstance(n1, list):
            logging.warning("node data is list... taking first")
            n1 = n1[0]
        if isinstance(n2, list):
            logging.warning("node data is list... taking first")
            n2 = n2[0]

        self.add_nx_node(G, n1)
        self.add_nx_node(G, n2)
        self.add_nx_edge(G, n1, n2, general_kind, special_kind)

    def add_nx_node(self, G, n):
        G.add_node(n['id'],
                   s_id=n['s_id'],
                   label=" ".join(n['text']).replace("'", ""))

    def add_nx_edge(self, G, n1, n2, general_kind, special_kind):
        G.add_edge(n1['id'], n2['id'],
                   general_kind=general_kind,
                   special_kind=special_kind,
                   label=general_kind + ' ' + special_kind)

    @coroutine
    def put_into_nx(self, G, general_kind):
        ''' This returns a subgraph of the graph, selected by the 'general_kind' param.

        :param general_kind: some string property of all members, that are added by this function
        :return: list of Predicate-dict-2tuples

        '''
        while True:
            data = (yield)
            if isinstance(data, tuple) and len(data) == 3:
                n1, n2, special_kind = data
                self.add_determined_expression_nx(G, general_kind, special_kind, n1, n2)
            elif isinstance(data, str) and data=='draw':
                self.draw_key_graphs(G)
            else:
                logging.error('Value could not be set because I don\'t know how to deal with the type')
                raise ValueError('Value could not be set because I don\'t know how to deal with the type')
        return None


    def draw_key_graphs(self, G):
        import pylab as plt
        path = './img/subject_aspects' +  str(next(self.subjasp_counter)) + ".svg"

        G.graph['graph'] = {'rankdir': 'LR','splines':'line'}
        G.graph['edges'] = {'arrowsize': '4.0'}

        A = nx.drawing.nx_agraph.to_agraph(G)
        A.layout('dot')
        A.draw(path)
        plt.clf()
