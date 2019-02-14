from pairix import Pairix
from predicatrix2 import Predication
from corpus_reader import nlp
from littletools.corutine_utils import coroutine
from littletools.generator_tools import count_up
from littletools.nested_list_tools import curry
from simmix import Simmix

import networkx as nx
import textwrap
import numpy as np
import logging
logging.captureWarnings(True)
logging.getLogger().setLevel(logging.INFO)


class Subjects_and_aspects(Pairix):
    ''' This module finds pairs of arguments, that are the subjects and aspects for a pair of a pair of expressions

    '''
    def __init__(self, corpus):
        self.similar = \
            Simmix([(20, Simmix.common_words_sim, 0.5, 1),
                    (1, Simmix.dep_sim, 0.1, 1),
                    (1, Simmix.pos_sim, 0.1, 1),
                    (1, Simmix.tag_sim, 0.1, 1),
                    ],
                   )

        self.filter1 = \
            Simmix([(18, Simmix.multi_sim( Simmix.common_words_sim, n=2), 0.1, 1),
                    #(    3, Simmix.multi_sim(Simmix.elmo_layer_sim(layer=[0, 1]) ), 0.2, 1),
                    (    4, Simmix.multi_sim(Simmix.head_dep_sim), 0.0, 1),
                    (1, Simmix.multi_sim(Simmix.pos_sim, n=2), 0.1, 1),
                    (1, Simmix.multi_sim(Simmix.tag_sim, n=2), 0.1, 1),
                    #(-1000, Simmix.multi_sim(Simmix.boolean_subsame_sim, n=2),      0.0, 0.1)
                    ],
                   n=100)

        self.theme_rheme = \
             Simmix ([(1, Simmix.multi_sim(Simmix.left_sim, n=2), 0.2, 1),
                     (-1000, Simmix.multi_sim(Simmix.boolean_subsame_sim, n=2), 0.0, 0.1)],
                     n=1)

        self.coreferential = \
             Simmix ([(1, Simmix.multi_sim(Simmix.coreferential_sim, n=2), 0, 0.1)],
                      n=1)

        global nlp
        standard_ex = nlp("The thing is round from the right side.")

        self.P = Predication (corpus)
        self.standard_predicate              = self.P.collect_all_predicates(standard_ex)
        self.standard_entity_exs             = self.standard_predicate[0]['arguments'][0:1]
        self.standard_aspect_exs             = self.standard_predicate[0]['arguments'][1:2]

        self.pattern_pair_ent = [(self.standard_entity_exs, self.standard_entity_exs)]
        self.pattern_pair_asp = [(self.standard_aspect_exs, self.standard_aspect_exs)]

        self.counter = count_up()


    def argument_or_reference_instead (self, arguments):
        ''' This exchanges in the list of arguments the ones, that are referencing to other nouns, and keep the ones,
        that are fine

        :param arguments: argument dicts
        :return: lists with some changes of same len

        '''
        new_arguments = []
        for argument in arguments:
            reference = argument['coreferenced'](argument['coref'])
            if reference:
                if reference == [[]]:
                    logging.error ('bad return value for coreferenced, taking normal argument')
                    new_arguments.append(argument)
                    continue
                new_arguments.append(reference[0])
            else:
                new_arguments.append(argument)
        assert len (new_arguments) == len (arguments)
        return new_arguments

    def get_arguments(self, argument):
        ''' Gets the arguments of the predicate

            :param argument: predicate-dict
            :return: argument-dict

        '''
        arguments = argument['arguments']
        return self.argument_or_reference_instead (arguments)


    def get_correlated (self, pair):
        ''' Returns pairs of similar arguments

            :param pair: opposed pair of predicate-dict-2tuples
            :return: correlated argument-dict-2tuples

        '''
        arguments = self.get_arguments(pair[0][0]), self.get_arguments(pair[1][0])
        return self.similar.choose(arguments, layout='1:1', out='ex')


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

        def subjectness_word(token):
            res = 0
            if token.pos_ == 'NOUN':
                res += 2
            if token.head.dep_ == 'ROOT':
                res += 20
            if token.head.dep_ in ['acl', 'rcl']:
                res += 1
            return res

        def aspectness_word(token):
            res = 0
            if token.dep_ in ['obj', 'pobj', 'dobj']:
                res += 2
            if token.head.dep_ != 'ROOT':
                res += 1
            return res

        def evaluate_subject_on_pair (pair, ness_fun = None):
            '''

            :param pair:
            :param ness_fun: a function like `subject-ness_word(token)` or `aspect-ness(token`
            :return:

            '''
            def check_subjectness_spans (expressions):
                return sum (map(check_subjectness_span, expressions))
            def check_subjectness_span  (expression):
                return check_subjectness_list_of_tokens(expression['full_ex'])
            def check_subjectness_list_of_tokens(span):
                return sum (map(ness_fun, span))
            return check_subjectness_spans(pair[0]) + check_subjectness_spans(pair[1])

        def subjects_aspects (correlates):
            score1 = np.array(list(map (curry(evaluate_subject_on_pair, ness_fun= subjectness_word), correlates)))
            score2 = np.array(list(map (curry(evaluate_subject_on_pair, ness_fun= aspectness_word), correlates)))
            subj_i = np.argmax(score1)
            aspe_i = np.argmax(score2)
            if subj_i == aspe_i:
                # It's better to put the aspect elsewhere, because the aspect can be hidden in nested phrases,
                # only of there are enough values
                if aspe_i < len (score2):
                    aspe_i = score2.argsort()[::-1][0]
            return correlates[subj_i], correlates[aspe_i]

        poss_subjects1, poss_aspects1 = subjects_aspects (poss_correlates1)
        poss_subjects2, poss_aspects2 = subjects_aspects (poss_correlates2)

        subjects_aspects = self.theme_rheme.choose(
            ([poss_subjects1, poss_subjects2], [poss_aspects2, poss_aspects1]),
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

        if not (
            Simmix.same_sent_sim(subject1, pred1) and
            Simmix.same_sent_sim(subject2, pred2) and
            Simmix.same_sent_sim(aspect1, pred3) and
            Simmix.same_sent_sim(aspect2, pred4) ):
            logging.warning ('subjects and aspects not from the same sentence')
            return []

        #coreferential = self.coreferential.choose(subjects_aspects, layout='n')

        #if coreferential:
        #    return []

        graph_coro_arg_binding.send ((pred1, subject1) + ('subject',))
        graph_coro_arg_binding.send ((pred2, subject2) + ('subject',))
        graph_coro_arg_binding.send ((pred3, aspect1) + ('aspect',))
        graph_coro_arg_binding.send ((pred4, aspect2) + ('aspect',))

        if paint_graph:
            #put_into_nx.send('draw')
            G = self.subj_aspect_to_nxdigraph(poss_correlates1+poss_correlates2, subjects_aspects[0][0], subjects_aspects[0][1])
            self.draw_subjects_aspects(G)
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


    def add_edge_between (self, dig, subject_args, aspect_args):
        for subject_arg in subject_args:
            for aspect_arg in aspect_args:
                key_asp1  = "poss_new" + aspect_arg    [0][0]['id']
                key_asp2  = "poss_new" + aspect_arg    [1][0]['id']

                key_subj1 = "poss_new" + subject_arg   [0][0]['id']
                key_subj2 = "poss_new" + subject_arg   [1][0]['id']


                dig.add_edge (key_subj2, key_subj1, label = "*subjects")
                dig.add_edge (key_subj1, key_subj2, label = "*subjects")

                dig.add_edge (key_asp1, key_asp2, label = "*aspects")
                dig.add_edge (key_asp2, key_asp1, label = "*aspects")

    def subj_aspect_to_nxdigraph (self, possible_subjs_asps, subjects, aspects):
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


        for ex1, ex2 in possible_subjs_asps:
            add_possible_correlation_node(ex1[0], kind = 'poss_new')
            add_possible_correlation_node(ex2[0], kind = 'poss_new')
            add_possible_correlation_edge(ex1[0], ex2[0], label="possibly correlated", kind="poss_new")

        self.add_edge_between(dig, subjects, aspects)
        return dig


    def draw_subjects_aspects(self, G):
        import pylab as P

        path = './img/subjectsaspects' + str (next(self.counter)) + ".svg"

        G.graph['graph'] = {'rankdir': 'LR', 'splines': 'line'}
        G.graph['edges'] = {'arrowsize': '4.0'}

        A = nx.drawing.nx_agraph.to_agraph(G)

        # Add contradictions to graph as cluster
        nbunch_cr = [n for n, d in G.nodes(data='kind') if d == 'contra']
        A.add_subgraph(nbunch=nbunch_cr,
                       name="cluster1",
                       style='filled',
                       color='lightgrey',
                       label='Found Contradictions')

        # Add correlations to graph as cluster
        nbunch_pn = [n for n, d in G.nodes(data='kind') if d == 'poss_new']
        A.add_subgraph(nbunch=nbunch_pn,
                       name="cluster2",
                       style='filled',
                       color='lightgrey',
                       label='Possible New Correlations')

        A.layout('dot')

        A.draw(path)
        P.clf()
