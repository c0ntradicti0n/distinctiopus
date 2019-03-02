from hardcore_annotated_expression import eT, apply_fun_to_nested, eL, eD
from pairix import Pairix
from littletools.corutine_utils import coroutine
from littletools.nested_list_tools import curry, flatten_reduce, recursive_map
from simmix import Simmix

import networkx as nx
import textwrap
import numpy as np
import logging
logging.captureWarnings(True)
logging.getLogger().setLevel(logging.INFO)


class Subjects_and_Aspects(Pairix):
    ''' This module finds pairs of arguments, that are the subjects and aspects for a pair of a pair of expressions

    '''
    def __init__(self, corpus):
        self.similar = \
            Simmix([(3, Simmix.fuzzytext_sim, 0.1, 1),
                    (2, Simmix.elmo_sim(), 0.1,1),
                    (1, Simmix.dep_sim, 0.1, 1),
                    (1, Simmix.pos_sim, 0.1, 1),
                    (1, Simmix.tag_sim, 0.1, 1)]
                   )

        self.theme_rheme = \
             Simmix ([#(1, Simmix.multi_sim(Simmix.left_sim, n=6), 0.0, 1),
                      (1, Simmix.multi_paral_tup_sim(Simmix.subj_asp_sim, n=4), 0.0, 1),
                      (-1000, Simmix.multi_sim(Simmix.boolean_subsame_sim, n=6), 0.0, 0.1)],
        n=1)


    def argument_or_reference_instead (self, arguments):
        ''' This exchanges in the list of arguments the ones, that are referencing to other nouns, and keep the ones,
        that are fine.

        :param arguments: argument dicts
        :return: lists with some changes of same len

        '''
        new_arguments = []
        for argument in arguments:
            reference = argument['coreferenced'](argument['coref'])
            if reference:
                new_arguments.extend(reference)
            else:
                new_arguments.append(argument)

        try:
            assert new_arguments and all (new_arguments)
        except AssertionError:
            print (arguments)
            raise

        return new_arguments


    def get_arguments(self, predicate_s):
        ''' Gets the arguments of the predicate

            :param predicate_s: predicate-dict or predicate list
            :return: argument-dict

        '''
        if isinstance(predicate_s, list):
            arguments = eL(flatten_reduce([self.get_arguments(pred) for pred in predicate_s]))
            if len (arguments.unique()) != len(arguments):
                logging.warning("INDEED AN EFFECT!!! %d" % (len (arguments.unique())- len(arguments)))
            return arguments.unique()


        arguments = predicate_s['arguments']
        try:
            assert (arguments)
        except:
            raise
        arguments_ref = self.argument_or_reference_instead (arguments)
        assert arguments_ref
        return arguments_ref


    def get_correlated (self, pair):
        ''' Returns pairs of similar arguments

            :param pair: opposed pair of predicate-dict-2tuples
            :return: correlated argument-dict-2tuples

        '''
        arguments = self.get_arguments(pair[0][0]), self.get_arguments(pair[1][0])
        if not all(arguments):
            raise ValueError ('no argument for predicate, that can be referenced?')
        return self.similar.choose(arguments, layout='n', n=100,  out='ex')


    def check_ness_spans(self, expressions, ness_fun=None):
        ''' Map some scoring function to a list of expressions (lists of tokens) and sum up the results

        :param expressions: list of list of spacy tokens
        :param ness_fun: function, that takes a list of spacy tokens
        :return: summed score

        '''
        x = apply_fun_to_nested (data=expressions, attribute='full_ex', fun=curry(self.check_ness_list_of_tokens, ness_fun=ness_fun), reduce=True)
        try:
             return [l[0] * l[1] for t in x for l in t]
        except TypeError:
             raise


    def check_ness_list_of_tokens(self, ex, ness_fun=None):
        ''' Apply the subject-aspect-ness function to the expression, get the scores and sum them up

        :param ex: list of spacy tokens
        :param ness_fun: scoring function
        :return: sum of scores

        '''
        return sum(map(ness_fun, ex))


    def score_expressions(self, exs, ness_fun=None):
        ''' gets a pair (or list of expression) and compute the score for every combination

        :param exs: expressions, lists/tuples of predicate dict
        :param ness_fun: scoring function
        :return: sum of score

        '''
        return [self.check_ness_spans(ex, ness_fun) for ex in exs]


    def subjects_aspects(self, correlates):
        '''

        :param correlates:
        :return:
        '''
        score1 = np.asarray(list(map(curry(self.score_expressions, ness_fun=self.subjectness_word), correlates)))
        score2 = np.asarray(list(map(curry(self.score_expressions, ness_fun=self.aspectness_word), correlates)))

        subj_i = np.argmax(score1)
        aspe_i = np.argmax(score2)

        if subj_i == aspe_i:
            # It's better to put the aspect elsewhere, because the aspect can be hidden in nested phrases,
            # only of there are enough values
            if aspe_i < len(score2):
                aspe_i = score2.argsort()[::-1][0]
        return correlates[subj_i], correlates[aspe_i]


    def annotate(self, clusters=None, graph_coro_subj_asp=None, graph_coro_arg_binding=None, paint_graph=True):
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

            :param clusters: 2tuple-2tuple-list-predicate-dicts, so 4 predicates in contradicting/corralating
                constellation
            :param graph_coro: coroutine to connect the subjects and aspects to debug their bindings
            :param graph_coro_arg_binding: coroutine to connect subject and aspects to their predicates

        '''
        #if paint_graph:
        #    G = nx.DiGraph()
        #    put_into_nx = self.put_into_nx(general_kind='contradiction', G=G)
        #    graph_coro_subj_asp = [graph_coro_subj_asp, put_into_nx ]
        graph_funs = [graph_coro_subj_asp]

        arguments_for_sides = apply_fun_to_nested (fun=self.get_arguments, attribute='predicate_id', data=clusters )

        to_correlate = eL(
            [eT(tuple(
                eL(flatten_reduce([sd['predicate_id'] for sd in cl['nucleus']['sides']])).unique()))
                for cl in arguments_for_sides]).unique()

        correlated = eL(
            [self.similar.choose(data=(to_corr.unique(), to_corr.unique()), layout='n', n=100, out='ex')
             for to_corr in to_correlate])

        to_distinguish =  eL(
            [eT([eL(flatten_reduce(list(corr[0]) + list(corr[1])))
                 for corr in cl])
             for cl in correlated])

        subjects_aspects = eL(
            [self.theme_rheme.choose(
            (corr, corr),
            n=4,
            minimize=False,
            layout='n',
            out='ex',
            type=("subjects", "aspects", "compared"))
             for corr in to_distinguish])                   # Put it in the graph

        def graph_write (x):
            x.type    = 'subj_aspe'
            x[0].type = 'subjects'
            x[0][0].type = 'subjects'

            x[1].type = 'aspects'
            x[1][0].type = 'aspects'

            query = "\n".join(x.neo4j_write())
            print (query)
            return [f(query) for f in graph_funs]

        written = apply_fun_to_nested (fun=graph_write,
                                       other_criterium=(
                                           lambda x: isinstance(x, eT) and isinstance(x[0], eL) and isinstance(x[1], eL)
                                                     and isinstance(x[0][0], eL) and isinstance(x[0][0][0], eD)),
                                       data=subjects_aspects )

        print (written)

        subject1 = subjects_aspects[0][0][0][0][0]
        subject2 = subjects_aspects[0][0][0][1][0]
        aspect1  = subjects_aspects[0][1][0][0][0]
        aspect2  = subjects_aspects[0][1][0][1][0]

        pred1    = oppo1[0][0]
        pred2    = oppo1[1][0]
        pred3    = oppo2[0][0]
        pred4    = oppo2[1][0]

        same_sentence = (Simmix.same_sent_sim(subject1, pred1) ,
            Simmix.same_sent_sim(subject2, pred2) ,
            Simmix.same_sent_sim(aspect1, pred3) ,
            Simmix.same_sent_sim(aspect2, pred4))
        if not all(same_sentence):
            logging.warning ('subjects and aspects not from the same sentence %s' % (str(same_sentence)))

        graph_coro_arg_binding.send (eT((pred1, subject1), type='subject'))
        graph_coro_arg_binding.send (eT((pred2, subject2), type='subject'))
        graph_coro_arg_binding.send (eT((pred3, aspect1), type='aspect'))
        graph_coro_arg_binding.send (eT((pred4, aspect2), type='aspect'))

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
        :return: list of Pred-dict-2tuples

        '''
        while True:
            data = (yield)
            if isinstance(data, tuple) and len(data) == 2:
                n1, n2 = data
                special_kind = data.type
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
