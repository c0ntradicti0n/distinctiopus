from hardcore_annotated_expression import eT, apply_fun_to_nested, eL, eD, ltd_ify, Argu
from littletools.generator_tools import count_up
from pairix import Pairix
from littletools.nested_list_tools import flatten_reduce, collapse
from simmix import Simmix
from time_tools import timeit_context

import logging
logging.captureWarnings(True)
logging.getLogger().setLevel(logging.INFO)


class Subjects_and_Aspects(Pairix):
    ''' This module finds pairs of arguments, that are the subjects and aspects for a pair of a pair of expressions

    '''
    def __init__(self, corpus):
        self.similar = \
            Simmix([(2, Simmix.elmo_sim(), 0.4,1)])

        self.subjects_aspects = \
             Simmix ([(1, Simmix.multi_paral_tup_sim(Simmix.subj_asp_sim, n=4), 0, 1),
                     (-1000, Simmix.multi_sim(Simmix.boolean_subsame_sim, n=100), 0, 0.001)])

    def annotate(self, clusters=None, graph_fun=None):
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
            :param graph_fun: neo4j driver

        '''
        arguments_for_sides = apply_fun_to_nested (fun=self.get_arguments, attribute='predicate_id', data=clusters )

        with timeit_context('collect arguments'):
            to_correlate = eL(
                [eT(tuple(
                    eL(flatten_reduce([sd['predicate_id'] for sd in cl['core_clusters']['sides']])).unique()))
                    for cl in arguments_for_sides]).unique()



        with timeit_context('computing correlates'):
            correlated = eL(
                [self.similar.choose(data=(to_corr.unique(),
                                           to_corr.unique()),
                                     layout='hdbscan',
                                     n=100)
                 for to_corr in to_correlate])

        with timeit_context('pairs'):
            self.neo4j_push(correlated, graph_fun )

        with timeit_context('computing subjects and aspects from correlates'):
            subjects_aspects = eL(
                [self.subjects_aspects.choose(
                (corr, corr),
                n=int(len(corr)/3),
                minimize=False,
                layout='n',
                out='ex')
                 for corr in correlated])                   # Put it in the graph

        with timeit_context('writing everything'):
            self.neo4j_write(graph_fun, subjects_aspects, clusters)
        return subjects_aspects

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

        assert all(isinstance(arg, Argu) for arg in new_arguments)

        return new_arguments


    def get_arguments(self, predicate_s):
        """ Gets the arguments of the predicate

            :param predicate_s: predicate-dict or predicate list
            :return: argument-dict

        """
        if isinstance(predicate_s, list):
            arguments = eL(flatten_reduce([self.get_arguments(pred) for pred in predicate_s]))
            # if len (arguments.unique()) != len(arguments):
            #    logging.warning("INDEED AN EFFECT!!! %d" % (len (arguments.unique())- len(arguments)))
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
        """ Returns pairs of similar arguments

            :param pair: opposed pair of predicate-dict-2tuples
            :return: correlated argument-dict-2tuples

        """
        arguments = self.get_arguments(pair[0][0]), self.get_arguments(pair[1][0])
        if not all(arguments):
            raise ValueError ('no argument for predicate, that can be referenced?')
        return self.similar.choose(arguments, layout='n', n=100,  out='ex')

    def neo4j_write (self, graph_fun, subjects_aspects, clusters):
        ''' push subjects and aspects to neo4j with appropriate node_labels

        :param graph_fun: neo4j driver
        :param subjects_aspects: annotated structure
        :param clusters: the correlating and contrasting clusters, that were used to make widows for the query of
            subjects and aspects

        '''
        with timeit_context('typing nested list for subject/aspect'):
            subjects_aspects = \
                ltd_ify(subjects_aspects,
                        node_type=['DENOTATION'],
                        stack_types=['SUBJECTS_ASPECTS_ALL', 'CLUSTER', 'A_S_TUPLES', ('SUBJECTS', 'ASPECTS'), 'GROUP', 'ARGUMENT'])

        with timeit_context('push results to neo4j'):
            self.neo4j_push (subjects_aspects, graph_fun)

        apply_fun_to_nested(fun=self.get_arguments, attribute='predicate_id', data=clusters)

        with timeit_context('neo4j cleanup'):
             self.merge_clean_up(graph_fun)

    cnt = count_up()

    def neo4j_push(self, x, graph_fun):
        ''' push nested annotation structure to neo4j

        :param x: nested eL, eT, eD-structure
        :param graph_fun: neo4j driver

        '''
        with timeit_context('generate query'):
            query = "".join(list(collapse(x.neo4j_write() + ['\n'])))
            with open("query %d.txt" % next(self.cnt), "w") as text_file:
                text_file.write(query)
        with timeit_context('neo4j'):
            graph_fun(query)

    def merge_clean_up(self, graph_fun):
        ''' Connect predicate and argument nodes and transit a node in the nested annotation

        :param graph_fun:
        :return:
        '''
        query = """MATCH (n:CONNOTATION),(a:ARGUMENT)
        WHERE a.id in n.arg_ids
        MERGE (n)-[:X]->(a)
        RETURN n,a"""
        graph_fun(query)

        query = """MATCH (n)-->(:GROUP)-->(s)
        CALL apoc.create.addLabels( id(s), labels(n) )
        YIELD node as n1
        MERGE (n)<-[:X]-(s)
        RETURN n"""
        graph_fun(query)
