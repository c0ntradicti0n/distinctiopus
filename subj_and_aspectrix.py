import itertools

from hardcore_annotated_expression import eT, apply_fun_to_nested, eL, eD, ltd_ify, Argu
from littletools.generator_tools import count_up
from pairix import Pairix
from littletools.nested_list_tools import flatten_reduce, collapse, flatten, flat_list_from, curry
from similaritymixer import SimilarityMixer
from time_tools import timeit_context

import logging
logging.captureWarnings(True)
logging.getLogger().setLevel(logging.INFO)


class Subjects_and_Aspects(Pairix):
    ''' This module finds pairs of arguments, that are the subjects and aspects for a pair of a pair of expressions

    '''
    def __init__(self, corpus):
        self.similar = \
            SimilarityMixer([(1, SimilarityMixer.elmo_simple_sim(), 0.6, 1)])

        self.subjects_aspects = \
             SimilarityMixer ([(1,     SimilarityMixer.multi_paral_tup_sim(SimilarityMixer.subj_asp_sim, n=4), 0, 1   ),
                               (-1000, SimilarityMixer.multi_sim(SimilarityMixer.same_expression_sim, n=100), 0, 0.001)])

    def annotate(self, clusters=None, graph_fun=None):
        ''' Annotates the correlations, that means expressions that are similar to each other and are DistinctFilter from the
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
        def get_arguments(predicate):
            args = self.get_arguments(predicate)
            return args



        with timeit_context('computing sameness for the words within these pairs and the subject-'):
            def correllate(pred_pairs_to_correlate):
                try:
                    arguments1 = (eL(flatten_reduce([get_arguments(preds_to_correlate[0]) for preds_to_correlate in pred_pairs_to_correlate]))).unique()
                    arguments2 = (eL(flatten_reduce([get_arguments(preds_to_correlate[1]) for preds_to_correlate in pred_pairs_to_correlate]))).unique()
                except IndexError:
                    return []
                if not arguments1 or not arguments2:
                    return []
                print (len(arguments1), len(arguments2))

                entities_aspects =  self.similar.choose(data=(arguments1, arguments2),
                                     layout='1:1')
                return entities_aspects

            argument_tuples_in_sides = apply_fun_to_nested (
                fun=correllate,
                attribute=['contrasts','coinings'], # If the attribute is a list, the function is applied to each single value in these lists
                out_attribute=['entities', 'aspects'],
                data=clusters)

        # 2. these tuples have a distance between these two words within, like name ~> thing in multiple sentences
        # they have a grammatical and semantical distance within. We compute this as a feature of these tuples and
        # feed them again into SimilarityMixer and use again hdbscan. So they must be converted to dicts

        # 3. look for the maximum distance with at least two tuples in these grouped tuples.

        # (things, things. things), (name answering to definition, name coresponding with the name) (name, name, name, name)

        with timeit_context('compute pairs of similar distance'):
            def distinct(entities_aspects):
                aspects, entities = entities_aspects

                if not aspects or not entities:
                    return []
                subject_aspect = self.subjects_aspects.choose(
                    (flat_list_from(entities), flat_list_from(aspects)),
                    n=1,
                    layout='n',
                    out='ex')
                return subject_aspect

            subjects_aspects_in_sides = apply_fun_to_nested (
                fun=distinct,
                attribute=('entities','aspects'), # If `attribute` is a tuple, the function is applied to pairs taken from this tuple
                out_attribute=('entity','aspect'),
                data=argument_tuples_in_sides)

        with timeit_context('writing everything'):
            self.neo4j_write(graph_fun, subjects_aspects_in_sides)
        return subjects_aspects_in_sides

    def argument_or_reference_instead (self, arguments):
        ''' This exchanges in the list of arguments the on

        #with timeit_context('neo4j cleanup'):
        #     self.merge_clean_up(graph_fun)es, that are referencing to other nouns, and keep the ones,
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
            logging.warning('not good reference for arguments found')
            new_arguments = arguments
            pass

        assert all(isinstance(arg, Argu) for arg in new_arguments)

        return new_arguments

    def get_arguments(self, predicate_s):
        """ Gets the arguments of the predicate

            :param predicate_s: predicate-dict or predicate list
            :return: argument-dict

        """
        if isinstance(predicate_s, list):
            arguments = eL(flatten_reduce([self.get_arguments(pred) for pred in predicate_s]))
            return arguments.unique()

        arguments = predicate_s['arguments']

        arguments_ref = self.argument_or_reference_instead (arguments)
        if not arguments:
            logging.warning('no referenced arguments found %s' % str(predicate_s))
            arguments_ref = arguments
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

    def neo4j_write (self, graph_fun, subjects_aspects):
        ''' push subjects and aspects to neo4j with appropriate node_labels

        :param graph_fun: neo4j driver
        :param subjects_aspects: annotated structure
        :param clusters: the correlating and contrasting clusters, that were used to make widows for the query of
            subjects and aspects

        '''

        # what = ['contrasts', 'coinings', ('entity', 'aspect')

        with timeit_context('push results to neo4j'):
            apply_fun_to_nested(fun=curry(self.neo4j_push, graph_fun), attribute='D', data=subjects_aspects)

        with timeit_context('neo4j cleanup'):
             self.merge_clean_up(graph_fun)

    cnt = count_up()

    def neo4j_push(self, x, graph_fun, type=None):
        ''' push nested annotation structure to neo4j

        :param x: nested eL, eT, eD-structure
        :param graph_fun: neo4j driver

        '''
        if not x['contrasts'] or not x['coinings'] or not x[('entity','aspect')]:
            logging.info('no subjects/aspects for some expressions found')
            return None
        conno = eL(flat_list_from(x['contrasts'])+flat_list_from(x['coinings'])).unique()
        for p in conno:
            p.node_type =  ['Nlp']
        deno = ltd_ify(x[('entity','aspect')][0], node_type=['DENOTATION'], stack_types=['X',('SUBJECTS','ASPECTS'),'PAIR','X','ARGUMENT'], d_max=4)

        with timeit_context('generate query'):
            query = "".join(list(collapse(conno.neo4j_write() + deno.neo4j_write()+ ['\n'])))
        with timeit_context('neo4j'):
            graph_fun(query)

    def merge_clean_up(self, graph_fun):
        ''' Connect predicate and argument nodes and transit a node in the nested annotation

        :param graph_fun:
        :return:
        '''
        collapse_helper_nodes_list = ['X','NLP', 'node_type_not_given', 'PAIR', 'HAS_ARGUMENTS' ]
        for nt in collapse_helper_nodes_list:
            query = """
                 MATCH (x)--(a:%s)--(y)
                 MERGE (x)-[:X]-(y)
                 DETACH DELETE a
                 WITH 1 as one
                 RETURN one""" % nt
            graph_fun(query)

        query="""MATCH (x)-[r:X]-(y)-[r2:X]-(x)
                 DETACH DELETE r
                 WITH 1 as one
                 RETURN one"""
        graph_fun(query)

        query="""MATCH (x)-[r:X]-(x)
                 DETACH DELETE r
                 WITH 1 as one
                 RETURN one"""
        graph_fun(query)

        query = """MATCH (a1:ARGUMENT)--(r1:ASPECTS)--(r3:SUBJECTS)--(a2:ARGUMENT)--(r4:SUBJECTS)--(r2:ASPECTS)--(a1)
CALL apoc.refactor.mergeNodes([r1,r2])  YIELD node as n1
CALL apoc.refactor.mergeNodes([r3,r4])  YIELD node as n2
RETURN n1"""
        graph_fun(query)

        query = """MATCH (A)-[r1:X]-(B)-[r2:X]-(A)
WHERE ID(r1)>ID(r2)
DELETE r2"""
        graph_fun(query)


        query = """MATCH(n:ARGUMENT)
        WHERE
        EXISTS(n.id)
        WITH
        n.id as id, collect(n) as nodes
        WHERE
        size(nodes) > 1
        call
        apoc.refactor.mergeNodes(nodes)
        yield node
        return count(*)"""
        graph_fun(query)

        query = """MATCH(: ACORE)--(:ASIDE)--(x:ACONNOTATION)--(:DENOTATION)-[s]-(:DENOTATION), (x)-[r]-(n)
        WHERE
        x: NCONNOTATION and not n:ACONNOTATION and not n: ARGUMENT and not n:ASIDE
        DELETE        r"""
        graph_fun(query)

        query = """MATCH(a: ASIDE)--(b:ACONNOTATION)--(c:ARGUMENT)
        SET
        a.abunch = ID(a), b.abunch = ID(a), c.abunch = ID(a)
        RETURN a, b, c"""
        graph_fun(query)

        query = """MATCH(a: NSIDE)--(b:NCONNOTATION) - -(c:ARGUMENT)
                SET
                a.nbunch = ID(a), b.nbunch = ID(a), c.nbunch = ID(a)
                RETURN a, b, c"""
        graph_fun(query)

        query = """MATCH (x:ARGUMENT)-[r:X]-(y:ARGUMENT)
                  DELETE r
                 WITH 1 as one
                 RETURN one"""
        graph_fun(query)

        query = """MATCH (x:ARGUMENT)-[r:X]-(y:Nlp)-[r2:X]-(x)
                  DELETE r2
                 WITH 1 as one
                 RETURN one"""
        graph_fun(query)


