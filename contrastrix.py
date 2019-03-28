#!/usr/bin/env python
# -*- coding: utf-8 -*-
from hardcore_annotated_expression import eL
from littletools.corutine_utils import coroutine
from littletools.generator_tools import count_up
from predicatrix import Predication
import word_definitions
from similaritymixer import SimilarityMixer

import networkx as nx
import textwrap

import logging

from time_tools import timeit_context

logging.captureWarnings(True)
logging.getLogger().setLevel(logging.INFO)


class Contrast:
    def __init__(self):
        ''' This module lets you find the contrasts between the predicates.

            The constructor sets up the filters to find them.
            Call then :func:`~contradix.Contradiction.find_contrasts` on predicates

            Example
            =======

            Get some annotated text data...

            >>> from corpus_reader import CorpusReader
            >>> corpus = CorpusReader(corpus_path='./corpora/aristotle_categories/import_conll', only=[9, 12, 14, 16]) # , 10, 11, 13, 15

            Extract some predicates

            >>> from predicatrix import Predication
            >>> P = Predication(corpus)
            >>> from littletools.nested_list_tools import type_spec, flatten_reduce
            >>> x = corpus.sentence_df.apply(P.analyse_predications, axis=1, result_type="reduce")
            >>> ps = P.post_processing(paint=True)
            >>> str(ps[1])[:111]
            'Some things , again , are present in a subject , but are never predicable of a subject . ~ [text=Some things ag'

            >>> C = Contrast()
            >>> p = C.find_constrasts(ps,ps)
            >>> print (p)
            [([0], [1, 2, 3]), ([1], [0, 2, 3]), ([2], [0, 1, 3]), ([3], [0, 1, 2])]

            This means, the sentence no 0 contrasts sentences 1,2,3 and so on
            Problems with this arise from faulty grammar, missing antonym information, bad sentence splitting or maybe the
            phrase has a more complex paraphrasing structure of the contrasting phrase

        '''
        fit_mix_neg = \
             SimilarityMixer ([(1, SimilarityMixer.elmo_complex_sim(key='elmo_embeddings_pred'), 0.98, 1),
                               #(1, SimilarityMixer.common_words_sim(), 0.5,0.99),
                               ],
                              n=None)
        self.NegationContrastFilter  = \
            SimilarityMixer([(1, SimilarityMixer.formula_excludes(fit_mix_neg), 0.1, 1),
                             (1, SimilarityMixer.sub_i, 0, 0.1)
                             ], n=30)

        antonym_filter = \
             SimilarityMixer ([(1, SimilarityMixer.elmo_complex_sim(), 0.86, 1),
                               (1, SimilarityMixer.detect_pair, 0.1, 1)
                               ], n=None)

        self.AntonymContrastFilter = \
            SimilarityMixer([(1, SimilarityMixer.formula_concludes(antonym_filter), 0.1, 1)
                             ], n=30)

        self.contra_counter = count_up()


    def find_constrasts (self, predicates1, predicates2, graph_coro=None, paint_graph=False, **kwargs):
        ''' Tbis function  searches in two lists of predicate-dict for contrasts, that are caused by antonym- and
            negation.

            These Pred have a special distribution of negation particles.
            See :func:`~predcicatrix.Predication.organize_negations`

            Antonyms come from nltk.WordNet and there is a list in `word_definitions`.
            See :data:`~word_definitions.antonym_dict`


            :param predicates1: list of predicate-dicts
            :param predicates2: list of predicate-dicts
            :param paint_graph: If a picture should be saved
            :param kwargs: keyword-args are propagated to :func:`~simmix.SimilarityMixer.choose`

            :return: Combined list of predicate-tuples

        '''
        if not predicates1 or not predicates2:
            logging.error("give predicates if you want contrasts")
            return []

        if paint_graph:
            G = nx.DiGraph()
            put_into_nx = self.put_into_nx(general_kind='constrast', G=G)
            graph_coro = [graph_coro, put_into_nx ]

        with timeit_context('contrast finding neg'):
            negation_constrasts = self.NegationContrastFilter.choose ((eL(predicates1), eL(predicates2)),
                                                                      type='negation',
                                                                      layout='n',
                                                                      out='i',
                                                                      graph_coro=graph_coro, **kwargs)
        with timeit_context('contrast finding anto'):
            antonym_constrasts  = self.AntonymContrastFilter.choose((eL(predicates1), eL(predicates2)),
                                                                      type='antonym',
                                                                      layout='n',
                                                                      out='i',
                                                                      graph_coro=graph_coro, **kwargs)
        logging.info("antonym : %s" % str (antonym_constrasts) + " negation: %s" % str (negation_constrasts))

        if paint_graph:
            #for p in predicates1 + predicates2:
            #    put_into_nx.send(p)
            put_into_nx.send('draw')

        try:
            return  negation_constrasts + antonym_constrasts
        except TypeError:
            return (negation_constrasts, antonym_constrasts)

    def wrap (self, line):
        ''' wrap the line

        :param line: just a string with text

        '''
        return textwrap.fill(line, 50)

    def add_possible_contrasting_node (self, G, predicate_dict, kind=None):
        ''' Add a node to networkx for debugging the constrast

            :param G: networkx graph
            :param predicate_dict: dict with 'key' and 'text'
            :param kind: special attribute for the node

        '''
        key_co1 = kind + predicate_dict['key']
        label = self.wrap(" ".join(predicate_dict['text']))
        G.add_node (key_co1, label=label, kind=kind)

    def add_correlation_edge (self, G, predicate_dict1, predicate_dict2, label=None, kind=None):
        ''' Add an edge to networkx for debugging purposes on the constrast

            :param G: networkx graph
            :param predicate_dict1: predicate dict with 'key'
            :param predicate_dict2: the same
            :param label: label for the edge
            :param kind: special attribute for the edge

        '''
        key_co1 = kind + predicate_dict1['key']
        key_co2 = kind + predicate_dict2['key']
        G.add_edge(key_co1, key_co2, label=label)

    def add_determined_expression_nx(self, G, general_kind, special_kind, n1, n2):
        ''' Throws node data into neo4j by expanding data as dictionary.

            :param general_kind: Property "GeneralKind" in the GDB
            :param special_kind: Property "SpecialKind" in the GDB ('anonym'/'negation')
            :param n1: predicate dict 1
            :param n2: predicate dict 2

        '''
        if isinstance(n1, list):
            logging.warning("node data is list... taking first")
            n1 = n1[0]
        if isinstance(n2, list):
            logging.warning("node data is list... taking first")
            n2 = n2[0]

        self.added_cluster1.append(n1['id'])
        self.added_cluster2.append(n2['id'])

        self.add_nx_node(G,n1)
        self.add_nx_node(G,n2)
        self.add_nx_edge(G,n1,n2, general_kind, special_kind)


    def add_nx_node(self, G, n):
        G.add_node(n['id'],
                   s_id=n['s_id'],
                   label=self.wrap(" ".join(n['text']).replace("'", "")))


    def add_nx_edge(self, G, n1, n2, general_kind, special_kind):
        G.add_edge (n1['id'], n2['id'],
             general_kind=general_kind,
             special_kind=special_kind,
             label = general_kind + ' ' + special_kind )


    @coroutine
    def put_into_nx(self, G, general_kind):
        ''' This returns a subgraph of the graph, selected by the 'general_kind' param.

        :param general_kind: some string property of all members, that are added by this function
        :return: list of Pred-dict-2tuples

        '''
        self.added_cluster1 = []
        self.added_cluster2 = []

        while True:
            data = (yield)
            if isinstance(data, tuple) and len(data) == 2:
                n1, n2 = data
                special_kind = data.type
                self.add_determined_expression_nx(G, general_kind, special_kind, n1, n2)

            elif isinstance(data, dict):
                self.add_nx_node(G, data)

            elif isinstance(data, str) and data=='draw':
                self.draw_key_graphs(G)

            else:
                logging.error('Value could not be set because I don\'t know how to deal with the type')
                raise ValueError('Value could not be set because I don\'t know how to deal with the type')
        return None


    def draw_key_graphs(self, G):
        ''' Draw a picture for debugging

        :param G: networkx directed graph
        :return:
        '''
        import pylab as plt
        path = './img/contrast' +  str(next(self.contra_counter)) + ".svg"

        G.graph['graph'] = {'rankdir': 'LR','splines':'line'}
        G.graph['edges'] = {'arrowsize': '4.0'}

        A = nx.drawing.nx_agraph.to_agraph(G)

        A.add_subgraph(nbunch=self.added_cluster1,
                       name="cluster1",
                       node="[style=filled]",
                       color='blue',
                       labeldistance='300',
                       label="Contrasts for:")

        A.add_subgraph(nbunch=self.added_cluster2,
                       name="cluster2",
                       node="[style=filled]",
                       color='blue',
                       labeldistance='300',
                       label="Contrasts are:")

        A.layout('dot')
        A.draw(path)
        plt.clf()