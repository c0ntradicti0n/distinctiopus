#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import networkx as nx
import textwrap
from corutine_utils import coroutine
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')
import pylab as plt

from contradictrix import Contradiction
from predicatrix2 import Predication
from correlatrix import Correlation
from subj_and_aspectrix import Subjects_and_aspects

import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

import py2neo

class DataframeCursorilyLogician:
    ''' This module handles all the databases for the operations on the text and does some transactions in between.

        * a DataFrame in the corpus, representing the grammatical information read from the input files, one
        line per word.
        * a DataFrame with all the predicative substructures of the sentences
        * a DataFrame with the actual sentences in it.
        and
        * a GraphDatabase (neo4j) for storing the results of the operations on the text as well as for querying for
        certain connections of such superficial rhetorical relations in the text

    '''
    def __init__(self, corpus):
        ''' Set up all the textsuntactical models and the graph database

        :param corpus: corpus instance with loaded conlls

        '''
        self.corpus = corpus
        self.sentence_df = self.corpus.sentence_df

        self.Predicatrix    = Predication(corpus)
        self.Contradictrix  = Contradiction ()
        self.Correlatrix    = Correlation()
        self.Subj_Aspectrix = Subjects_and_aspects(corpus)

        self.graph = py2neo.Graph("bolt://localhost:7687", auth=("s0krates", "password"))
        self.graph.run("match (n) optional match (n)-[r]-() delete n,r")

    def annotate_horizon (self, horizon=3):
        def horizon_from_row(x):
            return list(self.sentence_df.loc[x.name:x.name + horizon + 1].index)
        self.sentence_df['horizon'] = self.sentence_df.apply(
                                                   horizon_from_row,
                                                   result_type="reduce",
                                                   axis=1)


    def annotate_predicates (self):
        def analyse_predications(x):
            return self.Predicatrix.analyse_predication (doc=x['spacy_doc'], coref=x['coref'], s_id=x['s_id'])
        self.sentence_df['predication'] = self.sentence_df.apply(
                                                   analyse_predications,
                                                   result_type="reduce",
                                                   axis=1)
        def collect_predicates_from_rows(x):
            if x['horizon']:
                return list(itertools.chain.from_iterable(list(self.sentence_df.loc[x['horizon']]['predication'])))
            else:
                return None
        self.sentence_df['predications_in_range'] = self.sentence_df.apply(
                                                   collect_predicates_from_rows,
                                                   result_type="reduce",
                                                   axis=1)

    def get_predicates_in_horizon(self, s_id):
        ''' Collects the predicates, that are in the annotatet range, befor and after the sentence, where to look for
        intresting expressions

        :param s_id: id of the sentence
        :return: list of super-predicates, means they all have the property of having 'part_predications' under it

        '''
        return self.sentence_df.query('s_id==@s_id')['predications_in_range'].values[0]


    def get_predication(self, id):
        ''' Finds the predicate dict, that a special id belongs to by looking it up in the DataFrame in the Predication-
        module.

        :param id: id of that predicate
        :return: predicate-dict

        '''
        return self.Predicatrix.predicate_df.query('id==@id').to_dict(orient='records')


    def get_part_predication(self, s_id):
        ''' Collect all particular predications in that sentence. It also looks, if there were more blocks of predicates
        found.

        :param s_id: id of the sentense
        :return: list of predicate dicts.

        '''
        return [pp
                for pred_in_s in self.sentence_df.query('s_id==@s_id')['predication'].values.tolist()
                for x in pred_in_s
                for pp in x['part_predications']]


    def annotate_contradictions(self):
        ''' Looks first for pairs of phrases with negations and antonyms in an horizon
            and second it evaluates the similarity of these phrases, what would be the fitting counterpart for that one

        '''

        put_contradiction_into_gdb = self.put_into_gdb("contradiction")

        def get_contradictions_from_row(x):
            if x['horizon']:
                return self.Contradictrix.find_contradictive(
                      x['predication'],
                      x['predications_in_range'],
                      out='r',
                      G=put_contradiction_into_gdb)
            else:
                return None
        self.sentence_df['contradiction'] = self.sentence_df.apply(
                                                   get_contradictions_from_row,
                                                   result_type="reduce",
                                                   axis=1)

        def delete_none_findings (x):
            return [y + x['s_id'] for y in x['contradiction']] \
                if x['contradiction'] else None

        self.sentence_df['contradiction'] = self.sentence_df.apply(
                                                   delete_none_findings,
                                                   result_type="reduce",
                                                   axis=1)


    def get_correllation_preds(self, pred):
        '''  Collect the predicates, that can be modifyers to the predicate.

        These are either in the same sentence or the coreferenced predicates or in the sentence after

        :param pred: the predicate
        :return: list of predicate_tuples
        '''
        pred = pred[0]
        s_id = pred['s_id']
        poss_part = self.get_part_predication(s_id)
        coref_preds = self.get_coreferenced_preds (pred)
        # TODO next sentence also!
        return poss_part + coref_preds


    def get_addressed_coref (self, coref):
        ''' Analyses a coref mention and looks it up in the Database for predications.


        :param coref: dict  with sentence id, start and end of the mention
        :return:
        '''

        s_id  = coref['s_id']
        m_start = coref['m_start']
        m_end   = coref['m_end']
        mask = self.Predicatrix.predicate_df.query("s_id==@s_id")['full_ex_i'].apply(
            lambda ex_i: True if [m for m in range(m_start, m_end) if m in ex_i] else False)
        return self.Predicatrix.predicate_df.query("s_id==@s_id")[mask].to_dict(orient='records')


    def get_coreferenced_preds (self, pred):
        ''' Get the predicates, that are coreferenced by the coreference tags of another preducate.

        :param pred: predication tuple
        :return: list oft predicate dicts
        '''
        if any(pred['coref']):
            corefs_list = [x for x in pred['coref'] if x]
            preds = [p for corefs in corefs_list for coref in corefs for p in self.get_addressed_coref(coref)]
            return preds
        else:
            return []


    def annotate_correlations(self):
        ''' Look for pairs of hypotactical and paratactical and anaphorical expressions with similar semantics and
            modalities

            That means expressions, that both talk about examples or that both give reasons or are a modifyer to the
            seemingly contradictions, that are found.

            These routines reclassify some of the contradictions, because also talking about examples can seem to be
            anithetical, if the explanation of the instanciated concept is repeated.

        '''
        # Lookup what contradicitons were found
        contradictions           = list(self.get_from_gdb('contradiction'))

        # Coroutine for writing-tasks, no values stored here
        put_correlation_into_gdb = self.put_into_gdb("correlation")

        for contra1, contra2 in contradictions:
            poss_correl_l = self.get_correllation_preds(contra1)
            poss_correl_r = self.get_correllation_preds(contra2)

            self.Correlatrix.annotate_correlations(
                contradiction= (contra1, contra2),
                possible_to_correlate=(poss_correl_l, poss_correl_r),
                graph_coro=put_correlation_into_gdb)


    def get_opposed_constellation_gdb (self):
        ''' Returns the pattern, that gave a contradiction-opposition-pair

        :return: 2tuple-2tuple-list-predicate-dict
        '''
        query = \
            """Match (pred3)-[{GeneralKind:'correlation', SpecialKind:'correlated'}]-(pred1),
                     (pred1)-[{GeneralKind:'correlation', SpecialKind:'opposed'}]-(pred2),
                     (pred2)-[{GeneralKind:'correlation', SpecialKind:'correlated'}]-(pred4),
                     (pred4)-[{GeneralKind:'correlation', SpecialKind:'opposed'}]-(pred3),
                     
                     (pred1)-[{GeneralKind:'contradiction'}]-(pred2)

               Where ID(pred1)<ID(pred2)
               Return pred1, pred2, pred3, pred4"""

        logging.info("query neo4j for reading by this:\n%s" % query)
        records = self.graph.run(query).data()
        records = [
            (((self.get_predication(tuple4['pred1']['id']),
               self.get_predication(tuple4['pred2']['id'])),
              (self.get_predication(tuple4['pred3']['id']),
               self.get_predication(tuple4['pred4']['id']))))
            for tuple4 in records
        ]
        return records


    def annotate_subjects_and_aspects(self):
        ''' Look for some common arguments of the contradicting and correlating predications.
        These may may they be the logical subjects, the pre"""

        '''
        # Say, after annotation of the contradictions and their correlating modifyers we have a pair of 'opposed'
        # nodes, as annotated by the correlatrix.

        opposed_pair_pairs   = self.get_opposed_constellation_gdb()

        put_arg_into_gdb     = self.put_into_gdb('argument')
        put_deno_into_gdb    = self.put_into_gdb('denotation')


        for oppo in opposed_pair_pairs:
            self.Subj_Aspectrix.annotate(
                opposed_pair_pair   = oppo,
                graph_coro_subj_asp= put_arg_into_gdb,
                graph_coro_arg_binding=put_deno_into_gdb
                )


    kind_color_map = {
            'subject'           : '#5B6C5D',
            'predicate'         : '#9FFFF5',
            'example'           : '#59C9A5',
            'explanation'       : '#56E39F',
            'differential_layer': 'crimson',
            'correl'            : '#465775',
            'center'            : '#ECFEE8',
            'contra'            : '#EF6F6C'
        }
    edge_label_map = {
            'exclusive'         : 'side of distinction',
            'subject'           : 'subject',
            'predicate'         : 'sentence',
            'example'           : 'example',
            'explanation'       : 'explanation',
            'differential_layer': 'aspect',
            'correl'            : 'correlating part',
            'center'            : "let's distinguish",
            'contra'            : 'contradicting part',
            'computed'          : 'compared with',
            None                : '?'
        }


    def draw_as_dot_digraph(self, digraph, path):
        H = digraph
        def wrap (strs):
            return textwrap.fill(strs, 20)

        wrapped_node_labels = {n: {'label':wrap(d)} for n, d in H.nodes(data='label') if d}
        nx.set_node_attributes(H, wrapped_node_labels)

        edge_labels = dict(((u,v), {'xlabel': self.edge_label_map[d]}) for u,v, d in H.edges(data='xlabel'))
        nx.set_edge_attributes(H, edge_labels)

        node_colors =  {n: {'style':'filled', 'fillcolor':self.kind_color_map[d]} for n, d in H.nodes(data='kind') if d}
        nx.set_node_attributes(H, node_colors)

        H.graph['graph'] = {'rankdir': 'LR',
                            'style': 'filled',
                            'splines':'curved'}

        A = nx.drawing.nx_agraph.to_agraph(H)
        A.layout('dot')
        A.draw(path = "found_distinction.svg")
        return None


    def add_determed_expression (self, general_kind, special_kind, n1, n2):
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
        query = (
r"""               MERGE (a:Expression {id:'%s', s_id:'%s', text:'%s'}) 
                MERGE (b:Expression {id:'%s', s_id:'%s', text:'%s'}) 
                MERGE (a)-[:TextRelation {GeneralKind: '%s', SpecialKind:'%s'}]-(b)"""
                %
               (n1['id'], n1['s_id'],  " ".join(n1['text']).replace("'", ""),
                n2['id'], n2['s_id'],  " ".join(n2['text']).replace("'", ""),
                general_kind, special_kind))
        logging.info ("querying neo4j the following:\n %s" % query)
        self.graph.run(query)


    @coroutine
    def put_into_gdb (self, general_kind):
        ''' This returns a subgraph of the graph, selected by the 'general_kind' param.

        :param general_kind: Property "GeneralKind" in the GDB
        :return: list of Predicate-dict-2tuples
        '''
        while True:
            data = (yield)
            if isinstance(data, tuple) and len(data) == 3:
                n1, n2, special_kind =  data
                self.add_determed_expression(general_kind, special_kind, n1, n2)
            else:
                logging.error('Value could not be set because I don\'t know how to deal with the type')
                raise ValueError('Value could not be set because I don\'t know how to deal with the type')
        return None


    def get_from_gdb (self, kind):
        """ Returns pairs of in certain way connected nodes from neo4j

        :param kind: this certain kind of connection; its a property of the graph edges
        :yield: tuples of contradicting predicates
        """
        query = (
            r"""MATCH path = (a)-[:TextRelation {GeneralKind:'%s'}]->(b) 
                WHERE ID(a) < ID(b)
                RETURN a,b """ % kind
        )
        logging.info("query neo4j for reading by this:\n%s" % query)
        records = self.graph.run(query).data()
        records = [
            (self.get_predication(pair['a']['id']),
             self.get_predication(pair['b']['id']))
            for pair in records
        ]
        return records

    def query_distinctions (self):
        query = \
            r"""
Match (pred3)-[{GeneralKind:'correlation', SpecialKind:'correlated'}]-(pred1)-[{GeneralKind:'contradiction'}]-(pred2)-[{GeneralKind:'correlation', SpecialKind:'correlated'}]-(pred4)-[{GeneralKind:'correlation', SpecialKind:'opposed'}]-(pred3), 

(pred1)-[{GeneralKind:'argument',SpecialKind:'subject'}]-(arg1), (pred2)-[{GeneralKind:'argument',SpecialKind:'subject'}]-(arg2),
(pred3)-[{GeneralKind:'argument',SpecialKind:'aspect'}]-(arg3), (pred4)-[{GeneralKind:'argument',SpecialKind:'aspect'}]-(arg4),
(arg1)-[{SpecialKind:'subject_juncture'}]-(arg2),
(arg3)-[{SpecialKind:'aspect_juncture'}]-(arg4)

where ID(pred1)<ID(pred2)

Return arg1, arg2, pred1, pred2, pred3, pred4, arg3, arg4"""
        logging.info ("query neo4j for distinctions")
        self.distinction_df =  pd.DataFrame(self.graph.run(query).data()).applymap(
            lambda x: x['text']
        )
        return self.distinction_df