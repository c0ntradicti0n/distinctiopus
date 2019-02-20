#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import networkx as nx
import textwrap
from littletools.corutine_utils import coroutine
import pandas as pd

import matplotlib

from littletools.digraph_tools import neo4j2nx_root
from littletools.nested_list_tools import flatten_reduce

matplotlib.use('TkAgg')

from contradictrix import Contradiction
from predicatrix import Predication
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
        self.cleanup_debug_img()

        self.corpus = corpus
        self.sentence_df = self.corpus.sentence_df

        self.Predicatrix    = Predication(corpus)
        self.Contradictrix  = Contradiction ()
        self.Correlatrix    = Correlation()
        self.Subj_Aspectrix = Subjects_and_aspects(corpus)

        self.graph = py2neo.Graph("bolt://localhost:7687", auth=("s0krates", "password"))
        self.graph.run("MATCH (n) OPTIONAL MATCH (n)-[r]-() DELETE n,r")
        self.graph.run("CREATE INDEX ON :Nlp(s_id, i_s)")


    def annotate_horizon (self, horizon=3):
        ''' writes in a column 'horizon' a list of ids of the following n sentences

        :param horizon: nomber of sentences to look forward
        :return: None
        '''
        def horizon_from_row(x):
            return list(self.sentence_df.loc[x.name:x.name + horizon + 1].index)
        self.sentence_df['horizon'] = self.sentence_df.apply(
                                                   horizon_from_row,
                                                   result_type="reduce",
                                                   axis=1)


    def collect_predicates_from_rows(self, sentence_df_row):
        ''' Collect predicates from the df in the horizon and put them in the row of the predicate

        :param sentence_df_row: df with 'horizon'
        :return: the predicates in the horizon

        '''
        if sentence_df_row['horizon']:
            return list(itertools.chain.from_iterable(list(self.sentence_df.loc[sentence_df_row['horizon']]['predication'])))
        else:
            return None

    def annotate_predicates (self):
        ''' Call the function to annotate the predicates for each sentence and puts them into a column 'predication'

            :return: None
        '''

        self.sentence_df['predication'] = self.sentence_df.apply(
                                                   self.Predicatrix.analyse_predications,
                                                   result_type="reduce",
                                                   axis=1)

        self.sentence_df['predications_in_range'] = self.sentence_df.apply(
                                                   self.collect_predicates_from_rows,
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
        if isinstance(id, list):
            if len(id)!=0:
                id=id[0]
        id = str(id)
        return self.Predicatrix.predicate_df.query('id==@id').to_dict(orient='records')


    def get_part_predication(self, pred):
        ''' Collect all particular predications in that sentence. It also looks, if there were more blocks of predicates
            found.

            :param s_id: id of the sentense
            :return: list of predicate dicts.

        '''
        return [pp
                for pred_in_s in self.sentence_df.query('s_id==@s_id')['predication'].values.tolist()
                for x in pred_in_s
                for pp in x['part_predications']]


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


    def get_marked_predication(self, s_id, horizon=2):
        ''' Collect all predicates, that have a certain marker in them.

            It's  usefull, if you want to look for sentences with certain phrases like 'for example', 'in contrast',
            'except'. I searches for the the string, that makes the mark.

            :param s_id: id of the sentense
            :param horizon: how many sentences to look forward
            :return: list of predicate dicts.

        '''
        horizon = list(range(int(s_id) + 1, int(s_id) + horizon + 1))
        markers_pos = 'thus_ADV'
        markers_text = 'for instance'
        mask = (
            self.sentence_df['s_id'].astype(int).isin(horizon) &
            (self.sentence_df['text_pos_'].str.contains(markers_pos, case=False) |
            self.sentence_df['text'].str.contains(markers_text, case=False)))


        predicates =  flatten_reduce(self.sentence_df[mask]['predication'].values.tolist())
        if predicates:
            x = 1
        return predicates



    def annotate_contradictions(self):
        ''' Looks first for pairs of phrases with negations and antonyms in an horizon
            and second it evaluates the similarity of these phrases, what would be the fitting counterpart for that one

        '''
        put_contradiction_into_gdb = self.put_into_gdb("connotation", "contradiction")

        for index, x in self.sentence_df.iterrows():
            self.Contradictrix.find_contradictive(
                x['predication'],
                x['predications_in_range'],
                graph_coro=put_contradiction_into_gdb)


    def get_correllation_preds(self, pred):
        '''  Collect the predicates, that can be modifyers to the predicate.

            These are either in the same sentence or the coreferenced predicates or in the sentence after

            :param pred: the predicate
            :return: list of predicate_tuples

        '''
        if pred == []:
            raise ValueError ('got empty predicate')
        pred = pred[0]
        s_id = pred['s_id']
        predicates_in_sentence = self.get_part_predication(s_id)        # predicates from same sentence
        coref_preds            = self.get_coreferenced_preds (pred)     # predicates that are coreferencing/-ed
        marked_preds           = self.get_marked_predication (s_id)     # predicates with examples or meaning explanations ('e.g.', 'by saying that I mean')
        return predicates_in_sentence + coref_preds + marked_preds


    def get_addressed_coref (self, coref):
        ''' Analyses a coref mention and looks it up in the Database for predications.

            :param coref: dict  with sentence id, start and end of the mention
            :return: a list of coreferenced predicates

        '''
        s_id  = str(coref['s_id'])
        i_list = coref['i_list']
        mask = self.Predicatrix.predicate_df.query("s_id==@s_id")['i_s'].apply(
            lambda ex_i: True if [m for m in i_list if m in ex_i] else False)
        assert mask.any()
        return self.Predicatrix.predicate_df.query("s_id==@s_id")[mask].to_dict(orient='records')


    def get_coreferenced_preds (self, pred):
        ''' Get the predicates, that are coreferenced by the coreference tags of another preducate.

            :param pred: predication tuple
            :return: list oft predicate dicts or [] if not found

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

            :return: None
        '''
        # Lookup what contradicitons were found
        contradictions           = list(self.get_from_gdb('contradiction'))

        # Coroutine for writing the simmix ressults into the gdb
        put_correlation_into_gdb = self.put_into_gdb("Connotation", "correlation")

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
            """MATCH (pred1)-[:CORRELATION {SpecialKind:'correlated'}]-(pred3),
              (pred1)-[:CORRELATION {SpecialKind:'opposed'}]-(pred2),
              (pred2)-[:CORRELATION {SpecialKind:'correlated'}]-(pred4),
              (pred3)-[:CORRELATION {SpecialKind:'opposed'}]-(pred4)      
              RETURN pred1, pred2, pred3, pred4
              """

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

            :return: None

        '''
        # Say, after annotation of the contradictions and their correlating modifyers we have a pair of 'opposed'
        # nodes, as annotated by the correlatrix.
        opposed_pair_pairs   = self.get_opposed_constellation_gdb()

        put_arg_into_gdb     = self.put_into_gdb('denotation', 'argument')
        put_deno_into_gdb    = self.put_into_gdb('denotation', 'subjects_aspects')

        for oppo in opposed_pair_pairs:
            self.Subj_Aspectrix.annotate(
                opposed_pair_pair   = oppo,
                graph_coro_subj_asp= put_arg_into_gdb,
                graph_coro_arg_binding=put_deno_into_gdb
                )


    ''' Colors for the final graph'''
    kind_color_map = {
            'subject'           : '#5B6C5D',
            'aspect': '#5B6C5D',

            'predicate'         : '#9FFFF5',
            'example'           : '#59C9A5',
            'explanation'       : '#56E39F',
            'differential_layer': 'crimson',
            'correl'            : '#465775',
            'center'            : '#ECFEE8',
            'contra'            : '#EF6F6C'
        }


    '''Edge labels for the final picture'''
    edge_label_map = {
        'D_IN' : 'distinguished in',
        'D_TO' : 'this side',
        'SUBJECT': 'thema',
        'ASPECT': 'rhema',
        'SUBJECTS_ASPECTS': 'subj or aspect?',
        'what?': '???'
    }


    def draw(self, G, path):
        ''' Make a nice materialized graph from the distinction query

            :param G: nx.Multidigraph
            :param path: Where to put the file
            :return: None

        '''
        path = path + 'distinction.svg'

        from networkx.drawing.nx_agraph import graphviz_layout
        import pylab as pylab
        import matplotlib as plt
        fig = plt.pyplot.gcf()
        fig.set_size_inches(30, 30)

        def wrap (strs):
            return textwrap.fill(strs, 20)

        #wrapped_node_labels = {n: {'label':wrap(d)} for n, d in G.nodes(data='text') if d}
        wrapped_node_labels = {n: {'label': wrap(d)} for n, d in G.nodes(data='text') if d}

        edge_labels = {(u,v,1): {'xlabel': self.edge_label_map[d]} for u,v, d in G.edges(data='kind', default='what?')}
        node_colors =  {n: {'style':'filled', 'fillcolor':self.kind_color_map[d]} for n, d in G.nodes(data='SpecialKind') if d}

        nx.set_edge_attributes(G, edge_labels)
        nx.set_node_attributes(G, wrapped_node_labels)
        nx.set_node_attributes(G, node_colors)

        G.graph['graph'] = {'rankdir': 'LR',
                            'style': 'filled',
                            'splines':'curved'}
        kamada_kawai = nx.drawing.layout.kamada_kawai_layout(G)

        #spectral = nx.spectral_layout(G)
        spring = nx.spring_layout(G)
        dot_layout = graphviz_layout(G, prog='dot')
        pos = spring

        options = {
            'node_color': 'blue',
            'node_size': 100,
            'width': 3,
            'arrowstyle': '-|>',
            'arrowsize': 12,
        }

        nx.draw_networkx(G,
                         pos=pos,
                         labels=wrapped_node_labels,
                         with_labels=True,
                         font_size=10,
                         **options)
        #nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, rotate=False)

        #plt.title("\n".join([wrap(x, 90) for x in [title, wff_nice, wff_ugly]]), pad=20)
        #pylab.axis('off')

        pylab.savefig(path, dpi=200)
        pylab.clf()
        A = nx.drawing.nx_agraph.to_agraph(G)
        A.layout('dot')
        A.draw(path = "found_distinction.svg")


    def add_determined_expression (self, label, general_kind, special_kind, n1, n2):
        ''' Throws node data into neo4j by expanding data as dictionary.

            :param general_kind: some string property of all members, that are added by this function
            :param special_kind: some string property of all members, that is special for the new members each
            :param n1: predicate dict 1
            :param n2: predicate dict
            :return: None

        '''
        query = \
r"""            MERGE (a:Nlp {{s_id:{s_id1}, text:'{text1}', i_s:{i_s1}}}) 
                ON CREATE SET a.label='{label}', a.id={id1}
                ON MATCH SET a.id=a.id+[{id1}]   
                MERGE (b:Nlp {{s_id:{s_id2}, text:'{text2}', i_s:{i_s2}}}) 
                ON CREATE SET b.label='{label}', b.id={id2}
                ON MATCH SET b.id=b.id+[{id2}]
                MERGE (a)-[:{special_kind_u} {{SpecialKind:'{special_kind}'}}]-(b)
                MERGE (a)-[:{general_kind} {{SpecialKind:'{special_kind}'}}]-(b)""".format(
                label=label.upper(),
                general_kind=general_kind.upper(),
                special_kind=special_kind,
                special_kind_u=special_kind.upper(),

                id1=n1['id'],      s_id1=n1['s_id'],       i_s1=n1['i_s'],      text1=" ".join(n1['text']).replace("'", ""),
                id2=n2['id'],      s_id2=n2['s_id'],       i_s2=n2['i_s'],      text2=" ".join(n2['text']).replace("'", ""),
                )
        logging.info ("querying neo4j for %s" % general_kind)
        self.graph.run(query)


    @coroutine
    def put_into_gdb (self, label, general_kind):
        ''' This returns a subgraph of the graph, selected by the 'general_kind' param.

            :param general_kind: some string property of all members, that are added by this function
            :return: list of Predicate-dict-2tuples

        '''
        while True:
            data = (yield)
            if isinstance(data, tuple) and len(data) == 3:
                n1, n2, special_kind =  data
                self.add_determined_expression(label, general_kind, special_kind, n1, n2)
            else:
                logging.error('Value could not be set because I don\'t know how to deal with the type')
                raise ValueError('Value could not be set because I don\'t know how to deal with the type')
        return None


    def get_from_gdb (self, kind):
        """ Returns pairs of in certain way connected nodes from neo4j

            :param kind: this certain kind of connection; its a property of the graph edges
            :return: tuples of contradicting predicates

        """
        query = (
            r"""MATCH path = (a)-[:%s]->(b) 
                WHERE ID(a) < ID(b)
                RETURN a,b """ % kind.upper()
        )
        logging.info("query neo4j for %s" % kind)
        records = self.graph.run(query).data()
        
        records = [
            (self.get_predication(pair['a']['id']),
             self.get_predication(pair['b']['id']))
            for pair in records
        ]
        assert all (records)
        assert all (all (pair) for pair in records)

        return records

    def move_labels (self):
        ''' Calls a neo4j apoc function to give labels to the node from a property named 'label'

            :return: None

        '''
        query = """MATCH (n:Nlp)
CALL apoc.create.addLabels( id(n), [ n.label ] ) YIELD node
RETURN node"""
        self.graph.run(query)

    def query_distinctions (self):
        """ Makes the query for the distinctions in neo4j.

            It looks for at least two constrasting, correlated pairs of predicates with the same subjects and the same
            aspects

            :return: None

        """
        query = \
            r"""
            // group all contradictions by an id
            CALL algo.unionFind('CONNOTATION', 'CONTRADICTION', {write:true, partitionProperty:"cluster"})
            YIELD nodes as n1
            // group all the correlated hcains by an id
            CALL algo.unionFind('CONNOTATION', 'CORRELATED', {write:true, partitionProperty:"side"})
            YIELD nodes as n2
            
            // find the greater and smaller sections, what is distinguished
            MATCH (a:CONNOTATION)--(b:CONNOTATION)
            WHERE a.cluster = b.cluster and not a.side = b.side
            
            // create the nodes first
            MERGE (x:CORE {cluster:a.cluster, id:a.cluster})
            // create the sides
            MERGE (y:SIDE {cluster:a.cluster, side:a.side, id:a.side})
            MERGE (z:SIDE {cluster:a.cluster, side:b.side, id:b.side})
            
            // connect CORE and sides
            MERGE (x)-[:D_IN]->(y)
            MERGE (x)-[:D_IN]->(z)
            
            // connect CORE, SIDE
            MERGE (a)<-[:D_TO]-(y)
            MERGE (b)<-[:D_TO]-(z)
            
            return x
            """
        logging.info ("query neo4j for distinctions")
        self.distinction_df =  pd.DataFrame(self.graph.run(query).data()).applymap(
            lambda x: x['text']
        )

        G = neo4j2nx_root (self.graph, markers=['CORE', 'SIDE', 'CORRELATED', 'DENOTATION'])
        self.draw(G,'img/')
        return self.distinction_df


    def cleanup_debug_img(self):
        ''' Deletes the content of the './img' folder, that only new pictures are there.
        These pictures are usefull for debugging.

            :return: None

        '''
        import os
        folder = './img'
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                # elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)
