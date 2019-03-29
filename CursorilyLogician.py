#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import time
from functools import lru_cache

import neo4j
import networkx as nx
import textwrap

import matplotlib

from littletools.dict_tools import invert_dict

matplotlib.use('TkAgg')

from time_tools import timeit_context

from hardcore_annotated_expression import eL, eT, apply_fun_to_nested, ltd_ify
from littletools.corutine_utils import coroutine
from littletools.digraph_tools import neo4j2nx_root
from littletools.nested_list_tools import flatten_reduce
from contrastrix import Contrast
from predicatrix import Predication
from correlatrix import Correlation
from subj_and_aspectrix import Subjects_and_Aspects

import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())


class DataframeCursorilyLogician:
    """ This module handles all the databases for the operations on the text and does some transactions in between.

        * a DataFrame in the corpus, representing the grammatical information read from the input files, one
        line per word.
        * a DataFrame with all the predicative substructures of the sentences
        * a DataFrame with the actual sentences in it.
        and
        * a GraphDatabase (neo4j) for storing the results of the operations on the text as well as for querying for
        certain connections of such superficial rhetorical relations in the text

        """
    def __init__(self, corpus):
        """ Set up all the textsuntactical models and the graph database

            :param corpus: corpus instance with loaded conlls

            """
        self.cleanup_debug_img()
        self.corpus = corpus
        self.sentence_df = self.corpus.sentence_df

        self.Predicatrix    = Predication(corpus)
        self.Contradictrix  = Contrast ()
        self.Correlatrix    = Correlation()
        self.Subj_Aspectrix = Subjects_and_Aspects(corpus)

        self.graph = neo4j.Connector('http://localhost:7474', ("s0krates", "password"))
        self.graph.run("MATCH (n) OPTIONAL MATCH (n)-[r]-() DELETE n,r")
        self.graph.run("CREATE INDEX ON :Nlp(s_id, i_s)")

    def annotate_horizon (self, horizon=3):
        """ writes in a column 'horizon' a list of ids of the following n sentences

        :param horizon: nomber of sentences to look forward

            """
        def horizon_from_row(x):
            return list(self.sentence_df.loc[x.name:x.name + horizon + 1].index)
        self.sentence_df['horizon'] = self.sentence_df.apply(
                                                   horizon_from_row,
                                                   result_type="reduce",
                                                   axis=1)


    def collect_predicates_from_rows(self, sentence_df_row):
        """ Collect predicates from the df in the horizon and put them in the row of the predicate

            :param sentence_df_row: df with 'horizon'
            :return: the predicates in the horizon

            """
        if sentence_df_row['horizon']:
            return list(itertools.chain.from_iterable(list(self.sentence_df.loc[sentence_df_row['horizon']]['predication'])))
        else:
            return None

    def annotate_predicates (self):
        """ Call the function to annotate the predicates for each sentence and puts them into a column 'predication'

            """
        logging.info("ANNOTATE PREDICATES")
        with timeit_context('predicate annotating'):
            self.sentence_df['predication'] = self.sentence_df.apply(
                                                       self.Predicatrix.analyse_predications,
                                                       result_type="reduce",
                                                       axis=1)

            self.sentence_df['predications_in_range'] = self.sentence_df.apply(
                                                       self.collect_predicates_from_rows,
                                                       result_type="reduce",
                                                       axis=1)

            self.Predicatrix.post_processing(paint=True)


    def get_predicates_in_horizon(self, s_id):
        """ Collects the predicates, that are in the annotatet range, befor and after the sentence, where to look for
            intresting expressions

            :param s_id: id of the sentence
            :return: list of super-predicates, means they all have the property of having 'part_predications' under it

            """
        return self.sentence_df.query('s_id==@s_id')['predications_in_range'].values[0]

    def get_part_predication(self, pred):
        """ Collect all particular predications in that sentence. It also looks, if there were more blocks of predicates
            found.

            :param s_id: id of the sentense
            :return: list of predicate dicts.

            """
        return [pp
                for pred_in_s in self.sentence_df.query('s_id==@s_id')['predication'].values.tolist()
                for x in pred_in_s
                for pp in x['part_predications']]

    def get_part_predication(self, s_id):
        """ Collect all particular predications in that sentence. It also looks, if there were more blocks of predicates
            found.

            :param s_id: id of the sentense
            :return: list of predicate dicts.

            """
        return [pp
                for pred_in_s in self.sentence_df.query('s_id==@s_id')['predication'].values.tolist()
                for x in pred_in_s
                for pp in x['part_predications']]

    def get_marked_predication(self, s_id, horizon=2):
        """ Collect all predicates, that have a certain marker in them.

            It's  usefull, if you want to look for sentences with certain phrases like 'for example', 'in contrast',
            'except'. I searches for the the string, that makes the mark.

            :param s_id: id of the sentense
            :param horizon: how many sentences to look forward
            :return: list of predicate dicts.

            """
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

    def annotate_contrasts(self):
        """ Looks first for pairs of phrases with negations and antonyms in an horizon
            and second it evaluates the similarity of these phrases, what would be the fitting counterpart for that one

            """

        logging.info ('ANNOTATE CONTRASTS')
        put_acontradiction_into_gdb = self.put_into_gdb("aconnotation", "acontradiction")
        put_ncontradiction_into_gdb = self.put_into_gdb("nconnotation", "ncontradiction")


        with timeit_context('annotating contrasts'):
            for index, x in self.sentence_df.iterrows():
                self.Contradictrix.find_constrasts(
                    x['predication'],
                    x['predications_in_range'],
                    graph_coro=(put_acontradiction_into_gdb, put_ncontradiction_into_gdb))

    @lru_cache(maxsize=None)
    def get_correllation_preds(self, pred):
        """  Collect the predicates, that can be modifyers to the predicate.

            These are either in the same sentence or the coreferenced predicates or in the sentence after

            :param pred: the predicate
            :return: list of predicate_tuples

            """
        if pred == []:
            raise ValueError ('got empty predicate')
        pred = pred[0]
        s_id = pred['s_id']
        predicates_in_sentence = self.get_part_predication(s_id)        # predicates from same sentence
        coref_preds            = self.get_coreferenced_preds (pred)     # predicates that are coreferencing/-ed
        marked_preds           = self.get_marked_predication (s_id)     # predicates with examples or meaning explanations ('e.g.', 'by saying that I mean')
        return eL(predicates_in_sentence + coref_preds + marked_preds)

    def get_coreferenced_preds (self, pred):
        """ Get the predicates, that are coreferenced by the coreference tags of another preducate.

            :param pred: predication tuple
            :return: list oft predicate dicts or [] if not found

            """
        if any(pred['coref']):
            corefs_list = [x for x in pred['coref'] if x]
            preds = [p for corefs in corefs_list for coref in corefs for p in self.Predicatrix.get_addressed_coref(coref)]
            return preds
        else:
            return []

    def annotate_correlations(self):
        """ Look for pairs of hypotactical and paratactical and anaphorical expressions with similar semantics and
            modalities

            That means expressions, that both talk about examples or that both give reasons or are a modifyer to the
            seemingly contradictions, that are found.

            These routines reclassify some of the contradictions, because also talking about examples can seem to be
            anithetical, if the explanation of the instanciated concept is repeated.

            """
        logging.info ('ANNOTATE CORRELATIONS')
        # Lookup what contradicitons were found
        with timeit_context('retrieve constrast clusters'):
            acontradictions           = self.get_from_gdb('acontradiction')
            ncontradictions           = self.get_from_gdb('ncontradiction')


        # Coroutine for writing the simmix ressults into the gdb
        put_acorrelation_into_gdb = self.put_into_gdb("aconnotation", "acorrelation")
        put_ncorrelation_into_gdb = self.put_into_gdb("nconnotation", "ncorrelation")
        types_anto = ("opposed", "opposed", "acorrelated")
        types_nega = ("opposed", "opposed", "ncorrelated")


        with timeit_context('iterate through contradicting pairs'):
            for contra1, contra2 in acontradictions:
                with timeit_context('get possible preds'):
                    poss_correl_l = self.get_correllation_preds(contra1)
                    poss_correl_r = self.get_correllation_preds(contra2)

                with timeit_context('compare constrast pairs'):
                    self.Correlatrix.annotate_correlations(
                        contradiction= eT((contra1, contra2)),
                        possible_to_correlate=eT((poss_correl_l, poss_correl_r)),
                        types=types_anto,
                        graph_coro=put_acorrelation_into_gdb)

            for contra1, contra2 in ncontradictions:
                with timeit_context('get possible preds'):
                    poss_correl_l = self.get_correllation_preds(contra1)
                    poss_correl_r = self.get_correllation_preds(contra2)

                with timeit_context('compare constrast pairs'):
                    self.Correlatrix.annotate_correlations(
                        contradiction= eT((contra1, contra2)),
                        possible_to_correlate=eT((poss_correl_l, poss_correl_r)),
                        types=types_nega,
                        graph_coro=put_ncorrelation_into_gdb)




    def get_clusters (self):
        """ Returns the pattern, that gave a contradiction-opposition-pair

            :return: 2tuple-2tuple-list-predicate-dict

            """
        query = \
            """MATCH (b1:{CONNOTATION})-[:{correlated}]-(a1:{CONNOTATION})--({side}1:{SIDE})--(cl:{CORE})--({side}2:{SIDE})--(a2:{CONNOTATION})-[:{correlated}]-(b2:{CONNOTATION})
               WHERE id({side}1)>id({side}2) and (a1)-[:{contradiction}]-(a2) and (b1)-[:OPPOSED]-(b2) WITH
                    {{   {side}1 : {side}1.{side}, 
                        {side}2 : {side}2.{side}, 
                        contrasts: collect([a1.id,a2.id]),
                        coinings:  collect([b1.id,b2.id])
                    }} as D
               RETURN D
               """

        records1 = self.graph.run(query.format(**self.antonym_names))
        records2 = self.graph.run(query.format(**self.negation_names))

        records = records1 + records2
        predicates = apply_fun_to_nested(fun=self.Predicatrix.get_predication, attribute=['contrasts', 'coinings'], data=records)
        return predicates

    def annotate_subjects_and_aspects(self):
        """ Look for common arguments of the contradicting and correlating predications.

            These are evaluated to be more the subject of the sentences, that the distinction is applied to, or to be
            the aspects, that are some tokens for addressing the perspective which feature of the subject is focussed
            by the other expressions, that correlate to the expressions with the subject

            """
        logging.info ('ANNOTATE SUBJECTS ASPECTS')
        clusters = self.get_clusters()
        self.Subj_Aspectrix.annotate(
                clusters= clusters,
                graph_fun=self.graph.run,
                )

    """ Colors for the final graph """
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


    """ Edge labels for the final picture """
    edge_label_map = {
        'D'    : 'originates from',
        'D_IN' : 'distinguished in',
        'D_TO' : 'this side',
        'X':      'for',
        'D_APART':'distinguished in',
        'D_NPART': 'distinguished in',
        'ACORRELATED':'modifying',
        'NCORRELATED':'modifying',
        'ACORRELATION':'modyfying',
        'NCORRELATION': 'modyfying',
        'OPPOSED':'opposed',
        'ACONTRADICTION':'contrast',
                         'NCONTRADICTION':'contrast',
        'ANTONYM':'by antonym',
                  'NEGATION':'by antonym'

    }

    """ Node labels for final picture """
    node_label_map = {
        'SUBJECTS': 'SUBJECT',
        'ASPECTS':  'ASPECT',
        'REAL_ACORE': 'LETS\'S DISTINGUISH',
        'REAL_NCORE': 'LETS\'S DISTINGUISH',
        'ASIDE':'IN',
        'NSIDE': 'IN',

    }

    def draw(self, G, path):
        """ Make a nice materialized graph from the distinction query

            :param G: nx.Multidigraph
            :param path: Where to put the file
            :return: None

            """
        from networkx.drawing.nx_agraph import graphviz_layout
        import pylab as pylab
        import matplotlib as plt
        fig = plt.pyplot.gcf()
        fig.set_size_inches(30, 30)

        def wrap (strs):
            return textwrap.fill(strs, 20)

        wrappedtext_node_labels = {n: {'label': wrap(d)} for n, d in G.nodes(data='text') if d}
        mapped_node_labels = {n: {'label': " ".join([self.node_label_map[dd] for dd in d['kind'] if dd in self.node_label_map])} for n, d in G.nodes(data=True) if not 'text' in d}
        node_labels = {}
        node_labels.update(wrappedtext_node_labels)
        node_labels.update(mapped_node_labels)

        try:
            edge_labels = {(u,v,1):
                               {'xlabel': self.edge_label_map[d],
                                'headport':'e',
                                'tailport':'e',
                                'splines':'curved'}
                           for u,v, d in G.edges(data='kind', default='what?')}
        except:
            raise KeyError ("%s " % str({(u, v, 1): {'xlabel': (d, d in self.edge_label_map)} for u, v, d in G.edges(data='kind', default='what?')}))
        node_colors =  {n: {'style':'filled', 'fillcolor':self.kind_color_map[d]} for n, d in G.nodes(data='SpecialKind') if d}

        nx.set_edge_attributes(G, edge_labels)
        nx.set_node_attributes(G, node_labels)
        nx.set_node_attributes(G, node_colors)



        G.graph['graph'] = {'rankdir': 'LR',
                            'style': 'filled',
                            'splines':'curved'}
        #kamada_kawai = nx.drawing.layout.kamada_kawai_layout(G)

        #spectral = nx.spectral_layout(G)
        #spring = nx.spring_layout(G)
        dot_layout = graphviz_layout(G, prog='dot')
        #pos = spring

        options = {
            'node_color': 'blue',
            'node_size': 100,
            'width': 3,
            'arrowstyle': '-|>',
            'arrowsize': 12,
        }

        #nx.draw_networkx(G,
        #                 pos=pos,
        #                 labels=node_labels,
        #                 with_labels=True,
        #                 font_size=10,
        #                 **options)

        #pylab.savefig(path, dpi=200)
        #pylab.clf()



        A = nx.drawing.nx_agraph.to_agraph(G)

        all_nodes_rank = {n: [r] for n, r in G.nodes(data='rank')}
        all_rank_nodes = invert_dict(all_nodes_rank)
        for r, n_bunch in all_rank_nodes.items():
            A.add_subgraph(n_bunch, rank='same')

        """
        all_nodes_abunch = {n: [r] for n, r in G.nodes(data='abunch')}
        all_abunch_nodes = invert_dict(all_nodes_abunch)
        for r, n_bunch in all_abunch_nodes.items():
            if r != None:
                A.add_subgraph(n_bunch,
                               name="cluster%s" % str(r),
                               node="[style=filled]",
                               label='antonym'
                               )


        all_nodes_abunch = {n:[r] for n, r in G.nodes(data='nbunch')}
        all_abunch_nodes = invert_dict(all_nodes_abunch)
        for r, n_bunch in all_abunch_nodes.items():
            if r!= None:
                A.add_subgraph(n_bunch,
                         name = "cluster%s"%str(r),
                          node = "[style=filled]",
                          label='negation'                )
        """



        A.layout('dot')
        A.draw(path = path)

    def add_determined_expression (self, label, general_kind, special_kind, n1, n2, reason):
        """ Throws node data into neo4j by expanding data as dictionary.

            :param general_kind: some string property of all members, that are added by this function
            :param special_kind: some string property of all members, that is special for the new members each
            :param n1: predicate dict 1
            :param n2: predicate dict 2

            """
        if  isinstance(n1, list) and  isinstance(n2, list):
            for a,b in zip(n1,n2):
                self.add_determined_expression(label, general_kind, special_kind, a,b, reason)
            return
        elif isinstance(n1, list):
            n1 = n1[0]
            logging.warning ('unqual list balance')
        elif isinstance(n2, list):
            n2 = n2[0]
            logging.warning ('unqual list balance')

        query = \
r"""            MERGE (a:Nlp {{s_id:{s_id1}, text:'{text1}', arg_ids:{argument_ids1}}}) 
                ON CREATE SET a.label='{label}', a.id={id1}
                ON MATCH SET  a:{label}  
                MERGE (b:Nlp {{s_id:{s_id2}, text:'{text2}', arg_ids:{argument_ids2}}}) 
                ON CREATE SET b.label='{label}', b.id={id2}
                ON MATCH SET  b:{label}
                MERGE (a)-[:{special_kind_u} {{SpecialKind:'{special_kind}', Reason:{reason}}}]-(b)
                MERGE (a)-[:{general_kind} {{SpecialKind:'{special_kind}', Reason:{reason}}}]-(b)""".format(
                label=label.upper(),
                general_kind=general_kind.upper(),
                special_kind=special_kind,
                special_kind_u=special_kind.upper(),
                reason=sorted(list(reason)),

                id1=n1['id'],      s_id1=n1['s_id'],       i_s1=n1['i_s'],      text1=" ".join(n1['text']).replace("'", ""), argument_ids1=[int(i['id']) for i in n1['arguments']],
                id2=n2['id'],      s_id2=n2['s_id'],       i_s2=n2['i_s'],      text2=" ".join(n2['text']).replace("'", ""), argument_ids2=[int(i['id']) for i in n2['arguments']]
                )
        self.graph.run(query)
        self.graph.run(query)

    @coroutine
    def put_into_gdb (self, label, general_kind):
        """ This returns a subgraph of the graph, selected by the 'general_kind' param.

            :param general_kind: some string property of all members, that are added by this function
            :return: list of Pred-dict-2tuples

            """
        while True:
            data = (yield)
            if isinstance(data, tuple) and len(data) == 2:
                n1, n2 =  data

                special_kind = data.type
                if hasattr(data, 'reason'):
                    reason = data.reason
                else:
                    reason = []

                self.add_determined_expression(label, general_kind, special_kind, n1, n2, reason)
            else:
                logging.error('Value could not be set because I don\'t know how to deal with the type')
                pass
        return None

    def get_from_gdb (self, kind):
        """ Returns pairs of in certain way connected nodes from neo4j

            :param kind: this certain kind of connection; its a property of the graph edges
            :return: tuples of contradicting predicates

            """

        time.sleep(1)
        query = (
            r"""MATCH path = (a)-[:%s]-(b) 
                WHERE ID(a) < ID(b)
                RETURN a,b """ % kind.upper()
        )
        logging.info("query neo4j for %s" % kind)
        records = self.graph.run(query)
        time.sleep(0.2)

        records = eL([
            eT((self.Predicatrix.get_predication(pair['a']['id']),
                self.Predicatrix.get_predication(pair['b']['id'])))
            for pair in records
        ])
        assert all (records)
        assert all (all (pair) for pair in records)
        return records

    def move_labels (self):
        """ Calls a neo4j apoc function to give labels to the node from a property named 'label'

            :return: None

            """
        query = """MATCH (n:Nlp)
        CALL apoc.create.addLabels( id(n), [ n.label ] ) YIELD node
        RETURN node
        """
        res = self.graph.run(query)
        time.sleep(0.2)

    def cluster_distinctions (self):
        """ Makes the query for the distinctions in neo4j.

            It looks for at least two constrasting, correlated pairs of predicates with the same subjects and the same
            aspects

            :return: None

            """
        logging.info ("query neo4j for distinctions")

        self.move_labels()

        logging.info ("query neo4j for sides and cores")

        query_contradiction_clusters = \
            r"""
            // group all contradictions by an id
            CALL algo.unionFind('{CONNOTATION}', '{contradiction}', {{write:true, partitionProperty:"{cluster}"}})
            YIELD nodes"""

        query_side_clusters = \
            r"""// group all the correlated chains by an id
            CALL algo.unionFind('{CONNOTATION}', '{correlated}', {{write:true, partitionProperty:"{side}"}})
            YIELD nodes
            """

        query_build_cores = \
            r"""// create the nodes first
            MATCH (a:{CONNOTATION})-[c:{contradiction}]-(b:{CONNOTATION})
            WHERE a.{cluster}=b.{cluster} or a.{side} = b.{side}
            WITH a.{cluster} as d, b as b, a as a, c as c
            MERGE (x:{CORE} {{Reason:c.Reason, {cluster}:d}})
            SET b.{cluster} = a.{cluster}, x.id = a.{cluster}
            """

        query_connect_cores_sides = \
            r"""
            // create the {side}s
            MATCH (a:{CONNOTATION})--(b:{CONNOTATION}), (x:{CORE})
            WHERE a.{cluster} = b.{cluster} and not a.{side} = b.{side} and x.{cluster}=a.{cluster} and x.{cluster}=b.{cluster}
            MERGE (y:{SIDE} {{{side}:a.{side}, id:a.{side}}})
            MERGE (z:{SIDE} {{{side}:b.{side}, id:b.{side}}})        
            // connect {CORE} and {side}s
            MERGE (x)-[:D_IN]->(y)
            MERGE (x)-[:D_IN]->(z)            
            """

        query_super_cores = r"""            
                    CALL algo.unionFind(
                      'MATCH (c:{SIDE}) RETURN id(c) as id',
                      'MATCH (c1:{SIDE})<-[f1:D_IN]-(:{CORE})-[f2:D_IN]->(c2:{SIDE})
                       RETURN id(c1) as source, id(c2) as target',
                      {{graph:'cypher',write:true, partitionProperty:'{origin}'}}
                    )
                    Yield nodes
                    MATCH (s1:{SIDE})<-[f1:D_IN]-(c:{CORE})-[f2:D_IN]->(s2:{SIDE})
                    WHERE s1.{origin} = s2.{origin}
                    MERGE (x:REAL_{CORE} {{{origin}: s1.{origin}}})
                    MERGE (x)-[:D_APART]->(c)
                    MERGE (x)-[:D]->(s1)
                    MERGE (x)-[:D]->(s2)
                    """

        query_connect_connotation_sides = \
            r"""// connect
            Match (s:{SIDE}), (n:{CONNOTATION})
            WHERE s.{side} = n.{side}
            MERGE  (n)<-[: D_TO]-(s)
            return n,s
            """

        all_queries = [query_contradiction_clusters, query_side_clusters, query_build_cores, query_connect_cores_sides, query_super_cores, query_connect_connotation_sides]

        self.negation_names = {
            'contradiction':'NEGATION',
            'correlated':'NCORRELATED',
            'CONNOTATION': 'NCONNOTATION',
            'cluster': 'ncluster',
            'CORE':'NCORE',
            'side': 'nside',
            'SIDE':'NSIDE',
            'origin': 'norigin'}

        self.antonym_names = {
            'contradiction':'ANTONYM',
            'correlated': 'ACORRELATED',
            'CONNOTATION': 'ACONNOTATION',
            'cluster': 'acluster',
            'CORE':'ACORE',
            'side': 'aside',
            'SIDE':'ASIDE',
            'origin': 'aorigin'}

        self.graph.run("\nWITH count(*) as dummy\n".join(all_queries).format(**self.antonym_names))
        self.graph.run("\nWITH count(*) as dummy\n".join(all_queries).format(**self.negation_names))

        time.sleep(0.2)

    def draw_distinctions(self, path='img/'):
        """ Retrieve the final distinction graph, forward it to networkx and dump to a picture """
        G = neo4j2nx_root (
            self.graph,
            markers=[('REAL_ACORE'), ('ASIDE'), ('ACONNOTATION'), 'DENOTATION',  ('SUBJECTS', 'ASPECTS'), ('SUBJECTS', 'ASPECTS')])

        self.draw(G, path+'antonym_distinctions.svg')
        G = neo4j2nx_root (
            self.graph,
            markers=[('REAL_NCORE'), ('NSIDE'), ('NCONNOTATION'), 'DENOTATION',  ('SUBJECTS', 'ASPECTS'), ('SUBJECTS', 'ASPECTS')])
        self.draw(G, path+'negation_distinctions.svg')

    def cleanup(self):
        query = """MATCH (c)--(s)--(e)--(f)--(a)--(d)
WHERE (c:REAL_ACORE or c:REAL_NCORE) and (s:ASIDE or s:NSIDE) and (e:ACONNOTATION or e:NCONNOTATION) and (f:ACONNOTATION or f:NCONNOTATION) and (a:DENOTATION) and (d:SUBJECTS or d:ASPECTS)
with collect([a,s,d,e,c]) as result
unwind result as x
unwind x as y
with collect(ID(y)) as all_good
match (x)
where not ID(x) in all_good
detach delete x
        """
        res = self.graph.run(query)



    def cleanup_debug_img(self):
        """ Deletes the content of the './img' folder, that only new pictures are there.
        These pictures are usefull for debugging. """

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
