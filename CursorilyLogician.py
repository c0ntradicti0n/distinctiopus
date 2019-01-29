import itertools
import networkx as nx
import textwrap

import matplotlib
matplotlib.use('TkAgg')
import pylab as plt

from contradictrix import Contradiction
from predicatrix2 import Predication
from correlatrix import Correlation
from argumentatrix import Arguments
from corutine_utils import coroutine


import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())


from neo4j import GraphDatabase
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("s0krates", "password"))
with driver.session() as session:
    def clean_graph(tx):
        tx.run("match (n) optional match (n)-[r]-() delete n,r")
    session.write_transaction(clean_graph)

class DataframeCursorilyLogician:
    def __init__(self, corpus):
        self.corpus = corpus
        self.sentence_df = self.corpus.sentence_df

        self.Predicatrix   = Predication(corpus)
        self.Contradictrix = Contradiction ()
        self.Correlatrix   = Correlation()
        self.Argumentatrix = Arguments(corpus)


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
        return self.sentence_df.query('s_id==@s_id')['predications_in_range'].values[0]
    def get_predication(self, id):
        return self.Predicatrix.predicate_df.query('id==@id').to_dict(orient='records')

    def annotate_contradictions(self):
        """Looks first for pairs of phrases with negations and antonyms in an horizon
        and second it evaluates the similarity of these phrases, what would be the fitting counterpart for that one"""

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


    def annotate_correlations(self):
        """Look for pairs of hypotactical and paratactical and anaphorical expressions with similar semantics and
        modalities

        That means expressions, that both talk about examples or that both give reasons or are a modifyer to the
        seemingly contradictions, that are found.

        These routines reclassify some of the contradictions, because also talking about examples can seem to be
        anithetical, if the explanation of the instanciated concept is repeated.

        """
        #put_example_into_gdb     = self.put_into_gdb("example")
        #put_explanation_into_gdb = self.put_into_gdb("explanation")

        # Lookup what contradicitons were found
        contradictions           = list(self.get_from_gdb('contradiction'))

        # Coroutine for writing-tasks, no values stored here
        put_correlation_into_gdb = self.put_into_gdb("correlation")

        for contra1, contra2 in contradictions:
            s_id = contra1[0]['s_id']
            horizon_predicates = self.get_predicates_in_horizon(s_id)
            self.Correlatrix.annotate_correlations(
                contradiction= (contra1, contra2),
                possible_to_correlate=horizon_predicates,
                graph_coro=put_correlation_into_gdb)

        #self.G = self.Argumentatrix.annotate_example_nodes(contradictions, self.corpus), put_example_into_gdb)
        #self.G = self.Argumentatrix.annotate_explanations_nodes(contradictions, put_explanation_into_gdb)


    def annotate_subjects_and_aspects(self, linked_graph):
        """Look for some common arguments of the contradicting and correlating predications.
        These may may they be the logical subjects, the pre"""
        contradictions           = list(self.get_from_gdb('contradiction'))
        put_correlation_into_gdb = self.put_into_gdb("entity")
        self.Argumentatrix.annotate_common_arguments(contradictions, put_correlation_into_gdb)

    def center_groups (self, G):
        contradiction_nodes = [n for n, attribute in G.nodes(data=True) if 'contradiction' in attribute]
        contradiction_subgraph = nx.classes.function.induced_subgraph(G, contradiction_nodes)
        groups = [list(g) for g in nx.connected_components(contradiction_subgraph)]

        for i, group in enumerate(groups):
            center_node = "center%d" % i
            G.add_node(center_node, label="let's distinguish:", kind='center')
            G.remove_edges_from (list(contradiction_subgraph.edges(nbunch=group)))
            G.add_edges_from ([(node, center_node) for node in group], triggering_text_labels='exclusive')
        return G

    def subordinate_marked (self, G, kind, new_label):
        subgraph = nx.classes.function.induced_subgraph(
                G,
                [n for n, data in G.nodes(data='kind') if data == kind])

        nodes = list(subgraph.nodes)
        # G.remove_edges_from(list(G.edges(nbunch=nodes)))
        new_edges = []
        for n in nodes:
            found = False
            for dist in range(1,3):
                try:
                    new_node = str(int (n) - dist)
                except ValueError:
                    logging.error ('previously outsourced node, should be handled somewhere else')
                    continue
                if new_node in G and new_node not in nodes:
                    found = True
                    new_edges.append((new_node,n))
                    break
            if not found:
                logging.error ('For some %s nothing well found!' % (kind))
        G.add_edges_from([(u, v) for u, v in new_edges], triggering_text_labels=new_label)
        return G

    def to_digraph(self, G):
        diG = nx.DiGraph()
        centers = [node for node, da in G.nodes(data='kind') if da=='center']
        added = set()
        for center in centers:
            diG.add_node(center, label="let's distinguish:", kind='center', rank=0)
            added.add(center)

            # Predicates
            for neigh_rank1 in G[center]:
                if neigh_rank1 not in added and 'kind' not in G.nodes[neigh_rank1]:
                    diG.add_node(neigh_rank1, label=G.nodes[neigh_rank1]['predicate']['label'], kind='predicate', rank=1)
                    diG.add_edge(center, neigh_rank1, xlabel='exclusive', kind='exclusion')
                    added.add(neigh_rank1)

            # Correlations, Examples, Subjects, Explanations
            for neigh_rank1 in diG[center]:
                for neigh_rank2 in G[neigh_rank1]:
                    if neigh_rank2 not in diG:
                        #print (neigh_rank2)
                        if not 'kind' in G.nodes[neigh_rank2]:
                            print (G.nodes[neigh_rank2])
                            continue
                        diG.add_node(neigh_rank2, label=G.nodes[neigh_rank2]['label'], kind=G.nodes[neigh_rank2]['kind'], rank=2)
                        diG.add_edge(neigh_rank1,neigh_rank2, xlabel=G.nodes[neigh_rank2]['kind'])
                        added.add(neigh_rank2)
        for ed in G.edges(data=True):
            u, v, d = ed
            if 'computing' in d:
                if u in diG and v in diG:
                 diG.add_edge(u, v, xlabel='computed', constraint='false', headport='e', tailport='e', style='dotted')
        return diG

    def get_cleaned_graph(self, split_dict=None, sub_dir_nodes=None, sub_dir_edges=None):
        """A call of a filter function returns graphs with nodes as input und changed eges as output, to use that graph
        for further analysis, this egde data has to be shifted to the nodes and the edges are free to be renamed or
        changed."""
        cG = self.G.copy()

        for (n,c) in list(self.G.nodes(data=True)):
            old_attributes = dict(c)
            if not sub_dir_nodes:
                cG.add_node(n, **old_attributes)
            else:
                for k in list(self.G.node[n].keys()):
                    del cG.node[n][k]
                cG.add_node(n, **{sub_dir_nodes:old_attributes})

        node_labels = dict((n, {'label': d}) for n, d in self.G.nodes(data='label'))
        nx.set_node_attributes(cG, node_labels)

        for (u,v,c) in self.G.edges(data=True):
            uc, vc = list(zip(*c[split_dict]))

            if not sub_dir_edges:
                cG.add_node(u, **uc)
                cG.add_node(v, **vc)
            else:
                cG.add_node(u, **{sub_dir_edges: uc})
                cG.add_node(v, **{sub_dir_edges: vc})
        return cG

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

    def gdb_add_node (kind, data):
        with driver.session() as session:
            session.write_transaction()
        return None

    def add_determed_expression (self, tx, general_kind, special_kind, n1, n2):
        """ Throws node data into neo4j by expanding data as dictionary.

        :param tx:
        :param kind:
        :param data:
        :return:
        """
        if isinstance(n1, list):
            logging.warning("node data is list... taking first")
            n1 = n1[0]
        if isinstance(n2, list):
            logging.warning("node data is list... taking first")
            n2 = n2[0]
        query = (
r"""               MERGE (a:Expression {id:'%s', s_id:'%s', text:'%s'}) 
                MERGE (b:Expression {id:'%s', s_id:'%s', text:'%s'}) 
                MERGE (a)-[:TextRelation {GeneralKind: '%s', SpecialKind:'%s'}]->(b)"""
                %
               (n1['id'], n1['s_id'],  " ".join(n1['text']),
                n2['id'], n2['s_id'],  " ".join(n2['text']),
                general_kind, special_kind))
        logging.info ("querying neo4j the following:\n %s" % query)
        tx.run(query)

    @coroutine
    def put_into_gdb (self, general_kind):
        with driver.session() as session:
            while True:
                data = (yield)
                if isinstance(data, tuple) and len(data) == 3:
                    n1, n2, special_kind =  data
                    #n1, n2 = self.recursive_2tuple((n1,n2))
                    session.write_transaction(self.add_determed_expression, general_kind, special_kind, n1, n2)
                else:
                    logging.error('Value could not be set because I don\'t know how to deal with the type')
                    raise ValueError('Value could not be set because I don\'t know how to deal with the type')
        return None

    def recursive_2tuple (self, t):
        if not isinstance(t, tuple):
            raise ("not a tuple")
        t1, t2 = t
        if len(t) == 2 and isinstance(t1, dict) and isinstance(t2, dict):
            return t
        else:
            if isinstance(t1, tuple) or isinstance(t1, list):
                t1 = self.recursive_2tuple(t1)
            if len (t2) !=2:
                t2 = self.recursive_2tuple(t2)
            if len(t1) == 2 and len(t2) == 2:
                return t1, t2

    def get_determinded_expressions(self, tx, kind):
        query = (
r"""                MATCH path = (a)-[:TextRelation {GeneralKind:'contradiction'}]->(b) 
                WHERE ID(a) < ID(b)
                RETURN a,b """
                )
        logging.info ("query neo4j for reading by this:\n%s" % query)
        records = tx.run(query)
        records = [
            (self.get_predication(pair[0]._properties['id']),
             self.get_predication(pair[1]._properties['id']))
            for pair in records
        ]
        return records

    def get_from_gdb (self, kind):
        """ Returns pairs of in certain way connected nodes from neo4j

        :param kind: this certain kind of connection; its a property of the graph edges
        :yield: tuples of contradicting predicates
        """
        with driver.session() as session:
            while True:
                return session.read_transaction(self.get_determinded_expressions, kind)

import unittest


class TestCursorilyLogician(unittest.TestCase):

    def testNeo4j (self):
        from corpus_reader import CorpusReader
        corpus = CorpusReader(corpus_path="./corpora/aristotle_categories/import_conll", only=[7,9])
        Logician = DataframeCursorilyLogician(corpus)
        Logician.annotate_horizon(horizon=3)
        Logician.annotate_predicates()
        predicates = Logician.sentence_df['predications_in_range'].iloc[0]
        spam = Logician.put_into_gdb ("spam")
        spam.send ((predicates[0], predicates[1]))

if __name__ == '__main__':
    unittest.main()

