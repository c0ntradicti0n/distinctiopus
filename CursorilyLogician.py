import itertools
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

import matplotlib
matplotlib.use('TkAgg')
import pylab as plt

import textwrap

from contradictrix import Contradiction
from predicatrix2 import Predication
from correlatrix import Correlation
from argumentatrix import Arguments
from webanno_parser import Webanno_Parser
from grammarannotator import GrammarAnnotator,split_data_frame_list_beware

import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())


class DataframeCursorilyLogician:
    def __init__(self, corpus):
        self.corpus = corpus
        self.sentence_df = self.corpus.sentence_df

        self.Contradictrix = Contradiction ()
        self.Predicatrix   = Predication(corpus)
        self.Argumentatrix = Arguments(corpus)
        self.Correlatrix = Correlation()
        self.G = nx.Graph()
        return None

    def annotate_horizon (self, horizon):
        def horizon_from_row(x):
            return list(self.sentence_df.loc[x.name:x.name + horizon + 1].index)
        self.sentence_df['horizon'] = self.sentence_df.apply(
                                                   horizon_from_row,
                                                   result_type="reduce",
                                                   axis=1)
        return None

    def annotate_predicates (self):
        def analyse_predications(x):
            return self.Predicatrix.analyse_predication (doc=x['spacy_doc'], coref=x['coref'], id=x.name)
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
        return None

    def annotate_contradictions(self):
        """Looks first for pairs of phrases with negations and antonyms in an horizon
        and second it evaluates the similarity of these phrases, what would be the fitting counterpart for that one"""

        def get_contradictions_from_row(x):
            if x['horizon']:
                res =  self.Contradictrix.find_contradictive(
                      x['predication'],
                      x['predications_in_range'],
                      out='r',
                      G=self.G,
                      type='contradiction')
                if res:
                    print ("Contradiction(s) for %s in horizon: %s"
                             % (str (" ".join(x['predication'][0]['text'])),
                                str(", ".join([str(i) for i in res]))))
                    return res
                else:
                    return None
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
        return None

    def annotate_correlations(self, G):
        """Look for hypotactical and paratactical and anaphorical expressions with similar semantics and modalities
        (meaning, that these are expressions, that both talk about examples or that both give reasons etc.)"""
        self.G = self.Argumentatrix.annotate_example_nodes(G)
        self.G = self.Argumentatrix.annotate_explanations_nodes(G)
        self.G = self.Argumentatrix.annotate_correlations(G, self.corpus)
        return None

    def annotate_subjects(self, linked_graph):
        """Look for equal arguments of the predications, may they be the logical subjects, the predication is made of"""
        self.G = self.Argumentatrix.annotate_common_arguments(linked_graph)
        return None

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
        #A.write(path=path)


        return None

