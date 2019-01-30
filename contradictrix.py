#!/usr/bin/env python
# -*- coding: utf-8 -*-


from predicatrix2 import Predication
import word_definitions
import tdfidf_tool
from simmix import Simmix
from dict_tools import balance_complex_tuple_dict
import networkx as nx
import logging
logging.captureWarnings(True)
logging.getLogger().setLevel(logging.INFO)

class Contradiction:
    def __init__(self):
        self.G = nx.Graph()

        fit_mix_neg = \
             Simmix ( [(1,Simmix.elmo_sim(), 0.4,1),
                       (1, Simmix.common_words_sim, 0.5,1),
                       #(1,Simmix.fuzzystr_sim, 0.45,1)
                        ],
                      n=None)
        self.Contra_Neg  = \
            Simmix([(1, Simmix.formula_contradicts(fit_mix_neg), 0.1, 1)
                    ,(1, Simmix.sub_i,0,0.1)
                    ], n=30)

        fit_mix_ant = \
             Simmix ( [(1,Simmix.elmo_sim(), 0.55, 1),
                       (1, Simmix.excluding_pair_boolean_sim(word_definitions.antonym_dict), 0.1, 1)
                       ], n=None)
        self.Contra_Anto = \
            Simmix([(1, Simmix.formula_prooves(fit_mix_ant), 0.1, 1)
                    ], n=30)

    def find_contradictive (self, predicates1, predicates2, **kwargs):
        if not predicates1 or not predicates2:
            logging.error("no predicate found as noticed")
            return []
        negation_contradictions = self.Contra_Neg.choose ((predicates1, predicates2), type='negation', layout='n', **kwargs)
        antonym_contradictions  = self.Contra_Anto.choose((predicates1, predicates2), type='antonym',  layout='n', **kwargs)
        logging.info("Contradictions by Antonym : %s" % str (antonym_contradictions))
        logging.info("Contradictions by Negation: %s" % str (negation_contradictions))
        try:
            return  negation_contradictions + antonym_contradictions
        except TypeError:
            return (negation_contradictions, antonym_contradictions)

    def draw_graph(self, path):
        import textwrap
        import matplotlib as plt
        from networkx.drawing.nx_agraph import graphviz_layout
        import pylab

        fig = plt.pyplot.gcf()
        fig.set_size_inches(38.5, 38.5)

        def wrap (strs):
            return textwrap.fill(strs, 20)

        node_labels = dict((n, d['label']) for n, d in self.G.nodes(data=True))
        edge_labels = nx.get_edge_attributes(self.G, 'type')

        node_labels = {k:wrap(v) for k,v in node_labels.items()}
        edge_labels = {k:wrap(v) for k,v in edge_labels.items()}

        sprectral  =  nx.spectral_layout(self.G)
        spring     =  nx.spring_layout(self.G)
        dot_layout = graphviz_layout(self.G, prog='dot')
        pos = dot_layout

        options = {
            'node_color': 'blue',
            'node_size' : 100,
            'width'     : 3,
            'arrowstyle': '-|>',
            'arrowsize' : 12,
        }

        nx.draw_networkx(self.G,
                         pos=pos,
                         labels=node_labels,
                         with_labels=True,
                         font_size=10,
                         **options)
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, rotate=False)
        pylab.savefig (path, dpi=200)
        return None

    def draw_key_graphs(self):
        import pylab as P
        import textwrap
        def wrap (strs):
            return textwrap.fill(strs, 20)

        for u,v in self.G.edges:
            dig = nx.DiGraph()

            path = './img/contradicting_key_correlation'+u +" -- "+v+".svg"
            trigger = self.G.edges[u,v]['trigger']
            type  = self.G.edges[u,v]['type']
            node1_pred = trigger[0][0]['key']
            node2_pred = trigger[0][1]['key']
            node1 = trigger[0][0]['key']
            node2 = trigger[0][1]['key']


            dig.add_node(node1, label=wrap(str(node1_pred)))
            dig.add_node(node2, label=wrap(str(node2_pred)))
            dig.add_edge(node1,node2, xlabel=type)
            for (y,z) in trigger:
                node3 = y['key']
                labely = wrap(str(" ".join(y['text'])))
                node4 = z['key']
                labelz = wrap(str(" ".join(z['text'])))
                dig.add_node(node3, label=labely)
                dig.add_node(node4, label=labelz)
                dig.add_edge(node1, node3)
                dig.add_edge(node3, node4)
                dig.add_edge(node4, node2)

            dig.graph['graph'] = {'rankdir': 'LR','splines':'curved'}
            dig.graph['edges'] = {'arrowsize': '4.0'}

            A = nx.drawing.nx_agraph.to_agraph(dig)
            A.layout('dot')
            A.draw(path)
            P.clf()


import unittest
from corpus_reader import CorpusReader


class TestContradictrix(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestContradictrix, self).__init__(*args, **kwargs)
        self.default_C = Contradiction()
        corpus = CorpusReader(corpus_path='./corpora/aristotle_categories/import_conll', only=[7,8,9])
        self.default_P = self.P =  Predication(corpus)

    def test_try_out(self):
        corpus = CorpusReader(corpus_path='./corpora/aristotle_categories/import_conll', only=[10,11,12,13,14,15,16,17])

        from CursorilyLogician import DataframeCursorilyLogician
        Logician = DataframeCursorilyLogician(corpus)

        Logician.annotate_horizon()
        Logician.annotate_predicates()
        Logician.annotate_contradictions()

    def test_antonym_dict(self):

        type_keys = all([isinstance(x, str) or isinstance(x, tuple)
                                  for x in word_definitions.antonym_dict['lemma_'].keys()])
        type_values = \
                    all([isinstance(x, list)
                                  for x in word_definitions.antonym_dict['lemma_'].values()])
        type_values_in_values = \
                    all([isinstance(x,str) or isinstance(x,tuple)
                                  for lists in word_definitions.antonym_dict['lemma_'].values()
                                  for x in lists])
        print (type_keys, type_values, type_values_in_values)
        self.assertTrue(type_keys and type_values and type_values_in_values)

    def test_contradiction_symmetry(self):
        import networkx as nx
        import itertools

        import matplotlib as plt
        plt.rcParams['interactive'] = True

        """
        exs = [
            self.P.nlp("Of things themselves some are not predicable of a subject, and are never present in a subject."),
            self.P.nlp("Some things, again, are present in a subject, but are never predicable of a subject."),
            self.P.nlp("Other things, again, are both predicable of a subject and present in a subject."),
            self.P.conll_over_spacy(
                self.P.nlp("There is, lastly, a class of things which are neither present in a subject nor predicable of a subject, such as the individual man or the individual horse."),
                "./corpus/import_conll/17.conll"),
            self.P.nlp("On the other hand things are said to be named Univocally, which have the name and the definition answering to the name in common"),
            self.P.nlp("Things are named Derivatively which derive their name from some other name but differ from it in termination"),
            self.P.nlp("SECTION 1 Part 1 Things are said to be named Equivocally when though they have a common name the definition corresponding with the name differs for each"),
        ]"""

        # combinations, that must have symmetrical results
        combis = itertools.combinations(exs[::-1], 2)

        graphs = {}
        for i, combo in enumerate(combis):
            graphs[i] = {}
            for  j,(ex1, ex2) in enumerate(itertools.permutations(combo,2)):
                p1 = self.default_P.collect_all_predicates(ex1)
                p2 = self.default_P.collect_all_predicates(ex2)
                #self.default_P.draw_predicate_structure(p1, "p1.png")
                #self.default_P.draw_predicate_structure(p2, 'p2.png')
                if (len (p1) >1) or (len (p2) > 1):
                    self.default_P.print_predicates(p1)
                    self.default_P.print_predicates(p2)
                    raise RuntimeError ("More than one mother predicate for sentence.")
                    break

                G = nx.Graph()
                self.default_C.find_contradictive(p1,p2, out = 'nx', G=G)

                graphs[i].update({j:{"combo": (ex1,ex2), "graph":G, "nodes":list(G.nodes), "edges":str(G.edges)}})
            print (graphs)
            assert (nx.is_isomorphic(graphs[i][0]["graph"], graphs[i][1]["graph"]))

if __name__ == '__main__':
    unittest.main()


