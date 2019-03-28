import math
from collections import OrderedDict
from functools import lru_cache

import pandas as pd
import pyprover
import string
from allennlp.commands.elmo import ElmoEmbedder
import spacy

import logging

from time_tools import timeit_context

logging.getLogger(__name__).addHandler(logging.NullHandler())

from littletools.nested_list_tools import *
from littletools.digraph_tools import *
from littletools.dict_tools import *
from similaritymixer import SimilarityMixer
import word_definitions
from littletools import tdfidf_tool
from littletools.generator_tools import count_up


from hardcore_annotated_expression import PredMom, Pred, eL, Argu, eD


class Predication():
    def __init__(self, corpus=None):
        ''' This module lets you translate natural language expressions into predicative chunks, to analyse the
            parts of sentences and their relatitions.

            For the chunking intialize the model and call `analyse_predication` and you get back a hard-core-annotated expression

            These chunks are dicts of other properties, that represent a part of the sentence with different properties.
            A predicate for instance, contains

            Example 1
            =========

            >>> import spacy
            >>> nlp = spacy.load('en_core_web_sm')
            >>> P = Predication()
            >>> text = nlp ('What is going on here?')
            >>> pred = P.analyse_predication(doc=text, s_id='count_yourself')
            >>> pred[0]['dep_']
            ['nsubj', 'aux', 'ROOT', 'prt', 'advmod', 'punct']
            >>> pred[0]['text']
            ['What', 'is', 'going', 'on', 'here', '?']

            Or if you load a corpus, you can specify the predicates you want like this:

            Example 2
            =========

            >>> from corpus_reader import CorpusReader
            >>> corpus = CorpusReader(corpus_path='./corpora/aristotle_categories/import_conll', only=[16])
            >>> from predicatrix import Predication
            >>> P = Predication(corpus)
            >>> from littletools.nested_list_tools import type_spec, flatten_reduce
            >>> corpus.sentence_df.apply(P.analyse_predications, result_type="reduce", axis=1)


            There are special Datatypes defined 'eD, eL, eT, that simply manage
            some conversions (to string for printing), that simply inherit from dict, tuple, list.

            :param corpus: a corpus_reader_object, to overlook the document once, to precompute all lemmata and
            td-idf-values.

        '''
        if corpus:
            lemmatized_text = corpus.lemmatize_text()
            self.tdfidf = tdfidf_tool.tdfidf(lemmatized_text)
        else:
            self.tdfidf = tdfidf_tool.tdfidf("no text")

        options_file = './others_models/elmo_2x1024_128_2048cnn_1xhighway_options.json'
        weight_file = './others_models/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'
        self.elmo = ElmoEmbedder(options_file=options_file, weight_file=weight_file)

        self.attributive_arg_markers       = ['acl', 'prep', 'compound', 'appos']  # genitiv/dativ objects? amod?
        self.verbal_arg_markers            = ['nsubjpass', 'nsubj', 'obj', 'dobj', 'iobj', 'pobj', 'attr']
        self.attributive_too_deep_markers  = ['punct', 'advcl', 'relcl', 'ccomp']
        self.verbal_too_deep_markers       = ['punct', 'advcl', 'relcl', 'ccomp']
        self.ellipsis_too_deep_markers     = ['cc']

        self.uppercase_abc = list(string.ascii_uppercase)
        self.negation_list = word_definitions.logic_dict['~']
        self.antonym_dict  = word_definitions.antonym_dict
        self.stop_words    = word_definitions.stop_words
        self.logic_dict    = invert_dict(word_definitions.logic_dict)
        self.logic_dict    = {k: v[0] for (k, v) in self.logic_dict.items()}

        self.s_id_generator = count_up()
        self.id_generator   = count_up()
        self.pred_key_gen   = count_up()
        self.arg_key_gen    = count_up()

        if not self.negation_list:
            raise ValueError("give negation list")


        self.Counterpart =  SimilarityMixer(
            [   (1, SimilarityMixer.dep_sim, 0, 1),
                (1, SimilarityMixer.tag_sim, 0, 1),
                (1, SimilarityMixer.left_sim, 0, 1),
                (1, SimilarityMixer.fuzzystr_sim, 0, 1)
            ],
            n=1
        )
        self.dict_of_fun = \
            {'dep': Predication.spacy_dep,
             'dep_':Predication.spacy_dep_,
             'pos': Predication.spacy_pos,
             'pos_': Predication.spacy_pos_,
             'tag': Predication.spacy_tag,
             'tag_': Predication.spacy_tag_,
             'lemma': Predication.spacy_lemma,
             'lemma_': Predication.spacy_lemma_,
             'text': Predication.spacy_text,
             'i': Predication.spacy_i}

        self.predicate_df = pd.DataFrame()
        self.argument_df = pd.DataFrame()
        self.already_added_args = OrderedDict()


    def analyse_predications(self, sentence_df_row):
        ''' Collect all information from a sentence DataFrame row and analyse that predication with this tool

            :param sentence_df_row: df with 'spacy_doc' and 'coref' and 's_id' columns
            :return: predicate dict

        '''
        with timeit_context('total'):
            return self.analyse_predication (doc=sentence_df_row['spacy_doc'], coref=sentence_df_row['coref'], s_id=sentence_df_row['s_id'])

    def spacy_dep_ (ex):
        return set (x.dep_ for x in ex )
    def spacy_dep (ex):
        return set (x.dep for x in ex )
    def spacy_pos_ (ex):
        return set (x.pos_ for x in ex )
    def spacy_pos (ex):
        return set (x.pos for x in ex )
    def spacy_lemma_ (ex):
        return set (x.lemma_ for x in ex )
    def spacy_lemma (ex):
        return set (x.lemma for x in ex )
    def spacy_tag_ (ex):
        return set (x.tag_ for x in ex )
    def spacy_tag (ex):
        return set (x.tag for x in ex )
    def spacy_i (ex):
        return set (x.i for x in ex )
    def spacy_text (ex):
        return set (x.text for x in ex )

    def dictize_spacy (self,ex):
        return {kind:fun(ex) for kind, fun in self.dict_of_fun}

    def analyse_predication(self, doc=None, s_id=None, **kwargs):
        ''' Analyse a spacy doc for predications

            :param doc: spacy doctument
            :param s_id: id of sentence, use `s_id='count_yourself'` for not caring for this. In these cases, the td-idf
            information is not possible to compute, because one does not know, to which text this belongs
            :param kwargs:  :func:`~predcicatrix.Predication.collect_all_predicates`
            :return: list of predicate_dicts

        '''
        if s_id == 'count_yourself':
            s_id = next(self.s_id_generator)
        if not doc:
            raise KeyError("keyword argument 'doc'must be given")
        if s_id == None:
            raise KeyError("keyword argument 's_id' must be given")

        ps = self.collect_all_predicates(doc, s_id=s_id, **kwargs)
        return ps

    def get_attributive_roots(self,ex):
        return (e for e in ex if e.dep_ in ['nsubjpass', 'nsubj', 'obj', 'dobj', 'iobj', 'pobj', 'conj'] and e.tag_ not in ['WDT', 'DET', 'PRON'])

    def get_verbal_roots(self,ex):
        v_roots = list(e for e in ex if
                        e.pos_ in ['VERB']
                        and
                        e.dep_ not in ['acl'])
        v_roots += list(e for e in ex if
                        e.dep_ in ['conj']
                        and
                        e.head.dep_ in ['acomp'])

        if not v_roots:
            text = " ".join(x.text for x in ex)
            pos = " ".join(x.tag_ for x in ex)
            dep = " ".join(x.dep_ for x in ex)
            logging.warning (ValueError('Sentence without tag VERB, but dep ROOT, probably AnnotationError, "%s", %s, %s' % (text, pos, dep)))
            v_roots = tuple(e for e in ex if e.dep_ in ['ROOT'])
        return v_roots

    def get_sentence_roots(self, ex):
        v_roots = tuple(e for e in ex if e.dep_ in ['ROOT'] and e.pos_ not in ['NOUN'])
        return v_roots

    def build_predicate(self,predicate_i, arguments_i, full_ex_i, doc):
        ''' Build a dict from information over a predicate

            :param predicate_i: indices of predicates
            :param arguments_i: indices of arguments as list of list
            :param full_ex_i:  all indices
            :param doc: the spacy document
            :return: dict wth
                         "predicate"    : spacy tokens  of predicate,
                         "predicate_i"  : indices of that,
                         "arguments"    : spacy tokens of arguments,
                         "arguments_i"  : indices of that,
                         "full_ex"      : all spacy tokens,
                         "i_s"          : indices of that
                         "doc"          : spacy tokens
                         "text"         : list of words,
                         "key"          : a unique identifyer for that predicate

        '''
        try:
            predicate_ex =  [doc[x] for x in predicate_i]
            arguments_ex = [[doc[x] for x in arg_i] for arg_i in arguments_i]
            full_ex   =  [doc[x] for x in full_ex_i]
        except TypeError:
            raise ValueError ('Computed empty value error')

        predicate = Pred({"predicate": predicate_ex,
                     "predicate_i"  : predicate_i,
                     "arguments"    : arguments_ex,
                     "arguments_i"  : arguments_i,
                     "full_ex"      : full_ex,
                     "i_s"          : full_ex_i,
                     "doc"          : doc,
                     "text"         : [str(x.text) for x in full_ex][:],
                     "key"          : str(next(self.pred_key_gen))
                          })
        return predicate

    def post_process_arguments (self, arguments_i,doc):
        ''' Iterate through arguments and give back valid noun/pronoun-cores

            :param arguments_i: indices of possible argument tokens in the sentence (list(list))
            :param doc: spacy doc
            :return: list of argument core tokens

        '''
        for arg in arguments_i:
            arg = self.collect_any_argument(arg, doc)
            if arg:
                yield arg


    def collect_any_argument(self, arg_i, doc, out='i'):
        ''' From an argument expression take the noun-core and not its dependents

            :param arg_i: indices of possible argument tokens in the sentence
            :param doc: spacy document
            :return: list of spacy tokens

        '''
        arg = []
        for i in arg_i:
            if doc[i].pos_ in ['NOUN', 'PRON'] or doc[i].dep_ in ['nsubjpass','nsubj', 'obj', 'pobj', 'dobj']:
                if out == 'i':
                    arg.append(i) #+ [r.i for r in doc[i].rights if r.i not in too_deep_i]
                if out == 't':
                    arg.append(doc[i])
        return arg

    def collect_substantial_argument(self, arg_i, doc, out='i'):
        ''' From an argument expression take the noun-core and not its dependents

            :param arg_i: indices of possible argument tokens in the sentence
            :param doc: spacy document
            :return: list of spacy tokens

        '''
        arg = []
        for i in arg_i:
            if doc[i].pos_ in ['NOUN'] and doc[i].dep_ in ['nsubjpass','nsubj', 'obj', 'pobj', 'dobj']:
                if out == 'i':
                    arg.append(i) #+ [r.i for r in doc[i].rights if r.i not in too_deep_i]
                if out == 't':
                    arg.append(doc[i])
        return arg


    def arg_stop (self, s):
        arg = [s]
        for ch in s.children:
            if (
               ch.dep_ in ['appos ', 'det', 'amod', 'det', 'prep', 'pobj', 'csubj', 'nsubj', 'npasssubj', 'obj', 'dobj']
            and ch.text not in ['as', 'like']):
                arg.extend(self.arg_stop(ch))
        return arg


    def collect_grammatical_predicates(self, root_token, arg_markers, too_deep_markers, ellipsis_resolution=True,
                                       no_zero_args=True, mother = False, attributive_ordering=False):
        ''' Extract the predicative structure arising from a special triggering word, that can be handled like in
            predicate logic.

            A predicate is divided into two major parts:
            one expression for the information of the concept (the function-name, the Begriff), and its arguments (the
            arguments of the function, the individuals and variables for them).

            :param root_token: verbal or substantive root word
            :param arg_markers: what kinds of dependences to take arguments
            :param too_deep_markers: where to cut the expression because of deepness
            :param ellipsis_resolution: if coordinative bindings should be dissolved every time
                 (and the elliptical part doubbled for the second part)
            :param no_zero_args:
                 its not allowed for predications to have zero arugments
            :param mother:
                 the whole expression should appear as result
            :param attributive_ordering:
                 False (default):
                     A verbal is the root of the predicative structure and dependents with noun or pronoun cores are the 'arguments'
                 True:
                     If the predicate is triggered by an adjective-substantive-junction, then the substantive (with its
                     dependents) is the lonly argument and the adjective with its dependents is the predicate

            :return: dict from :func:`~predicatrix.Predication.build_predicate`

        '''
        predicate_i    = []
        too_deep_i      = []
        lpar_toodeep_i = []
        if not root_token:
            return None

        def is_counterpart(t):
            return any([(r.dep_ in ['cc'] and r.text  in word_definitions.counterpart_words) for r in t.head.rights])

        if (ellipsis_resolution
            and root_token.head.dep_ != 'ROOT'
            and root_token.dep_ in ['conj']
            and list(root_token.head.rights)
            and any(r.dep_ in ['cc', 'mark'] for r in root_token.head.rights if r.text not in word_definitions.counterpart_words)):
            predicate_i += [root_token.head.head.i] + [x.i for x in root_token.head.rights  if x.dep_ in ['cc'] ]

        for subleaf in root_token.subtree:
            predicate_i.append(subleaf.i)

            sl_is_counterpart = is_counterpart(subleaf)

            if (root_token.i != subleaf.i) and (subleaf.dep_ in ['conj', 'cc'] and not sl_is_counterpart)   and not mother:
                if len ([s.i for s in subleaf.subtree]) >2 and subleaf.dep_ != 'cc':
                    lpar_toodeep_i += [s.i for s in subleaf.subtree] + [subleaf.i]

            if ((root_token.i != subleaf.i) and (subleaf.dep_ in too_deep_markers)
                or (subleaf.dep_ in ['conj'] and not sl_is_counterpart and not subleaf == root_token)
               ) and not mother:
                if (True or len ([s.i for s in subleaf.subtree]) >2) and subleaf.dep_ != 'cc':
                    counterpart = [s.i for s in subleaf.subtree]
                    conjunction = [c.i
                                   for c in subleaf.head.rights
                                   if c.dep_ in ['cc'] and c.i<subleaf.i ]
                    too_deep_i  += counterpart + conjunction

        predicate_i = list(set(predicate_i) - set(too_deep_i[:]))

        if root_token.dep_ == 'conj':
            if root_token.head.dep_ in ['acomp','dobj','pobj','aobj'] :
                head = root_token.head.head
            else:
                head = root_token.head

            lefts = set([s.i for s in head.lefts])
            children_of_lefts = set([s.i for left in head.lefts for s in left.subtree])
            children_of_lefts.update(lefts)
            elliptical   = children_of_lefts - set (too_deep_i[:])
            predicate_i  = list(elliptical) + predicate_i

        arguments_i    = []
        for j in predicate_i:
            subleaf = root_token.doc[j]
            if subleaf.dep_ in arg_markers:
                argument_i = [x.i for x in self.arg_stop(subleaf)]
                arguments_i.append(argument_i)

        predicate_i =  set(predicate_i) - set(flatten(arguments_i[:]))
        arguments_i = [list(set(arg_i) - set(too_deep_i[:])) for arg_i in arguments_i]

        if not predicate_i:
            return None
        if no_zero_args and not list(flatten(arguments_i[:])):
            return None

        predicate_i =  sorted(predicate_i)
        arguments_i = [sorted(arg_i) for arg_i in arguments_i if arg_i]
        full_ex_i   =  sorted(set(predicate_i) | set(flatten(arguments_i)))

        if ellipsis_resolution:
            full_ex_i = sorted(list(set(full_ex_i) - set(lpar_toodeep_i)))
            predicate_i = sorted(list(set(predicate_i) - set(lpar_toodeep_i)))
            arguments_i = [sorted(set(arg_i) - set(lpar_toodeep_i)) for arg_i in arguments_i if arg_i]

        if attributive_ordering:
            predicate_i, arguments_i = list(flatten_list(arguments_i)), [predicate_i]

        doc = root_token.doc
        arguments_i = list(self.post_process_arguments(arguments_i, doc))
        predicate = self.build_predicate(predicate_i,arguments_i,full_ex_i,doc)
        return predicate

    p_counter = itertools.count()

    def attribute_negation_particle (self, p):
        p['negation_particle'] = [x for x in p['full_ex'] if x.text in self.negation_list]
        p['contained_antonyms']          = OrderedDict({x: self.antonym_dict['lemma_'][x] for x in p['lemma_'] if x in self.antonym_dict['lemma_']})
        p['antonyms']                    = OrderedDict()
        return None

    def organize_negations (self, ps):
        """ Annotate negations in nested structure

        :param ps: list of predicates
        :return:
        """
        for s in ps:
            for p in s['part_predications']:
                self.attribute_negation_particle(p)
                p['negation_count'] = 0

        for s in ps:
            if 'containment_structure' not in s:
                logging.warning('no graph for this predicate')
                continue
            G = s['containment_structure']

            for edge in G.edges():
                x_i, y_i = edge
                x = G.nodes[x_i]['predicate']
                y = G.nodes[y_i]['predicate']

                common_negation    = [n1 for n1 in x['negation_particle'] for n2 in y['negation_particle']
                                              if n1==n2]

                common_antonyms    = {n1:x['contained_antonyms'][n1] for n1 in x['contained_antonyms'] for n2 in y['contained_antonyms']
                                               if n1==n2}

                G.edges[edge]['shared_negations'] =  tuple(common_negation)
                G.edges[edge]['antonym suspicious']  =  tuple(common_antonyms )

        try:
            self.tree_distribute_marker_edges(G, key_of_edge_marker='shared_negations', key_to_write='negation_count')
            self.tree_distribute_marker(G, key_of_marker='negation_particle', key_to_write='negation_count')

            self.tree_distribute_marker_edges(G, key_of_edge_marker='antonym suspicious', key_to_write='antonyms', out='l')
            self.tree_distribute_marker(G, key_of_marker='contained_antonyms', key_to_write='antonyms', out='l')
        except UnboundLocalError:
            logging.error("predicates empty")
            pass

        return ps

    def tree_distribute_marker_edges(self, G, key_of_edge_marker, key_to_write, out='n'):
        # Negations that occur in other branch
        for path in source_sink_generator(G):
            rath = list(reversed(path))
            for i, n in enumerate(rath[:-1]):
                edge = G.edges[(rath[i + 1], rath[i])]
                if edge[key_of_edge_marker]:
                    negation_containing_node = rath[i]
                    G.nodes[negation_containing_node]['predicate'][key_to_write] = len(edge[key_of_edge_marker]) if out == 'n' else edge[key_of_edge_marker]
                    break

    def tree_distribute_marker(self, G, key_of_marker, key_to_write, out='n'):
        # Marker in single leaves
        for node, attrs in G.nodes(data=True):
            markers = attrs['predicate'][key_of_marker]
            neighbors = G.neighbors(node)
            markers_of_neighbors = list(
                flatten([G.nodes[neighbor]['predicate'][key_of_marker] for neighbor in neighbors]))

            if any([n not in flatten(markers_of_neighbors) for n in markers]):
                just_these_markers = list(mak for mak in markers if mak not in markers_of_neighbors)
                attrs['predicate'][key_of_marker] = just_these_markers
                attrs['predicate'][key_to_write] = len(just_these_markers)  if out == 'n' else OrderedDict((n,markers[n]) for n in just_these_markers)


    def organize_subpredicates(self, ps):
        ps = sorted(ps, key=lambda x:-len(x['i_s']))

        Sub_sim = SimilarityMixer([(1, SimilarityMixer.sub_i, 0.1, 1),
                                   (-1, SimilarityMixer.boolean_same_sim, 0, 0.1)],
                                  n = None)

        if not ps:
            logging.error ("empty expression_list can't contain any predicate.")
            return []
        contain =  Sub_sim.choose ((eL(ps[:]), eL(ps[:])), out='2t', layout='n')

        if not contain:
            # If there is no dependent structure, its just one predicate, that contains itself for convenience
            for p in ps:
                p['part_predications'] = eL([p])
            return ps

        edges = contain

        containment_structure = nx.DiGraph()
        containment_structure.add_nodes_from([i for i,p in enumerate(ps)])
        attrs = {i: {'label': p['key']+ "  " + " ".join(p['text']), 'predicate':p} for i,p in enumerate(ps)}
        nx.set_node_attributes(containment_structure, attrs)
        containment_structure.add_edges_from(edges)
        containment_structure = transitive_reduction(containment_structure)
        source_nodes = [node for node, indegree in containment_structure.in_degree(containment_structure.nodes()) if indegree == 0]

        p_new = []
        for r in source_nodes:

                 mother_node = eD(ps[r])
                 mother_node['id'] =  next(self.id_generator)
                 descendants = nx.algorithms.descendants(containment_structure, r)
                 descendants.update({r})
                 sub_predicates                 = [ps[x] for x in descendants]  # the node itself and its descendants
                 mother_node['part_predications']     = eL(sorted(sub_predicates, key=lambda x:len(x['i_s'])))
                 nx.set_node_attributes(containment_structure, {n:{"__subgraph__": mother_node['id']} for n in descendants})
                 mother_node['containment_structure'] = containment_structure
                 p_new.append(mother_node)

        return p_new


    def collect_verbal_attributive_grammatical_predicates (self, ex):
        a_roots = list(self.get_attributive_roots(ex))
        v_roots = list(self.get_verbal_roots(ex))
        m_roots = list(self.get_sentence_roots(ex))

        a_ps = []
        for x in a_roots:
            p = self.collect_grammatical_predicates(
                    x,
                    self.attributive_arg_markers,
                    self.attributive_too_deep_markers,
                    no_zero_args=True,
                    attributive_ordering=True)
            a_ps.append(p)

        v_ps = []
        for x in v_roots:
            p = self.collect_grammatical_predicates(
                    x,
                    self.verbal_arg_markers,
                    self.verbal_too_deep_markers,
                    no_zero_args=True,
                    ellipsis_resolution=True)
            v_ps.append(p)


        m_ps = []
        for x in m_roots:
            p = self.collect_grammatical_predicates(
                        x,
                        self.verbal_arg_markers,
                        self.verbal_too_deep_markers,
                        no_zero_args=True,
                        mother = True,
                        ellipsis_resolution=False)
            v_ps.append(p)

        ps = v_ps + a_ps + m_ps
        ps = [x for x in ps if x]

        ps_new = []
        for p in ps:
            if not any(set(p['i_s']) == set(p2['i_s']) for p2 in ps_new):
                ps_new.append(p)
        ps = ps_new

        for predicate in ps:
            predicate["doc"] = ex[0].doc
            predicate["dep"] = [ex[x].dep for x in predicate['i_s']]
            predicate["pos"] = [ex[x].pos for x in predicate['i_s']]
            predicate["tag"] = [ex[x].tag for x in predicate['i_s']]
            predicate["dep_"] = [ex[x].dep_ for x in predicate['i_s']]
            predicate["pos_"] = [ex[x].pos_ for x in predicate['i_s']]
            predicate["tag_"] = [ex[x].tag_ for x in predicate['i_s']]
            predicate["text"] = [ex[x].text for x in predicate['i_s']]
            predicate["lemma"] = [ex[x].lemma for x in predicate['i_s']]
            predicate["lemma_"] = [ex[x].lemma_ for x in predicate['i_s']]
            predicate["lemma_tag_"] = [ex[x].lemma_ + '_' + ex[x].tag_ for x in predicate['i_s']]
        return ps


    def collect_all_predicates(self, ex, coref=None, s_id=None,  paint_graph=True):
        ''' Extracts a multitude of properties from a natural language expression, including grammar
            imformation, word2vec, some importance weight in the document, some kind of cursorily logical
            formula with a dictionary, to what expressions the constants in the formula belong to.

            :param ex:
                string or spracy instance of the language expression
            :return:
                dictionary of properties of the expression

        '''
        if not ex:
            return None
        try:
            logging.info('sentence %s: predicates found for %s' % (s_id, " ".join(x.text for x in ex)))
        except AttributeError:
            logging.error('expression is no spacy doc? sentence %s %s' % (s_id, str(ex)))
            return []

        with timeit_context('embed'):
            elmo_embeddings = self.elmo.embed_sentence([x.text for x in ex])

            # Take out negations for semantics, they don't have good embeddings and disturb all other things
            indices_of_negations = [i for i, x in enumerate(ex) if x.lemma_ in self.negation_list]
            indices_of_stopwords = [i for i, x in enumerate(ex) if x.lemma_ in self.stop_words]
            elmo_embeddings[:,[indices_of_negations + indices_of_stopwords]] = 0

        with timeit_context('weight'):
            try:
                importance = self.tdfidf.sentence2vec([x.lemma_ for x in ex])
            except AttributeError:
                logging.warning("No td-idf information was precomputed for this expression, filling it up with 0")
                importance = self.tdfidf.half_importance([x.lemma_ for x in ex])

        with timeit_context('filter predicate from spacy'):
            ps = self.collect_verbal_attributive_grammatical_predicates(ex)

        with timeit_context('build nx tree'):
            dep_tree = self.build_dependency_graph(ex)

        if not ps:
            logging.error("no predicates found for %s" % str(ex))

        if not coref:
            coref = [[]]*len (ex[0].doc)

        with timeit_context('stuff'):
            for p in ps:
                try:
                    c = [coref[i] for i in p['i_s']]
                except IndexError:
                    raise


                p["id"]                      = str(next(self.id_generator))
                p['s_id']                    = s_id
                p["elmo_embeddings"]         = elmo_embeddings[:,p['i_s']].sum(axis=1)
                p["elmo_embeddings_pred"]    = elmo_embeddings[:,p['predicate_i']].sum(axis=1)
                p["coref"]                   = [coref[i] for i in p['i_s']]
                p["elmo_embeddings_per_word"]= elmo_embeddings[:,p['i_s']]
                p["elmo_embeddings_full"]    = elmo_embeddings
                p["importance"]              = importance[p['i_s']]
                p["importance_full"]         = importance
                p['dep_tree']                = dep_tree
                p['arguments']               = eL(self.sp_imp_elmo_dictize_ex (
                                             p['arguments'],
                                             coref,
                                             elmo_embeddings,
                                             importance,
                                             s_id,
                                             dep_tree))(node_type=['HAS_ARGMENTS'])
                p = Pred(p)

        with timeit_context('contained'):
            ps = self.organize_subpredicates(ps)

        with timeit_context('organize negations'):
            ps = self.organize_negations(ps)

        with timeit_context('formalize'):
            for p in ps:
                p = self.formalize(p)

        if not ps:
            text = " ".join([x.text for x in ex])
            logging.error("No predication found in expression: '%s'." % text )
            return []

        with timeit_context('databasing'):
            self.organize_dfs(ps)
        return ps

    def organize_dfs(self, ps):
        ps = eL([PredMom(p) for p in ps])
        self.append_to_predicate_df(ps)


        arguments_mother = flatten_reduce(list(map(lambda x: flatten_reduce([x['arguments']]), ps)))
        arguments_parts = flatten_reduce(list(map(
            lambda x: flatten_reduce([part_pred['arguments']
                                      for part_pred in x['part_predications']]),
            ps)))
        arguments = eL(arguments_mother + arguments_parts)

        self.append_to_argument_df(arguments)



    def build_dependency_graph (self, doc):
        edges = []
        for t in doc:
            edges.append((t.head.i, t.i, {'dep':t.dep_, 'pos':t.pos_, 'tag':t.tag}))
        G = nx.Graph(edges)
        return G


    def append_to_predicate_df (self, ps):
        ''' There is a predicate df to look up coreferences and ids, if a predicate expression is addressed by coref or
            some external database, to what we don't want to tell everything

            :param ps: list of predicates with 'part_predications'

        '''
        #self.predicate_df = pd.DataFrame.from_records(ps)
        #self.predicate_df = self.predicate_df.append(ps)
        self.predicate_df = self.predicate_df.append(ps+[part_p for p in ps for part_p in p['part_predications']])


    def get_predication(self, id):
        ''' Finds the predicate dict, that a special id belongs to by looking it up in the DataFrame in the Predication-
            module.

            :param id: id of that predicate
            :return: predicate-dict

        '''
        if isinstance(id, list):
            if len(id)!=0:
                return [self.get_predication(i) for i in id]

        id = str(id)
        rec = self.predicate_df.query('id==@id').to_dict(orient='records')
        return eL([Pred(r) for r in rec])


    def get_addressed_coref (self, coref):
        ''' Analyses a coref mention and looks it up in the Database for predications.

            :param coref: dict  with sentence id, start and end of the mention
            :return: a list of coreferenced predicates

        '''
        s_id  = str(coref['s_id'])
        i_list = coref['i_list']

        df_part = self.predicate_df.reset_index().query("s_id==@s_id")

        df_part['score'] = df_part['i_s'].apply(lambda ex_i:
                                                           (len ([m for m in i_list if m in ex_i])
                                                          - len ([m for m in i_list if m not in ex_i]) )
                                                          / len(ex_i)
                                                         )

        n = 2
        nhighest = df_part.nlargest(n=n, columns='score')
        acceptable = nhighest[nhighest['score']>0]
        rec = acceptable.to_dict(orient='record')

        return eL([Pred(r) for r in rec])



    def append_to_argument_df (self, arguments):
        ''' There is a DataFrame with all the argumentes found to look up coreferences and ids, if an expression is
            addressed by coref or some external database, to what we don't want to tell everything

            :param ps: list of arguments

        '''
        new_arguments = []

        for arg in arguments:
            pos = (arg['s_id'], tuple(arg['i_s']))
            # If that arg is not in the df, append it
            if not (arg['s_id'], tuple(arg['i_s'])) in self.already_added_args:
                new_arguments.append(arg)
                self.already_added_args.update({pos:arg['id']})
                continue
        # Else take the 'id' from the stored ids and override it
            else:
                arg['id'] = self.already_added_args[pos]

        with timeit_context('append'):

            df = pd.DataFrame(new_arguments)
            self.argument_df = pd.concat([self.argument_df,df])

    def get_coreferenced_arguments(self, corefs):
        if not corefs:
            return []
        else:
            arguments = eL([x for x in flatten_reduce(list(map(self.get_coreferenced_argument, corefs))) if x])
            return arguments

    def get_coreferenced_argument(self, coref):
        return self.get_coreferenced_argument_cacheble (eD(coref))


    @lru_cache (maxsize=100)
    def get_coreferenced_argument_cacheble(self, coref):
        ''' We get the coreferenced arguments from the arguments-dataframe, that was build while constructing all
        predicate or we reconstruct it.

        There two ways of getting the argument:
        * Either these noun-cored expressions are in the ext as they are, like:
             The [blue wolf] roars to the moon. He (= blue wolf) was spayed blue by conservationists.
        * Or you have to construct it from a verbal expression:
             The the wolf was spayed blue and the ants were spayed green by conservationists. The latter (= the green
              ants) are save now.

        :param coref: coreference mention with i_list and s_id
        :return: list of Argu_s

        '''
        s_id  = str(coref['s_id'])
        i_list = coref['i_list']
        print (i_list)

        mask = self.argument_df.query("s_id==@s_id").apply(
                lambda ex: all (m in ex['i_s'] for m in i_list) and ('NOUN' in ex['pos_']),
            axis=1)

        if mask.any():
            # If there is something in the df, return it
            referenced = self.argument_df.query("s_id==@s_id")[mask].nsmallest(n=1, columns='len').to_dict(orient='records')
            return [Argu(x) for x in referenced]
        else:
            # If it's not found before (e.g. contructed from multiple parts of the sentence). construct it
            doc, elmo_embeddings, importance, dep_tree = self.argument_df.query("s_id==@s_id")[['doc', 'elmo_embeddings_full', 'importance_full', 'dep_tree']].values[0]
            arg_tokens = self.collect_substantial_argument(i_list, doc, out='t')
            if arg_tokens == []:
                return []
            arg = Argu(self.sp_imp_elmo_dictize_ex(ex=arg_tokens, coref=[[]]*len(doc), elmo_embeddings=elmo_embeddings, importance=importance, s_id=s_id, dep_tree=dep_tree))
            self.append_to_argument_df([arg])
            return [arg]

    def post_processing(self, paint=False):
        ''' resolves als coreferences and gives the option to draw all predicates

        :param paint: True or False, print the file to the outputfolder
        '''
        def to_replace (tok):
            return tok.pos in ['PRON']

        def resolve_lemma(x, deep=False):
            resolutions = []

            for pos, crs in enumerate(x['coref']):
                for cr in crs:
                    if to_replace(x['full_ex'][pos]):
                        poss_noun_arg = self.get_coreferenced_argument(cr)
                        if poss_noun_arg :
                            resolutions += [(cr,self.get_coreferenced_argument(cr), pos)]

            for r in resolutions:
                x['lemma_'] = [r[1][0]['lemma_'][0] if i == r[2] else w     for i, w in enumerate (x['lemma_']) ]
                x['text']   = [r[1][0]['text'][0] if i == r[2] else w     for i, w in enumerate (x['text']) ]
                x['full_ex']= [r[1][0]['full_ex'][0] if i == r[2] else w     for i, w in enumerate (x['full_ex']) ]
                x["label"]  = x['text']

            if 'part_predications' in x: # For part predicates this is nan
                if isinstance(x['part_predications'], float):
                    return x
                for pp in x['part_predications']:
                    if not deep:
                        resolve_lemma(pp, deep=True)
            return x

        self.predicate_df = self.predicate_df.apply(resolve_lemma, result_type="reduce",
                                                   axis=1)

        ps = self.predicate_df.to_dict(orient='records')

        if paint:
            for p in ps:
                if isinstance(p['part_predications'], float): # For part predicates this is nan
                    continue
                self.draw_predicate_structure([p], "./img/predicate_new" + p['key'] + ".svg")

        return eL([PredMom(p)             for p in ps
                if not isinstance(p['part_predications'], float)])

    def sp_imp_elmo_dictize_ex(self, ex, coref, elmo_embeddings, importance, s_id, dep_tree):
        if not ex:
            logging.warning("No expression given for argument?")
            return {}
        if isinstance(ex, list) and not isinstance(ex[0], spacy.tokens.token.Token ):
            return list (self.sp_imp_elmo_dictize_ex(e, coref, elmo_embeddings, importance, s_id, dep_tree) for e in ex)

        i_s = [x.i for x in ex]
        try:
            d = Argu({
                "id": str(next(self.id_generator)),
                "s_id"            : s_id,
                "i_s"             : i_s,
                'len'             : len(i_s),

                "text"            : [x.text for x in ex],
                "full_ex"         : ex,
                "doc"             : ex[0].doc,

                #"dep"             : [x.dep for x in ex],
                "dep_"            :  [x.dep_ for x in ex],

                #"pos"             : [x.pos for x in ex],
                "pos_"            : [x.pos_ for x in ex],

                #"tag"             : [x.tag for x in ex],
                "tag_"            : [x.tag_ for x in ex],

                #"lemma"           : [x.lemma for x in ex],
                "lemma_"          : [x.lemma_ for x in ex],
                #"lemma_tag_"      : [x.lemma_ + '_' + x.tag_ for x in ex],

                "coref"           : flatten_reduce([coref[i] for i in i_s]),
                "coreferenced"    : self.get_coreferenced_arguments,

                "dep_tree"        : dep_tree,

                "importance"      : importance[i_s],
                "elmo_embeddings" : elmo_embeddings[:,i_s].sum(axis=1),
                "elmo_embeddings_full": elmo_embeddings,
                "importance_full": importance,

                "subj_score"      : sum([self.subjectness_word(t, dep_tree, s_id) for t in ex])/len(ex),
                "aspe_score"      : sum([self.aspectness_word(t, dep_tree, s_id) for t in ex])/len(ex),

                "key"             : "arg" + str(next(self.arg_key_gen))

            })
        except IndexError:
            raise IndexError ('indices out of range of coref')
        return d


    def subjectness_word(self, token, dep_tree, s_id):
        ''' Give score for something being a theme of a certain token.

        If an expression is a noun and is more connected to the ROOT, then its probably the subject.

        :param token:
        :return: score

        '''

        root_dep_dist = [nx.shortest_path_length(dep_tree, source=token.i, target=s.root.i) +50
                     for s in token.doc.sents
                     if nx.has_path(dep_tree, token.i, s.root.i)]

        root_pos_dist = [token.i - s.root.i + 50
                         for s in token.doc.sents
                         if nx.has_path(dep_tree, token.i, s.root.i)]

        s_id_score = 0 # int(s_id)

        res = math.pow(1.0001, -100/(sum(root_dep_dist)+1) - sum(root_pos_dist) - (1000 * s_id_score))
        if res <0:
            raise ValueError
        return res


    def aspectness_word(self, token, dep_tree, s_id):
        ''' Give score for something being a theme of a certain token.

        If an expression is a noun and is more connected to the ROOT, then its probably the subject.

        :param token:
        :return: score

        '''
        root_dep_dist = [nx.shortest_path_length(dep_tree, source=token.i, target=s.root.i) + 50
                     for s in token.doc.sents
                     if nx.has_path(dep_tree, token.i, s.root.i)]

        root_pos_dist = [token.i - s.root.i+50
                     for s in token.doc.sents
                     if nx.has_path(dep_tree, token.i, s.root.i)]

        s_id_score = 0 # int(s_id)

        res = math.pow(1.0001, sum(root_dep_dist) + sum(root_pos_dist)) + (1000 * s_id_score/10)
        if res <0: raise ValueError

        return res
        return res


    def coreferenced_expressions (self, corefs):
        return list(map (self.coreferenced_expression, corefs))
    def coreferenced_expression (self, coref):
        s_id = coref['s_id']
        i_list = coref['i_list']
        mask = self.Predicatrix.predicate_df.query("s_id==@s_id")['i_s'].apply(
            lambda ex_i: True if [m for m in i_list if m in ex_i] else False)
        return self.Predicatrix.predicate_df.query("s_id==@s_id")[mask].to_dict(orient='records')


    def logical_subjunct_predicates (self, ps, coordinator):
        almost_wff = \
             ["%s pyprover.logic.Prop('%s')"
              % ("".join(['~'] * p['negation_count']),
                 p['key'] )
              for p in ps]
        wff = coordinator.join(almost_wff)
        return  wff

    def formalize (self,pred):
        if not "part_predications" in pred:
            logging.error("No 'part_predications'! for %s " % (str(pred['text'])))

        ps = pred["part_predications"]
        ascribed_keys = {p['key']: p for p in ps}


        now_wff_and    = self.logical_subjunct_predicates(ps, "&")
        now_wff_or     = self.logical_subjunct_predicates(ps, "|")

        pred["wff_dict"]  = {key: val  for key,val in ascribed_keys.items()}

        try:
            pred["wff_nice_and"]       = str(pyprover.simplify(eval(now_wff_and)))
            pred["wff_nice_or"]        = str(pyprover.simplify(eval(now_wff_or)))
        except:
            raise SyntaxError("Syntax Error in pyprover formula: %s for expressions %s "% (now_wff_and, str(pred['full_ex'])))

        pred["wff_comp_and"]  = now_wff_and
        pred["wff_comp_or"]   = now_wff_or

        pred ["label"]        = " ".join(pred['text'])
        return pred


    def draw_predicate_structure(self, ps, path):
        import textwrap

        G = ps[0]['containment_structure']

        def wrap(strs, width=40):
            return textwrap.fill(strs, width)

        def dict_to_nice(dic):
            res = "<" + r"""<BR ALIGN="LEFT"/>""".join(
                ["%s: %s" % (str(atrr), wrap(str(", ".join([x.text if hasattr(x, 'text') else str(x) for x in val])))) for atrr, val in dic.items() if val]) + ">"
            return res

        def get_label (n, attrs):
            keys = [p['key'] for p in ps[0]['part_predications']]
            if attrs['predicate']['key'] in keys:
                x = [" ".join(p['text']) for p in  ps[0]['part_predications'] if attrs['predicate']['key'] == p['key']]
                return x[0]
            else:
                return wrap(attrs['label'])


        nx.set_edge_attributes(G, {(u, v): dict_to_nice(attrs) for (u, v, attrs) in G.edges(data=True)}, 'label')
        nx.set_node_attributes(G, {n: get_label(n, attrs) for (n, attrs) in G.nodes(data=True)}, 'label')

        G.graph['graph'] = {
            'rankdir': 'TB',
            'splines': 'line',
            'fontname':'helvetica'
            }

        G.graph['node'] = {
            'fontname': 'helvetica'
        }

        G.graph['edge'] = {
            'fontname':'helvetica'
        }

        A = nx.drawing.nx_agraph.to_agraph(G)

        # different mother predicates in one
        sub_graphs_ids = collect(ps, 'id')
        for sub_pred_id, p in zip(sub_graphs_ids, ps):

            sentence = " ".join(p['text'])
            wff_nice = p['wff_nice_and']
            wff_ugly = p['wff_comp_and']
            title = "\n".join([sentence, wff_ugly, wff_nice])

            nbunch_cr = [n for n, d in G.nodes(data='__subgraph__') if d == sub_pred_id]
            A.add_subgraph(nbunch=nbunch_cr,
                           name="cluster%s" % sub_pred_id,
                           node="[style=filled]",
                           color='blue',
                           labeldistance='300',
                           label=title)

        A.layout('dot')
        A.draw(path)
        return
