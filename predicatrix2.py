import pandas
import pyprover
import itertools
import string
import pprint
from allennlp.commands.elmo import ElmoEmbedder
import spacy

import logging


logging.getLogger(__name__).addHandler(logging.NullHandler())

from nested_list_tools import *
from digraph_tools import *
from dict_tools import *
from simmix import Simmix
import word_definitions
import tdfidf_tool
from generator_tools import count_up


class Predication():
    def __init__(self, corpus=None):
        ''' This module lets you translate natural language expressions into predicative chunks, to analyse the
        parts of sentence further.

        These chunks are dicts of other properties, that represent a part of the sentence with different properties.
        A predicate for instance, contains

        >>> import spacy
        >>> nlp = spacy.load('en_core_web_sm')
        >>> P = Predication()
        >>> text = nlp ('What is going on here?')
        >>> pred = P.analyse_predication(doc=text, s_id='count_yourself')
        >>> pred[0]['dep_']
        ['nsubj', 'aux', 'ROOT', 'prt', 'advmod', 'punct']
        >>> pred[0]['text']
        ['What', 'is', 'going', 'on', 'here', '?']

        :param corpus: a corpus_reader_object, to overlook the document once, to precompute all lemmata and
        td-idf-values.
        '''

        if corpus:
            lemmatized_text = corpus.lemmatized_text()
            self.tdfidf = tdfidf_tool.tdfidf(lemmatized_text)
        else:
            self.tdfidf = tdfidf_tool.tdfidf("no text")

        options_file = './others_models/elmo_2x1024_128_2048cnn_1xhighway_options.json'
        weight_file = './others_models/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'
        self.elmo = ElmoEmbedder(options_file=options_file, weight_file=weight_file)

        self.attributive_arg_markers       = ['amod', 'acl', 'prep', 'compound', 'appos']  # genitiv/dativ objects?
        self.verbal_arg_markers            = ['nsubjpass', 'nsubj', 'obj', 'dobj', 'iobj', 'pobj', 'attr']
        self.attributive_too_deep_markers  = ['punct', 'advcl']
        self.verbal_too_deep_markers       = ['punct', 'advcl', 'relcl']
        self.ellipsis_too_deep_markers     = ['cc']

        self.uppercase_abc = list(string.ascii_uppercase)
        self.negation_list = word_definitions.logic_dict['~']
        self.logic_dict    = invert_dict(word_definitions.logic_dict)
        self.logic_dict    = {k: v[0] for (k, v) in self.logic_dict.items()}

        self.s_id_generator = count_up()
        self.id_generator   = count_up()
        self.pred_key_gen   = count_up()
        self.arg_key_gen    = count_up()

        if not self.negation_list:
            raise ValueError("give negation list")


        self.Counterpart =  Simmix(
            [   (1, Simmix.dep_sim,  0, 1),
                (1, Simmix.tag_sim,  0, 1),
                (1, Simmix.left_sim, 0, 1),
                (1, Simmix.fuzzystr_sim, 0, 1)
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

        self.predicate_df = pandas.DataFrame()
        self.argument_df = pandas.DataFrame()

        return None

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
            information is not possible to compute
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
        return (e for e in ex if e.dep_ in ['nsubjpass', 'nsubj', 'obj', 'dobj', 'iobj', 'pobj'])

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

    def build_predicate(self,predicate_i, arguments_i, full_ex_i, logicatom_i, doc):
        predicate =  [doc[x] for x in predicate_i]
        arguments = [[doc[x] for x in arg_i] for arg_i in arguments_i]
        full_ex   =  [doc[x] for x in full_ex_i]
        logicatom =  [doc[x] for x in logicatom_i]

        predicate = {"predicate": predicate,
                     "predicate_i"  : predicate_i,
                     "arguments"    : arguments,
                     "arguments_i"  : arguments_i,
                     "full_ex"      : full_ex,
                     "full_ex_i"    : full_ex_i,
                     "i"            : full_ex_i,
                     "doc"          : doc,
                     "text"         : [str(x.text) for x in full_ex][:],
                     "logic_atom"   : logicatom,
                     "key"          : str(next(self.pred_key_gen))
            }
        return predicate

    def collect_predicates(self,root_token, arg_markers, too_deep_markers,  ellipsis_resolution=True,
                no_zero_args=True, mother = False):
        predicate_i    = []
        toodeep_i      = []
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
                    toodeep_i  += counterpart + conjunction

        predicate_i = list(set(predicate_i) - set(toodeep_i[:]))

        if root_token.dep_ == 'conj':
            counterpart1 = set([s.i for s in root_token.head.rights]) - set (toodeep_i[:])

            if root_token.head.dep_ == 'acomp':
                head = root_token.head.head
            else:
                head = root_token.head

            lefts = set([s.i for s in head.lefts])
            children_of_lefts = set([s.i for left in head.lefts for s in left.subtree])
            children_of_lefts.update(lefts)
            elliptical   = children_of_lefts - set (toodeep_i[:])
            predicate_i  = list(elliptical) + predicate_i

        arguments_i    = []
        for j in predicate_i:
            subleaf = root_token.doc[j]
            if subleaf.dep_ in arg_markers:
                argument_i = [s.i for s in subleaf.subtree]
                arguments_i.append(argument_i)

        predicate_i =  set(predicate_i) - set(flatten(arguments_i[:]))
        arguments_i = [list(set(arg_i) - set(toodeep_i[:])) for arg_i in arguments_i]

        if not predicate_i:
            return None
        if ( no_zero_args and (
             not list(flatten(arguments_i[:])))):
            return None

        predicate_i =  sorted(predicate_i)
        arguments_i = [sorted(arg_i) for arg_i in arguments_i if arg_i]
        full_ex_i   =  sorted(set(predicate_i) | set(flatten(arguments_i)))
        logicatom_i =  sorted(list(set(full_ex_i) - set(lpar_toodeep_i)))

        if ellipsis_resolution:
            full_ex_i = sorted(list(set(full_ex_i) - set(lpar_toodeep_i)))
            predicate_i = sorted(list(set(predicate_i) - set(lpar_toodeep_i)))
            arguments_i = [sorted(set(arg_i) - set(lpar_toodeep_i)) for arg_i in arguments_i if arg_i]

        doc = root_token.doc
        predicate = self.build_predicate(predicate_i,arguments_i,full_ex_i,logicatom_i,doc)
        return predicate

    p_counter = itertools.count()

    def attribute_negation_particle (self, p):
        p['negation_particle'] = [x for x in p['full_ex'] if x.text in self.negation_list]
        return None

    def attribute_negation_sentence_predicates (self, ps):
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

                try:
                    common_negation    = list([n1 for n1 in x['negation_particle'] for n2 in y['negation_particle']
                                               if n1==n2])
                except TypeError:
                    logging.error ("Something not iterable:\n%s or %s \n\nfor expression1 %s \nand expression2 %s" %
                                     (str (x['negation_particle']),
                                      str(y['negation_particle']),
                                      str(x['full_ex']),
                                      str(y['full_ex'])))
                    continue
                G.edges[edge]['shared_negations'] =  tuple(common_negation)

            for path in source_sink_generator(s['containment_structure']):
                rath = list(reversed(path))
                for i, n in enumerate(rath[:-1]):
                     edge = G.edges[(rath[i+1], rath[i])]
                     if edge['shared_negations']:# and not edge['used']:
                         negation_containing_node = \
                             rath[i] #if len(s['part_predications'][n]['full_ex']) > len(s['part_predications'][rath[i+1]]['full_ex']) else rath[i+1]
                         G.nodes[negation_containing_node]['predicate']['negation_count'] = len(edge['shared_negations'])
                         break

            # lonly negations
            for node, attrs in G.nodes(data=True):
                 negations = attrs['predicate']['negation_particle']
                 neighbors = G.neighbors(node)
                 negations_of_neighbors = flatten([G.nodes[neighbor]['predicate']['negation_particle'] for neighbor in neighbors])
                 try:
                     if (n in flatten (negations_of_neighbors) for n in negations) or negations and not negations_of_neighbors:
                         negations = attrs['predicate']['negation_particle'] = len(list(neg not in negations_of_neighbors for neg in negations))
                 except  TypeError:
                     logging.error ("negations is int? %s" % (str(negations)))
        return ps

    def attribute_contained_predicates(self,ps, expand = False):
        ps = sorted(ps, key=lambda x:-len(x['full_ex_i']))

        Sub_sim = Simmix([(1, Simmix.sub_i,  0.1, 1),
                          (-1, Simmix.boolean_same_sim, 0,0.1)],
                          n = None)

        if not ps:
            logging.error ("empty expression_list can't contain any predicate.")
            return []
        contain =  Sub_sim.choose ((ps[:], ps[:]), out='2t', layout='n')

        if not contain:
            # If there is no dependent structure, its just one predicate
            for p in ps:
                p['part_predications'] = [p]
            return ps

        nodes = flatten (contain[:])
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
                 sub_predicates                 = [ps[r]] + [ps[x] for x in nx.algorithms.descendants(containment_structure, r)]
                 ps[r]['part_predications']     = sorted(sub_predicates, key=lambda x:len(x['full_ex_i']))
                 ps[r]['containment_structure'] = containment_structure
                 if len(sub_predicates) != len(containment_structure.nodes) - 1:
                     logging.error("graph bigger than predicates!")
                 p_new.append(ps[r])
        return p_new

    def collect_all_simple_predicates (self,ex):
        a_roots = list(self.get_attributive_roots(ex))
        v_roots = list(self.get_verbal_roots(ex))
        m_roots = list(self.get_sentence_roots(ex))

        a_ps = []
        for x in a_roots:
            p = self.collect_predicates(
                    x,
                    self.attributive_arg_markers,
                    self.attributive_too_deep_markers,
                    no_zero_args=True)
            a_ps.append(p)

        v_ps = []
        for x in v_roots:
            p = self.collect_predicates(
                    x,
                    self.verbal_arg_markers,
                    self.verbal_too_deep_markers,
                    no_zero_args=True,
                    ellipsis_resolution=True)
            v_ps.append(p)


        m_ps = []
        for x in m_roots:
            p = self.collect_predicates(
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
            if not any(set(p['full_ex_i']) == set(p2['full_ex_i']) for p2 in ps_new):
                ps_new.append(p)
        ps = ps_new

        for predicate in ps:
            predicate["doc"] = ex[0].doc
            predicate["dep"] = [ex[x].dep for x in predicate["full_ex_i"]]
            predicate["pos"] = [ex[x].pos for x in predicate["full_ex_i"]]
            predicate["tag"] = [ex[x].tag for x in predicate["full_ex_i"]]
            predicate["dep_"] = [ex[x].dep_ for x in predicate["full_ex_i"]]
            predicate["pos_"] = [ex[x].pos_ for x in predicate["full_ex_i"]]
            predicate["tag_"] = [ex[x].tag_ for x in predicate["full_ex_i"]]
            predicate["text"] = [ex[x].text for x in predicate["full_ex_i"]]
            predicate["lemma"] = [ex[x].lemma for x in predicate["full_ex_i"]]
            predicate["lemma_"] = [ex[x].lemma_ for x in predicate["full_ex_i"]]
            predicate["lemma_tag_"] = [ex[x].lemma_ + '_' + ex[x].tag_ for x in predicate["full_ex_i"]]
        return ps

    def collect_all_predicates(self,ex, coref=None, s_id=None,  paint_graph=True):
        ''' Extracts a multitude of properties from a natural language expression, including grammar
        imformation, word2vec, some importance weight in the document, some kind of cursorily logical
        formula with a dictionary, to what expressions the constants in the formula belong to.

        :param ex:
            string or spracy instance of the language expression
        :return:
            dictionary of properties of the expression
        '''
        global nlp

        if not ex:
            return None
        if (isinstance(ex, str)):
            ex = nlp(ex)
        if (isinstance(ex, list)):
            def listwrapper (ex):
                for x in ex:
                    yield self.collect_all_predicates(x)
            return sum(listwrapper(ex),[])


        logging.info ("predicates for '%s'," % str(ex))

        elmo_embeddings = self.elmo.embed_sentence([x.text for x in ex])

        try:
            importance = self.tdfidf.sentence2vec([x.lemma_ for x in ex])
        except AttributeError:
            logging.warning("No td-idf information was precomputed for this expression, filling it up with 0")
            importance = self.tdfidf.half_importance([x.lemma_ for x in ex])

        ps = self.collect_all_simple_predicates(ex)

        if not ps:
            logging.error("no predicates found for %s" % str(ex))

        if not coref:
            coref = [[]]*len (ex[0].doc)

        for p in ps:
            p["id"]                      = str(next(self.id_generator))
            p['s_id']                    = s_id
            p["elmo_embeddings"]         = elmo_embeddings[:,p["full_ex_i"]].sum(axis=1)
            p["coref"]                   = [coref[i] for i in p["full_ex_i"]]
            p["elmo_embeddings_per_word"]= elmo_embeddings[:,p["full_ex_i"]]
            p["elmo_embeddings_full"]    = elmo_embeddings
            p["importance"]              = importance[p["full_ex_i"]]
            p["importance_full"]         = importance
            p['arguments']               = self.sp_imp_elmo_dictize_ex (
                                         p['arguments'],
                                         coref,
                                         elmo_embeddings,
                                         importance,
                                         s_id)

        ps = self.attribute_contained_predicates(ps)
        ps = self.attribute_negation_sentence_predicates(ps)

        for p in ps:
            p = self.formalize(p, elmo_embeddings, importance)
        if not ps:
            text = " ".join([x.text for x in ex])
            logging.error("No predication found in expression: '%s'." % text )

        if paint_graph:
            self.draw_predicate_structure(ps,"./img/predicate chunks " + ps[0]['key']+".svg")

        self.append_to_predicate_df(ps)
        self.append_to_argument_df(ps)
        return ps

    def append_to_predicate_df (self, ps):
        self.predicate_df = self.predicate_df.append(ps)
        self.predicate_df = self.predicate_df.append([part_p for p in ps for part_p in p['part_predications']])


    def append_to_argument_df (self, ps):
        arguments = flatten_reduce( list( map (lambda x: x['arguments'], ps) ))
        self.argument_df = self.argument_df.append(arguments)


    def get_coreferenced_arguments(self, corefs):
        return list (map(self.get_coreferenced_argument, corefs))


    def get_coreferenced_argument(self, coref):
        if not coref:
            return None
        s_id  = coref['s_id']
        m_start = coref['m_start']-1
        m_end   = coref['m_end']-1
        mask = self.argument_df.query("s_id==@s_id").apply(
                lambda ex:
                True
                if (
                        all(m in ex['i_s']
                            for m in range(m_start, m_end))
                        and 'NOUN' in ex['pos_'])
                else False,
            axis=1)

        referenced = self.argument_df.query("s_id==@s_id")[mask].nsmallest(n=1, columns='len').to_dict(orient='records')

        if referenced:
            if referenced == [[[]]]:
                a=1
            return referenced[0]
        else:
            return []

    def sp_imp_elmo_dictize_ex(self, ex, coref, elmo_embeddings, importance, s_id):
        if not ex:
            logging.warning("empty expression for argument?")
            return {}
        if isinstance(ex, list) and not isinstance(ex[0], spacy.tokens.token.Token ):
            return list (self.sp_imp_elmo_dictize_ex(e, coref, elmo_embeddings, importance, s_id) for e in ex)

        i_s = [x.i for x in ex]
        try:
            d = {
                "id": str(next(self.id_generator)),
                "s_id"            : s_id,
                "i_s"             : i_s,
                'len'             : len(i_s),

                "text"            : [x.text for x in ex],
                "full_ex"         : ex,
                "doc"             : ex[0].doc,

                "dep"             : [x.dep for x in ex],
                "dep_"            :  [x.dep_ for x in ex],

                "pos"             : [x.pos for x in ex],
                "pos_"            : [x.pos_ for x in ex],

                "tag"             : [x.tag for x in ex],
                "tag_"            : [x.tag for x in ex],

                "lemma"           : [x.lemma for x in ex],
                "lemma_"          : [x.lemma_ for x in ex],
                "lemma_tag_"      : [x.lemma_ + '_' + x.tag_ for x in ex],

                "coref"           : flatten_reduce([coref[i] for i in i_s]),
                "coreferenced"    : self.get_coreferenced_arguments,

                "importance"      : importance[i_s],
                "elmo_embeddings" : elmo_embeddings[:,i_s].sum(axis=1),

                "key"             : "arg" + str(next(self.arg_key_gen))
            }
        except IndexError:
            raise IndexError ('indices out of range of coref')
        return d


    def coreferenced_expressions (self, corefs):
        return list(map (self.coreferenced_expression, corefs))
    def coreferenced_expression (self, coref):
        s_id = coref['s_id']
        m_start = coref['m_start']
        m_end = coref['m_end']

        mask = self.Predicatrix.predicate_df.query("s_id==@s_id")['full_ex_i'].apply(
            lambda ex_i: True if [m for m in range(m_start, m_end) if m in ex_i] else False)
        return self.Predicatrix.predicate_df.query("s_id==@s_id")[mask].to_dict(orient='records')




    def formalize (self,pred, elmo_embeddings, importance):
        if not "part_predications" in pred:
            logging.error("No 'part_predications'! for %s " % (str(pred['text'])))

        ps = pred["part_predications"]
        ascribed_keys = {p['key']: p for p in ps}

        def logical_subjunct_predicates (ps, coordinator):
            almost_wff = \
                 ["%s pyprover.logic.Prop('%s')"
                  % ("".join(['~'] * p['negation_count']),
                     p['key'] )
                  for p in ps]
            wff = coordinator.join(almost_wff)
            return  wff

        now_wff_and    = logical_subjunct_predicates(ps, "&")
        now_wff_or     = logical_subjunct_predicates(ps, "|")

        pred["wff_dict"]  = {key: val  for key,val in ascribed_keys.items()}

        try:
            pred["wff_nice_and"]       = str(pyprover.simplify(eval(now_wff_and)))
            pred["wff_nice_or"]        = str(pyprover.simplify(eval(now_wff_or)))
        except:
            raise SyntaxError("Syntax Error in pyprover formula: %s for expressions %s "% (now_wff_and, str(pred['full_ex'])))

        pred["wff_comp_and"]  = now_wff_and
        pred["wff_comp_or"]   = now_wff_or

        pred ["id"]           = str(next(self.id_generator))
        pred ["label"]        = " ".join(pred['text'])
        return pred

    def print_predicates(self,predicates, debug = False):
        for p in predicates:
            self.print_predicate(p,debug=debug)
        return None

    def print_predicate(self,predicate, debug = False):
        print("Predicate: "+" ".join(predicate['text']))
        pprint.pprint(str(predicate['wff_nice_and']))
        pprint.pprint(str(predicate['wff_nice_or']))
        self.pprint_key_dict(predicate)
        if (debug):
            pprint.pprint(str(predicate['wff_comp_and']))
            pprint.pprint(str(predicate['wff_comp_or']))
        return None

    def pprint_key_dict (self,predicate):
        pprint.pprint({k: val['full_ex'] for k, val in predicate['wff_dict'].items()})
        return None

    def ps_to_file(self,fp, ps):
        for p in ps:
            self.p_to_file(self,fp, p)
        return None

    def p_to_file(self,fp, predicate):
        try:
            fp.write(str(predicate['text'])+"\n")
            fp.write(str(predicate['wff'])+"\n")
            fp.write(str(({ str(k): str([x.text for x in val['full_ex']]) + "\n" for k, val in predicate['wff_dict'].items()}))+"\n")
        except:
            logging.error("Predicate is a scalar? " + str(predicate))
        return None

    def draw_predicate_structure(self, ps, path):
        import textwrap
        import matplotlib as plt
        from networkx.drawing.nx_agraph import graphviz_layout
        import pylab


        G = nx.algorithms.operators.all.compose_all(p['containment_structure'] for p in ps)
        title = " ".join(" ".join(p['text']) for p in ps)
        wff_nice = " ".join(p['wff_nice_and'] for p in ps)
        wff_ugly = " ".join(p['wff_comp_and'] for p in ps)

        fig = plt.pyplot.gcf()
        fig.set_size_inches(15, 10)

        def wrap (strs, width = 30):
            return textwrap.fill(strs, width)
        def dict_to_nice (dic, width = 20):
            res = "/n".join(["%s: %s" % (str(atrr), wrap(str(val))) for atrr, val in dic.items() if val])
            return res

        node_labels = dict((n,d['label']) for n, d in G.nodes(data=True))
        edge_labels = {(u,v): dict_to_nice(attrs) for (u,v,attrs) in G.edges(data=True)}

        node_labels = {k: wrap(str(v)) for k,v in node_labels.items()}
        edge_labels = {k: wrap(str(v)) for k,v in edge_labels.items()}

        sprectral  =  nx.spectral_layout(G)
        spring     =  nx.spring_layout(G)
        dot_layout = graphviz_layout(G, prog='dot')
        pos = dot_layout

        options = {
            'node_color': 'blue',
            'node_size': 100,
            'width': 3,
            'arrowstyle': '-|>',
            'arrowsize': 12,
        }

        nx.draw_networkx(G,
                         pos=pos,
                         labels=node_labels,
                         with_labels=True,
                         font_size=10,
                         **options)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, rotate=False)

        pylab.title("\n".join([wrap(x,90) for x in [title,wff_nice,wff_ugly]]), pad=20)
        pylab.axis('off')
        pylab.savefig (path, dpi=200)
        pylab.clf()
        return None


class TestPredicates(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestPredicates, self).__init__(*args, **kwargs)
        self.gen_dummy_dependent_information()

    def gen_dummy_dependent_information(self):
        # initialized Predication Module
        self.P = Predication()

    def test_all_keys_in_formula(self):
        ex = self.P.nlp("Thus Man is predicable of the individual man and is not never present in a subject")
        #ex2 = self.P.nlp("Other things again are both predicable of a subject and present in a subject")
        p = self.P.collect_all_predicates(ex)[0]
        def count_upper_letters (string):
            return sum(1 for c in string if c.isnumeric())
        print ('\n',p['wff_nice_and'])
        self.P.pprint_key_dict(p)
        print (count_upper_letters(p['wff_nice_and']), len(p['wff_dict']))
        self.assertTrue(count_upper_letters(p['wff_nice_and']) == len(p['wff_dict']))

    def test_attribute_predicate(self):
        ex = self.P.nlp("A coloured man standing behind a garage is wearing shirt .")
        ps = self.P.collect_all_predicates(ex)
        self.P.print_predicates(ps)
        self.assertTrue(ps)

    def test_coin(self):
        standard_ex = self.P.nlp("The thing is round from the right side.")

        #standard_ex = self.P.nlp(
        #               "Things are named Derivatively which derive their name from some other name but differ from it in termination")

        ps = self.P.collect_all_predicates(standard_ex)
        self.P.print_predicates(ps)
        self.assertTrue(ps)

    def test_attribute_predicate_before_behind(self):
        ex = self.P.nlp("A man in a blue shirt standing in front of a garage-like structure painted with geometric designs.")
        ps = self.P.collect_all_predicates(ex)
        self.P.print_predicates(ps)
        self.assertTrue(ps)


    def test_num_of_predicates(self):
        exs = self.P.load_conll([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,16], './corpus/import_conll')
        print (exs)
        for i, ex in enumerate(exs):
            ps = self.P.collect_all_predicates(ex)
            self.P.print_predicates(ps)
            self.assertTrue(len(ps)==1)
            self.P.draw_predicate_structure(ps, "predicate chunks %d.svg" % i)

    def test_attribute_predicate_before_behind(self):
        ex = self.P.nlp(
            "Lying, Sitting, are terms indicating position, Shod, Armed, state, Tolance, Tocauterize, action, Tobelanced, Tobecauterized, affection.")
        ps = self.P.collect_all_predicates([ex])
        self.P.print_predicates(ps)

        self.assertTrue(ps)

    def test_negations(self):
        ex = self.P.nlp("Of things themselves some are predicable of a subject, and are never present in a subject.")
        ex = self.P.nlp("Some things, again, are present in a subject, but are never predicable of a subject.")
        ex = self.P.nlp("There is, lastly, a class of things which are neither present in a subject nor predicable of a subject.")
        ex = self.P.nlp("A body, being white, is said to be whiter at one time than it was before, or, being warm, is said to be warmer or less warm than at some other time.")


        ps = self.P.collect_all_predicates(ex)
        self.P.print_predicates(ps)
        self.P.draw_predicate_structure(ps, "predicate.png")

        """
        exs = [
            nlp("Some things, again, are present in a subject, but are never predicable of a subject."),
            nlp("Other things, again, are both predicable of a subject and present in a subject."),
            nlp("There is, lastly, a class of things which are neither present in a subject nor predicable of a subject."),
            
            nlp(
                "On the other hand things are said to be named Univocally, which have the name and the definition answering to the name in common"),
            nlp("Things are named Derivatively which derive their name from some other name but differ from it in termination"),
            nlp(
                "SECTION 1 Part 1 Things are said to be named Equivocally when though they have a common name the definition corresponding with the name differs for each"),
        ]
        """

    def test_negation(self):
        ex = self.P.nlp(
            "Of things themselves some are not predicable of a subject, and are never present in a subject.")

        res_sub_predicates = ["Of things themselves some are not predicable of a subject",
                              "Of things themselves some are never present in a subject",
                              "Of things themselves some are not predicable of a subject and are never present in a subject."]




        ps = self.P.collect_all_predicates(ex)
        self.P.print_predicates(ps)


        self.assertTrue(ps[0]['wff_comp_and'].count('~') == 2)
        self.assertTrue(ps[0]['wff_comp_and'].count('pyprover') >= 3 )

        res = []
        for res_p in res_sub_predicates:
            t_or_f = any (all(r in s['text'] for r in res_p.split()) for s in ps[0]['part_predications'])
            print ("Checking for %s in sub_predicates --> %s" % (res_p, str(t_or_f)))
            res.append(t_or_f)
        assert(all(res))



if __name__ == "__main__":
    import doctest
    doctest.testmod()
