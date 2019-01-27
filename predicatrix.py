import re
import pyprover
import itertools
import string
import numpy as np
import collections

from nested_list_tools import *
from digraph_tools import *
from dict_tools import *
from simmix import Simmix
import word_definitions
import pprint
import tdfidf_tool

import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

from allennlp.commands.elmo import ElmoEmbedder
import spacy

raise DeprecationWarning("Better version predicatrix2")

class Predication():
    def __init__(self):
        # big
        # options_file = '/home/bingobongo/PycharmProjects/Sokrates3/elmo_2x4096_512_2048cnn_2xhighway_options.json'
        # weight_file = '/home/bingobongo/PycharmProjects/Sokrates3/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'

        # small
        options_file = './3rdparty/elmo_2x1024_128_2048cnn_1xhighway_options.json'
        weight_file = './3rdparty/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'
        self.elmo = ElmoEmbedder(options_file=options_file, weight_file=weight_file)

        with open('./corpus/aristotle-categories-edghill-spell-checked.lemmas', 'r') as cf:
             corpus = cf.read()

        self.tdfidf = tdfidf_tool.tdfidf(corpus)
        self.nlp = spacy.load('en_core_web_sm')

        self.attributive_arg_markers = ['amod', 'acl', 'prep', 'compound', 'appos']  # genitiv/dativ objects?
        self.verbal_arg_markers = ['nsubjpass', 'nsubj', 'obj', 'dobj', 'iobj', 'pobj', 'attr']
        self.attributive_too_deep_markers = ['punct', 'advcl']
        self.verbal_too_deep_markers = ['punct']

        self.uppercase_abc = list(string.ascii_uppercase)
        self.negation_list = word_definitions.logic_dict['~']
        if not self.negation_list:
            logging.error("No list of Negations given!")
        self.logic_dict    = invert_dict(word_definitions.logic_dict)
        self.logic_dict    = {k: v[0] for (k, v) in self.logic_dict.items()}

        self.id_generator =  Predication.count_up()

        if not self.negation_list:
            raise ValueError("give negation list")
        return None

    def analyse_predication (self, doc=None, id=None):
        """Analyses spacy doc and returns all found predicates"""
        if not doc or id:
            raise KeyError("keyword arguments 'doc' and 'id' must be given")
        predicate = self.collect_all_predicates (doc, id=id)
        return predicate

    def count_up():
        i=0
        while (True):
            yield i
            i +=1


    def get_attributive_roots(self,ex):
        return (e for e in ex if e.dep_ in ['nsubjpass', 'nsubj', 'obj', 'dobj', 'iobj', 'pobj'])

    def get_verbal_roots(self,ex):
        v_roots = tuple(e for e in ex if
                        e.pos_ in ['VERB']
                        and
                        e.dep_ not in ['acl'])

        if not v_roots:
            text = " ".join(x.text for x in ex)
            pos = " ".join(x.tag_ for x in ex)
            dep = " ".join(x.dep_ for x in ex)
            logging.warning (ValueError('Sentence without tag VERB, but dep ROOT, probably AnnotationError, "%s", %s, %s' % (text, pos, dep)))
            v_roots = tuple(e for e in ex if e.dep_ in ['ROOT'])
        return v_roots


    def build_predicate(self,predicate_i, arguments_i, full_ex_i, logicatom_i, doc):
        predicate =  [doc[x] for x in predicate_i]
        arguments = [[doc[x] for x in arg_i] for arg_i in arguments_i]
        full_ex   =  [doc[x] for x in full_ex_i]
        logicatom =  [doc[x] for x in logicatom_i]

        predicate = {"predicate": predicate,
                     "predicate_i": predicate_i,
                        "arguments"    : arguments,
                        "arguments_i"  : arguments_i,
                        "full_ex"     : full_ex,
                        "full_ex_i"    : full_ex_i,
                        "i"        : full_ex_i,
                        "text": [str(x.text) for x in full_ex][:],
                        "logic_atom"  : logicatom,}
        return predicate

    def collect_predicates(self,root_token, arg_markers, too_deep_markers,
                no_zero_args=True):
        predicate_i    = []
        arguments_i    = []
        toodeep_i      = []
        lpar_toodeep_i = []
        if not root_token:
            return None

        for subleaf in root_token.subtree:
            predicate_i += [subleaf.i]
            if subleaf.dep_ in arg_markers:
                argument_i = [s.i for s in subleaf.subtree]
                arguments_i += [argument_i]

            if ((root_token.i != subleaf.i) and subleaf.dep_ in ['conj', 'cc']):
                lpar_toodeep_i += [s.i for s in subleaf.subtree] + [subleaf.i]

            if (root_token.i != subleaf.i) and subleaf.dep_ in too_deep_markers:
                toodeep_i += [s.i for s in subleaf.subtree]

        predicate_i = list(set(predicate_i) - set(flatten(arguments_i[:])) - set(toodeep_i[:]))
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

        doc = root_token.doc

        predicate = self.build_predicate(predicate_i,arguments_i,full_ex_i,logicatom_i,doc)


        return predicate



    p_counter = itertools.count()

    def attribute_contained_predicates(self,ps, expand = False):
        p_new = []

        # needs text  and  i keys for the predicate
        Sub_sim = Simmix([(1, Simmix.sub_i,  0.1, 1),
                          (-1, Simmix.boolean_same_sim, 0,0.1)],
                         n = None)

        if not ps:
            logging.error ("empty expression_list can't contain any predicate.")
            return []
        contain =  Sub_sim.choose ((ps[:], ps[:]), out='i')

        if not contain:
            # If there is no dependent structure, its just one predicate
            for p in ps:
                p['contained']      = p['full_ex']
                p['sub_predicates'] = p['full_ex']
            return ps

        nodes = flatten (contain[:])

        list_of_edges = [list(itertools.product(n1,n2)) for n1, n2 in contain]
        edges = [y for x in list_of_edges for y in x]

        containment_structure = nx.DiGraph()

        containment_structure.add_nodes_from(nodes)
        containment_structure.add_edges_from(edges)

        #plt.clf()
        #nx.draw(containment_structure)
        #plt.savefig(str("./img/containment_graphs/" + str(next(p_counter)) + ".png"))

        root_edges = find_roots(containment_structure)
        #nx.algorithms.dominating_set(containment_structure)

        for r in root_edges:
             to_be_stacked = [ps[r]["full_ex"]] + [ ps[x]["full_ex"] for x in nx.algorithms.descendants(containment_structure, r)]

             if not to_be_stacked:
                 logging.error ("No predicates found, that can be stacked\n%s,\n%s,\n%s\n " %
                                (str(ps),
                                 str(root_edges),
                                 str(nx.algorithms.descendants(containment_structure, r))))
                 continue

             ps[r]["sub_predicates"] = to_be_stacked
             ps[r]["contained"] = stack_matryoshka(to_be_stacked)
             p_new.append( ps[r])

        return p_new

    def collect_all_simple_predicates (self,ex):
        a_roots = list(self.get_attributive_roots(ex))
        v_roots = list(self.get_verbal_roots(ex))
        a_ps = [self.collect_predicates(
            x,
            self.attributive_arg_markers,
            self.attributive_too_deep_markers,
            no_zero_args=True)
            for x in a_roots]
        v_ps = [self.collect_predicates(
            x,
            self.verbal_arg_markers,
            self.verbal_too_deep_markers,
            no_zero_args=True)
            for x in v_roots]
        ps = v_ps + a_ps
        ps = [x for x in ps if x]
        for predicate in ps:
            predicate["dep"] = [ex[x].dep for x in predicate["full_ex_i"]]
            predicate["pos"] = [ex[x].pos for x in predicate["full_ex_i"]]
            predicate["tag"] = [ex[x].tag for x in predicate["full_ex_i"]]
            predicate["text"] = [ex[x].text for x in predicate["full_ex_i"]]
            predicate["lemma"] = [ex[x].lemma for x in predicate["full_ex_i"]]
            predicate["lemma_"] = [ex[x].lemma_ for x in predicate["full_ex_i"]]
            predicate["lemma_tag_"] = [ex[x].lemma_ + '_' + ex[x].tag_ for x in predicate["full_ex_i"]]

        return ps

    def collect_all_predicates(self,ex):
        """ Extracts a multitude of properties from a natural language expression, including grammar
        imformation, word2vec, some importance weight in the document, some kind of cursorily logical
        formula with a dictionary, to what expressions the constants in the formula belong to.

        :param ex:
            string or spracy instance of the language expression
        :return:
            dictionary of properties of the expression
        """

        if not ex:
            return None
        if (isinstance(ex, str)):
            ex = self.nlp(ex)
        if (isinstance(ex, list)):
            def listwrapper (ex):
                for x in ex:
                    yield self.collect_all_predicates(x)
            return sum(listwrapper(ex),[])


        elmo_embeddings = self.elmo.embed_sentence([x.text for x in ex])
        print (ex)
        importance = self.tdfidf.sentence2vec([x.lemma_ for x in ex])

        ps = self.collect_all_simple_predicates(ex)
        if not ps:
            logging.error("no predicates found for %s" % str(ex))
        ps = self.attribute_contained_predicates(ps)
        for p in ps:

            p["importance"]      = importance[p["full_ex_i"]]
            p["elmo_embeddings"] = elmo_embeddings[:,p["full_ex_i"]]
            p["elmo_embeddings_full"] = elmo_embeddings
            p["importance_full"] = importance

            # this adds also p.wff, wff_comp and key_dict by side-effects
            p = self.formalize(p, elmo_embeddings, importance)

        if not ps:
            text = " ".join([x.text for x in ex])
            logging.error("No predication found in expression: '%s'." % text )
        return ps

    def symbolize(self,x):
        if x.lemma_ in self.logic_dict:
            tag = self.logic_dict[x.lemma_]
            if tag != "~":
                pos_rel = x.head.i
                if tag == ">>":
                    pos_rel = x.head.head.i
                tag += '-' if pos_rel < x.i else '+'
            return tag
        else:
            return x.i

    def unique_occurrence(self, replace_iterator = None, words_atom_dict={}):
        """seriell-unique, so semi-unique elements

        Groups, (the x in the statements below) of None or ~ should be sorted ~ to left, the rest to the right.
        """
        def unique_occurrence_ (l):
            """
            0. Bracket out junctors: ['>>-', 'A'] -> ..., '>>-',['>>', 'A']

            1. Sort the groups
            2. replace them in the nested list
            """

            for group_number, e in reversed(list(enumerate(l))):
                if isinstance(e, list):
                    for j, se in enumerate(e):
                        if se in ['&-', '>>-', '|-']:
                            l.insert(group_number, se)
                            continue
                        else:
                            break

            def normal_and_negated_and_sub(key):
                return isinstance(key, int)  or key == '~' or isinstance(key, list)
            def not_subordinated_preds(key):
                return not isinstance(key, list)
            def not_neg(key):
                return not key == '~'
            def not_neg_and_not_list(key):
                return (key != '~') and (not isinstance(key,list))
            def normal_words(key):
                return isinstance(key, int)
            def words_and_subordinated_preds(key):
                return isinstance(key, int) or  isinstance(key, list) or key in ['(', ')'] or key in uppercase_abc

            groups = []
            boolean_replace_flags = []
            for to_replace, g in itertools.groupby(l, normal_and_negated_and_sub):
                groups.append(list(g))
                boolean_replace_flags.append(to_replace)

            replaced_lol = []
            for group_number, to_replace in enumerate(boolean_replace_flags):
                if to_replace:
                    group_to_sort = groups[group_number]

                    replace_tag  = next(replace_iterator)
                    # sort the negations to the left and replace only nonlogical words with some logical variable
                    group_to_sort = sorted(group_to_sort, key=not_subordinated_preds)
                    #sorted_group  = sorted(group_to_sort, key=not_neg)
                    sorted_group  = sorted(group_to_sort, key=not_neg_and_not_list)

                    # If there is a negation inside a nested predication, put it into the brackets (illogically, for sure)
                    for group_number, x in enumerate(sorted_group):
                        if isinstance(x,str) and x == '~':
                            continue
                        #sorted_group.insert(group_number, '(')
                        #sorted_group.append(')')

                        break

                    # Generate the tags in the formula
                    for is_key_to_replace, group_num in itertools.groupby (sorted_group, normal_words):
                        content = list(group_num)
                        if is_key_to_replace:
                            replaced_lol.append(replace_tag)
                            # Generate keys for the fomula
                            # for key_num, group_num in itertools.groupby(sorted_group, normal_words):
                            for is_key_to_replace, group_num in itertools.groupby(sorted_group,
                                                                                  normal_and_negated_and_sub):
                                content = list(group_num)
                                if is_key_to_replace:
                                    content_simplified = flatten(list(content)[:])
                                    # Include
                                    content_simplified = [x if isinstance(x, int)
                                                          else words_atom_dict[x]
                                                          for x in content_simplified
                                                          if x not in ['(', ')', '~', '&+', '|+', '&-', '|-']]

                                    words_atom_dict[replace_tag] = list(set(flatten(content_simplified[:])))

                        if not is_key_to_replace:
                            for x in content:
                                replaced_lol.append(x)


                else:
                    for x in list(groups[group_number]):
                        replaced_lol.append(x)


            return replaced_lol  #, key_dict
        return unique_occurrence_

    def sortneg(self,x):
        def neg(x):
            if x == "~":
                return False
            else:
                return True
        x = sorted(x, key=neg)
        return x


    # Print Functions

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

    def confusing_string_processing(self,almost_wff):
        l2 = almost_wff
        almost_wff = l3 = almost_wff.replace("[", "(").replace("]", ")")
        almost_wff = l4 = almost_wff.replace(",", "")
        almost_wff = l5 = almost_wff.replace("'", "")
        almost_wff = l6 = almost_wff.replace("-", "").replace("+", "")
        almost_wff = l7 = re.sub(r"((^|(?:[\[\(]))+)\s*((>>|\||&|\s)+)", r"\1", almost_wff)
        almost_wff = l8 = re.sub(r"(((>>|\||&)\s?)+)\s*((^|(?:[\]\)]))+)", r"\4", almost_wff)
        almost_wff = l9 = re.sub(r"(\s*((>>)|(\|)|(&))){2,}", r"\1", almost_wff)

        almost_wff   = l10 = re.sub(r"([A-Z])", r"pyprover.\1", almost_wff)


        realistic_exceptional_wff   = r11 = re.sub(r"(pyprover.[A-Z]) \| ~\)", r"\1 | ~ \1)", almost_wff)
             # "wether ...  or not" makes this shape and is a clear ellipsis



        logging.info ("steps to wff: \n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n" %(l2,l3,l4,l5,l6,l7,l8,l9,l10,r11))
        return realistic_exceptional_wff

    def formalize (self,pred, elmo_embeddings, importance):
        p = pred["contained"]

        almost_wff = on_each_level(p[:], self.symbolize, out=[])

        ascribed_keys  = {}
        atom_counter = (x for x in self.uppercase_abc)

        almost_wff = on_each_iterable(
            almost_wff,
            self.unique_occurrence(
                atom_counter,
                words_atom_dict=ascribed_keys),
            )

        # My interpretation of subordinated predicates like attributive and verbal version:
        # (Wisdom is like (a big sea))  --> ((A) & B)
        # Negation before is optional: ((A) ~ B))  --> (A) & ~ B

        def isex_or_in_brackets_or_negation(x):
            return isinstance(x, list) or (isinstance(x, str) and x.isupper()) or (x=="~")
        def isex_or_in_brackets(x):
            return isinstance(x, list) or (isinstance(x, str) and x.isupper())

        # replace list, list ----> list, &, list
        # For a WFF with & and | at subcombinitations of two predicates, think also about mathematical vectors!
        def logical_subjunct_predicates (almost_wff, coordinator):
            pattern = [isex_or_in_brackets, isex_or_in_brackets_or_negation]
            replacement_and = ["\\0", coordinator, "\\1"]
            almost_wff    = l1 = on_each_iterable(almost_wff,  curry(replace_pattern, pattern, replacement_and))
            # now confusing string operations
            almost_wff    = l2  = str(almost_wff)
            return  self.confusing_string_processing(almost_wff)

        now_wff_and    = logical_subjunct_predicates(almost_wff, "&")
        now_wff_or     = logical_subjunct_predicates(almost_wff, "|")

        doc = pred["full_ex"][0].doc
        pred["wff_dict"]  = {key: {"full_ex":[doc[i] for i in val],
                                   "text":[doc[i].text for i in val],
                                   "dep": [doc[i].dep for i in val],
                                   "tag": [doc[i].tag for i in val],
                                   "pos": [doc[i].pos for i in val],
                                   "elmo_embeddings": np.array([elmo_embeddings[:,i] for i in val]),
                                   "importance": np.array ([importance[i] for i in val]),
                                   "key": key,
                             }  for key,val in ascribed_keys.items()}


        try:
            pred["wff_nice_and"]       = str(pyprover.simplify(eval(now_wff_and)))
            pred["wff_nice_or"]        = str(pyprover.simplify(eval(now_wff_or)))
        except:
            raise SyntaxError("Syntax Error in pyprover formula: %s for expressions %s "% (now_wff_and, str(pred['full_ex'])))

        pred["wff_comp_and"]  = now_wff_and
        pred["wff_comp_or"]   = now_wff_or

        pred ["id"]           =  str(next(self.id_generator))
        pred ["label"]        =  " ".join(pred['text'])

        return pred


class TestPredicates(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestPredicates, self).__init__(*args, **kwargs)
        self.gen_dummy_dependent_information()


    def gen_dummy_dependent_information(self):
        # initialized Predication Module
        self.P = Predication()

    def test_all_keys_in_formula(self):
        ex = self.P.nlp("Thus Man is predicable of the individual man and is never present in a subject")
        #ex2 = self.P.nlp("Other things again are both predicable of a subject and present in a subject")
        p = self.P.collect_all_predicates(ex)[0]
        def count_upper_letters (string):
            return sum(1 for c in string if c.isupper())
        print ('\n',p['wff_nice_and'])
        self.P.pprint_key_dict(p)
        print (count_upper_letters(p['wff_nice_and']), len(p['wff_dict']))
        self.assertTrue(count_upper_letters(p['wff_nice_and']) == len(p['wff_dict']))

    def test_attribute_predicate(self):
        ex = self.P.nlp("A coloured man standing behind a garage is wearing shirt .")
        ps = self.P.collect_all_predicates(ex)
        self.P.print_predicates(ps)

        self.assertTrue(ps)

    def test_attribute_predicate_before_behind(self):
        ex = self.P.nlp("A man in a blue shirt standing in front of a garage-like structure painted with geometric designs.")
        ps = self.P.collect_all_predicates(ex)
        self.P.print_predicates(ps)

        self.assertTrue(ps)



if __name__ == '__main__':
    unittest.main()
