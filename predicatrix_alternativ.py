import re
import pyprover
import itertools
import string
import numpy as np

from nested_list_tools import *
from digraph_tools import *
from dict_tools import *
from simmix import Simmix
from word_definitions import logic_dict, negation_list
import pprint

import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

from allennlp.commands.elmo import ElmoEmbedder

#big
#options_file = '/home/bingobongo/PycharmProjects/Sokrates3/elmo_2x4096_512_2048cnn_2xhighway_options.json'
#weight_file = '/home/bingobongo/PycharmProjects/Sokrates3/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'
#small

options_file = './elmo_2x1024_128_2048cnn_1xhighway_options.json'
weight_file =  './elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'
elmo = ElmoEmbedder(options_file=options_file, weight_file=weight_file)

import spacy
nlp = spacy.load('en_core_web_sm')

attributive_arg_markers = ['amod', 'acl', 'prep', 'compound', 'appos']  # genitiv/dativ objects?

verbal_arg_markers = ['nsubjpass', 'nsubj', 'obj', 'dobj', 'iobj', 'pobj', 'attr']

attributive_too_deep_markers = ['punct', 'advcl', *attributive_arg_markers]


verbal_too_deep_markers = ['punct']

uppercase_abc = list(string.ascii_uppercase)



negation_list = logic_dict['~']
logic_dict = invert_dict(logic_dict)
logic_dict = {k:v[0] for (k, v) in logic_dict.items()}
if not negation_list:
    raise ValueError("Dictionary of atrributes and values can't be " + str(attrib_dict))

def get_attributive_roots(ex):
    return (e for e in ex if e.dep_ in ['nsubjpass', 'nsubj', 'obj', 'dobj', 'iobj', 'pobj'])

def get_verbal_roots(ex):
    v_roots = tuple(e for e in ex if e.pos_ in ['VERB'])

    if not v_roots:
        text = " ".join(x.text for x in ex)
        pos = " ".join(x.tag_ for x in ex)
        dep = " ".join(x.dep_ for x in ex)
        logging.warning (ValueError('Sentence without tag VERB, but dep ROOT, probably AnnotationError, "%s", %s, %s' % (text, pos, dep)))
        v_roots = tuple(e for e in ex if e.dep_ in ['ROOT'])
    return v_roots


def build_predicate(predicate_i, arguments_i, full_ex_i, logicatom_i, doc):
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

def collect_predicates(root_token, arg_markers, too_deep_markers,
            no_zero_args=True, invert=False):
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

        if (root_token.i != subleaf.i) and subleaf.dep_ in too_deep_markers and subleaf.head != root_token:
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

    if not invert:
        res = build_predicate(predicate_i,arguments_i,full_ex_i,logicatom_i,doc)

    else:
        predicate_form_arguments =  [x for sub_list in arguments_i for x in sub_list]
        res = build_predicate(predicate_form_arguments,[predicate_i],full_ex_i,logicatom_i,doc)

    return res



p_counter = itertools.count()

def attribute_contained_predicates(ps, expand = False):
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

def collect_all_simple_predicates (ex):
    a_roots = list(get_attributive_roots(ex))
    v_roots = list(get_verbal_roots(ex))
    a_ps = [collect_predicates(
        x,
        attributive_arg_markers,
        attributive_too_deep_markers,
        no_zero_args=True,
        invert=True)
        for x in a_roots]
    #a_ps = list(flatten(a_ps))
    v_ps = [collect_predicates(
        x,
        verbal_arg_markers,
        verbal_too_deep_markers,
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

def collect_all_predicates(ex, tdfidf):
    if not ex:
        return None
    elmo_embeddings = elmo.embed_sentence([x.text for x in ex])
    importance = tdfidf.sentence2vec([x.lemma_ for x in ex])

    ps = collect_all_simple_predicates(ex)
    if not ps:
        logging.error("no predicates found for %s" % str(ex))
    ps = attribute_contained_predicates(ps)
    for p in ps:

        p["importance"]      = importance[p["full_ex_i"]]
        p["elmo_embeddings"] = elmo_embeddings[:,p["full_ex_i"]]
        p["elmo_embeddings_full"] = elmo_embeddings
        p["importance_full"] = importance

        # this adds also p.wff, wff_comp and key_dict by side-effects
        p = formalize(p, elmo_embeddings, importance)

    if not ps:
        text = " ".join([x.text for x in ex])
        logging.error("No predication found in expression: '%s'." % text )
    return ps

def symbolize(x):
    if x.lemma_ in logic_dict:
        tag = logic_dict[x.lemma_]
        if tag != "~":
            pos_rel = x.head.i
            if tag == ">>":
                pos_rel = x.head.head.i
            tag += '-' if pos_rel < x.i else '+'
        return tag
    else:
        return x.i

def unique_occurrence(replace_iterator = None, words_atom_dict={}):
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
                    sorted_group.insert(group_number, '(')
                    sorted_group.append(')')

                    break

                # Generate the tags in the formula
                for is_key_to_replace, group_num in itertools.groupby (sorted_group, normal_words):
                    content = list(group_num)
                    if is_key_to_replace:
                        replaced_lol.append(replace_tag)

                    if not is_key_to_replace:
                        for x in content:
                            replaced_lol.append(x)

                # Generate keys for the fomula
                #for key_num, group_num in itertools.groupby(sorted_group, normal_words):
                for is_key_to_replace, group_num in itertools.groupby(sorted_group, normal_and_negated_and_sub):
                    content = list(group_num)
                    if is_key_to_replace:
                        content_simplified = flatten(list(content)[:])
                        # Include
                        content_simplified = [x if isinstance(x, int)
                                              else words_atom_dict[x]
                                              for x in content_simplified
                                              if x not in ['(', ')', '~', '&+', '|+','&-','|-']]

                        words_atom_dict[replace_tag] = list(set(flatten(content_simplified[:])))

            else:
                for x in list(groups[group_number]):
                    replaced_lol.append(x)


        return replaced_lol  #, key_dict
    return unique_occurrence_

def sortneg(x):
    def neg(x):
        if x == "~":
            return False
        else:
            return True
    x = sorted(x, key=neg)
    return x


# Print Functions

def print_predicates(predicates):
    for p in predicates:
        print_predicate(p)
    return None

def print_predicate(predicate):
    print("Predicate: "+" ".join(predicate['text']))
    pprint.pprint(str(predicate['wff_nice_and']))
    pprint.pprint(str(predicate['wff_nice_or']))
    pprint.pprint(str(predicate['wff_comp_and']))
    pprint.pprint(str(predicate['wff_comp_or']))

    pprint.pprint({k: val['full_ex'] for k, val in predicate['wff_dict'].items()})
    return None


def ps_to_file(fp, ps):
    for p in ps:
        p_to_file(fp, p)
    return None

def p_to_file(fp, predicate):
    try:
        fp.write(str(predicate['text'])+"\n")
        fp.write(str(predicate['wff'])+"\n")
        fp.write(str(({ str(k): str([x.text for x in val['full_ex']]) + "\n" for k, val in predicate['wff_dict'].items()}))+"\n")
    except:
        logging.error("Predicate is a scalar? " + str(predicate))
    return None

def confusing_string_processing(almost_wff):
    l2 = almost_wff
    almost_wff = l3 = almost_wff.replace("[", "(").replace("]", ")")
    almost_wff = l4 = almost_wff.replace(",", "")
    almost_wff = l5 = almost_wff.replace("'", "")
    almost_wff = l6 = almost_wff.replace("-", "").replace("+", "")
    almost_wff = l7 = re.sub(r"((^|(?:[\[\(]))+)\s*((>>|\||&|\s)+)", r"\1", almost_wff)
    almost_wff = l8 = re.sub(r"(((>>|\||&)\s?)+)\s*((^|(?:[\]\)]))+)", r"\4", almost_wff)
    almost_wff = l9 = re.sub(r"(\s*((>>)|(\|)|(&))){2,}", r"\1", almost_wff)
    real_wff   = l10= re.sub(r"([A-Z])", r"pyprover.\1", almost_wff)
    logging.info ("steps to wff: \n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s" %(l2,l3,l4,l5,l6,l7,l8,l9,l10))
    return real_wff

def formalize (pred, elmo_embeddings, importance):
    p = pred["contained"]

    almost_wff = on_each_level(p[:], symbolize, out=[])

    ascribed_keys  = {}
    atom_counter = (x for x in uppercase_abc)

    almost_wff = on_each_iterable(
        almost_wff,
        unique_occurrence(
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
        return  confusing_string_processing(almost_wff)

    now_wff_and    = logical_subjunct_predicates(almost_wff, "&")
    now_wff_or     = logical_subjunct_predicates(almost_wff, "|")

    doc = pred["full_ex"][0].doc
    pred["wff_dict"]  = {key: {"full_ex":[doc[i] for i in val],
                               "text":[doc[i].text for i in val],
                               "elmo_embeddings": np.array([elmo_embeddings[:,i] for i in val]),
                               "importance": np.array ([importance[i] for i in val]),
                               "key": key,
                         }  for key,val in ascribed_keys.items()}

    pred["wff_nice_and"]       = str(pyprover.simplify(eval(now_wff_and)))
    pred["wff_nice_or"]        = str(pyprover.simplify(eval(now_wff_or)))

    pred["wff_comp_and"]  = now_wff_and
    pred["wff_comp_or"]   = now_wff_or
    return pred
