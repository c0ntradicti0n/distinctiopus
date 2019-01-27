#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This module is intended for overlapping an open-end variety of similarty measures on complex data in
natural language processing. Its like an organ with different registers for text-comparisons.

Choices are made from two lists of dict elements, that are representing some phrases. They are bundles
of properties, as their text, grammar, also a logical formula to represent their logical structure (pyprover), semantical vectors (and syntactical, like elmo-embeddings from allen-nlp) , tdf-idf-importance and some more. These are predefined here, if you know some of your own,its possible to give custom functions.

p1 =
  {
  'text': ['The', 'cause', 'of', 'cancer', 'is', 'not', 'a', 'virus', 'and', 'the', 'cause', 'of', 'cancer', 'is', 'not', 'a', 'fungus'],
  'lemma_': ['the', 'because', 'of', 'cancer', 'be', 'not', 'a', 'virus', 'and', 'the', 'because', 'of', 'cancer', 'be', 'not', 'a', 'fungus'],
  'dep',
  'importance': array([0.53395703, ...  0.2       ])
  A ∧ ¬C ∧ B ∧ ¬D
((((pyprover.A )) & ~ pyprover.C ) & (((pyprover.B )) & ~ pyprover.D ))
  }

p2 = ...
q1 = ...
q2 = ...
ps = [p1,p2,p3 ...]
qs = [q1,q2,q3 ...]

Two list of these elements have to be provided for the Simmix.choose ((p1,p2))-function.





Also there are functions in Simmix to compare these and different sorts of how choices can be made.

The functions and parameters for computing the overall-similarity,, that should be used in a complex comparison have to
be initialized like this:

 superficial_sim  = \
    Simmix([(1, Simmix.fuzzystr_sim, 0.72, 2.01), # fuzzy-string-comparison
            (1, Simmix.dep_sim, 0.4, 1)       # levensthein distance of the dependency measures
            ])

The list of tuple parameter is represents a pipeline, what should be computed:
1                     -- The weight of their normalized results.
Simmix.fuzzystr_sim   -- which function
0.72                  -- minimum-score
2.01                  -- maximum-score



There are also different kinds, how choices can be made. Simmix.choose(data) takes optional arguments.

def choose (self, data, out=None, layout=None, n=None):

Default behavior: the best and only one element.

If n is set, the best first n results are chosen.

If out is set to "i", you get the indices of your input-bundles data for the made choice.

If layout is set to "1:1:, for each input-bundle in the lists of the data exactly one element is taken from both lists, in the realm of the given thresholds and the number of items given.


"""
import types
import scipy
import numpy as np
import numpy_indexed as npi
from sklearn import preprocessing
import itertools
import collections
import pyprover

import string
import re
from pyxdameraulevenshtein import damerau_levenshtein_distance
import logging
from nested_list_tools import check_for_tuple_in_list, flatten,  flatten_reduce
import dict_tools



def invert_dict(d):
    inverse = dict()
    for key in d:
        # Go through the list that is saved in the dict:
        for item in d[key]:
            # Check if in the inverted dict the key exists
            if item not in inverse:
                # If not create a new list
                inverse[item] = [key]
            else:
                inverse[item].append(key)
    return inverse


uppercase_abc = list(string.ascii_uppercase)
uppercase_bca = list(string.ascii_uppercase)[::-1]

# https://stackoverflow.com/questions/11144513/numpy-cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
# Generalized N-dimensional products
def cartesian_product_itertools(arrays):
    return np.array(list(itertools.product(*arrays)))



class Simmix:
    """This module contains utils to overlap different similarity measuring functions. 
    """

    def __init__ (self, similarity_composition, n=1, elmo=None, type=None):
        funs           = [np.nan] * len(similarity_composition) 
        weights        = [np.nan] * len(similarity_composition) 
        thresholds_min = [np.nan] * len(similarity_composition) 
        thresholds_max = [np.nan] * len(similarity_composition)

        for i, composition_tuple in enumerate(similarity_composition):
            if len(composition_tuple) == 2:
                (weight, fun) = composition_tuple
                funs[i]       = fun
                weights[i]    = weight
            elif len(composition_tuple) == 4:
                (weight, fun,  threshold_min, threshold_max) = composition_tuple
                funs[i]           = fun
                weights[i]        = weight
                thresholds_min[i] = threshold_min
                thresholds_max[i] = threshold_max

            else:
                raise ValueError("len of tuples must be 2, (weight, fun) or 3 (weight, fun, threshold_min, threshold_max)" +
                                 str(similarity_composition))
        self.funs           = np.array(funs)
        self.weights        = np.array(weights)
        self.thresholds_min = np.array(thresholds_min)
        self.thresholds_max = np.array(thresholds_max)
        self.n              = n
        self.elmo           = elmo
        self.beam           = {}
        self.type           = type

        try:
            mins = [f.min for f in funs]
            maxs = [f.max for f in funs]
            self.minmax_defaults = np.array([mins,maxs])
        except AttributeError:
            raise AttributeError ("At least one of the funs doesn't have standard_range decorator! %s " % (str (funs)))

        self.scaler = preprocessing.MinMaxScaler()

    def standard_range (minim, maxim):
        def wrapper(f):
            f.min = minim
            f.max = maxim
            return f
        return wrapper

    def join_choices_ (list_of_funs):
        def collate_choices(data):
            return list(itertools.chain.from_iterable([fun(data) for fun in list_of_funs]))
        return collate_choices

    def apply_sim_fun (self, fee):
        fun = fee[2]
        ex1 = fee[0]
        ex2 = fee[1]
        try:
            res, d = fun(ex1,ex2)
            dict_tools.update (self.beam, d)
        except TypeError:
            raise TypeError ("Function doesn't return beam. %s, %s " % (str(fun(ex1,ex2)), str(fun)))
        return res

    def choose(self, data, out=None, layout=None, n=None, G=None, type=None, output=False,
               minimize=False):
        """Does the choice in different manners.
/
        :param data:
            tuple of lists of expressions, meaning the dictionaries or what is compared for you
        :param out:
            defines the structure of the returned value
            default=None: the chosed expressions are returned
            'i': only the indices
            'r': only the indices of the right value are returned
            '2t': pairs of tuples (for building a graph
            'nx': directed networkxgraph
        :param G:
            Graph instance for out=='nx'
        :param layout:
            default=None behavior like n=1
            '1:1' means for each one element of the one expression is eactly one chosen from the other
            'n:m' for samples of n elements are fitting m other elements chosen from the other expression
        :param n:
            default=1
            count of best n solutions, also clusters of expressions of the one are chosen to fit to a cluster of the other
        :return:
            list of tuples of lists of ints or indexed expressions, according to the chosen 'out' parameter
        """

        exs1 = None
        exs2 = None
        self.beam = {}
        if n:
            self.n = n

        if isinstance(data, list) or isinstance(data, types.GeneratorType):
            tuples = []
            for tup in data:
                tuples += self.choose(tup, out=out, layout=layout)
            return tuples

        elif isinstance(data, tuple):
            exs1, exs2 = (data[0], data[1])

            # expressions1 * expressions2 * funs
        sim_cube = cartesian_product_itertools([
            exs1,
            exs2,
            self.funs
        ])

        # apply funs   res_cube.reshape(len(exs1), len(exs2), -1)
        try:
            res_cube = np.apply_along_axis(self.apply_sim_fun, 1, sim_cube)
            res_cube = res_cube.reshape(len(exs1), len(exs2), -1)
        except np.core._internal.AxisError as e:
            raise ValueError('expression is empty %s' % (str((exs1, exs2))))

        # get cubic shape back and floating points for the numbers
        res_cube = res_cube.reshape((-1, len(self.funs)))
        res_cube = res_cube.astype(np.float64)

        # add some minmax_vectors to have comparable scaling for normalization
        # but first, check, if these values
        exceeds_min = res_cube < self.minmax_defaults[0]
        exceeds_max = res_cube > self.minmax_defaults[1]
        if exceeds_min.any() or exceeds_max.any():
            logger = logging.getLogger(__name__)
            exceeds_range_msg = \
                ("Similarity results exceed minmax defaults, required for constant scaling in normalization."
                 + "\nfuns=" + str(self.funs)
                 + "\nminmax_defaults=" + np.array2string(self.minmax_defaults,
                                                          formatter={'float_kind': lambda x: "%.2f" % x})
                 + "\ncube=\n" + np.array2string(res_cube, formatter={'float_kind': lambda x: "%.2f" % x})
                 + "\nmin on mask \n" + str(exceeds_min)
                 + "\nmax on mask \n" + str(exceeds_max))
            logger.error(exceeds_range_msg)
            print(exceeds_range_msg)
            # raise ValueError("Similarity results exceed minmax defaults, required for constant scaling in normalization. Info printet above.")

        res_cube = np.append(res_cube, self.minmax_defaults, axis=0)
        # scale
        trans_cube = self.scaler.fit_transform(res_cube)
        # reshape matrix
        trans_cube = trans_cube[:trans_cube.shape[0] - 2, :]
        if output == True:
            print(trans_cube)

        # apply weights: matrix dot vector = summed up vector with one value for each vector
        weighted_res = trans_cube.dot(self.weights)  # Matrix times weight
        weighted = weighted_res.reshape(len(exs1), len(exs2))
        # apply filters
        # they are not boolean mask, because the max_n function returns only indices
        # if we got some nan
        is_possible = np.where([~np.isnan(weighted_res)])[1]

        # within epsilon intervall: [min ... max]
        if (~np.isnan(self.thresholds_min)).all():
            is_above_lower_boundary = np.where((trans_cube >= self.thresholds_min).all(axis=1))
            is_possible = np.intersect1d(is_above_lower_boundary, is_possible)

        if (~np.isnan(self.thresholds_max)).all():
            is_under_upper_boundary = np.where((trans_cube <= self.thresholds_max).all(axis=1))
            is_possible = np.intersect1d(is_under_upper_boundary, is_possible)

        if (self.n != None and weighted_res.size > self.n):
            if minimize:
                weighted_res = - weighted_res
            is_biggest_n = np.argpartition(weighted_res, -self.n)[-self.n:]
            # and also nonzero!
            best_n = np.intersect1d(is_biggest_n, is_possible)
        else:
            best_n = is_possible

        left_value = None
        right_values = None
        if layout == "1:1":
            weighted_res = weighted_res.reshape((len(exs1), len(exs2)))

            # check which values are within the realm of the thresholds
            higher = np.full(weighted_res.shape, False)  # building some boolean mask
            lower = np.full(weighted_res.shape, False)
            if (~np.isnan(
                    self.thresholds_min)).all():  # lower and higher catch are true, if it's within, if all is false, there can be no result
                lower = trans_cube >= self.thresholds_min
            if (~np.isnan(self.thresholds_max)).all():
                higher = trans_cube <= self.thresholds_max

            # higher and lower are masks on the normalized 3-d matrix
            # weighted_res is the summed up and weighted 2 d matrix

            is_within_thresholds = (higher & lower).all(axis=1)
            is_within_thresholds = is_within_thresholds.reshape((len(exs1), len(exs2)))

            if not is_within_thresholds.any():
                return []

            if weighted_res.shape != is_within_thresholds.shape:
                print(higher, lower)
                raise ValueError(
                    "Some mismatch between the shapes of the weighted result and the thresholds. shape= %s, \nexs1 = %s, \nexs2 = %s" % (
                    str((weighted_res.shape, is_within_thresholds.shape)), str([x['text'] for x in exs1]),
                    str([x['text'] for x in exs2])))

            mask = is_within_thresholds
            left_value, right_values = Simmix.one_to_one(weighted_res, exs1, exs2, mask)

        elif layout == "(x<=m):(y<=n)" or layout == None:
            chords = np.divmod(best_n, len(exs2))
            chords = np.column_stack(chords)
            left_value = np.unique(chords[:, 0])  # unique values, centers of different
            # clusters
            right_values = npi.group_by(chords[:, 0]).split(chords[:, 1])  # groupby the other values, that form
            # the clusters and concat all these
            # ex1 is comparable to all these

        if not G == None:
            l_s = Simmix.expressions_list(left_value, right_values, exs1, exs2)
            t_s = Simmix.reduce_i_s_pair_tuples(l_s)

            for (ex1, ex2) in t_s:
                if not G.has_node(ex1['id']):
                    G.add_node(ex1['id'], **ex1)
                if not G.has_node(ex2['id']):
                    G.add_node(ex2['id'], **ex2)
                # print (self.beam[ex1['id']][ ex2['id']])
                G.add_edge(ex1['id'], ex2['id'], **self.beam[ex1['id']][ex2['id']], type=type)

        if not out or out == 'ex':
            return Simmix.expressions_list(left_value, right_values, exs1, exs2)
        elif out == 'i':
            return Simmix.i_list(left_value, right_values)
        elif out == 'r':
            i_s = Simmix.i_list(left_value, right_values)
            return Simmix.reduce_i_s_right_values(i_s)
        elif out == '2t':
            i_s = Simmix.i_list(left_value, right_values)
            return Simmix.reduce_i_s_pair_tuples(i_s)
        elif out == 'nx':
            if G == None:
                raise ValueError("parameter G is required for use with Graph!")
            return G
        elif out == '(i,ex)':
            return (Simmix.i_list(left_value, right_values),
                    Simmix.expressions_list(left_value, right_values, exs1, exs2))
        else:
            raise NotImplementedError("What the else could out be returned? Wrong parameter for 'out'")

    def one_to_one(weighted_res, exs1, exs2, mask):
        """
        compute combinations of input-lists with a solution connecting one to one expression.
        Solved with the hungarian method, filtered by the mask of acceptable solutions within the allowed range of values.

        :param exs1:  list of comparaed values  in x direction
        :param exs2:  list of compared values   in y direction
        :paramm mask: np boolean mask of fine solutions
        :return: tuple of list of indices to the exs1 list and of indices of the exs2 list
        """
        res = scipy.optimize.linear_sum_assignment(-weighted_res)
        res_mask = mask[res]
        l_values = res[0][res_mask].tolist()
        r_values = [[r] for r in res[1][res_mask].tolist()]
        return l_values, r_values

    def expressions_list (left_value, right_values, exs1, exs2):
        return [([exs1[left_value[l]]], [exs2[r] for r in right_values[l]])#, trans_cube[n[0]])
                  for l in range(len(left_value))]
    def i_list (left_value, right_values):
        return [([left_value[l]], [r for r in right_values[l]])#, trans_cube[n[0]])
                  for l in range(len(left_value))]
    def reduce_i_s_right_values(x):
        if not x:
            return []
        else:
            return list(flatten(r for l, r in x if x))
    def reduce_i_s_pair_tuples (x):
        if not x:
            return []
        else:
            return list((y,z) for l, r in x if x for z in r for y in l)

    @standard_range(0, 2)
    def print_exs (ex1, ex2):
        print ("ex1", ex1, type(ex1))
        print ("ex2", ex2, type(ex2))
        print ([x.lemma_ for x in ex1])
        print ([x.lemma_ for x in ex2])
        return 1, {}
    @standard_range(-1, 0)
    def pos_sim (ex1, ex2):
        grammar1  = ex1['pos']
        grammar2  = ex2['pos']
        return -damerau_levenshtein_distance(grammar1, grammar2) /(len(ex1) + len(ex2)), {}
    @standard_range(-1, 0)
    def dep_sim (ex1, ex2):
        grammar1  = ex1["dep"]
        grammar2  = ex2["dep"]
        return -damerau_levenshtein_distance(grammar1, grammar2) / (len(ex1) + len(ex2)) , {}
    @standard_range(-1, 0)
    def lemma_sim (ex1, ex2):
        grammar1  = ex1["lemma"]
        grammar2  = ex2["lemma"]
        return -damerau_levenshtein_distance(grammar1, grammar2) / (len(ex1) + len(ex2)), {}
    @standard_range(-1, 0)
    def tag_sim (ex1, ex2):
        grammar1  = ex1["tag"]
        grammar2  = ex2["tag"]
        return -damerau_levenshtein_distance(grammar1, grammar2) / (len(ex1) + len(ex2)) , {}
    @standard_range(-1, 0)
    def fuzzystr_sim (ex1, ex2):
        str1  = ex1["text"]
        str2  = ex2["text"]
        return -damerau_levenshtein_distance(str1, str2) / (len(ex1) + len(ex2)) , {}
    @standard_range(-0.6, 0.6)
    def common_words_sim (ex1, ex2):
        str1  = ex1["lemma"]
        str2  = ex2["lemma"]
        res = (len([ex1['importance'][i] for i,x in enumerate(str1) if x in str2]) +
               len([ex2['importance'][i] for i, x in enumerate(str2) if x in str1]) -
               len([ex1['importance'][i] for i, x in enumerate(str1) if x not in str2]) -
               len([ex2['importance'][i] for i, x in enumerate(str2) if x in str1]))/(len(ex1) + len(ex2))
        print (res)
        return res, {}
    @standard_range(-1000, 1000)
    def vecs_sim (gensim_model):
        def vecs_sim_(ex1,ex2):
            lemmata1  = [x.lemma_ for x in ex1 if x.lemma_ in gensim_model.wv.vocab]
            lemmata2 =  [x.lemma_ for x in ex2  if x.lemma_ in gensim_model.wv.vocab]
            if lemmata1 and lemmata2:
                return gensim_model.wv.n_similarity(lemmata1, lemmata2) * 10000
            else:
                nlemmata1 = [x.lemma_ for x in ex1 if x.lemma_ not in gensim_model.wv.vocab]
                nlemmata2 = [x.lemma_ for x in ex2 if x.lemma_ not in gensim_model.wv.vocab]
                logger = logging.getLogger(__name__)
                logger.error ("gensim meager, words are missing: " + str(nlemmata1) + str(nlemmata2) )
                return 0
        return vecs_sim_
    def elmo_sim():
        @Simmix.standard_range(-6, 0.5)
        def elmo_sim_generated (ex1,ex2):
            vectors1 = ex1["elmo_embeddings"]
            vectors2 = ex2["elmo_embeddings"]
            total_length = max(len(ex1["full_ex"]), len (ex2["full_ex"]))
            try:
                return \
                    -(np.log( (total_length/np.e+np.e)) *
                     scipy.spatial.distance.cosine(vectors1[0,:],vectors2[0,:])+
                     scipy.spatial.distance.cosine(vectors1[1,:],vectors2[1,:])+
                     scipy.spatial.distance.cosine(vectors1[2,:],vectors2[2,:])), {}
            except IndexError:
                raise IndexError
        return elmo_sim_generated
    def elmo_multi_sim(n=3):
        @Simmix.standard_range(-6*n, 0.5*n)
        def elmo_sim_generated (exs1,exs2):
            sim = 0
            for ex1, ex2 in itertools.product(flatten_reduce(exs1), flatten_reduce(exs2)):
                vectors1 = ex1["elmo_embeddings"]
                vectors2 = ex2["elmo_embeddings"]
                total_length = max(len(ex1["full_ex"]), len (ex2["full_ex"]))

                sim = sim + \
                    -(np.log( (total_length/np.e+np.e)) *
                      scipy.spatial.distance.cosine(vectors1[0,:],vectors2[0,:])+
                      scipy.spatial.distance.cosine(vectors1[1,:],vectors2[1,:])+
                      scipy.spatial.distance.cosine(vectors1[2,:],vectors2[2,:])
                      )
            return sim, {}
        return elmo_sim_generated
    def elmo_layer_sim(layer = [0,1,2]):
        @Simmix.standard_range(-10/len(layer), 0.5/len(layer))
        def elmo_sim_generated (ex1,ex2):
            vectors1 = ex1["elmo_embeddings"]
            vectors2 = ex2["elmo_embeddings"]
            total_length = max(len(ex1["full_ex"]), len (ex2["full_ex"]))
            try:
                return \
                    -( np.log( (total_length/np.e+np.e))
                       * sum(
                          scipy.spatial.distance.cosine(vectors1[l,:],vectors2[l,:])
                          for l in layer)), {}
            except IndexError:
                raise IndexError
        return elmo_sim_generated
    @standard_range(-20, 20)
    def elmo_weighted_sim():
        def elmo_sim_generated (ex1,ex2):
            total_length = max(len(ex1["full_ex"]), len (ex2["full_ex"]))
            vector1  = ex1["elmo_embeddings_per_word"]
            weights1 = ex1["importance"]
            vector2  = ex2["elmo_embeddings_per_word"]
            weights2 = ex2["importance"]
            try:
                return \
                    -(np.log( (total_length/np.e+np.e)) *
                     (#scipy.spatial.distance.cosine( vector1[0,:,:].sum(axis=0), vector2[0,:,:].sum(axis=0))+
                      #scipy.spatial.distance.cosine( vector1[1,:,:].sum(axis=0), vector2[1,:,:].sum(axis=0))+
                      scipy.spatial.distance.cosine( weights1.dot(vector1[2,:,:]).sum(axis=0), weights2.dot(vector2[2,:,:]).sum(axis=0))
                            )), {}
            except IndexError:
                raise IndexError
        return elmo_sim_generated
    @standard_range(0, 1)
    def boolean_same_sim (ex1, ex2):
        text1        = " ".join(ex1["text"])
        text2        = " ".join(ex2["text"])
        return text2 == text1, {}
    @standard_range(0, 1)
    def boolean_subsame_sim (ex1, ex2):
        text1        = " ".join(ex1["text"])
        text2        = " ".join(ex2["text"])
        return (text2 in text1) or (text1 in text2), {}
    @standard_range(0, 1)
    def sub_i (ex1, ex2):
        is1 = set(ex1["i"])
        is2 = set(ex2["i"])
        try:
            return is2.issubset(is1) and ex1['doc'] == ex2['doc'], {}
        except KeyError:
            logging.warning ("Why is there no 'doc'-key? Proceed... ")
            return is2.issubset(is1), {}
    @standard_range(-100,0)
    def longer_sim (ex1, ex2):
        is1 = set(ex1["i"])
        is2 = set(ex2["i"])
        return -len (is1)-len(is2), {}
    def multi_sim(fun=None, n=4):
        b = {}
        if not fun:
            raise ValueError("fun must be given")
        @Simmix.standard_range(fun.min*n,fun.max*n)  # depends on fun!
        def multi_generated (exs1,exs2):
            sim = 0
            for ex1, ex2 in itertools.product(flatten_reduce(exs1), flatten_reduce(exs2)):
                res, d = fun (ex1, ex2)
                sim += res
                b.update(d)
            return sim, b
        return multi_generated
    @standard_range(-1000, 100)
    def nextleft_sim (ex1, ex2):
        minL1  = min(ex1["i"])
        maxL2  = max(ex2["i"])
        return -minL1 - maxL2, {}
    @standard_range(-1000, 100)
    def nextright_sim (ex1, ex2):
        maxR1  = max(ex1["i"])
        minL2  = min(ex2["i"])
        return -maxR1 - minL2, {}
    def boolean_sim (attrib_dict):
        if not attrib_dict:
            raise ValueError("Dictionary of atrributes and values can't be " + str(attrib_dict))
        @Simmix.standard_range(0, 10)
        def boolean_sim_(ex1,ex2):
            cost = 0
            for attr, single_attrib_dict in attrib_dict.items():
                for key, item in single_attrib_dict.items():
                    if key in [getattr(x,attr) for x in ex1]:
                        that_key_cost =  int(item in [getattr(x,attr) for x in ex2])
                        if isinstance(item,list):
                            for it in item:
                                that_key_cost +=  int(it in [getattr(x,attr) for x in ex2])
                        cost += that_key_cost
            return cost, {}
        return boolean_sim_
    def excluding_pair_boolean_sim (attrib_dict):
        """ The defined pair of values mustn't occur in one of the statements at once and in the other. """
        if not attrib_dict:
            raise ValueError("Dictionary of atrributes and values can't be " + str(attrib_dict))
        @Simmix.standard_range(0, 4)
        def excluding_pair_boolean_sim_generated(ex1,ex2):
            cost = 0
            ex1 = ex1["full_ex"]
            ex2 = ex2["full_ex"]
            def match (needle, stack):
                val  = (needle in stack or
                        (isinstance(needle, tuple) and check_for_tuple_in_list(stack,needle)))
                return val

            for attr, single_attrib_dict in attrib_dict.items():
                comp_elements1 = [getattr(x, attr) for x in ex1]

                for key, item in single_attrib_dict.items():
                    if (match(key, comp_elements1)):
                            comp_elements2 = [getattr(x,attr) for x in ex2]
                            that_key_cost =  int(item in comp_elements2)
                            if isinstance(item,list):
                                for it in item:
                                    if (match(it,comp_elements2)
                                       and not  match(it,  comp_elements1)    # decisive point, that the antonympair should not itself appear in the one or other phrase
                                       and not  match(key, comp_elements2)
                                       ):
                                        actual_cost = int(it in comp_elements2
                                                             or
                                                             isinstance(it, tuple) and check_for_tuple_in_list(
                                                             comp_elements2, it))
                                        that_key_cost += actual_cost
                                        if actual_cost:
                                            logging.info('pair of antonyms found -- "%s" and "%s"' % (key, it))


                            cost += that_key_cost


            return cost, {}
        return excluding_pair_boolean_sim_generated
    def formula_prooves(fit_mix):
        import pyprover
        @Simmix.standard_range(0, 4)
        def formula_prooves_generated(ex1, ex2):
            """
            :param ex1:
            :param ex2:
                    pyproover formulas with "pyproover." prefix for proposition "A"

            :return:
                    1 True for both, 0.7 for one direction of contradicting, 0 for nothing
            """
            f1 = ex1["wff_comp_and"][:]
            f2 = ex2["wff_comp_and"][:]
            keys1 = list(ex1["wff_dict"].values())
            keys2 = list(ex2["wff_dict"].values())

            key_to_key = fit_mix.choose((keys1, keys2), out="ex", layout="1:1")

            if None in flatten(key_to_key[:]) or not key_to_key:
                logging.debug("No antonyms found!")
                return 0, {}

            key_char_to_key_char = {k1[0]["key"]: k2[0]["key"] for k1, k2 in key_to_key}

            cost = 0
            triggers = []

            for k1, k2 in key_to_key:
                new_cost = Simmix.do_logic(
                    formulas = (f1, f2),
                    key_rel = {k1[0]['key']: k2[0]['key']},
                    fun = pyprover.proves,
                    negate_one=True)
                if new_cost:
                    triggers.append((k1[0],k2[0]))
                cost += new_cost

            beam = {}
            if cost:
                logging.info("contradiction by antonyms")
                beam = {ex1['id']:
                            {ex2['id']:
                                 {"trigger"   : triggers,
                                  "key_to_key": str(key_char_to_key_char),
                                 }
                            }
                        }
            return cost, beam
        return formula_prooves_generated

    def formula_contradicts (fit_mix, symmetric=False):
        fit_keys = fit_mix
        @Simmix.standard_range(0, 4)
        def formula_contradicts_generated (ex1,ex2):
            """
            :param ex1:
            :param ex2:
                    pyproover formulas with "pyproover." prefix for proposition "A"

            :return:
                    1 True for both, 0.7 for one direction of contradicting, 0 for nothing
            """
            if symmetric:
                res1, b1 = formula_contradicts_generated(ex1, ex2)
                res2, b2 = formula_contradicts_generated(ex2, ex1)
                return (res1+res2)/2, b1

            f1 = ex1["wff_comp_and"][:]
            f2 = ex2["wff_comp_and"][:]
            keys1 = list(ex1["wff_dict"].values())
            keys2 = list(ex2["wff_dict"].values())

            key_to_key = fit_keys.choose((keys1, keys2), out="ex", layout="1:1")

            if None in flatten(key_to_key[:]) or not key_to_key:
                logging.debug("empty negation result?")
                return 0, {}

            key_char_to_key_char = {k1[0]["key"]: k2[0]["key"] for k1, k2 in key_to_key}

            cost = 0
            triggers = []

            for k1, k2 in key_to_key:
                new_cost = Simmix.do_logic(
                    formulas = (f1, f2),
                    key_rel = {k1[0]['key']: k2[0]['key']},
                    fun = pyprover.proves)
                if new_cost:
                    triggers.append((k1[0],k2[0]))
                cost += new_cost

            beam = {}
            if cost :
                logging.info("contradiction by negation")
                beam = {ex1['id']:
                            {ex2['id']:
                                 {"trigger"   : triggers,
                                  "key_to_key":str(key_char_to_key_char),
                                 }
                            }
                        }
            return cost, beam
        return formula_contradicts_generated

    key_regex = re.compile("pyprover\.logic\.Prop\(\\'([a-zA-Z0-9])+\\'\)")
    @classmethod
    def do_logic(cls, formulas=None, key_rel=None, fun=None, negate_one=False):
        f1, f2 =  formulas

        k_in_1 = re.findall(Simmix.key_regex, f1)
        k_in_2 = re.findall(Simmix.key_regex, f2)

        k_to_correlate_1 = list(key_rel.keys())
        k_to_correlate_2 = list(key_rel.values())

        k_not_to_correlate_1 = list(set(k_in_1) - set(k_to_correlate_1))
        k_not_to_correlate_2 = list(set(k_in_2) - set(k_to_correlate_2))

        not_in_other_formula = eval(
            "(x for x in uppercase_bca)")  # "eval" because there is a complication, that the generator is really created every call of this function
        in_other_formula = eval(
            "(x for x in uppercase_abc)")  # "eval" because there is a complication, that the generator is really created every call of this function

        def pyprover_key(k):
            return "pyprover.logic.Prop('" + k + "')"

        for k1 in k_not_to_correlate_1:
            f1 = f1.replace(pyprover_key(k1), pyprover_key(next(not_in_other_formula)))
        for k2 in k_not_to_correlate_2:
            f2 = f2.replace(pyprover_key(k2), pyprover_key(next(not_in_other_formula)))

        relevant_keys = []

        for k1, k2 in zip(k_to_correlate_1, k_to_correlate_2):
            shared_key = pyprover_key("common" + next(in_other_formula))
            f1 = f1.replace(pyprover_key(k1), shared_key)

            if negate_one:
                shared_key = '~' + shared_key

            f2 = f2.replace(pyprover_key(k2), shared_key)

            relevant_keys.append(shared_key)
            # relevant_keys_nice.append (k1)

        set_keys = "&".join(relevant_keys)

        f = '(' + f1 + ') & (' + f2 + ')'
        to_proove =  '(' + set_keys + ') & ~ (' + set_keys + ')'

        #if not set_keys:
        #    logging.debug("No keys set? %s" % (str(key_to_key)))

        cost = int(pyprover.proves(eval(f), pyprover.logic.false)) ##eval(to_proove)))

        return cost


import unittest

class TestSimmix(unittest.TestCase):
    def test_one_to_one_simple(self):
        import nested_list_tools
        exs2 = [1, 2, 3]
        exs1 = [1]
        weighted_res = np.array([[0.], [0.], [0.2]])

        print(Simmix.one_to_one(weighted_res, exs1, exs2))
        print(Simmix.one_to_one(weighted_res.T, exs2, exs1))

        self.assertTrue (
                set(nested_list_tools.flatten(Simmix.one_to_one(weighted_res, exs1, exs2))) \
                == \
                set(nested_list_tools.flatten(Simmix.one_to_one(weighted_res.T, exs2, exs1))))


    def test_one_to_one_complex (self):
        weighted_res =\
             np.array( [[0.,0.,0.,0.97233996,   0.,0.,0.,0.,0.],
                        [0.,0.,0.,0.,0.95478777,0.,0.,0.,0.],
                        [0.,0.4,0.,0.,0.,0.,0.,0.,0.],
                        [0.,0.,0.,0.3,0.,0., 0.,0.,0.],
                        [0.,0.,0.,0.,0.,0.,0.,0.,0.],
                        [0.,0.,0.,0.,0.,0., 0.,0.,0.]] )
        exs1 = [0,1,2,3,4,5,6,7,8]
        exs2 = [0,1,2,3,4,5]

        print(Simmix.one_to_one(weighted_res, exs1, exs2))
        print(Simmix.one_to_one(weighted_res.T, exs2, exs1))



    def test_excluding_pair (self):
        import spacy
        nlp = spacy.load('en_core_web_sm')
        from dict_tools import balance_complex_tuple_dict
        import word_definitions

        s1 = nlp("On the other hand things are said to be named Univocally, which have both the name the definition answering to the name in common")
        s2 = nlp('Things are said to be named Equivocally when though they have a common name the definition corresponding with the name differs for each')
        p1 = {"full_ex":s1[13:25]}
        p2 = {"full_ex":s2[8:24]}


        d = {
                #'differ': [ 'derive'],
                'differ': [('differ', 'in'), 'differ', ('have', '*', 'in', 'common')]}
        d =  balance_complex_tuple_dict(d)
        fun = Simmix.excluding_pair_boolean_sim({'lemma_':d})
        fun = Simmix.excluding_pair_boolean_sim(word_definitions.antonym_dict)

        print (p1,p2, )
        print (fun(p2,p1))

        print (fun(p1,p2))
        self.assertTrue (fun(p2,p1)>0)
        self.assertTrue (fun(p1,p2)>0)


    def test_antonym_dict_for_symmetry(self):
        import word_definitions
        from dict_tools import dict_compare

        d1 = invert_dict(word_definitions.antonym_dict['lemma_'])
        d2 = word_definitions.antonym_dict['lemma_']

        added, removed, modified, same=\
            dict_compare(d1, d2, ignore_order=True)
        self.assertTrue (not added and not removed and not modified)

        if not  (not added and not removed and not modified):
            print (dict_compare(d1,d2, ignore_order=True))

    def test_antonym_dict_for_symmetry(self):
        import word_definitions
        from dict_tools import dict_compare

        def key_in_val(d):
            print ([(k,v)  for k,v in d.items() if k in v])
            return (any ([k in v  for k,v in d.items()]))

        self.assertFalse (key_in_val(word_definitions.antonym_dict['lemma_']))


if __name__ == '__main__':
    unittest.main()