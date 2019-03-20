#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' This module allows to overlap different similarity measures. Give it two lists or two lists of pairs of expressions and it find the ones, that belong together in a way, how you defined 'belonging together'.


    This module is intended for overlapping an open-end variety of similarty measures on complex data in
    natural language processing. Its like an organ with different registers for text-comparisons.

    Choices are made from two lists of dict elements, that are representing some phrases. They are bundles
    of properties, as their text, grammar, also a logical formula to represent their logical structure (pyprover), semantical vectors (and syntactical, like elmo-embeddings from allen-nlp) , tdf-idf-importance and some more. These are predefined here, if you know some of your own,its possible to give custom functions.

    Example
    -------

    Predicates have the form of dictionaries to reveal their different properties in computer linguistics.
    They can be tagged differently (pos, tag, dep), lemmatized, vectors and other kinds of represantations.

    >>> p1 = {'text':  ['Black', 'bird', 'singing', 'in', 'the', 'dead', 'of', 'night'],
    ...       'lemma': ['black', 'bird', 'sing', 'in', 'the', 'dead', 'of', 'night'],
    ...       'dep_':  ['amod', 'subj', 'compound', 'prep', 'pobj', 'det', 'prep', 'pobj']}
    >>> p2 = {'text':  ['All', 'your', 'life'],
    ...       'lemma': ['all', 'your', 'life'],
    ...       'dep_' : ['det', 'poss', 'subj']}
    >>> p3 = {'text':  ['You', 'love', 'all' 'your', 'life'],
    ...       'lemma':  ['You', 'love', 'all' 'your', 'life'],
    ...       'dep_' :  ['nsubj', 'ROOT', 'det' 'poss', 'dobj'],}

   This model now goes on to compare two lists of such predicate-dictionaries. But first you have to specify, how you
   want to compare them. You can compare app. Because indeed apples and pears and carrots are comparable, but you can
   choose, which perspective to use for that comparison, maybe by weight, by volume, color, shape or whatsoever.

   Here we define, that we want to compare this time the text of the expressions with string-fuzzy logic.
   We instantiate the SimilarityMixer-module and give it this list of tuple:

   >>> function_tuples = [
   ...             (1, SimilarityMixer.fuzzystr_sim, 0.72, 2.01), # fuzzy-string-comparison
   ...             (1, SimilarityMixer.dep_sim, 0.4, 1)           # levensthein distance of the dependency measures
   ...             ]


   And then the SimilarityMixer module can be initialized with that:

   These 4-tuples consist in

   .. code-block:: python

        function_tuple = (
            weight,                                   # weight of this measure
            function,                                 # function without argument that takes Dict[str:?], Dict[str: ?] -> float
            maximal_threshold,                        # minimum threshold
            minimal_threshold,                        # maximum threshold
            )

   And to apply that to the defined predicates:

   >>> superficial_sim  = \
   ...    SimilarityMixer(function_tuples)
   >>> superficial_sim.choose(([p1],[p2,p2]))

'''
import collections
import types
from functools import lru_cache

from addict import Dict
import scipy
import numpy as np
import numpy_indexed as npi
from cytoolz.dicttoolz import copy
from sklearn import preprocessing
import hdbscan
import itertools
import pyprover
import string
import re
from pyxdameraulevenshtein import damerau_levenshtein_distance
import logging
from littletools.nested_list_tools import check_for_tuple_in_list, flatten, flatten_reduce, flatten_list, type_spec, \
    existent, collapse
from littletools import dict_tools, nested_list_tools

from hardcore_annotated_expression import eT, eL, ltd_ify

uppercase_abc = list(string.ascii_uppercase)
uppercase_bca = list(string.ascii_uppercase)[::-1]


def match(needle, stack):
    val = (needle in stack or
           (isinstance(needle, tuple) and check_for_tuple_in_list(stack, needle)))
    return val

def match_comp_els(item, key, comp_elements1, comp_elements2):
    that_key_cost = int(item in comp_elements2)
    antonym_pair = None
    if isinstance(item, list):
        for it in item:
            if (match(it, comp_elements2)
                    and not match(it,
                                  comp_elements1)  # decisive point, that the antonympair should not itself appear in the one or other phrase
                    and not match(key, comp_elements2)
            ):
                actual_cost = int(it in comp_elements2
                                  or
                                  isinstance(it, tuple) and check_for_tuple_in_list(
                    comp_elements2, it))
                that_key_cost += actual_cost
                if actual_cost:
                    antonym_pair = ((key, it))
                    logging.info('pair of antonyms found -- "%s" and "%s"' % (key, it))
                    return that_key_cost, antonym_pair
    return False, antonym_pair




def cartesian_product_itertools(arrays):
    ''' Cartesian product, that works with functions and combinations of parameters and returns an ordered output, that
        can be sliced with numpy and axis-indexing

        Example
        =======

        >>> import addict
        >>> from numpy import array
        >>> from itertools import product
        >>> a = addict.Dict({'y':{'a':1}})
        >>> b = addict.Dict({'t':{'o':2}})
        >>> f = lambda x:x+1
        >>> array(list(product([a],[b],[f])))

    '''
    # https://stackoverflow.com/questions/11144513/numpy-cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
    # Generalized N-dimensional products
    return np.array(list(itertools.product(*arrays)))


class SimilarityMixer:
    def __init__ (self, similarity_composition, n=1):
        r''' Setup, which functions are taken to compute a similarity matrix to choose from some natural language expressions, that are already analysed in the manner of :mod:`~predicatrix.Predication`.

            Under the hood a similarity matrix is computed by a cartesian product of the two samples of expressions as
            two vectors and a vector valued function,

               .. math:: \overrightarrow{f}(x_{1,n}, x_{2,m}) = \langle f_1(x_{1,n}, x_{2,m}), f_2(x_{1,n}, x_{2,m}), \dots f_k(x_{1,n}, x_{2,m}) \rangle

            The function matrix then is:

            .. image:: /_static/doc_external/aslant_matrix.png

            Then the sum of the results of the matrix are normalized by attaching two vectors for minimum and maximum of the functions to each of the planes.
            Then the results are normalized.
            And the added maximum and miminum vectors are taken off.

            Then the matrix is weighted by building the scalar product.

            .. math::

                \overrightarrow{w} \bullet \sum D\overrightarrow{f}(\overrightarrow{x_1}, \overrightarrow{x_2})  \Delta n = \overrightarrow{w} \bullet
                \begin{bmatrix}
                     \sum_{n=1}^l f_n(x_{1,1}, x_{2,1}) & \sum_{n=1}^l f_n(x_{1,2}, x_{2,1}) & \cdots & \sum_{n=1}^l f_n(x_{1,x}, x_{2,1})\\
                     \sum_{n=1}^l f_n(x_{1,1}, x_{2,2}) & \sum_{n=1}^l f_n(x_{1,2}, x_{2,2}) & \cdots & \sum_{n=1}^l f_n(x_{1,x}, x_{2,2})\\
                      \vdots  & \vdots  & \ddots & \vdots  \\
                     \sum_{n=1}^l f_n(x_{1,1}, x_{2,y}) & \sum_{n=1}^l f_n(x_{1,2}, x_{2,y}) & \cdots & \sum_{n=1}^l f_n(x_{1,x}, x_{2,y})
                 \end{bmatrix}

            You can set a range of valid results of the similarity functions. If the value for one function, the expression is excluded.

            After this, one can adjust different modes, how to make a choice based on this similarity matrix.
            Either to take exactly one with the maximal similarity, or to take the best n, or get an optimal 1:1 alignment.

            Last but not least there are different possibilities, how to get back the results.

            * the expressions pairs, that are the chosen
                - :func:`~simmix.SimilarityMixer.expressions_list`
            * just the indices of the input expressions or just of the left resp. right expression list.
                - :func:`~simmix.SimilarityMixer.expressions.i_list`
            * write it into a graph, if you present a routinr for writing in a graph (a coroutine, to receive the result with `data=(yield) and send with `corou.send(data...)` send).
                - :func:`~simmix.SimilarityMixer.write_to_graph`

            :param similarity_composition:
                list of tuples (
                    weight<float>,
                    fun<function, accepting expressiondicts as arguments>,
                    min-val<float>,
                    max-val<float>)
            :param n:
                number of results to return, can be overridden with choose(...)

        '''
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
        self.n              = n

        self.funs           = np.array(funs)
        self.weights        = np.array(weights)
        self.thresholds_min = np.array(thresholds_min)
        self.thresholds_max = np.array(thresholds_max)

        try:
            mins = [f.min for f in funs]
            maxs = [f.max for f in funs]
            self.minmax_defaults = np.array([mins,maxs])
        except AttributeError:
            raise AttributeError ("At least one of the funs doesn't have standard_range decorator! %s " % (str (funs)))
        self.scaler = preprocessing.MinMaxScaler()


    def standard_range (minim, maxim):
        ''' Decorator for similarity computing functions, it is later used for normalisation

        :param maxim: maximum value
        :param maxim: minimum value
        :return: decorated function

        '''
        def wrapper(f):
            f.min = minim
            f.max = maxim
            return f
        return wrapper


    def multi_fun_doc (fun):
        ''' Decorator for similarity computing functions, it is later used for normalisation

        :param maxim: maximum value
        :param maxim: minimum value
        :return: decorated function

        '''
        def wrapper(f):
            f.fun = fun
            return f
        return wrapper



    def apply_sim_fun (self, fee):
        ''' Apply the field of functions to the arguments

        The argument are 3-tuples, with the function and the two expressions, these expressions are feeded into the functions

        :param fee: tuple
        :return: return value of the function
        :raises: TypeError if the function does not give a 2-tuple of value and dictionary information for backtracking
        '''
        fun = fee[2]
        ex1 = fee[0]
        ex2 = fee[1]
        try:
            fun(ex1, ex2)
        except ValueError:
            pass
        try:
            res, d = fun(ex1,ex2)
            dict_tools.update (self.beam, d)
        except TypeError:
            raise TypeError ("Function doesn't return beam. %s, %s " % (str(fun(ex1,ex2)), str(fun)))
        except ValueError:
            raise ValueError(" ". join ([str(fun), str(ex1), str(ex2)]))
        return res

    def choose(self, data,
               input='tuple',
               out=None,
               layout=None,
               n=None,
               graph_coro=None,
               type=None,
               output=False,
               minimize=False):
        '''Choose in different manners building a similarity matrix.

        :param data:
            tuple of lists of expressions, meaning the dictionaries or what is compared for you
        :param out:
            defines the structure of the returned value
            default=fNone: the chosed expressions are returned
            'i': only the indices
            'r': only the indices of the right value are returned
            '2t': pairs of tuples (for building a graph
            'nx': directed networkxgraph
        :param graph_coro:
            graph_coroutine, that accepts a 3-tuple of two dictionaries and a `type` string
        :param type:
            If using `G`, one has to specify the type-argument,
                * either in case of input lists as data a string,
                * or a 3tuple `(type_l_pair, type_r_pair, type_between)=type` in case of input datatypes like the
            list-tuple-list-dict, that this module returns
        :param layout:
            default=None behavior like n=1
            '1:1' means for each one element of the one expression is eactly one chosen from the other
            'hdbscan':
                 for samples of n elements are fitting m other elements chosen from the other expression, works only
                 for samples with >10 elements
        :param n:
            default=1
            count of best n solutions, also clusters of expressions of the one are chosen to fit to a cluster of the other
        :return:
            list of tuples of lists of ints or indexed expressions, according to the chosen 'out' parameter

        '''
        my_args = locals().copy()

        exs1 = None
        exs2 = None
        self.beam = {}
        if n:
            self.n = n
        if layout == None:
            raise ValueError ("'layout' must be given")

        if isinstance(data, tuple) and len(data) >  2:
            results = []
            for combo in itertools.combinations(data, 2):
                my_args['data'] = combo
                my_args['n'] = 100
                results.append(SimilarityMixer.choose(**my_args))
            return results

        elif isinstance(data, list) or isinstance(data, types.GeneratorType):
            tuples = []
            for tup in data:
                my_args['data'] = tup
                tuples += SimilarityMixer.choose(**my_args)
            return tuples

        elif isinstance(data, tuple) and len(data) == 2:
            exs1, exs2 = (data[0], data[1])

        sim_cube = cartesian_product_itertools([
            exs1,
            exs2,
            self.funs
        ])

        try:
            res_cube = np.apply_along_axis(self.apply_sim_fun, 1, sim_cube)
            res_cube = res_cube.reshape(len(exs1), len(exs2), -1)
        except np.core._internal.AxisError as e:
            raise ValueError('expression is empty %s' % (str((exs1, exs2))))
        except TypeError as e:
            raise TypeError (str(e) + "\nor you forgot the multi-wrapper, when using a function for tuples as simmix expressions")

        # get cubic shape back and floating points for the numbers
        res_cube = res_cube.reshape((-1, len(self.funs)))
        res_cube = res_cube.astype(np.float64)

        # add some minmax_vectors to have comparable scaling for normalization
        # but first, check, if these values
        exceeds_min = res_cube < self.minmax_defaults[0]
        exceeds_max = res_cube > self.minmax_defaults[1]

        if exceeds_min.any() or exceeds_max.any():
            if exceeds_min.any():
                exceeding_min, i_f_min = np.unique(np.where(exceeds_min)[0]), np.unique(np.where(exceeds_min)[1])
                logging.error(
                     "\n\n***********************************************************************************\n"
                     "Similarity results exceed minimum defaults, required for constant scaling in normalization.\n"
                     "Function: {fun} {fun_in_fun} from threshold: {thresh} to maximally: {value}".format(
                        fun=str(self.funs[i_f_min]),
                        fun_in_fun = "".join(collapse(['(',[str(f.fun) for f in self.funs[i_f_min] if hasattr(f, 'fun')], ')'])),
                        thresh=str(self.minmax_defaults[0][i_f_min]),
                        value=str(res_cube[exceeds_min[:]].min(axis=0))))
            if exceeds_max.any():
                exceeding_max, i_f_max = np.unique(np.where(exceeds_max)[0]), np.unique(np.where(exceeds_max)[1])
                logging.error(
                     "\n\n***********************************************************************************\n"
                     "Similarity results exceed maximum defaults, required for constant scaling in normalization.\n"
                     "Function: {fun} {fun_in_fun} from threshold: {thresh} to minimally: {value}".format(
                        fun=str(self.funs[i_f_max]),
                        fun_in_fun="".join(collapse(['(', [str(f.fun) for f in self.funs[i_f_max] if hasattr(f, 'fun')], ')'])),
                        thresh=str(self.minmax_defaults[1][i_f_max]),
                        value=str(res_cube[exceeds_max[:]].max(axis=0))))


        res_cube = np.append(res_cube, self.minmax_defaults, axis=0)
        # scale
        trans_cube = self.scaler.fit_transform(res_cube)
        # reshape matrix
        trans_cube = trans_cube[:trans_cube.shape[0] - 2, :]
        if output == True:
            print(trans_cube)

        # apply weights: matrix dot vector = summed up vector with one value for each vector
        weighted_res = trans_cube.dot(self.weights)  # Matrix times weight

        if layout=='hdbscan':
            matrix = self.scaler.fit_transform(weighted_res.reshape(-1, 1))
            matrix = matrix.reshape(len(exs1), len(exs2))
            np.fill_diagonal(matrix, 1)
            dist_matrix = abs(1-matrix)

            #matrix[matrix <= 0.3] = np.inf
            #matrix[matrix >= 1] = np.inf
            if not exs1 == exs2 and len(matrix)<1:
                raise ValueError ({"Expression lists for hdbscan must be the same, and the matrix symmetric"})

            clusterer = hdbscan.HDBSCAN(min_cluster_size=2,
                                        #min_samples=1,
                                        metric='precomputed',
                                        cluster_selection_method='leaf',
                                        #algorithm='generic'
                                        )

            try:
                cluster_labels = clusterer.fit_predict(dist_matrix)
            except ValueError:
                raise
            #print ("hdbscan:", cluster_labels)

            if all(cluster_labels == -1):
                 from sklearn.cluster import SpectralClustering
                 sc = SpectralClustering(n_clusters=int(len(exs1)/2), affinity='precomputed', n_init=100,
                                    assign_labels='discretize')
                 sc.fit_predict(matrix)
                 #print("sprectral clustering:", sc.labels_)
                 cluster_labels = sc.labels_

            grouped_indices = npi.group_by(cluster_labels).split(list(range(0,len(exs1))))
            if (sorted(cluster_labels)[0]==-1):
                grouped_indices = grouped_indices [1:]
            #print (grouped_indices)
            return ltd_ify([eT(tuple([exs1[x] for x in arr])) for arr in grouped_indices])

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
            left_value, right_values = SimilarityMixer.one_to_one(weighted_res, exs1, exs2, mask)

        elif layout == 'n':
            ''' compute choordinates within in the cube by divmod, that means, try to divide through the length of the rows (=y)
            and the rest is the the position in the col (=x)
            
            for instance this means:
            coordinates =  <class 'tuple'>: (array([0]), array([6])) '''
            coodinates = np.divmod(best_n, len(exs2))

            ''' zip the numpy arrays '''
            coodinates = np.column_stack(coodinates)

            ''' decompose and uniquify the arrays, we need a list for left and a list of lists for right'''
            ''' these are the centers of the similar clusters'''
            left_value = np.unique(coodinates[:, 0])

            ''' these are different nodes, that belong to them'''
            right_values = npi.group_by(coodinates[:, 0]).split(coodinates[:, 1])

        if not graph_coro == None:
            self.write_to_graph (graph_coro=graph_coro, type=type, exs1=exs1, exs2=exs2, left_value=left_value, right_values=right_values)

        if not out or out == 'ex':
            return self.expressions_list(left_value, right_values, exs1, exs2)
        elif out == 'i':
            return self.i_list(left_value, right_values)
        elif out == 'r':
            i_s = self.i_list(left_value, right_values)
            return SimilarityMixer.reduce_i_s_right_values(i_s)
        elif out == 'lx':
            return [exs1[i] for i in left_value]
        elif out == 'rx':
            return [exs1[i] for r in right_values for i in r]
        elif out == '2t':
            i_s = self.i_list(left_value, right_values)
            return SimilarityMixer.reduce_i_s_pair_tuples(i_s)
        elif out == 'nx':
            if graph_coro == None:
                raise ValueError("parameter graph_coro is required for use with a Graph!")
            return graph_coro
        elif out == '(i,ex)':
            return (self.i_list(left_value, right_values),
                    self.expressions_list(left_value, right_values, exs1, exs2))
        else:
            raise NotImplementedError("What the else could out be returned? Wrong parameter for 'out'")

    def one_to_one(weighted_res, exs1, exs2, mask):
        ''' Compute combinations of input-lists with a solution connecting one to one expression.
            Solved with the hungarian method, filtered by the mask of acceptable solutions within the allowed range of values.

            :param exs1:  list of comparaed values  in x direction
            :param exs2:  list of compared values   in y direction
            :param mask:  np boolean mask of fine solutions
            :return:      tuple of list of indices to the exs1 list and of indices of the exs2 list

        '''
        res = scipy.optimize.linear_sum_assignment(-weighted_res)
        res_mask = mask[res]
        l_values = res[0][res_mask].tolist()
        r_values = [[r] for r in res[1][res_mask].tolist()]
        return l_values, r_values


    def expressions_list (self, left_value, right_values, exs1, exs2):
        ''' Collects the expressions from the indices

        :param left_value: list of ints, indices for exs1
        :param right_values: list of list of indices for exs2, they are indices, because they can be more than one if found
        :param exs1: list of dicts or list of tuples of list of dicts
        :param exs2: list of dicts or list of tuples of list of dicts

        :return: list of tuples of lists of dicts or
                 list of tuple of list of tuple of list of dict

        '''
        try:
            triggers = [[self.beam[exs1[left_value[l]]['id']][exs2[r]['id']] for r in right_values[l]]
                      for l in range(len(left_value))]
        except KeyError:
            logging.warning ('beam has wrong ids %s ' % self.funs)
            triggers = [[]] * len(left_value)
        except TypeError:
            pass
            triggers = [[]] * len(left_value)


        exss = [(eL([exs1[left_value[l]]]),
                 eL([exs2[r] for r in right_values[l]]))
                  for l in range(len(left_value))]
        return eL(eT(tup, trigger=trig) for tup, trig in zip (exss, triggers))


    def i_list (self, left_value, right_values):
        return [([left_value[l]], [r for r in right_values[l]])
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

    @standard_range(-1.5, 0)
    def pos_sim (ex1, ex2):
        ''' Compares the pos tags. Levenstejn distance per total length of both expressions

        :param ex1: dict with ['pos']
        :param ex2: dict with ['pos']
        :return: float and bracktracking {}

        '''
        grammar1  = ex1['pos']
        grammar2  = ex2['pos']
        res = -damerau_levenshtein_distance(grammar1, grammar2) /(len(ex1) + len(ex2))
        beam = Dict()
        beam[ex1['id']][ex2['id']].reason = res
        return res, beam

    @standard_range(-1.5, 0)
    def dep_sim (ex1, ex2):
        ''' Compares the dep tags. Levenstejn distance per total length of both expresions

        :param ex1: dict with ['dep']
        :param ex2: dict with ['dep']
        :return: float and bracktracking {}

        '''
        grammar1  = ex1["dep"]
        grammar2  = ex2["dep"]
        res = -damerau_levenshtein_distance(grammar1, grammar2) / (len(ex1) + len(ex2))
        beam = Dict()
        beam[ex1['id']][ex2['id']].reason = res
        return res, beam

    @standard_range(-1, 0)
    def lemma_sim (ex1, ex2):
        ''' Compares the lemma tags. Levenstejn distance per total length of both expresiions

            :param ex1: dict with ['lemma']
            :param ex2: dict with ['lemma']
            :return: float and bracktracking {}

            '''
        grammar1  = ex1["lemma"]
        grammar2  = ex2["lemma"]
        return -damerau_levenshtein_distance(grammar1, grammar2) / (len(ex1) + len(ex2)), {}

    @standard_range(-1, 0)
    def tag_sim (ex1, ex2):
        ''' Compares the tag tags. Levenstejn distance per total length of both expresiions

            :param ex1: dict with ['tag']
            :param ex2: dict with ['tag']
            :return: float and bracktracking {}

            '''
        grammar1  = ex1["tag"]
        grammar2  = ex2["tag"]
        return -damerau_levenshtein_distance(grammar1, grammar2) / (len(ex1) + len(ex2)) , {}

    @standard_range(-1.5, 0)
    def fuzzystr_sim (ex1, ex2):
        ''' Compares the text tags. Levenstejn distance per total length of both expresiions

            :param ex1: dict with ['text']
            :param ex2: dict with ['text']
            :return: float and bracktracking {}

            '''
        str1  = ex1["text"]
        str2  = ex2["text"]
        res = -damerau_levenshtein_distance(str1, str2) / (len(ex1) + len(ex2))
        beam = Dict()
        beam[ex1['id']][ex2['id']].reason = res
        return res, beam

    @lru_cache(maxsize=None)
    @standard_range(-1, 0)
    def fuzzytext_sim (ex1, ex2):
        ''' Compares the text tags. Levenstejn distance per total length of both expresiions

            :param ex1: dict with ['text']
            :param ex2: dict with ['text']
            :return: float and bracktracking {}

            '''
        str1  = " ".join (ex1["text"])
        str2  = " ".join (ex2["text"])
        res = -damerau_levenshtein_distance(str1, str2) / max (len(ex1),len(ex2))
        beam = Dict()
        beam[ex1['id']][ex2['id']].reason = res
        return res, beam


    def common_words_sim(invert = False):
        if not invert:
            factor = 1
        else:
            factor = -1

        @lru_cache(maxsize=None)
        @SimilarityMixer.standard_range(0, 20)
        def common_words_sim_generated (ex1, ex2):
            ''' This function gives weighted score for common words and negative weighted score for words, that appear only
                in one expressions.

                The weight of the weighted score is the tdidf-score in the document.

                .. math::

                    c = \dfrac{\sum_{n=1}^{|W_{x_1} \cup W_{x_2}|} 2 \text{tf-idf}(w_n,x_1, D) -    \text{tf-idf}(w_n,x_1, D) -  \text{tf-idf}(w_n,x_2, D)}{|W_{x_1} \cup W_{x_2}|}

                :param ex1: dict with 'importance' and 'lemma'
                :param ex2: dict with 'importance' and 'lemma'
                :return: c and backtracking beam {}

                '''
            # A bit something else than this: https://en.wikipedia.org/wiki/Overlap_coefficient
            str1  = ex1["lemma"]
            str2  = ex2["lemma"]
            sum1  = sum(ex1['importance'][i]**factor for i,x in enumerate(str1))
            sum2  = sum(ex2['importance'][i]**factor for i,x in enumerate(str2))
            res = ((sum([ex1['importance'][i]**factor for i,x in enumerate(str1) if x in str2]) +
                   sum([ex2['importance'][i]**factor for i, x in enumerate(str2) if x in str1]) -
                   sum([ex1['importance'][i]**factor for i, x in enumerate(str1) if x not in str2]) -
                   sum([ex2['importance'][i]**factor for i, x in enumerate(str2) if x in str1])) + 3 )/(sum1 + sum2)
            if res < 0:
                res = 0
            beam = Dict()
            beam[ex1['id']][ex2['id']].reason = res
            return res, beam
        return common_words_sim_generated

    def convolve_sim(layer=None):
        ''' Compares embeddings with a convolution

            :param ex1: dict with ['elmo_embeddings_full']
            :param ex2: dict with ['elmo_embeddings_full']
            :return: float and bracktracking {}

            '''
        import scipy.signal as sg
        if not layer:
            raise ValueError('layers must be set!')
        @SimilarityMixer.standard_range(-5, 5)
        def _convolve_sim (ex1, ex2):
            vectors1 = ex1["elmo_embeddings_full"]
            vectors2 = ex2["elmo_embeddings_full"]
            return sg.convolve(vectors1, vectors2).mean()

        return _convolve_sim

    def elmo_sim():
        ''' Compares the expressions by their elmo embeddings, all three layers are taken into account.

            There is a problem with comparing a really long expression with a short one, then the similarity is normaly
            higher than with shorter expressions. You can imagine this, if you put different vectors  together by summing
            them up, you get statistically a vector, that goes in every direction and is more like a snowball, all snowballs
            look equal, but not totally the same.
            If you have just around 5, then its quite more directed. So I decided to divide it with a the natural logarithm
            of the biggstes lengths of both. Maybe I can't back up that scientificially, ok.

            ..math::

                c = \ln (\dfrac{\text(max)}{2e}) * (      \text{cosine distance}(\overrightarrow{x_{1,0}}, \overrightarrow{x_{2,0}}) +       \text{cosine distance}(\overrightarrow{x_{1,1}}, \overrightarrow{x_{2,1}}) +       \text{cosine distance}(\overrightarrow{x_{1,2}}, \overrightarrow{x_{2,2}})
            :param ex1: dict with ['elmo_embeddings', 'full_ex']
            :param ex2: dict with ['elmo_embeddings', 'full_ex]
            :return: float and bracktracking {}

            TODO Parentheses bug!

            '''
        #@lru_cache(maxsize=None)
        @SimilarityMixer.standard_range(-6, 0.5)
        def elmo_sim_generated (ex1,ex2):
            vectors1 = ex1["elmo_embeddings"]
            vectors2 = ex2["elmo_embeddings"]
            total_length = max(len(ex1["full_ex"]), len (ex2["full_ex"]))
            res = (-(np.log( (total_length/np.e+np.e)) *
                     scipy.spatial.distance.cosine(vectors1[0,:],vectors2[0,:])+
                     scipy.spatial.distance.cosine(vectors1[1,:],vectors2[1,:])+
                     scipy.spatial.distance.cosine(vectors1[2,:],vectors2[2,:])))

            beam = Dict()
            beam[ex1['id']][ex2['id']].reason = res
            return res, beam
        return elmo_sim_generated

    @lru_cache(maxsize=None)
    def elmo_multi_sim(n=3):
        @SimilarityMixer.standard_range(-6 * n, 0.5 * n)
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
        @SimilarityMixer.standard_range(-10 / len(layer), 0.5 / len(layer))
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

    @lru_cache(maxsize=None)
    @standard_range(-20, 20)
    def elmo_weighted_sim():
        ''' Use a tf-idf-importance weighted, length-normalized measure based on the most semantical layer of elmo-
        embeddings

        ..math::
             c = \ln (\dfrac{\text(max)}{2e}) * (      \text{cosine distance}(\overrightarrow{x_{1,0}}, \overrightarrow{x_{2,0}}) +       \text{cosine distance}(\overrightarrow{x_{1,1}}, \overrightarrow{x_{2,1}}) +       \text{cosine distance}(\overrightarrow{x_{1,2}}, \overrightarrow{x_{2,2}})

        :param ex1: dict with 'elmo_embeddings_per_word', 'importance'
        :param ex2: dict with 'elmo_embeddings_per_word', 'importance'
        :return: -20 to +20 , {}
        '''
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
        ''' Is the text the same in the other?

        :param ex1: dict with 'text'
        :param ex2: dict with 'text
        :return: 0-1 , {}

        '''
        text1        = " ".join(ex1["text"])
        text2        = " ".join(ex2["text"])
        return text2 == text1, {}


    @standard_range(0, 1)
    def same_expression_sim (ex1, ex2):
        ''' Is the text of the one contained in the other?

        :param ex1: dict with 'text'
        :param ex2: dict with 'text
        :return: 0-1  , {}

        '''
        text1        = ex1["text"]
        text2        = ex2["text"]
        res =  int (all(w in text1 for w in text2) or all(w in text2 for w in text1))
        beam = Dict()
        beam[ex1['id']][ex2['id']].reason = res
        return res, beam

    @lru_cache(maxsize=None)
    @standard_range(0, 1)
    def sub_i (ex1, ex2):
        ''' Are the one tokens a subset of the other tokens?

        :param ex1: dict with 'i_s'
        :param ex2: dict with 'i_s
        :return: 0-1

        '''
        is1 = set(ex1["i_s"])
        is2 = set(ex2["i_s"])
        try:
            return is2.issubset(is1) and ex1['doc'] == ex2['doc'], {}
        except KeyError:
            logging.warning ("Why is there no 'doc'-key? Proceed... ")
            return is2.issubset(is1), {}


    @standard_range(0,1)
    def same_sent_sim (ex1,ex2):
        ''' Check if both expressions are from the same sentence.

        :param ex1: dict with 's_id'
        :param ex2: dict with 's_id'
        :return: 0 or 1

        '''
        return int(ex1['s_id'] == ex2['s_id'])


    @standard_range(0,1)
    def coreferential_sim (ex1,ex2):
        ''' Check if the expressions are coreferential.

        That means, either they refer to the same expression or they refer to one another

        :param ex1: dict with 'coref', 's_id', 'full_ex_i'
        :param ex2: dict with 'coref', 's_id', 'full_ex_i'
        :return: 0 or 1

        '''
        coref1 = ex1['coref']
        coref2 = ex2['coref']

        def look_for_the_other(corefs, ex):
            for corefs_per_word in existent(corefs):
                for coref in corefs_per_word:
                    if coref['s_id'] == ex['s_id']:
                        if any(i in ex['i_s'] for i in range(coref['m_start'], coref['m_end'])):
                            return True
                        else:
                            continue
                    else:
                        continue
                return False
            return False

        def same_reference(coref1, coref2):
            return any([c in coref1 for c in coref2 if c])

        coreferential = (same_reference(coref1, coref2)
                         or look_for_the_other(coref1, ex2)
                         or look_for_the_other(coref2, ex1))
        return int (coreferential), {}



    @standard_range(-100,0)
    def longer_sim (ex1, ex2):
        ''' Difference between the lenght of the expressions.

        ..math::

            c = |W_{x_1} \cup W_{x_2}|

        :param ex1: expression1
        :param ex2: expression2
        :return: abs (len(ex1['i']) - len ['i'], {}

        '''
        is1 = set(ex1["i"])
        is2 = set(ex2["i"])
        return abs (len (is1)-len(is2)), {}

    @lru_cache(maxsize=None)
    def multi_cross2tup_sim(fun, n=2):
        ''' This wrapper turns a functions, that evaluate pairs of expression, into functions, that work with tuples (!)
        of expressions, to evaluate these tuples in a crossing over way. So with (1,2) and (3,4) you gonna compare 1
        with 4 and 2 with 3.

        Example
        -------
        With input like this, every value value in a pair is compared to the cross over value of the other pair

        >>> data = ([{'id': '1','text': ['I', 'am', 'here']}],[{'id': '2', 'text': ['I', 'am', 'there']}]), ([{'id': '3', 'text': ['I', 'was', 'here']}],[{'id': '4','text': ['I', 'was', 'there']}])
        >>> SimilarityMixer.multi_cross2tup_sim(SimilarityMixer.fuzzystr_sim, n=2)(*data)
        (-1.0, {'1': {'4': {'reason': -0.5}}, '2': {'3': {'reason': -0.5}}})

        So, with a result of 2*-.5, you get -1 for this fuzzy logic cross over comparison of these two tuples in data.
        Beware, that in the tuples the predicate dicts must be encapsulated in lists.

        >>> data1 = ([{'id': '1','text': ['You', 'are', 'here']}],[{'id': '2', 'text': ['You', 'was', 'there']}]), ([{'id': '3', 'text': ['I', 'am']}],[{'id': '4','text': ['I', 'was', 'there', 'not']}])
        >>> data2 = ([{'id': '1','text': ['I', 'am', 'here']}],[{'id': '2', 'text': ['I', 'am', 'there']}]), ([{'id': '3', 'text': ['I', 'am']}],[{'id': '4','text': ['I', 'was', 'there', 'not']}])
        >>> data3 = ([{'id': '1','text': ['I', 'am', 'here']}],[{'id': '2', 'text': ['I', 'am', 'there']}]), ([{'id': '3', 'text': ['I', 'am']}],[{'id': '4','text': ['am','here']}])

        >>> SimilarityMixer.multi_cross2tup_sim(SimilarityMixer.same_expression_sim, n=2)(*data1)
        (0, {'1': {'4': {'reason': 0}}, '2': {'3': {'reason': 0}}})
        >>> SimilarityMixer.multi_cross2tup_sim(SimilarityMixer.same_expression_sim, n=2)(*data2)
        (1, {'1': {'4': {'reason': 0}}, '2': {'3': {'reason': 1}}})
        >>> SimilarityMixer.multi_cross2tup_sim(SimilarityMixer.same_expression_sim, n=2)(*data3)
        (2, {'1': {'4': {'reason': 1}}, '2': {'3': {'reason': 1}}})

        If you want to know, if one of the expressions is contained in the opposite position, do this


        :param n: How many values are maximally expected in these lists? The expected range of the multi-fun is the
            square of this
        :return: wrapped function

        '''
        n **= 2
        beam = Dict()

        @SimilarityMixer.multi_fun_doc(fun)
        @SimilarityMixer.standard_range(fun.min * n, fun.max * n)  # n depends on fun!
        def multi_cross2tup_sim (exs1,exs2):
            b = copy(beam)
            sim = 0
            if not exs1 or not exs2:
                raise ValueError ("one of the expressions is empty")

            ex11, ex12 = exs1
            ex21, ex22 = exs2

            res1, d1 = fun (ex11[0], ex22[0])
            res2, d2 = fun (ex12[0], ex21[0])
            res = res1 + res2

            b.update(d1)
            b.update(d2)
            return res, b
        return multi_cross2tup_sim



    def multi_paral2tup_sim(fun, n=2):
        ''' This wrapper turns a functions, that evaluate pairs of expression, into functions, that work with tuples (!)
        of expressions, to evaluate these tuples in a parallel way. So with (1,2) and (3,4) you gonna compare 1
        with 2 and 3 with 4, as it is in these tuples.

        Example
        -------
        With input like this, every value value in a pair is compared to the cross over value of the other pair

        >>> data = ([{'id': '1','text': ['I', 'am', 'here']}],[{'id': '2', 'text': ['I', 'am', 'there']}]), ([{'id': '3', 'text': ['I', 'was', 'here']}],[{'id': '4','text': ['I', 'was', 'there']}])
        >>> SimilarityMixer.multi_paral2tup_sim(SimilarityMixer.fuzzystr_sim, n=2)(*data)
        (-1.0, {'1': {'4': {'reason': -0.5}}, '2': {'3': {'reason': -0.5}}})

        So, with a result of 2*-.5, you get -1 for this fuzzy logic cross over comparison of these two tuples in data.
        Beware, that in the tuples the predicate dicts must be encapsulated in lists.

        >>> data1 = ([{'id': '1','text': ['You', 'are', 'here']}],[{'id': '2', 'text': ['You', 'was', 'there']}]), ([{'id': '3', 'text': ['I', 'am']}],[{'id': '4','text': ['I', 'was', 'there', 'not']}])
        >>> data2 = ([{'id': '1','text': ['I', 'am', 'here']}],[{'id': '2', 'text': ['I', 'am', 'there']}]), ([{'id': '3', 'text': ['I', 'am']}],[{'id': '4','text': ['I', 'was', 'there', 'not']}])
        >>> data3 = ([{'id': '1','text': ['I', 'am', 'here']}],[{'id': '2', 'text': ['I', 'am', 'there']}]), ([{'id': '3', 'text': ['I', 'am']}],[{'id': '4','text': ['am','here']}])

        >>> SimilarityMixer.multi_paral2tup_sim(SimilarityMixer.same_expression_sim, n=2)(*data1)
        (0, {'1': {'4': {'reason': 0}}, '2': {'3': {'reason': 0}}})
        >>> SimilarityMixer.multi_paral2tup_sim(SimilarityMixer.same_expression_sim, n=2)(*data2)
        (1, {'1': {'4': {'reason': 0}}, '2': {'3': {'reason': 1}}})
        >>> SimilarityMixer.multi_paral2tup_sim(SimilarityMixer.same_expression_sim, n=2)(*data3)
        (2, {'1': {'4': {'reason': 1}}, '2': {'3': {'reason': 1}}})

        If you want to know, if one of the expressions is contained in the opposite position, do this


        :param n: How many values are maximally expected in these lists? The expected range of the multi-fun is the
            square of this
        :return: wrapped function

        '''
        n **= 2
        beam = Dict()

        @SimilarityMixer.multi_fun_doc(fun)
        @SimilarityMixer.standard_range(fun.min * n, fun.max * n)  # n depends on fun!
        def multi_paral2tup_sim (exs1,exs2):
            b = copy(beam)
            sim = 0
            if not exs1 or not exs2:
                raise ValueError ("one of the expressions is empty")

            ex11, ex12 = exs1
            ex21, ex22 = exs2

            res1, d1 = fun (ex11[0], ex12[0])
            res2, d2 = fun (ex12[0], ex22[0])
            res = res1 + res2

            b.update(d1)
            b.update(d2)
            return res, b
        return multi_paral2tup_sim

    def multi_paral_tup_sim(fun, n=4):
        ''' This wrapper turns functions, that evaluate pairs of expression, into functions, that work with tuples (!)
        of expressions. This is done by comparing each value in these list with each value in the other list.

        Example
        -------
        define the function like this

        >>> sx = SimilarityMixer (
               [(1, SimilarityMixer.multi_sim(SimilarityMixer.fuzzystr_sim, n=7), 0.5, 1)]
               )

        and work with data that are a tuple lists of tuples of tuples lists of predicate-dicts

        >>> data = (
                    [([{'text': ..., ...}],[{'text': ..., ...}])],
                    [([{'text': ..., ...}],[{'text': ..., ...}])])

        That you need, if you want to filter the results of simmix again with simmix, for example what pairs of
        expressions fit to other expressions.

        :param n: How many values are maximally expected in these lists? The expected range of the multi-fun is the
            square of this
        :return: wrapped function

        '''
        n **= 2
        beam = Dict()

        @SimilarityMixer.standard_range(fun.min * n, fun.max * n)  # depends on fun!
        def multi_paral_tup_generated(exs1, exs2):
            b = copy(beam)
            sim = 0
            if not exs1 or not exs2:
                raise ValueError("one of the expressions is empty")

            number = len(exs1) * len(exs2)
            for ex1, ex2 in itertools.product(flatten_reduce(exs1), flatten_reduce(exs2)):

                try:
                    res, d = fun(ex1, ex2)
                except:
                    pass

                try:
                    res, d = fun(ex1, ex2)
                except TypeError:
                    raise NotImplementedError("return beam from the distance measure! %s" % str(fun))
                except ValueError:
                    raise

                if not d:
                    raise NotImplementedError("beam empty! %s" % str(fun))


                sim += res / number
                b.update(d)
            return sim, b
        return multi_paral_tup_generated

    @lru_cache(maxsize=None)
    def multi_sim(fun, n=2):
        ''' This wrapper turns functions, that evaluate pairs of expression, into functions, that work with tuples (!)
        of expressions. This is done by comparing each value in these list with each value in the other list.

        Example
        -------
        define the function like this

        >>> sx = SimilarityMixer (
               [(1, SimilarityMixer.multi_sim(SimilarityMixer.fuzzystr_sim, n=7), 0.5, 1)]
               )

        and work with data that are a tuple lists of tuples of tuples lists of predicate-dicts

        >>> data = (
                    [([{'text': ..., ...}],[{'text': ..., ...}])],
                    [([{'text': ..., ...}],[{'text': ..., ...}])])

        That you need, if you want to filter the results of simmix again with simmix, for example what pairs of
        expressions fit to other expressions.

        :param n: How many values are maximally expected in these lists? The expected range of the multi-fun is the
            square of this
        :return: wrapped function

        '''
        n **= 2
        beam = Dict()
        @SimilarityMixer.multi_fun_doc(fun)
        @SimilarityMixer.standard_range(fun.min * n, fun.max * n)  # depends on fun!
        def multi_generated (exs1,exs2):
            b = copy(beam)
            sim = 0
            if not exs1 or not exs2:
                raise ValueError ("one of the expressions is empty")

            for ex1, ex2 in itertools.product(flatten_reduce(exs1), flatten_reduce(exs2)):

                try:
                    res, d = fun (ex1, ex2)
                except TypeError:
                    raise NotImplementedError ("return beam from the distance measure! %s" % str(fun))
                if not d:
                    raise NotImplementedError ("beam empty! %s" % str(fun))

                sim += res
                b.update(d)
            return sim, b
        return multi_generated


    @standard_range(0, 1)
    def left_sim (ex1, ex2):
        ''' Is the expression2 left of expression1?

            :param ex1: predicate-tuple
            :param ex2: predicate-tuple
            :return: 0 or 1 (if left), {}

        '''
        pos1  = max(ex1["i_s"]) + int(ex1['s_id'])*1000
        pos2  = min(ex2["i_s"]) + int(ex2['s_id'])*1000
        res = int(pos1<pos2)
        beam = Dict()
        beam[ex1['id']][ex2['id']].reason = res
        return res, beam


    @standard_range(0, 1)
    def right (ex1, ex2):
        ''' Is the expression2 right of expression1?

            :param ex1: predicate-tuple
            :param ex2: predicate-tuple
            :return: 0 or 1 (if right), {}

        '''
        pos1  = max(ex1["i_s"]) + int(ex1['s_id'])*1000
        pos2  = min(ex2["i_s"]) + int(ex2['s_id'])*1000
        return  int(pos1>pos2), {}


    def boolean_sim (attrib_dict):
        ''' According to a dictionary, that says, which attribute in the one of the pair must also in the attribute
            of the other one, if a match should be there.

            Example
            -------

            >>> d = {'lemma':
            ...            {'black':['life'],
            ...             'gray': [('shade', 'of', 'gray')]
            ...              },
            ...      'dep_': {'subj': 'subj'}}
            >>> p1 = {'text':  ['Black', 'bird', 'singing', 'in', 'the', 'dead', 'of', 'night'],
            ...       'lemma': ['black', 'bird', 'sing', 'in', 'the', 'dead', 'of', 'night'],
            ...       'dep_':  ['amod', 'subj', 'compound', 'prep', 'pobj', 'det', 'prep', 'pobj']}
            >>> p2 = {'text':  ['All', 'your', 'life'],
            ...       'lemma': ['all', 'your', 'life'],
            ...       'dep_' : ['det', 'poss', 'subj']}
            >>> f = SimilarityMixer.boolean_sim (d)
            >>> f (p1,p2)
            (2, {})

        :param attrib_dict: dict of dict of lists of strings

        :return: number of matched conditions and {}, the beam to follow the computation in some cases

        '''
        @SimilarityMixer.standard_range(0, 10)
        def boolean_sim_(ex1,ex2):
            cost = 0
            for attr, single_attrib_dict in attrib_dict.items():
                for key, item in single_attrib_dict.items():
                    if key in ex1[attr]:
                        that_key_cost =  int(item in ex2[attr])
                        if isinstance(item,list):
                            for it in item:
                                that_key_cost +=  int(it in ex2[attr])
                        cost += that_key_cost
            return cost, {}
        return boolean_sim_


    def detect_pair (attrib_dict):
        """ The defined pair of values mustn't occur in one of the statements at once and in the other. """
        if not attrib_dict:
            raise ValueError("Dictionary of atrributes and values can't be " + str(attrib_dict))
        #attrib_dict = {k: collections.OrderedDict({kk: vv for kk, vv in v.items()}) for k, v in
        #               attrib_dict.items()}
        #attrib_dict = collections.OrderedDict(attrib_dict)

        @SimilarityMixer.standard_range(0, 4)
        def excluding_pair_boolean_sim_generated(ex1, ex2):
            cost = 0
            antonym_pairs = []
            beam = Dict()  # addict Dict for nested dict

            fex1 = ex1["full_ex"]
            fex2 = ex2["full_ex"]

            for attr, single_attrib_dict in attrib_dict.items():
                comp_elements1 = [getattr(x, attr) for x in fex1]

                for key, item in single_attrib_dict.items():
                    if (match(key, comp_elements1)):
                            comp_elements2 = [getattr(x,attr) for x in fex2]
                            c, ap = match_comp_els(item, key, comp_elements1, comp_elements2)
                            cost += c
                            antonym_pairs.append(ap)
            beam[ex1['id']][ex2['id']].reason = [a for a in antonym_pairs if a]
            return cost, beam
        return excluding_pair_boolean_sim_generated




    def formula_concludes(fit_mix):
        @lru_cache(maxsize=None)
        @SimilarityMixer.standard_range(0, 4)
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

            key_to_key = fit_mix.choose((eL(keys1), eL(keys2)), out="ex", layout="1:1")

            if None in flatten(key_to_key[:]) or not key_to_key:
                logging.debug("No antonyms found!")
                return 0, {}

            key_char_to_key_char = {k1[0]["key"]: k2[0]["key"] for k1, k2 in key_to_key}

            cost = 0
            triggers = []

            for t in key_to_key:
                k1, k2 = t
                new_cost = SimilarityMixer.do_logic(
                    formulas = (f1, f2),
                    key_rel = {k1[0]['key']: k2[0]['key']},
                    negate_one=True)
                if new_cost:
                    t.reason = set(flatten([d['reason'] for d in t.trigger]))
                    triggers.append(t)
                cost += new_cost

            beam = Dict()
            if cost:
                reasons = set(flatten([d['reason'] for t in triggers for d in t.trigger]))
                beam[ex1['id']][ex2['id']].trigger = triggers
                beam[ex1['id']][ex2['id']].reason  = reasons
                beam[ex1['id']][ex2['id']].key_to_key = key_char_to_key_char
            return cost, beam
        return formula_prooves_generated

    def formula_excludes (fit_mix, symmetric=False):
        fit_keys = fit_mix
        @lru_cache(maxsize=None)
        @SimilarityMixer.standard_range(0, 4)
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

            key_to_key1 = fit_keys.choose((eL(keys1), eL(keys2)), out="ex", layout="1:1")
            key_to_key2 = fit_keys.choose((eL(keys2), eL(keys1)), out="ex", layout="1:1")

            key_to_key = key_to_key1 + key_to_key2

            if None in flatten(key_to_key[:]) or not key_to_key:
                logging.debug("empty negation result?")
                return 0, {}

            key_char_to_key_char = {k1[0]["key"]: k2[0]["key"] for k1, k2 in key_to_key}

            cost = 0
            triggers = []

            for t in key_to_key:
                k1, k2 = t
                new_cost = SimilarityMixer.do_logic(
                    formulas = (f1, f2),
                    key_rel = {k1[0]['key']: k2[0]['key']}
                )
                if new_cost:
                    t.reason = 'neg'
                    triggers.append(t)
                cost += new_cost

            beam = Dict()
            if cost :
                beam[ex1['id']][ex2['id']].trigger = triggers
                beam[ex1['id']][ex2['id']].reason  = ['negation']
                beam[ex1['id']][ex2['id']].key_to_key = key_char_to_key_char
            return cost, beam
        return formula_contradicts_generated


    key_regex = re.compile("pyprover\.logic\.Prop\(\\'([a-zA-Z0-9])+\\'\)")
    @classmethod
    def do_logic(cls, formulas=None, key_rel=None, negate_one=False):
        ''' This function does the logic and returns true, it can generate a contradiction

            It renames the keys in the formula: These keys, that don't belong to each other, get some other name.
            The relationated keys get the same key.

            If  negate_one` is set, then for one relationated key a negation is added. That is used, if the expression, that
            this key stands for contains a negation

            :param formulas:   formulas 1 and 2 as tuple
            :param key_rel:    the keys to combine
            :param negate_one: add a negation to one of the keys of key_rel
            :return: 1 for a contradiction, 0 for no one

        '''
        f1, f2 =  formulas

        k_in_1 = re.findall(SimilarityMixer.key_regex, f1)
        k_in_2 = re.findall(SimilarityMixer.key_regex, f2)

        k_to_correlate_1 = list(key_rel.keys())
        k_to_correlate_2 = list(key_rel.values())

        k_not_to_correlate_1 = list(set(k_in_1) - set(k_to_correlate_1))
        k_not_to_correlate_2 = list(set(k_in_2) - set(k_to_correlate_2))

        not_in_other_formula = eval(
            "(x for x in uppercase_bca)")  # `eval` because there is a complication,
                                           # that the generator is really created every call of this function
        in_other_formula = eval(
            "(x for x in uppercase_abc)")  # the same

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

        f = '(' + f1 + ') & (' + f2 + ')'

        cost = int(pyprover.proves(eval(f), pyprover.logic.false))
        return cost


    def write_to_graph(self, graph_coro, type, exs1, exs2, left_value, right_values):
        ''' Write a result to the graph using a coroutine

        :param graph_coro:
             coroutine to send a (dict, dict, type)-tuple to. This also can be a list con coroutines and the same
             arguments will be applied to them
        :param type: the type added to the pair of dicts
        :param exs1: expression list 1
        :param exs2: expression list 2
        :param left_value: list of left indices
        :param right_values: list of right indices
        :return: None

        '''
        if isinstance(graph_coro, list):
            for gc in graph_coro:
                if not gc == None:
                    self.write_to_graph(gc, type, exs1, exs2, left_value, right_values)
            return

        l_s = SimilarityMixer.expressions_list(self, left_value, right_values, exs1, exs2)
        t_s = SimilarityMixer.reduce_i_s_pair_tuples(l_s)

        if isinstance(exs1[0], dict) and isinstance(exs2[0], dict):
            all_triggers = flatten_list([self.beam[ex1['id']][ex2['id']]['trigger'] for ex1, ex2 in t_s])
            for trigger in all_triggers:
                trigger.type = type
                graph_coro.send(trigger)

        elif isinstance(exs1[0], tuple):
            (type_l_pair, type_r_pair, type_between) = type
            for trigger in t_s:
                try:
                    ex11 = trigger[0][0][0]
                    ex12 = trigger[0][1][0]
                    ex21 = trigger[1][0][0]
                    ex22 = trigger[1][1][0]

                    orig_l_edge = eT((ex11, ex12), type=type_l_pair)
                    orig_r_edge = eT((ex21, ex22), type=type_r_pair)
                    lr1_edge = eT((ex11, ex21), type=type_between)
                    lr2_edge = eT((ex12, ex22), type=type_between)

                    graph_coro.send(orig_l_edge)
                    graph_coro.send(orig_r_edge)
                    graph_coro.send(lr1_edge)
                    graph_coro.send(lr2_edge)
                except TypeError:
                    raise TypeError(
                        "one of the edges didn't have the list-tuple-dict-list-specification\n"
                        "types: \n%s,\n%s,\n%s,\n%s" %
                        (type_spec(orig_l_edge),
                         type_spec(orig_r_edge),
                         type_spec(lr1_edge),
                         type_spec(lr2_edge)))

    #@lru_cache(maxsize=None)
    @standard_range(0, 1000)
    def subj_asp_sim (ex1, ex2):
        beam = Dict()
        res = (ex1['subj_score']/ex2['subj_score']) + (ex2['aspe_score']/ex1['aspe_score'])
        beam[ex1['id']][ex2['id']].reason = res
        return res, beam

import unittest

class TestSimmix(unittest.TestCase):
    def test_one_to_one_simple(self):
        exs2 = [1, 2, 3]
        exs1 = [1]
        weighted_res = np.array([[0.], [0.], [0.2]])

        print(SimilarityMixer.one_to_one(weighted_res, exs1, exs2))
        print(SimilarityMixer.one_to_one(weighted_res.T, exs2, exs1))

        self.assertTrue (
                set(nested_list_tools.flatten(SimilarityMixer.one_to_one(weighted_res, exs1, exs2))) \
                == \
                set(nested_list_tools.flatten(SimilarityMixer.one_to_one(weighted_res.T, exs2, exs1))))


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

        print(SimilarityMixer.one_to_one(weighted_res, exs1, exs2))
        print(SimilarityMixer.one_to_one(weighted_res.T, exs2, exs1))


    def test_excluding_pair (self):
        import spacy
        nlp = spacy.load('en_core_web_sm')
        from littletools.dict_tools import balance_complex_tuple_dict
        import word_definitions

        s1 = nlp("On the other hand things are said to be named Univocally, which have both the name the definition answering to the name in common")
        s2 = nlp('Things are said to be named Equivocally when though they have a common name the definition corresponding with the name differs for each')
        p1 = {"full_ex":s1[13:25]}
        p2 = {"full_ex":s2[8:24]}


        d = {'differ': [('differ', 'in'), 'differ', ('have', '*', 'in', 'common')]}
        d =  balance_complex_tuple_dict(d)
        #fun = SimilarityMixer.detect_pair({'lemma_':d})
        fun = SimilarityMixer.detect_pair(word_definitions.antonym_dict)

        print (p1,p2, )
        print (fun(p2,p1))

        print (fun(p1,p2))
        self.assertTrue (fun(p2,p1)>0)
        self.assertTrue (fun(p1,p2)>0)


    def test_antonym_dict_for_symmetry(self):
        import word_definitions
        from littletools.dict_tools import dict_compare

        d1 = dict_tools.invert_dict(word_definitions.antonym_dict['lemma_'])
        d2 = word_definitions.antonym_dict['lemma_']

        added, removed, modified, same=\
            dict_compare(d1, d2, ignore_order=True)
        self.assertTrue (not added and not removed and not modified)

        if not  (not added and not removed and not modified):
            print (dict_compare(d1,d2, ignore_order=True))

    def test_antonym_dict_for_symmetry(self):
        import word_definitions

        def key_in_val(d):
            print ([(k,v)  for k,v in d.items() if k in v])
            return (any ([k in v  for k,v in d.items()]))
        self.assertFalse (key_in_val(word_definitions.antonym_dict['lemma_']))


if __name__ == '__main__':
    unittest.main()