from expression_views import *


from typing import Iterable
from itertools import combinations
from addict import Dict

from littletools.list_and_dict_type import L, T
import cytoolz

import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

from littletools.nested_list_tools import recursive_map, curry, flatten_reduce, flatten


import collections

class eD (collections.OrderedDict, forward_mapping_neo4j_view):
    ''' expression dictionary

    '''
    def __init__ (self, *args, **kwargs):
        super(eD, self).__init__(*args, **kwargs)


    def __str__(self):
        if 'text' in self:
            return "text={text} (id={id} s_id={s_id} i_s={i_s}) ".format(
                id=str(self['id']),
                s_id=str(self['s_id']),
                i_s=str(self['i_s']),
                text=' '.join(self['text']))
        else:
            return super.__str__(self)

    def __hash__(self):
        if 's_id' in self and 'i_s' in self:
            return hash(self['s_id']+str(self['i_s']))
        else:
            return hash(str(self))

    def set_property(self, prop, val):
        self.__setattr__(prop, val)
        return self


class eT (T, iterable_neo4j_view):
    ''' expression tuple

    '''
    def __new__(self, t, **kwargs):
        return super(eT, self).__new__(self, t, **kwargs)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self):
        return "(" +', '.join([str(e) for e in self]) + ')'

    def __hash__(self):
        return id(self)

    def unique (self):
        seen = set()
        res = eT(tuple([seen.add(hash(obj)) or obj for obj in self if hash(obj) not in seen]),  **self.__dict__)
        return res

    def set_property(self, prop, val):
        self.__setattr__(prop, val)
        return self


class eL (L, iterable_neo4j_view):
    ''' expression list

    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self):
        return "[" +'\n, '.join([str(e) for e in self]) + ']'

    def __hash__(self):
        return id(self)

    def unique (self):
        seen = set()
        try:
            res =  eL([seen.add(hash(obj)) or obj for obj in self if hash(obj) not in seen],  **self.__dict__)
        except:
            raise
        return res

    def set_property(self, prop, val):
        self.__setattr__(prop, val)
        return self


def ltd_ify (nltd, d=0, node_type=['NLP'], stack_types=[], d_max = 6):
    ''' Parse all list, tuple, dict types in nested expressions to these special list, tuple, dicts defined here

    Eample
    ======

    >>> ltd_ify([{'side': {'side': 1, 'p_id': [91, 50, 47]}}, {'side': {'side': 0, 'p_id': [15, 0, 4]}}, {'side': {'side': 5, 'p_id': [136, 140]}}])
    [{'side': {'side': 1, 'p_id': [91, 50, 47]}}, {'side': {'side': 0, 'p_id': [15, 0, 4]}}, {'side': {'side': 5, 'p_id': [136, 140]}}]

    :param nltd: nested lists, tuples, dicts within each other
    :return: typed nltd

    '''
    rest = []
    if len(stack_types)>1:
        stack_type, *rest = stack_types
        stack_type = [stack_type]
    else:
        stack_type=stack_types

    if d>d_max:
        return nltd

    if isinstance(nltd, tuple) or isinstance(nltd, list):
        if isinstance(nltd, tuple):
            if rest and isinstance(rest[0], tuple):
                tup, *rest_of_rest = rest
                res = eT(ltd_ify(x, d=d + 1, node_type=node_type, stack_types= [st] + rest_of_rest, d_max=d_max) if isinstance(x, Iterable) else x for x, st in zip(nltd, tup))
            else:
                res = eT(ltd_ify(x, d=d + 1, node_type=node_type, stack_types=rest, d_max=d_max) if isinstance(x, Iterable) else x for x in nltd)

        if  isinstance(nltd, list):
            if rest and isinstance(rest[0], tuple):
                try:
                    tup, *rest_of_rest = rest
                    res = eL(ltd_ify(x, d=d + 1, node_type=node_type, stack_types=[st] + rest_of_rest, d_max=d_max) if isinstance(x, Iterable) else x
                             for x, st in zip(nltd, tup))
                except:
                    raise
            else:
                res = eL(
                    ltd_ify(x, d=d + 1, node_type=node_type, stack_types=rest, d_max=d_max) if isinstance(x, Iterable) else x for x
                    in nltd)

        res.set_property('node_type', node_type + stack_type)
        return res

    elif isinstance(nltd, dict):
        res = (eD if not isinstance(nltd, eD) else type(nltd)) ({k:ltd_ify(x, d=d + 1, node_type=node_type, stack_types=rest, d_max=d_max) if isinstance(x, Iterable) else x for k, x in nltd.items()})
        if (hasattr(nltd, 'set_property')):
            res.set_property('node_type', node_type + stack_type)
        return res

    #elif isinstance(nltd, eD):
    #    res = nltd
    #    res.set_property('node_type', node_type + stack_type)
    #    return res

    return nltd


def apply_fun_to_attribute_of_ex(ex, fun, *args, attribute=None, other_criterium=None, reduce=False, **kwargs):
    ''' Take one value of the ex. It's 'currying' of the dict, forwarding the rest.
    It's good for projextions on the dicts.

    Example
    =======

    >>> apply_fun_to_attribute_of_ex({'side': 1, 'p_id': [91, 50, 47]},
    ...                 fun=str,
    ...                 attribute='p_id')
    {'side': 1, 'p_id': '[91, 50, 47]'}


    :param expression: predicate dict
    :param reduce: if the result should replace the whole dict
    :param **kwargs: parameters, like a function to be used with the ex
    :return: the same like the input, but the ex is limited to the attribute

    '''
    if isinstance(ex, dict) and attribute in ex:
        if reduce:
            return fun(ex[attribute], *args, **kwargs)
        else:
            ex.update({attribute: fun (ex[attribute], *args, **kwargs)})
    if other_criterium:
            return fun(ex, *args, **kwargs)
    return ex


def apply_fun_to_nested(fun=None, attribute=None, other_criterium=None, data=None, reduce=False):
    ''' This function runs through a nested structure of eD's, eL's and eT's, and allows to apply a function in the
        depth of this tree. It works like working in a forest, climbing on all the trees and doing there something
        if the property named by `attribute` applies.

        It's usefull, when it's complicated or annoying to write the looping yourself. It works recursively.

    :param fun:       function, that gets one argument (you can curry  the function, if you have multiple arguments there)
    :param attribute: the attribute in the tree
    :param data:      the nesting  eD's, eL's and eT's
    :param reduce:    if the attribute should be kept there or is just replaced by the return value of the function
    :return:  changed nested structure

    '''
    if not fun or not (attribute or other_criterium) or not data:
        raise ValueError ("All parameters must be given. {args}".format(args={'fun':fun, 'attribute':attribute, 'data': bool(data)}))
    return ltd_ify(recursive_map(curry(apply_fun_to_attribute_of_ex, attribute=attribute, fun=fun, reduce=reduce, other_criterium=other_criterium), data, other_criterium=other_criterium))



class PredMom(eD, pred_neo4j_view):
    def __init__ (self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node_type = ['PREDICATE']


    def __str__(self):
        if 'text' in self and 'part_predications' in self:
            return "{text} >>> {predicates}". format(
                text =  ' '.join(self['text']),
                predicates = str (self['part_predications']))
        elif 'text' in self:
            return ' '.join(self['text'])
        else:
            return super.__str__(self)


class Pred(pred_neo4j_view, eD):
    def __init__ (self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node_type = ['PREDICATE']


class Argu (argu_neo4j_view, eD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node_type = ['ARGUMENT']

    def __str__(self):
        return "{a} > sc={sc}, as={ac}".format(
            sc=self['subj_score'],
            ac=self['aspe_score'],
            a=super().__str__())
