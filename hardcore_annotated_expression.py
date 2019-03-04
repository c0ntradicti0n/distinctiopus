from typing import Iterable
from itertools import combinations

from addict import Dict

from littletools.generator_tools import generate_new_string
from littletools.list_and_dict_type import L, T
import cytoolz

import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())


# addict after possible bugfix
from littletools.nested_list_tools import recursive_map, curry, flatten_reduce, flatten

variable_generator = generate_new_string()




class eD (dict):
    ''' expression dictionary

    '''
    def __init__ (self, *args, **kwargs):
        super().__init__(*args, **kwargs)


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
            return hash(self.__str__(self))


    def set_property(self, prop, val):
        self.__setattr__(prop, val)
        return self


    def neo4j_write(self):
        global variable_generator
        self.neo4j_name=next(variable_generator)
        return \
            """MERGE ({my_name}:{node_type} {{s_id:{s_id1}, text:'{text1}'}})\n""" \
            """ON CREATE SET {my_name}.id={id1}""".format(
                my_name=self.neo4j_name,
                node_type=':'.join(self.node_type),
                id1   = self['id'],
                s_id1 = self['s_id'],
                i_s1  = self['i_s'],
                text1 = " ".join(self['text']).replace("'", "")
                )



class iterable_neo4j_view:
    def __init__(self):
        pass

    def neo4j_write(self):
        global variable_generator
        self.neo4j_name = next(variable_generator)
        childrens_births = list(flatten([n.neo4j_write() for n in self]))
        names = [n.neo4j_name for n in self]

        if hasattr(self, 'type'):
            type = self.type
        else:
            type = 'unknown'
        if hasattr(self, 'reason'):
            reason = self.reason
        else:
            reason = 'unknown'

        create_me = """MERGE ({my_name}:{node_type}:{utype} {{AbcName:'{my_name}'}})""".format(
            my_name=self.neo4j_name,
            node_type=':'.join(self.node_type),
            utype=type.upper())

        return \
           childrens_births + \
           [create_me] + \
           ["""MERGE ({my_name})-[:NAMELY {{SpecialKind:'{type}', Reason:'{reason}'}}]->({x})""".format(
                my_name=self.neo4j_name,
                x=x, type=type,
                utype=type.upper(),
                reason=reason)
           for x in names] + \
           ["""MERGE ({y})-[:GROUP {{SpecialKind:'{type}', Reason:'{reason}', group:'{my_name}'}}]->({x})""".format(
               my_name=self.neo4j_name,
               x=x,
               y=y,
               type=type,
               utype=type.upper(),
               reason=reason)
           for x, y in  combinations(names, 2)]
class eT (T, iterable_neo4j_view):
    ''' expression tuple

    '''
    def __new__(self, t, **kwargs):
        return super(eT, self).__new__(self, t, **kwargs)


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def __str__(self):
        return "(" +', '.join([str(e) for e in self]) + ')'


    def unique (self):
        try:
            return eL(set(self), **self.__dict__)
        except ValueError:
            logging.error ("eT, set bug, set throws {ValueError}The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()")
            return self


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
        try:
            return eL(set(self), **self.__dict__)
        except ValueError:
            logging.error ("eL, set bug, set throws {ValueError}The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()")
            return self


    def set_property(self, prop, val):
        self.__setattr__(prop, val)
        return self


def ltd_ify (nltd, d=0, node_type=['NLP']):
    ''' Parse all list, tuple, dict types in nested expressions to these special list, tuple, dicts defined here

    Eample
    ======

    >>> ltd_ify([{'side': {'side': 1, 'p_id': [91, 50, 47]}}, {'side': {'side': 0, 'p_id': [15, 0, 4]}}, {'side': {'side': 5, 'p_id': [136, 140]}}])
    [{'side': {'side': 1, 'p_id': [91, 50, 47]}}, {'side': {'side': 0, 'p_id': [15, 0, 4]}}, {'side': {'side': 5, 'p_id': [136, 140]}}]

    :param nltd: nested lists, tuples, dicts within each other
    :return: typed nltd

    '''
    if d>6:
        return nltd

    if isinstance(nltd, tuple):
        res = eT(ltd_ify(x, d=d + 1, node_type=node_type) if isinstance(x, Iterable) else x for x in nltd)
        res.set_property('node_type', node_type)
        return res

    elif isinstance(nltd, list):
        res = eL(ltd_ify(x, d=d + 1, node_type=node_type) if isinstance(x, Iterable) else x for x in nltd)
        res.set_property('node_type', node_type)
        return res

    elif isinstance(nltd, dict) and not isinstance(nltd, eD):
        res = eD({k:ltd_ify(x, d=d + 1, node_type=node_type) if isinstance(x, Iterable) else x for k, x in nltd.items()})
        res.set_property('node_type', node_type)
        return res

    elif isinstance(nltd, eD):
        res = nltd
        res.set_property('node_type', node_type)
        return res

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



class PredMom(eD):
    def __init__ (self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def __str__(self):
        if 'text' in self and 'part_predications' in self:
            return "{text} >>> {predicates}". format(
                text =  ' '.join(self['text']),
                predicates = str (self['part_predications']))
        elif 'text' in self:
            return ' '.join(self['text'])
        else:
            return super.__str__(self)


class Pred(eD):
    def __init__ (self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Argu (eD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self):
        return "{a} > sc={sc}, as={ac}".format(
            sc=self['subj_score'],
            ac=self['aspe_score'],
            a=super().__str__())
