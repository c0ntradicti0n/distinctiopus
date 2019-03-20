import itertools

from more_itertools import replace
import collections
import functools
import operator

def existent(iterable):
    for element in iterable:
        if element:
            yield element

def flatten(iterable):
    for el in iterable:
        if isinstance(el, collections.Iterable) and not isinstance(el, str):
            yield from flatten(el)
        else:
            yield el

def flatten_reduce(iterable):
    if not iterable:
        return []
    try:
        if not isinstance(iterable[0], dict):
            return functools.reduce(operator.add, iterable)
        else:
            return iterable
    except IndexError:
        raise ValueError("Value empty")
    except KeyError:
        raise ValueError("Value is a dict, list expected!")
    except ValueError:
        raise
    except TypeError:
        raise

def collapse(lst):
    for x in lst:
        if isinstance(x, collections.Iterable) and not isinstance(x, str):
            for y in flatten(x):
                yield y
        else:
            yield x

def flatten_list(l):
    for el in l:
        if isinstance(el, list):
            yield from flatten_list(el)
        else:
            yield el


def collect_gen (it, key):
    for i in (it):
        yield i[key]


def collect (it, key):
    return set(list(collect_gen(it, key)))


from itertools import tee

def pairwise(iterable):
    """ Iterate on pairs through the list, return iterator
        s -> (s0,s1), (s1,s2), (s2, s3),

        >>> list(pairwise([1,2,3,4]))
        [(1, 2), (2, 3), (3, 4)]

        :param: iterator/sequence
        :return: pair iterator

    """
    # From itertools recipes. now it's heavily documented
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def curry(function, *args, **kwargs):
    ''' Apply arguments to a function, but leave the last one out

    :param function: function with multiple arguments
    :param args: normal arguments
    :param kwargs: keyword arguments
    :return: fun with one argument

    '''
    def fun_new(x):
        return function (x, *args, **kwargs)
    return fun_new


def on_each_level (lol, fun, out = []):
    ''' If you have a nested list and you want to apply a function on the elements of the list, you have to go through
    all the levels, this functions does this recursively and applys the function

    :param lol: nested list
    :param fun: function with one argument
    :param out: output list, default (for starting) = []
    :return:

    '''
    for x in lol:
        if not isinstance(x, list):
            out.append(fun(x))
        else:
            out.append(on_each_level(x, fun, []))
    return out


from collections import Sequence

def recursive_map (func, seq, d=0, x_max_d=5, other_criterium=None):
    if d>= x_max_d:
        return seq

    if other_criterium:
        if other_criterium(seq):
            #seq = type(seq)((k, recursive_map(func, v, d=d+1, x_max_d=x_max_d, other_criterium=other_criterium)) for k, v in seq.items())
            res = func(seq)
            return res

    if isinstance(seq, dict) and not other_criterium:
        seq = type(seq)((k, recursive_map(func, v, d=d+1, x_max_d=x_max_d, other_criterium=other_criterium)) for k, v in seq.items())
        res = func(seq)
        return res

    elif isinstance(seq, Sequence):
        res = []
        for item in seq:
            if isinstance(item, Sequence):
                res.append(recursive_map(func, item, d=d+1, x_max_d=x_max_d, other_criterium=other_criterium))
            elif isinstance(item, dict):
                res.append(recursive_map(func, item, d=d+1, x_max_d=x_max_d, other_criterium=other_criterium))
            else:
                res.append(func(item))
        return res

    else:
        return seq

def ret (seq):
    yield seq


def flat_on_level(it, d=-1, level=None):
    """

    >>> list(flat_on_level([[[['a']]]], level=3))
    ['a']

    """
    if isinstance(it, dict):
        y = it.values()
    else:
        y = it
    if not isinstance(y, collections.Iterable):
        yield y
    else:
        for x in y:
            if d == level:
                yield x
            else:
                yield from flat_on_level(x, d=d + 1, level=level)


def flatt_on_level(it, d=-1, level=None):
    """
    >>> list(flatt_on_level([[[['a']]]], level=3))
    [['a']]

    """
    if d == -1:
        return list(flatt_on_level(it, d=d + 1, level=level))

    if d == level:
        return (i for i in [it])

    res = []
    for x in it:
        res.extend( flatt_on_level(x, d=d+1, level=level))
    return res



def a (l, level, d):
    """
    >>> flat2l([[[['a']]]], 2, 0)

    :param l:
    :param level:
    :param d:
    :return:
    """
    if d==level:
        return d
    for x in flat2l(l, level=level, d=d+1):
        yield x


def a(x):
    """
    >>> list(a([[[1],[0]]]))
    [[[1, 1, 2]]]

    :param x:
    :return:
    """
    for r in x:
        yield from b(r)

def b(x):
    """
    >>> list(b([1,1,2]))
    [1, 1, 3]

    :param x:
    :return:
    """
    for r in x:
        yield from c(r)

def c(x):
    return (z for z in [x])




def on_each_iterable (lol, fun):
    out = []
    if isinstance(lol, collections.Iterable) and not isinstance(lol, str):
        for x in lol:
            out.append(on_each_iterable(x, fun))
        out = fun(out)
    else:
        out = lol
    return out


def stack_matryoshka(list_to_nest):
    ''' Put lists, that fit into the elements of a list together.

    Example
    -------

    >>> l = [[4],
    ... [3,4,5],
    ... [2,3,4,5,6],
    ... [1,2,3,4,5,6,7]]
    >>> stack_matryoshka(l)
    [1, [2, [3, [4], 5], 6], 7]

    But remember, graphs are the most time better solutions to this or to build a string and parse it as python

    :param nesting_list: lists, that may fit together, but if you want to get a good result, you should give a list
    with all elements in it.
    :return: nested list

    '''
    list_to_nest = sorted(list_to_nest, key=lambda x: len(x))
    n = 0
    while n < (len(list_to_nest) - 1):
        to_fit_there = list_to_nest[n]
        flatted_to_fit_there = list(flatten(to_fit_there[:]))

        def is_fitting(*xs):
            flatted_compared = list(flatten(xs[:]))
            if type(flatted_compared[-1]) == object:
                return False
            try:
                decision = flatted_compared == list(flatted_to_fit_there)
            except TypeError:
                return False
            return decision

        for m in range(n + 1, len(list_to_nest)):
            through = list(list_to_nest[m])

            def replacing_fun(x):
                return list(replace(list(x), is_fitting, [to_fit_there], window_size=len(to_fit_there)))

            list_to_nest[m] = on_each_iterable(through, replacing_fun)

        n = n + 1
    return (list_to_nest[-1])


def replace_pattern(lst, pattern_sequence, replacement, expand=False):
    out = lst[:]
    len_difference = 0
    for i, e in enumerate(lst):

        if pattern_sequence[0](e):
            i1 = i
            f = 1
            if len(lst) - i < len(pattern_sequence):
                f = 0
            for fun_e1, e2 in zip(pattern_sequence, lst[i:]):
                if not fun_e1(e2):
                    f = 0
                    break
                i1 += 1
            if f == 1:
                del out[i+ +len_difference : i1  + len_difference]
                if expand:
                    for x in list(replacement):
                        out.insert(i + len_difference, x)
                else:
                    for j,x in enumerate(list(replacement)):
                        if '\\' in x:
                            n = int(x[1])
                            out.insert(i+j + len_difference, lst[i+n])
                        else:
                            out.insert(i+j + len_difference, x)
                    len_difference += len(replacement) - len(pattern_sequence)

    return out

def check_for_tuple_in_list (l, t, wildcard ='*'):
    found = False
    suspend = False
    for i1, e1 in enumerate(l):
        i2 = 0
        if i2 == len(t):
            break
        e2 = t[i2]

        while i1 < len(l):
            found = False
            if e1 == e2:
                found = True
                i2+=1
                if i2 == len(t):
                    return True
                e2 = t[i2]
                i1+=1
                if i1 == len(l):
                    return False
                e1 = l[i1]
                #if found == False:
                #    #raise Warning("Python Bug?")
                continue
            elif e2 == wildcard:
                i2 += 1
                if i2 == len(t):
                    return True
                e2 = t[i2]
                found = True
                suspend = True
                continue
            elif suspend:
                i1 += 1
                if i1 == len(l):
                    return False
                e1 = l[i1]
            else:
                found = False
                break
        else:
            return False
    return found


def guess_seq_len(seq):
    guess = 1
    max_len = int(len(seq) / 2)
    for x in range(2, max_len):
        if seq[0:x] == seq[x:2*x] :
            return x
    return guess


from suffix_trees import STree
def occurrence_of_string_sequence (strings, min_len):
    st = STree.STree(strings)
    longest = st.lcs()
    if len(longest) >= min_len:
        occurrences = st.find_all(longest)
        return len(occurrences)
    else:
        return 0


_iterable_cache_ = []
def check_for_hash_teration(obj):
    #global _iterable_cache_
    #_iterable_cache_ += [str(hash(str(obj)))]
    #if occurrence_of_string_sequence(_iterable_cache_, 16) > 10:
    #    _iterable_cache_ = []
    #    return "<***>"
    return False

maximal_depth = 20
def type_spec_iterable(obj, name, depth, max_depth=maximal_depth):
    depth += 1
    if depth > max_depth:
        return "?*"

    check = check_for_hash_teration(obj)
    if check:
        return check

    tps = set(type_spec(e, depth) for e in obj)
    if len(tps) == 1:
        return name + "<" + next(iter(tps)) + ">"
    else:
        return name + "<...>"


def type_spec_dict(obj, depth, max_depth=maximal_depth):
    depth += 1
    if depth > max_depth:
        return "?*"

    check = check_for_hash_teration(obj)
    if check:
        return check

    tps = set((type_spec(k, depth), type_spec(v, depth)) for (k,v) in obj.items())
    keytypes = set(k for (k, v) in tps)
    valtypes =  set(v for (k, v) in tps)
    kt = next(iter(keytypes)) if len(keytypes) == 1 else "..."
    vt = next(iter(valtypes)) if len(valtypes) == 1 else "..."
    return "dict<%s, %s>" % (kt, vt)


def type_spec_tuple(obj, depth, max_depth=maximal_depth):
    depth += 1
    if depth > max_depth:
        return "?*"

    check = check_for_hash_teration(obj)
    if check:
        return check
    return "tuple<" + ", ".join(type_spec(e, depth) for e in obj) + ">"


def type_spec(obj, depth = 0, max_depth=maximal_depth):
    depth += 1
    check = check_for_hash_teration(obj)
    if check:
        return check
    if depth > max_depth:
        return "?*"

    t = type(obj)
    res = {
        int: "int",
        str: "str",
        bool: "bool",
        float: "float",
        type(None): "(none)",
        list: lambda o: type_spec_iterable(o, 'list', depth),
        set: lambda o: type_spec_iterable(o, 'set', depth),
        dict: lambda o: type_spec_dict(o, depth),
        tuple: lambda o: type_spec_tuple(o, depth),
    }.get(t, lambda o: type(o).__name__)
    return res if type(res) is str else res(obj)


def partition(items, predicate=bool):
    a, b = tee((predicate(item), item) for item in items)
    return ((item for pred, item in a if not pred),
            (item for pred, item in b if pred))

########################################

import unittest

class TestNLT(unittest.TestCase):
    def test_check_for_tuple_in_list(self):
        l  = [1,2,3,4,5,6,7,8,9]
        t1 = (1,2,3)
        t2 = (1,2,4)
        t3 = (8,9,10)
        t4 = (7,8,9)
        t5 = (1,2,3,4,5,6,7,8,9,0)

        t6 = (1,'*',3)

        t7 = (1,'*',4)
        t8 = (1,'*',9)
        t9 = (6,'*',9)
        t10= (6,'*',10)
        t11= (6,'*',7)
        t12 = (1,2)


        self.assertTrue (True  == check_for_tuple_in_list(l,t1))
        self.assertTrue (False == check_for_tuple_in_list(l,t2))
        self.assertTrue (False == check_for_tuple_in_list(l,t3))
        self.assertTrue (True  == check_for_tuple_in_list(l,t4))
        self.assertTrue (False == check_for_tuple_in_list(l,t5))
        self.assertTrue (True  == check_for_tuple_in_list(l,t6))
        self.assertTrue (True  == check_for_tuple_in_list(l,t7))
        self.assertTrue (True  == check_for_tuple_in_list(l,t8))
        self.assertTrue (True  == check_for_tuple_in_list(l,t9))
        self.assertTrue (False == check_for_tuple_in_list(l,t10))
        self.assertTrue (True  == check_for_tuple_in_list(l,t11))
        self.assertTrue (True  == check_for_tuple_in_list(l,t12))


        t12 = 'derive'
        l = ['have', 'both', 'the', 'name', 'the', 'definition', 'answer', 'to', 'the', 'name', 'in', 'common']
        self.assertTrue (False  == check_for_tuple_in_list(l,t12))



if __name__ == '__main__':
    unittest.main()
    import doctest

    doctest.testmod()