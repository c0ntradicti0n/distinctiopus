import collections



def invert_dict(d):
    ''' Get the anti-dict, where the keys become the values and the values become the keys.

        Example
        -------
            >>> from dict_tools import *
            >>> d = {
            ...             1:[1,2,3],
            ...             6:[3,4,1]
            ...     }
            >>> invert_dict(d)
            {1: [1, 6], 2: [1], 3: [1, 6], 4: [6]}

        :param d: dict of keyable values (also with lists)
        :return: inverted dict

    '''
    inverse = dict()
    for key in d:
        if isinstance(d[key], list):
            # Go through until a list in the dict:
            for item in d[key]:
                # Check if in the inverted dict the key exists
                if item not in inverse:
                    # If not create a new list
                    inverse[item] = [key]
                else:
                    inverse[item].append(key)
    return inverse



def balance_complex_tuple_dict(d, _sort=False):
    ''' Balancing a complex dict, meaning, that the keys and tghe values point transitive to another.

        The dict can have strings, lists of strings and tuples of strings as keys. The lists are split up
        when rearranging the keys and combined, if packed to a value-of the balanced dict.


        Example
        -------
            >>> from dict_tools import *
            >>> d = {
            ...             ('differ'):['equal'],
            ...             ('have','*','in', 'common'):[('differ','in'), 'differ', 'derive']
            ...     }
            >>> balance_complex_tuple_dict(dict(d), _sort=True)
            {'differ': [('have', '*', 'in', 'common'), 'equal'], ('have', '*', 'in', 'common'): [('differ', 'in'), 'differ', 'derive'], 'equal': ['differ'], ('differ', 'in'): [('have', '*', 'in', 'common')], 'derive': [('have', '*', 'in', 'common')]}

        :param d:
            dict with strings and lists and lists of tuples in it, that will be balanced
        :return:
    '''
    inverse = d.copy()
    for key in d:
        # Go through the list that is saved in the dict:
        for item in d[key]:
            # Check if in the inverted dict the key exists
            if item not in inverse:
                # If not create a new list
                inverse[item] = [key]
            else:
                if (item == key):
                    continue
                inverse[item].append(key)
        if _sort:
            for v in d.values():
                if isinstance(v, list):
                    v = sorted(v, key=lambda x: hash(str(x)))

    for key, val in inverse.items():
        inverse[key] = list(set(val))
    return inverse


def dict_compare(d1, d2, ignore_order=False):
    ''' This function compares two dicts.

        Example
        -------

        >>> d1 = dict(zip(range(0,10), [list(range(x-1,x+2)) for x in range(10,20)]))
        >>> d2 = dict(zip(range(1,9), [list(range(x+2,x-1,-1)) for x in range(10,20)]))
        >>> d1
        {0: [9, 10, 11], 1: [10, 11, 12], 2: [11, 12, 13], 3: [12, 13, 14], 4: [13, 14, 15], 5: [14, 15, 16], 6: [15, 16, 17], 7: [16, 17, 18], 8: [17, 18, 19], 9: [18, 19, 20]}
        >>> d2
        {1: [12, 11, 10], 2: [13, 12, 11], 3: [14, 13, 12], 4: [15, 14, 13], 5: [16, 15, 14], 6: [17, 16, 15], 7: [18, 17, 16], 8: [19, 18, 17]}
        >>> dict_compare(d1,d2)
        ({0, 9}, set(), {1: ([10, 11, 12], [12, 11, 10]), 2: ([11, 12, 13], [13, 12, 11]), 3: ([12, 13, 14], [14, 13, 12]), 4: ([13, 14, 15], [15, 14, 13]), 5: ([14, 15, 16], [16, 15, 14]), 6: ([15, 16, 17], [17, 16, 15]), 7: ([16, 17, 18], [18, 17, 16]), 8: ([17, 18, 19], [19, 18, 17])}, set())

    :param d1: dict
    :param d2: dict
    :param ignore_order: ignore the order of the keys
    :return:
         4-tuple of set: added, removed, modified, same

    '''
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    intersect_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys
    modified = {o : (d1[o], d2[o]) for o in intersect_keys if d1[o] != d2[o]}
    if ignore_order:
        modified = {o : (d1[o], d2[o]) for o in intersect_keys
                    if set(d1[o]) != set(d2[o])}

    same = set(o for o in intersect_keys if d1[o] == d2[o])
    return added, removed, modified, same


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


if __name__ == "__main__":
    import doctest
    doctest.testmod()
