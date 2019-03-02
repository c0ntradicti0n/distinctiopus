import itertools

def count_up():
    i = 0
    while (True):
        yield i
        i += 1


def generate_new_string():
    """ Get some lowercase strings for using them as variable names

    Example
    =======

    >>> g = generate_new_string()
    >>> [next(g) for x in range(1,30)]
    ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'aa', 'ab', 'ac', 'ad', 'ae', 'af']

    :yield: next name, will be also a string, but another
    """
    for r in range (1,100):
        for string in map(''.join, itertools.product('abcdefghijklmnopqrstuvw', repeat=r)):
            yield string




