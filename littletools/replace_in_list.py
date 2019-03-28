import collections

lists = [[1,2,3,4,5,6,7,8,9,10],[2,3,4,5,6,7],[4,5,6], [2,3]]

def flatten(iterable):
    for el in iterable:
        if isinstance(el, collections.Iterable) and not isinstance(el, str):
            yield from flatten(el)
        else:
            yield el

def replacer(lst, old, new):
    if not lst or not old or not new:
        return None
    def match (i):
        falsificate = True
        k = 1
        for j, o in enumerate(old[1:]):
            if not lst[i+j+1]==o:
                falsificate =  False
                break
        return falsificate

    for i, el in enumerate(lst):
        if el == old[0]:
            if not match(i):
               continue
            else:
               return lst[:i] + new + lst[i + len(old):]
    return None

def replace_matroshka (lists):
    lists = sorted(lists, key=lambda x: len(x))
    res = lists[0]
    for w in lists [1:]:
        #print (w)
        r = replacer(w, list(flatten(res[:])), [res])
        if not r:
            res.append(w)
        else:
            res = r
    return res

r = replace_matroshka(lists)
print (r)





def replace(sequence, replacement, lst, expand=False):
    out = list(lst)
    for i, e in enumerate(lst):
        if e == sequence[0]:
            i1 = i
            f = 1
            for e1, e2 in zip(sequence, lst[i:]):
                if e1 != e2:
                    f = 0
                    break
                i1 += 1
            if f == 1:
                del out[i:i1]
                if expand:
                    for x in list(replacement):
                        out.insert(i, x)
                else:
                    out.insert(i, replacement)
    return out

print (replace([1,2,3,4,5,6,7,8,9], [4,5,6,7], [4,5,6,7]))

def replace_matroshka (lists):
    lists = sorted(lists, key=lambda x: len(x))
    res = lists[0]
    for w in lists [1:]:
        print (w)
        #r = replacer(w, list(flatten(res[:])), [res])
        r = replace(list(flatten(res[:])), res, w)
        if not r:
            res.append(w)
        else:
            res = r
    return res

r = replace_matroshka(lists)
print (r)
