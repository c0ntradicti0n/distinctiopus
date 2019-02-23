from addict import Dict
from littletools.list_and_dict_type import L, T
import cytoolz


# addict after possible bugfix
class HAE (dict):
    ''' "Hardcore annotated expression"

    '''
    def __init__ (self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_coreferenced(self):
        if 'coref' in self:
            pass


    def __str__(self):
        if 'text' in self:
            return ' '.join(self['text'])
        else:
            return super.__str__(self)



class PredMom(HAE):
    def __init__ (self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def __str__(self):
        if 'text' in self and 'part_predications' in self:
            return "{text}: {predicates}". format(
                text =  ' '.join(self['text']),
                predicates = str (self['part_predications']))
        elif 'text' in self:
            return ' '.join(self['text'])
        else:
            return super.__str__(self)


class Pred(HAE):
    def __init__ (self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Argu (HAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class HEAT (T):
    def __new__(self, t, **kwargs):
        return super(HEAT, self).__new__(self, t, **kwargs)


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def __str__(self):
        return "(" +', '.join([str(e) for e in self]) + ')'


class HEAL (L):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self):
        return "[" +', '.join([str(e) for e in self]) + ']'