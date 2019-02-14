class HardcoreExpression (dict):
    def __init__ (self, d=None):
        s_id
        i_s
        text
        spacy_tokens
        doc
        dep_
        dep
        pos_
        pos
        tag_
        tag
        lemma_
        lemma
        importance
        elmo_origninal
        elmo_sum

    def get_coreferenced(self):

    def __str__(self):
        return ...



class PredicateMother(HardcoreExpression)
    def __init__ (self, d=None):
        sub_preds

class Predicate(HardcoreExpression):
    def __init__ (self, d=None):
        arguments

class Argument (HardcoreExpression):
    def __init__ (self, d=None):

class HardcoreTuple (tuple):

class HardcoreList (list):