"""
This module computes some 'importance' of words for a corpus with the tdf-idf-measure as a whole document.

>>> test = tdfidf('This seems to be a document about Abraham. this seems to be another document Abraham. that seems not to be a document about Abraham. An this seems to be just shit Abrahamn Obrobobom Abraham Abraham.')

>>> print (test.get_vector('this'))
0.29559878344928797

>>> print (test.importance_of_word('this'))
0.7044012165507121

>>> print (test.sentence2vec(["this","seems","not"]))
[0.70440122 0.60586829 0.90146707]
"""


import numpy as np

import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

class tdfidf:
    def __init__(self, corpus_lemmata):
        from sklearn.feature_extraction.text import TfidfTransformer
        transformer = TfidfTransformer(smooth_idf=False)
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform([corpus_lemmata])
        counts = X.toarray()
        self.tdf_idf = transformer.fit_transform(counts)
        self.tdf_idf_dict = {v:i for i, v in enumerate(vectorizer.get_feature_names())}

    def get_vector(self, word):
        return self.tdf_idf[0,self.tdf_idf_dict[word]]

    def importance_of_word(self, word):
        try:
            return 1- self.tdf_idf[0,self.tdf_idf_dict[word]]
        except KeyError:
            if not (word.isdigit()) and (word not in ['.',',','?','!',':',';']):
                logging.warning ("'%s' not in tdfidf-vocabulary." % word)
            return 0.2

    def sentence2vec(self, str_list):
        return np.array([self.importance_of_word(w) for w in str_list]) ** 3

    def sentence2relevantwords(self, str_list, min_thr, max_thr):
        return [w
                for w in str_list
                if self.importance_of_word(w) > min_thr
                    and self.importance_of_word(w) < max_thr]


    def zero_importance(self, str_list):
        return np.array([0 for w in str_list])



if __name__ == "__main__":
    import doctest
    doctest.testmod()


