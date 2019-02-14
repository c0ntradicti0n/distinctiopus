#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gzip
import gensim 
import logging
import spacy
nlp = spacy.load('en_core_web_md')
import re
import sys


class Replacement(object):
    def __init__(self, dict, fun, replacement):
        self.replacement = replacement
        self.occurrences = []
        self.dict        = dict
        self.fun         = fun

    def __call__(self, match):
        global quotes
        matched = match.group(0)
        replaced = self.fun(match.expand(self.replacement))
        self.dict.update({matched:replaced})

        self.occurrences.append((matched, replaced))
        #print ("'"+replaced+"'")
        return replaced

quotes = {}
def clean_text(string):
    global quotes
    # Text cleaning
    #string = string.lower()                                                                  # big letters
    #string = re.sub(ur"([‚‘])",                           "'", string)
    string = re.sub(r"""e\.g\.""",                " exempli gratia ", string)
    string = re.sub(r"""([\d]+[\w]+.)""",                     "", string)

    string = re.sub(r"""(\{[\(\)A-Za-z,.;:\-\s"']*\})""", "", string)                         # original text annotations
    string = re.sub(r"""(\([\w‚‘\s,.;:"']+\))"""        , "", string)                         # text in parentheses
    string = re.sub(r"""\[([A-Za-z,.;:\-\s"']*)\]""",  "\\1", string)                         # own manipulation my inputs

    def modifier(s): return " " + s.replace(" ","").replace("-","").title().strip()
    replace_the_cites = Replacement (quotes, modifier, " \\2")                                # other language layer by quoting
    string = re.sub(r"""(?:^|\s)(["'])((?:\\.|[^\\])*?)(\1)""", 
                                               replace_the_cites, string)
    #for matched, replaced in replace_the_cites.occurrences:
    #    print (matched, '=>', replaced)
    string = re.sub("[:;]",                               ",", string)                        # punctuation of sentences
    string = string.replace("-","")                                                           # dashs 
    string = string.replace("—",", ") 
    
    string = " ".join(string.split())                                                         # all kinds of whitespaces, tabs --> one space
    return string

def read_input(input_file):
    """This method reads the input file which is in gzip format"""
    global quotes
    logging.info("reading file {0}...this may take a while".format(input_file))
    with open(input_file, 'r', encoding='utf-8') as infile:
        for i, line in enumerate(infile):
 
            if (i % 3000 == 0):
                logging.info("read {0} reviews".format(i))

            line = clean_text(line)
            yield line.split()

def train_model(documents, args, epochs):
    model = gensim.models.Word2Vec(
        documents,
        **args)
    model.train(documents, total_examples=len(documents), epochs=epochs)

    logging.info("gensim vocab  " + str(model.wv.vocab))
    return model

def test_phrase_similarity (model,  w1, w2, output = True, attribute=None):
    import time

    w1_l = []
    w2_l = []
    if attribute:
        w1_l = [getattr(x, attribute) for x in nlp(" ".join(w1))]
        w2_l = [getattr(x, attribute) for x in nlp(" ".join(w2))]
    else:
        w1_l = w1
        w2_l = w2

    #print (w1)
    #print (w2)

    logging.info("testing similarity for  " + str (w1)  + "  and  " + str(w2) +" : ")
    logging.info("lemmata               = " + str(w1_l) + "  and  " + str(w2_l))

    if output:
        print ("testing similarity for  " + str (w1)  + "  and  " + str(w2) +" : ")
        print ("                      = " + str(w1_l) + "  and  " + str(w2_l))


    start_time = time.time()

    try:
        res = model.n_similarity(w1_l, w2_l)

        logging.info("res                   = " + str (res))
        logging.info("                        --- %s seconds ---" % (time.time() - start_time))

        if output:
            print ("res                   = " + str (res))
            print ("                        --- %s seconds ---" % (time.time() - start_time))

        
        return res

    except KeyError as e:
        logging.warning( "word was "+ str (e))
        print ( "word was "+ str (e))

        return "KeyError"


def create_model_for_spacy_property (documents, output_prefix, attribute, gensim_args, gensim_epochs):
    documents = [x for x in list(documents) if x]
    docs = [nlp(" ".join(doc)) for doc in documents if doc] 
    #print (docs)
    documents = [[getattr(x, attribute) for x in doc] for doc in docs]
    logging.info('text lemmatized')

    #print (documents)
    model     = train_model(documents, gensim_args, gensim_epochs)
    logging.info('model trained')

    fname = './'+output_prefix+'-'+attribute+'-gensim.w2v'
    model.wv.save(fname)
    logging.info("model saved in " + fname) 

    vocab_file_name = './'+output_prefix+'-'+attribute+'-vocab.txt'
    with open(vocab_file_name,'w') as f:
        f.write('\n'.join(list(model.wv.vocab.keys())))
    logging.info("vocabulary saved in " + vocab_file_name) 

    return model

def main():
    global quotes
    logging.basicConfig(filename='./gensim_vectors.log', filemode='w', level=logging.DEBUG)
    logging.info('Started')

    input_file_name = "./aristotle-categories-edghill-spell-checked.txt"
    output_file_prefix = "aristotle"

    documents = list(read_input(input_file_name))
    logging.info('text read and cleaned')

    text_args = \
        {
        'size'       : 2000,
        'window'     : 10,
        'min_count'  : 1,
        'workers'    : 4
        }

    dep_args = \
        {
        'size'       : 50,
        'window'     : 3,
        'min_count'  : 1,
        'workers'    : 4
        }

    dep_model       = create_model_for_spacy_property (documents, output_file_prefix, 'dep_'  , dep_args, 3000)
    

    def test_lemmatized (w1, w2, output = True): 
        res = test_phrase_similarity(lemma_model,  w1, w2, output, attribute='lemma_')
        return res

    def test_dep_ (w1, w2, output=True): 
        res = test_phrase_similarity(dep_model, w1, w2, output, attribute=None)
        return res


    """
    lemma_model     = create_model_for_spacy_property (documents, output_file_prefix, 'lemma_', text_args,  dep_args, 260)


    w1 = ['animal']
    w2 = ['man']
    test_lemmatized(w1, w2)

    w1 = ['Presentinasubject']
    w2 = ['more']
    test_lemmatized(w1, w2)
    w1 = ['most']
    w2 = ['more']
    test_lemmatized(w1, w2)
    w1 = ['beautiful']
    w2 = ['less']
    test_lemmatized(w1, w2)
    w1 = ['I']
    w2 = ['you']
    test_lemmatized(w1, w2)
    w1 = ['Presentinasubject']
    w2 = ['you']
    test_lemmatized(w1, w2)    
    w1 = ['definition']
    w2 = ['name']
    test_lemmatized(w1, w2)
    w1 = ['more']
    w2 = ['less']
    test_lemmatized(w1, w2)
    w1 = ['more']
    w2 = ['small']
    test_lemmatized(w1, w2)
    w1 = ['more']
    w2 = ['quantity']
    test_lemmatized(w1, w2)
    w1 = ['the', 'definition', 'be', 'the', 'same']
    w2 = ['the', 'definition', 'be', 'equal', 'and']
    test_lemmatized(w1, w2)
    w1 = ['the', 'definition', 'be', 'the', 'same']
    w2 = ['the', 'quantity', 'have', 'nothing', 'and']
    test_lemmatized(w1, w2)
    w1 = ['the', 'definition', 'be', 'the', 'same']
    w2 = ['the', 'quantity', 'have', 'nothing', 'and']
    test_lemmatized(w1, w2)
    """
    

    w1 = [ 'advmod', 'punct', 'mark', 'nsubj', 'advcl', 'det', 'amod', 'dobj', 'punct']
    w2 = [ 'advcl', 'preconj', 'det', 'nsubj', 'cc', 'det', 'conj', 'dobj', 'prep', 'det', 'pobj', 'prep', 'amod', 'punct']
    test_dep_(w1,w2)

    w1 = ['advmod']
    w2 = ['punct']
    test_dep_(w1,w2)

    w1 = ['nsubj']
    w2 = ['nsubjpass']
    test_dep_(w1,w2)

    dep_vocab = list([*dep_model.wv.vocab])
    print (dep_vocab)

 
    import pandas as pd
    dict =  {y: [test_dep_ ([x], [y], output=False) for x in dep_vocab] for y in dep_vocab}
    #print (dict)
    df = pd.DataFrame(dict).round(2)
    df.index = dep_vocab

    import numpy as np
    np.fill_diagonal(df.values, 0)
    df = df.ix[:, df.max().sort_values(ascending=False).index]

    print (df) 


    from pandas.plotting import scatter_matrix
    import matplotlib.pyplot as plt

    plt.figure(); 
    scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')
    df.plot();
    plt.show()


    writer = pd.ExcelWriter('dep_.xlsx')
    df.to_excel(writer,'Sheet1')
    writer.save()

    logging.info('Finished')


if __name__ == '__main__':
    main()
