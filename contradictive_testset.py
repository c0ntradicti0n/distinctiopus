from sklearn.metrics import f1_score
import numpy as np

import predicatrix
import tdfidf_tool
from contradictrix import find_contradictive
from predicatrix import collect_all_predicates

import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(filename='./tests/test_contradictions.log',level=logging.DEBUG)
logging.captureWarnings(True)


import spacy
nlp = spacy.load('en_core_web_sm')

path_rte = './tests/RTE2_test_negated_contradiction.xml'
def read_rte_testset(path_rte):
    import xml.etree.ElementTree as ET
    tree = ET.parse(path_rte)
    root = tree.getroot()

    corpus = ""
    testset = []
    for pair in root:
        exs1 = [nlp(pair.find('t').text)]
        exs2 = [nlp(pair.find('h').text)]
        corpus += " ".join([w.lemma_ for x in exs1 for w in x] + [w.lemma_ for x in exs2 for w in x])
        yn  = True if (pair.attrib['contradiction'] == 'YES') else False
        testset.append((exs1, exs2, yn))
    return corpus, testset


path_snli = 'tests/snli_1.0/snli_1.0_dev.jsonl'
def read_snli_testset(path_snli, max = None):
    import json
    data = []
    for line in open(path_snli, 'r'):
        data.append(json.loads(line))

    corpus = ""
    testset = []

    for i, json_dict in enumerate(data):
        exs1 = [nlp(json_dict['sentence1'])]
        exs2 = [nlp(json_dict['sentence2'])]
        corpus += " ".join([w.lemma_ for x in exs1 for w in x] + [w.lemma_ for x in exs2 for w in x])
        print  ("sample no %5i, %-5s" %
                 (i,
                  str(json_dict["annotator_labels"] == ['contradiction'] * 5)
                  )
                )
        yn = True if (json_dict["annotator_labels"] == ['contradiction'] * 5) else False
        testset.append((exs1, exs2, yn))
        if max and i>=max:
            break
    return corpus, testset

#corpus, testset = read_rte_testset(path_rte)
corpus, testset = read_snli_testset(path_snli, max = 450)



tdfidf = tdfidf_tool.tdfidf(corpus)

path_fp = "./tests/results/false_positive.txt"
path_tp = "./tests/results/true_positive.txt"
path_fn = "./tests/results/false_negative.txt"
path_tn = "./tests/results/true_negative.txt"
import os
os.system("rm  %s %s %s %s " %(path_fn,path_fp,path_tn,path_tp))

def result_to_file(path, contradictions, exs1, exs2):
    with open(path, mode="a") as fp:
        fp.write("\n\nFor Predications:\n")
        fp.write(" ".join ([x.text for x in  exs1]))
        fp.write('\nand\n')
        fp.write(" ".join([x.text for x in  exs2]))

        for p1, p2 in contradictions:
            fp.write ("\n\nContradiction between:\n")
            predicatrix.ps_to_file(fp, p1)
            predicatrix.ps_to_file(fp, p2)
    return None

def contradictive_filter(exs1, exs2):
    predicates1 = []
    predicates2 = []
    for ex1 in exs1:
        try:
            predicates1.extend(collect_all_predicates(ex1, tdfidf ))
        except Exception as e:
            if hasattr(e, 'message'):
                e.message += " Extracting predicates \n " + str(exs1)
            raise e
    for ex2 in exs2:
        try:
            predicates2.extend(collect_all_predicates(ex2, tdfidf ))
        except Exception as e:
            if hasattr(e, 'message'):
                e.message += " Extracting predicates \n " + str(exs2)
            raise e

    try:
        contradictions = find_contradictive(predicates1, predicates2)
    except Exception as e:
        if hasattr(e, 'message'):
            e.message += " Detecting contradiction, with these values: \n      " + str (predicates1) + "\n       " + str(predicates2)
        raise e

    return contradictions

y_true = []
y_pred = []
errors = []

for exs1,exs2, yn in testset:
    contradictions = contradictive_filter(exs1, exs2)
    found = (int (len (contradictions) > 0))

    # res to different files
    if       found and not yn:
            result_to_file(path_fp, contradictions, exs1, exs2)
    elif not found and     yn:
            result_to_file(path_fn, contradictions, exs1, exs2)
    elif     found and     yn:
            result_to_file(path_tp, contradictions, exs1, exs2)
    elif not found and not yn:
            result_to_file(path_tn, contradictions, exs1, exs2)
    y_pred.append(found)
    y_true.append(int (yn))

    print(y_true)
    print(y_pred)
    print("F1 score        =  % 6.5f" % f1_score(y_true, y_pred, average='macro'))
    y_random = np.random.randint(2, size=len(y_true))

    alle  = len (y_true)
    tp = np.sum(y_true)
    fp = alle-tp
    random = np.random.choice(np.arange(0, 2), p=[fp/alle, tp/alle])
    print("F1 score random =  % 6.5f" % f1_score(y_true, y_random, average='macro'))


### Result at end

print (y_true)
print (y_pred)
print (y_pred==y_true)
print (len(y_true))
print ("F1 score        =  ", f1_score(y_true, np.array(y_pred)>0,   average='macro') )

y_random = np.random.randint(2, size=len(y_true))
print ("F1 score random =  ", f1_score(y_true, y_random, average='macro') )

import pprint
pprint.pprint (errors)
