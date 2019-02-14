#!/usr/bin/env python
# -*- coding: utf-8 -*-

from corpus_reader import CorpusReader
from CursorilyLogician import DataframeCursorilyLogician
from webanno_parser import Webanno_Parser

import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

import argparse
parser = argparse.ArgumentParser(description='Sokrates distinction tool')
parser.add_argument("-c","--conll", help="directory with conll-files to read corpus from")
parser.add_argument("-w","--webanno_tsv", help="webanno_tsv for testing results")
parser.add_argument("-o","--output", help="output file, image file ('svg', 'jpg') or 'json', resp. matplotlib draw(..)")
parser.add_argument("-wd","--work_dir", help="directory to write to, default is ./img", default='./img')
parser.add_argument("-r","--only", help="index numbers to test a piece of the corpus like '[7,8,9]", default = "None")

args = parser.parse_args()

def logging_setup():
    logging.basicConfig(
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M',
        filename='CursorilyLogician.log',
        filemode='w',
        level=logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    return None

def parse_webanno():
    grammarian = GrammarAnnotator(import_dir=args.conll)
    def get_tsv():
        import os
        for file in os.listdir(args.webannotsv):
            if file.endswith(".tsv"):
                yield (os.path.join(args.webannotsv, file))
    webanno_parsed = Webanno_Parser(next(get_tsv()))
    grammarian.annotate(webanno_parsed)
    return None



def main():
    logging_setup()
    if args.webanno_tsv:
        parse_webanno()

    corpus = CorpusReader(corpus_path=args.conll, only=eval(args.only))

    Logician = DataframeCursorilyLogician(corpus)
    Logician.annotate_horizon(horizon=3)
    Logician.annotate_predicates()
    Logician.annotate_contradictions()
    Logician.annotate_correlations()
    Logician.annotate_subjects_and_aspects()

    Logician.move_labels()
    df = Logician.query_distinctions()
    print(df.to_excel(args.conll+"/../output.xlsx"))

    #Logician.draw_as_dot_digraph(digraph, args.work_dir + "/found_distinctions." + args.output)
    return 0

if __name__ == '__main__':

    import cProfile
    import pstats
    import io
    pr = cProfile.Profile()
    pr.enable()
    main()
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats()

    with open('stats/cprofile.txt', 'w+') as f:
        f.write(s.getvalue())
