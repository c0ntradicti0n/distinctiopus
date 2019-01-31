#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import logging
import os
import pandas as pd
import collections
import pickle


import spacy
nlp = spacy.load('en_core_web_sm')

quotes = {}
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
        return replaced


def split_data_frame_list(df, target_column):
    """
    Splits a column with lists into rows
    
    Keyword arguments:
        df -- dataframe
        target_column -- name of column that contains lists        
    """
    
    # create a new dataframe with each item in a seperate column, dropping rows with missing values
    col_df = pd.DataFrame(df[target_column].dropna().tolist(),index=df[target_column].dropna().index)

    # create a series with columns stacked as rows         
    stacked = col_df.stack()

    # rename last column to 'idx'
    index = stacked.index.rename(names="idx", level=-1)
    new_df = pd.DataFrame(stacked, index=index, columns=[target_column])
    return new_df


def split_data_frame_list_beware(df, 
                       target_column):
    ''' 
    Accepts a column with multiple types and splits list variables to several rows.

    df: dataframe to split
    target_column: the column containing the values to split
    output_type: type of all outputs
    returns: a dataframe with each entry for the target column separated, with each element moved into a new row. 
    The values in the other columns are duplicated across the newly divided rows.
    '''
    row_accumulator = []
    def split_list_to_rows(row):
        split_row = row[target_column]
        if isinstance(split_row, collections.Iterable):
          for i, s in enumerate(split_row):
              new_row = row.to_dict()
              new_row[target_column] = s
              new_row.update({'idx':i})  
              row_accumulator.append(new_row)
          if split_row == []:
              new_row = row.to_dict()
              new_row[target_column] = None
              new_row.update({'idx':0})  
              row_accumulator.append(new_row)
        else:
          new_row = row.to_dict()
          new_row[target_column] = split_row
          new_row.update({'idx':0})  
          row_accumulator.append(new_row)
    df.apply(split_list_to_rows, axis=1)
    new_df = pd.DataFrame(row_accumulator)
    return new_df

def df2tuples (df):
    return [tuple(x) for x in df.values]

def split_data_frame_list(df, target_column):
    """
    Splits a column with lists into rows
    
    Keyword arguments:
        df -- dataframe
        target_column -- name of column that contains lists        
    """
    
    # create a new dataframe with each item in a seperate column, dropping rows with missing values
    col_df = pd.DataFrame(df[target_column].dropna().tolist(),index=df[target_column].dropna().index)

    # create a series with columns stacked as rows         
    stacked = col_df.stack()

    # rename last column to 'idx'
    index = stacked.index.rename(names="idx", level=-1)
    new_df = pd.DataFrame(stacked, index=index, columns=[target_column])
    return new_df


class GrammarAnnotator:
    def __init__(self, df, import_dir=None):
        self.import_dir = import_dir

        self.df = df
        if self.import_dir:
            self.df, self.sentence_df = GrammarAnnotator.import_parses (self.import_dir, df)
        else:
            self.df, self.sentence_df= self.add_parses (df, debug=True)
        return None

    def dump(df, sentence_df, filename):
        pickle.dump((sentence_df), open( "save.p", "wb" ) )
        
    def load(filename):
        df, sentence_df = pickle.load( open( "save.p", "rb" ) )
        return df, sentence_df

    def collapse_sentences(df):
        """merges a token df to single senetences by groupung thenm by a column 'sentence_id'"""
        def parse_coref_str(x):
            mfas = [re.finditer (r"((?P<s_id>\d+)->\[(?P<m_start>\d+):(?P<m_end>\d+)\])+", y) for y in x]
            def parse_ints(d):
                return {a: int(x) for a, x in d.items()}
            return  [[parse_ints(m.groupdict()) for m in mfa] if mfa else None for mfa in mfas ]

        sentence_df = df.groupby('s_id').agg({'text':  lambda x: "%s" % ' '.join(x),
                                              'coref': lambda x: list(parse_coref_str(x))
                                              })

        sentence_df = sentence_df.sort_index()
        sentence_df = sentence_df.reset_index()
        return sentence_df

    def add_parses(self, df, debug=False):
        sentence_df = GrammarAnnotator.collapse_sentences(df)
        sentence_df['spacy_doc'] = sentence_df['text'].apply(lambda x: nlp(x))
        def f (x):
            df_for_sentence = df[df['s_id'] == x['s_id']]
            return self.load_conll_over_spacy_doc(
                x['spacy_doc'],
                df_for_sentence=df_for_sentence)

        y = sentence_df.apply(f, axis=1)
        sentence_df['spacy_doc']
        return GrammarAnnotator.merge_sentence_df (df, sentence_df)

    def export_parses (path, sentence_df):
        if not os.path.exists(path):
            os.makedirs(path)
        print (sentence_df.columns)
        sentence_df.apply(lambda x: GrammarAnnotator.save_conll(
            path + str(x.name) + '.conll',
               x['spacy_doc'] 
               ), axis=1)

    def import_parses (path, df):
        """ import parse tree information like head and dep_ from a directory containing conll-files, that have the 'sentence_id' as filename and 'conll' as suffix.
        These files can be produced by GrammarAnnotator.export_parses
        # conll editor: https://arborator.ilpga.fr/q.cgi
        """
        sentence_df = GrammarAnnotator.collapse_sentences(df)

        sentence_df['text'] = sentence_df['text'].apply(lambda x: GrammarAnnotator.clean_text(x))
        sentence_df['spacy_doc'] = sentence_df['text'].apply(lambda x: nlp(x))

        sentence_df.apply(lambda x: GrammarAnnotator.load_conll_over_spacy_doc(
            path + "/" + str(x['s_id']) + '.conll',
               x['spacy_doc'] 
               ), axis=1)

        return GrammarAnnotator.merge_sentence_df (df, sentence_df)

    def merge_sentence_df (df, sentence_df, debug = False):
        splitted = split_data_frame_list_beware(sentence_df, target_column='spacy_doc')
        splitted = splitted.reset_index()
        splitted['idx'] = splitted['idx'] + 1
        splitted['spacy_token'] = splitted['spacy_doc'].apply(lambda x: x.text)
        splitted['spacy_dep_'] = splitted['spacy_doc'].apply(lambda x: x.dep_)
        splitted['spacy_lemma_'] = splitted['spacy_doc'].apply(lambda x: x.lemma_)
        splitted['new_swid'] = splitted['s_id'].astype(str) +'-' + splitted['idx'].astype(str)

        df = df.merge(splitted, left_on='s_id', right_on='s_id', how='outer')
        df = df.sort_values(by='s_id')
        return df, sentence_df

    conll_format = "%d\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s"
    def doc2conll (doc):
        for i, word in enumerate(doc):
            if word.head is word:
                head_idx = 0
            else:
                head_idx = doc[i].head.i+1

            yield (GrammarAnnotator.conll_format % (
                     i+1, # There's a word.i attr that's position in *doc*
                     word.text,
                     '_',
                     word.pos_, # Coarse-grained tag
                     word.tag_, # Fine-grained tag
                     '_',
                     head_idx,
                     word.dep_, # Relation
                     '_', '_'))

    def save_conll(conll_path, doc):
        with open(conll_path, 'w') as f:
            print (doc)
            f.write('\n'.join(GrammarAnnotator.doc2conll(doc)))

    def load_conll_over_spacy_doc(self, doc, df_for_sentence=None):
        """ read conll_files over a spacy document with same length. they may be manipulated."""
        if df_for_sentence.empty:
            raise ValueError("no DataFrame with conll_information")

        for index, row in df_for_sentence.iterrows():
             i = row ['id']
             if i >= len(doc) or row['head_id'] >= len(doc):
                 logging.error("%d not in doc %s" % (i, str(doc)))
                 return doc
             doc[i].head   = doc[row['head_id']]
             doc[i].dep_   = row["dep_"]
             doc[i].tag_   = row["tag_"]
             doc[i].pos_   = row["pos_"]
             doc[i].lemma_ = row['lemma']
        return doc

    conll_pattern = re.compile(
                            r"""(?P<id>\d+)((?:\.)?(?P<hidden>\d+))?      # i (as well es hidden node-ids in conll-u format)
                                 \t(?P<text>.*?)    
                                 \t(?P<lemma>.*?)  
                                 \t(?P<pos_>.*?) 
                                 \t(?P<tag_>.*?)
                                 \t(?P<nothing2>.*?)
                                 \t(?P<head_id>\d+)((?:\.)?(?P<head_hidden>\d+))?
                                 \t(?P<dep_>.*?)
                                 \t(?P<spacy_i>.*?)
                                 \t(?P<coref>.*)
                                     """, re.VERBOSE
                              )
    def conll_line2match(line):
        match = GrammarAnnotator.conll_pattern.match(line)
        return match

    def clean_text(string):
        global quotes
        # Text cleaning
        #string = string.lower()                                                              # big letters
        #string = re.sub(ur"([‚‘])",                              "'", string)
        string = re.sub(r"""e\.g\.""",                            " exempli gratia ", string)
        string = re.sub(r"""e\.g""",                              " exempli gratia ", string)

        string = re.sub(r"""([\d]+[\w]+.)""",                     "", string)

        string = re.sub(r"""(\{[\(\)A-Za-z,.;:\-\s"']*\})""",     "", string)# original text annotations
        string = re.sub(r"""(\([\w‚‘\s,.;:"']+\))"""        ,     "", string)# text in parentheses
        string = re.sub(r"""\[([A-Za-z,.;:\-\s"']*)\]""",         "\\1", string)# own manipulation

        def modifier(s): return " " + s.replace(" ","").replace("-","").title().strip()
        replace_the_cites = Replacement (quotes, modifier, " \\2")                                # other language layer by quoting
        string = re.sub(r"""(?:^|\s)(["'])((?:\\.|[^\\])*?)(\1)""", 
                                                   replace_the_cites, string)   
        #for matched, replaced in replace_the_cites.occurrences:
        #    print (matched, '=>', replaced)

        string = re.sub("[:;]",                                   ",", string)# punctuation of sentences
        string = string.replace("-",                              "")                                   # dashs 
        string = string.replace("—",                              ", ") 
        
        string = " ".join(string.split())                                 # all kinds of whitespaces, tabs --> one space
        return string