import re
import os
from itertools import count
from littletools.nested_list_tools import flatten_reduce, curry
import fnmatch

import pandas as pd
import copy

import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

import spacy
nlp = spacy.load('en_core_web_sm')

def find_position_in_doc_by_approx(doc, text_token, pos, deviation=10):
    deviator = iterate_away(pos, deviation)
    for p in deviator:
        if p < 0:
            continue
        if p >= len(doc):
            continue
        if text_token == doc[p].text or ((text_token in ['’'] or len(text_token)<2) and text_token in doc[p].text):
            return p
    else:
        logging.error("Token '%s' not seen in spacy doc (search tokens: '%s')! returning starting position, '%s" %
                      (text_token,
                       str([w.text for w in doc[pos - deviation:pos + deviation]]),
                       str(doc[pos])))
        return pos

def iterate_away(pos, deviation):
    yield pos
    for d in range(1, deviation):
        yield pos + d
        yield pos - d

class CorpusReader:
    def __init__(self, corpus_path=None, only=None):
        ''' This module reads conll files into a DataFrama.

        Conll files have here to be read, they are written line per word.
        It is collapsed into one line per sentence here.

        Also some datatypes are parsed into a more appropriate datastructure than strings
        The spacy doc's are build here, overwritten with grammar from the conll files to use the datastructure of spracy.
        Also the coreference mention tags are parsed to some dict.

        :param corpus_path: path to the dir that contains a folder 'import', where the conll files are
        :param only:  restrict the conll files that should be read.

        '''
        if not corpus_path:
            raise AttributeError("path must be given!")
        self.corpus_path = corpus_path

        # loand conlls into dicts and build dataframe with one line per word
        all_sentences, all_conlls = self.load_all_conlls(corpus_path, only=only)
        self.df = pd.DataFrame(all_conlls)

        # translate also normal conll-nodes as well as invisible knodes to some nice id
        # groupby makes the grouped value invisible to the function, copy it
        self.df['s_id2'] = self.df ['s_id'] # 
        self.df = self.df.groupby('s_id').apply(lambda x: self.renumerate_word_ids(x))

        # parse columns to int
        num_cols = ["s_id", "head_id", "id"]
        self.df[num_cols] = self.df[num_cols].astype(int)

        # from dataframe with one line per word get another with one line to one sentence
        self.sentence_df = self.df.groupby('s_id').apply(self.aggregate_sentences)
        

    def renumerate_word_ids(self, x):
        ''' In conll files normal id's are ints, but invisible nodes have to be re-enumerated.

         If you annotated `hidden nodes` with this syntax for the id: '1.1', they are recounted here.

         See this discussion here:
         .. _Null subjects: http://www.python.org/https://github.com/UniversalDependencies/docs/issues/589
         That also happens, if there are coordinative bindings in the sentences, that re-use the tokens for their
         grammatical well-formedness

        :param x: pandas Series of Enhanced UD
        :return: index list
        '''
        invisible_nodes_translation_dict = dict(zip(x['inv_id'],list(range(len(x)))))
        x['id'] = invisible_nodes_translation_dict.values()
        x['head_id'] = [ invisible_nodes_translation_dict[h] for h in x['inv_head_id']]
        return x


    def override_spacy (self, tldpth):
        t = tldpth[-1]
        t.lemma_ = tldpth [0]
        t.dep_ = tldpth [1]
        t.pos_ = tldpth[2]
        t.tag_ =  tldpth[3]
        t.head = t.doc[tldpth[4]]
        return t


    def aggregate_sentences(self, x):
        doc = nlp(' '.join(x['text']))
        doc = [self.override_spacy (tup) for tup in zip(x['lemma'], x['dep_'], x['pos_'], x['tag_'], x['head_id'], doc)]
        corefs = self.parse_coref_str(x['coref'])
        return pd.Series({'text': ' '.join(x['text']),
                          'text_pos_': ' '.join([text + '_' + pos for text, pos in zip(x['text'], x['pos_'])]),
                          's_id': x['s_id2'].iloc[0],
                          'spacy_doc': doc,
                          'coref': corefs
                          })

    def parse_coref_str(self, coref_string):
        ''' parses coref mentions in two formats. Either ranges or lists of token indices

        7->[6,1,2,3]
        1->[22:23]

        :param coref_string: such a string
        :return: coref dict with 's_id' and 'i_list'

        '''
        mfas = [re.finditer(r"(((?P<s_id_r>\d+)->\[(?P<m_start>\d+):(?P<m_end>\d+)\])|((?P<s_id_i>\d+)->\[(?P<i_list>(?:\d+)(?:,\s*\d+)*)\]))+", y) for y in coref_string]
        return [[self.parse_coref_dict(m.groupdict()) for m in mfa] if mfa else [] for mfa in mfas]



    def parse_coref_dict(self, d):
        ''' Parses the content of the coref_dict to some normal data

            :param d: dict with either 's_id_r' and 'm_start', 'm_end' or 's_id_i' and 'i_list'
            :return: coref dict with 's_id' and 'i_list'

        '''
        if d['s_id_r']:
            return {'s_id': int (d['s_id_r']),
                    'i_list' : list(range (int(d['m_start']), int(d['m_end'])))}
        elif d['s_id_i']:
            return {'s_id': int(d['s_id_i']),
                    'i_list':  eval(d['i_list'])}
        else:
            raise ValueError ('coref dict with bad indices')


    def lemmatize_text(self):
        return " ".join(self.df['lemma'].tolist())

    def read_one_conll (self,fname, s_id_dict):
        sentence = []
        conll_lines = []

        with open(fname, 'r') as fh:
            for i, line in enumerate (fh):
                try:
                    sentence.append(re.search(r'(?:^\d+\.?\d*\t)([^\t]+)', line).group(1))
                    conll_line_dict = self.conll_line2match(line).groupdict()
                    conll_line_dict.update(s_id_dict)
                    conll_lines.append(conll_line_dict)
                except AttributeError:
                    raise SyntaxError(
                        "wrong syntax in file %s, line no. %d line:\n'%s' (Maybe an empty"
                        "line at the end of your conll" % (fname, i, line))
                if not line.strip():
                    line = last
                    break
                last = line
                pass

        return conll_lines, " ".join(sentence)

    def load_all_conlls (self, path, only=None):
        all_sentences = []
        all_conlls = []

        def hasNumbers(inputString):
            return any(char.isdigit() for char in inputString)

        for filename in sorted(os.listdir(path), key =lambda x: int(''.join(filter(str.isdigit , x))) if hasNumbers(x) else 0):
            if fnmatch.fnmatch(filename, '*.conll'):
                s_id_dict = re.search("(?P<s_id>[0-9]+)" , filename).groupdict()
                if (only and int (s_id_dict['s_id']) not in only):
                    continue
                filename = os.path.join(path, filename)
                conll_lines, sentence = self.read_one_conll(filename, s_id_dict)
                all_sentences.append (sentence)
                all_conlls.append (conll_lines)
        all_conlls = flatten_reduce(all_conlls)

        return all_sentences, all_conlls

    def load_conll (self, i, corpus_path):
        if isinstance(i, list):
            docs = []
            for j in i:
                print (j)
                docs.append(self.load_conll(j, corpus_path))
            return docs

        fname = corpus_path + "/" + str (i) + '.conll'
        sentence = []
        last = ''
        with open(fname, 'r') as fh:
            for line in fh:
                try:
                    sentence.append(re.search(r'(?:^\d+\t)([^\t]+)', line).group(1))
                except AttributeError:
                    print (i, "'"+line+"'")
                    raise
                if not line.strip():
                    line = last
                    break
                last = line

                pass
        doc = self.nlp(" ".join(sentence))
        new_doc = self.conll_over_spacy(doc, fname)
        return new_doc

    pattern = re.compile(   r"""(?P<inv_id>(?P<id>\d+)((?:\.)?(?P<hidden>\d+))?)      # i (as well es hidden node-ids in conll-u format)
                                 \t(?P<text>.*?)    # whitespace, next bar, n1
                                 \t(?P<lemma>.*?)   # whitespace, next bar, n1
                                 \t(?P<pos_>.*?)    # whitespace, next bar, n2
                                 \t(?P<tag_>.*?)    # whitespace, next bar, n1
                                 \t(?P<nothing2>.*?)# whitespace, next bar, n1
                                 \t(?P<inv_head_id>(?P<head_id>\d+)((?:\.)?(?P<head_hidden>\d+))?)   # head_id (as well es hidden node-ids in conll-u format)
                                 \t(?P<dep_>.*?)    # whitespace, next bar, n2
                                 \t(?P<spacy_i>.*?)# whitespace, next bar, n1
                                 \t(?P<coref>.*)# whitespace, next bar, n1
                                 """, re.VERBOSE)

    def conll_line2match(self, line):
        match = self.pattern.match(line)
        return match

    col_set = ['i','text', 'lemma','pos','tag','nothing','head','dep','spacy_i','coref']
    def conll_over_spacy(self, doc, dir, i, no_cols={}):
        to_change = set(self.col_set) - set(no_cols)
        fname = str (i) + '.conll'
        path  = dir + "/" + fname

        # read conll_files, may manipulated over spacy
        with open(path) as f:
            for line in f:
                match = self.conll_line2match(line)
                i = int(match.group("id")) - 1
                head_i = int(match.group("head_id")) - 1
                doc[i].set_extension('coref', default = list(), force=True)
                try:
                    if 'head' in to_change:
                        doc[i].head = doc[head_i]
                    if 'lemma' in to_change:
                        doc[i].lemma_ = match.group("pos_")
                    if 'pos' in to_change:
                        doc[i].pos_ = match.group("pos_")
                    if 'tag' in to_change:
                        doc[i].tag_ = match.group("tag_")
                    if 'dep' in to_change:
                        doc[i].dep_ = match.group("dep_")
                    #if 'spacy_i' in to_change:
                    #    doc[i].i      = match.group("spacy_i")
                    if 'coref' in to_change:
                        doc[i]._.coref= match.group("coref")

                except IndexError:
                    raise ValueError("Shape of the spacy doc and conll file incongruent, look for the number of tokens! '%s'" % (str(doc)))
        return doc

    def coref_lookup(self, corefs, what= None):
        if not corefs:
            return []
        for coref in corefs:
            s_id = coref['s_id']
            m_start = coref['m_start']
            m_end = coref['m_end']
            if what =="sub_pred":
                all_subpreds = flatten_reduce(flatten_reduce(
                    [[p['part_predications'] for p in ps]
                     for ps in self.sentence_df.query("s_id == @s_id")['predication']]
                ))
                return [sp for sp in all_subpreds if all([m in sp['full_ex_i'] for m in range(m_start, m_end)])]
            else:
                return self.sentence_df.query("s_id == @s_id")['spacy_doc'].values[0][m_start-1:m_end-1]

    conll_format = "%d\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s"
    def export_dict (self, doc, index=None):
        res = []
        w_counter = count(0)

        for word in doc:
            i = next(w_counter)
            if word.head is word:
                head_idx = 0
            else:
                head_idx = doc[i].head.i+1
            #coref = self.extract_coref_from_spacy_neucoref (doc, word, i)

            res.append(
                          {  's_id'   : index,
                             'i'      : i+1,
                             'text'   : word.text,
                             'lemma'  : word.lemma_,
                             'pos'    : word.pos_, #
                             'tag'    : word.tag_, #
                             'unknown': '_',
                             'head'   : head_idx,
                             'dep'    : word.dep_, # Relation
                             'corp_id': str(index)+'-'+str(word.i), # Generation_i
                             'doc_i'  : word.i,
                             #'coref'  : coref
                          }
                      )
        return res


    def explode(df, column_to_explode):
        """
        Similar to Hive's EXPLODE function, take a column with iterable elements, and flatten the iterable to one element
        per observation in the output table

        :param df: A dataframe to explod
        :type df: pandas.DataFrame
        :param column_to_explode:
        :type column_to_explode: str
        :return: An exploded data frame
        :rtype: pandas.DataFrame
        """

        # Create a list of new observations
        new_observations = list()

        # Iterate through existing observations
        for row in df.to_dict(orient='records'):

            # Take out the exploding iterable
            explode_values = row[column_to_explode]
            del row[column_to_explode]

            # Create a new observation for every entry in the exploding iterable & add all of the other columns
            for explode_value in explode_values:
                # Deep copy existing observation
                new_observation = copy.deepcopy(row)

                # Add one (newly flattened) value from exploding iterable
                new_observation[column_to_explode] = explode_value

                # Add to the list of new observations
                new_observations.append(new_observation)

        # Create a DataFrame
        return_df = pd.DataFrame(new_observations)

        # Return
        return return_df

    def annotate_corefs (self, doc, df):
        df['coref'] =  [[] for _ in range(len(df))]

        def element_rest (l):
            for i, e in  enumerate (l):
                yield e, l[:i]+l[i+1:]
        def ref_from_row (r):
            try:
                row = df.query('doc_i in @r')
            except KeyError:
                print ("not found?")
            if len (row.s_id.values) == 0 :
                ba =' ta'
                return 'out of range?'
            return  str(row.s_id.values[0]) + "->" + str(row.i.values[0])

            return ",".join(other_sents)

        if doc._.has_coref:
            for cl in doc._.coref_clusters:
                for ment, rest_ments in element_rest (cl):
                    ids = range(ment.start, ment.end)
                    other_sents = [ref_from_row(range(r.start, r.end)) for r in rest_ments]
                    df.loc[df['doc_i'].isin(ids), 'coref'] += other_sents

        df.coref = df.coref.apply (lambda x: ",".join(x) if x else '_')
        return None

    def write_conll_by_df_group(self, x):
        x = x
        conll_lines = []
        for row in x.itertuples():
            conll_lines.append(CorpusReader.conll_format % (
                row.i,  # There's a word.i attr that's position in *doc*
                row.text,
                row.lemma,
                row.pos,  # Coarse-grained tag
                row.tag,  # Fine-grained tag
                row.unknown,
                row.head,
                row.dep,  # Relation
                row.corp_id,  # Generation_i
                row.coref))

        conll_path = self.export_dir + '/' + str(row.sent_id) + '.conll'
        with open(conll_path, 'w+') as f:
            f.write ("\n".join (conll_lines) +'\n')

        return None




