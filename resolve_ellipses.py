
import logging
import itertools

from webanno_parser import Webanno_Parser
from grammarannotator import GrammarAnnotator
from grammarannotator import split_data_frame_list_beware
from simmix import Simmix


class Ellipses_Resolver:
    def __init__(self):
        return None

    ellipsis_conditions = { "text": {"than":"than",
                                     "one":"the",
                                     "one":"same",
            .                         "by":"by"},
                            "pos_": {"VERB":"VERB"} }

    Ellipsis_sx = Simmix(
        [(Simmix.deps_alike, Simmix.Damerau_levenshtein_kwargs(deletion=1, insertion=1, transposition=1,substitution=1), +1),
         (Simmix.tags_alike, +1),
         (Simmix.vecs_alike, +1),
         (Simmix.boolean_same, (-3)), 
         (Simmix.boolean_alike, Simmix.Boolean_same_kwargs(attrib_dict=ellipsis_conditions), +1),
         (Simmix.nextleft_alike, +1),
         (Simmix.fuzzystr_alike, +1)], 
         gensim_model = '../similarity/aristotle-gensim.w2v')

    def ellipsis_split (spacy_token, sentence_id, debug = True):
        logging.debug  ("\n\nSplit Ellipsis" + str(spacy_token.doc))

        head             = spacy_token.head 
        ancestorchildren = [x for x in head.children] 

        conjunction      = [x for x in ancestorchildren if (x.dep_ == 'cc' or x.dep_ == "punct")]
        if not conjunction :

           logging.warning ('no conjunction found in ' + str(sentence_id) + str ())
           logging.warning ('nothing in here: ' + str( ancestorchildren))
           return False
        conjunction      = conjunction[0] 

        preconjunction   = [x for x in ancestorchildren if x.dep_ == 'preconj']

        conjunct         = [x for x in spacy_token.subtree] 
        conjunct         = conjunct + [x for x in conjunction.subtree if x != conjunction]   
        # Strange exception, that tokens can be children of the conjunction itself, e.g.:
        # a man may contend that Much is the contrary of Little , or Great of Small , 
        # but of definite quantitative terms no contrary exists .
                

        # grammatical correct ones
        pot_conjungens1   = [([head] + list(x.subtree)) for x in ancestorchildren if (
                                        x != spacy_token and
                                        x != conjunction and
                                        x not in  preconjunction)
                            ]
        

        # search leftwards head is something else than correctly annotated
        lefts            = [conjunction.doc[conjunction.i - i] for i in range(conjunction.i,0,-1)]  

        pot_conjungens2   = [(list(x.subtree)) for x in lefts]
        logging.debug ("\nbefore cleaning: " + str( pot_conjungens2))
        pot_conjungens2   = [[x for x in s_l if (
                                        x.i < conjunction.i and # nothing right of the conjunction
                                        x not in preconjunction)] 
                             for s_l in pot_conjungens2
                            ]
        pot_conjungens2   = [x for x in pot_conjungens2 if x]
        logging.debug ("\nafter cleaning: " + str( pot_conjungens2))


        pot_conjungens = pot_conjungens1 + pot_conjungens2

        if not pot_conjungens:
            pot_conjungens = [head]

        pot_conjungens = sorted(pot_conjungens, key=lambda x: int(x[0].i), reverse=True)

        if debug:

            #if not pot_conjungens:
            logging.warning ('\nspacy_token            = ' + spacy_token.text +
                                 '\npot conjungens (empty) = ' + str(pot_conjungens) +
                                                 "\nlemmata             : " + str([x.lemma_ for x in spacy_token.doc]) +

                                 "\nsentence               = " + " ".join(str(spacy_token.doc)))

        conjungens       = list(Ellipses_Resolver.Ellipsis_sx.mixed_choose (pot_conjungens, conjunct))
        
        if debug:
            logging.debug ("\nlist of possible first parts = " + str([" ".join ([y.text for y in x]) for x in pot_conjungens])  +
                           "\nlist of possible first parts = " + str([" ".join ([y.tag_ for y in x]) for x in pot_conjungens]))
            def j_s(x): return '"' + " ".join([y.text for y in x]) + '"'
            logging.debug (
                "\nspacy_token         : " + spacy_token.text +
                "\nsentence            : " + str(int (sentence_id)) + " " + str(spacy_token.doc) +
                "\n1rst part           : " + j_s(conjungens) +
                "\n2nd part            : " + j_s(conjunct) + 
                "\nconjunction         : " + j_s([conjunction]) + 
                "\npreconjunction      : " + j_s(preconjunction) 

                )
        
        return conjungens, conjunct
    
    def newsentences (sentence, have_del_tuple):
        if (have_del_tuple):
            deletions = [y for x,y in have_del_tuple]
            return [x for x in sentence for d in deletions if x not in d]
        else:
            return None

    def annotate (self, webanno_parsed, debug=True):
        sentence_df = webanno_parsed.sentence_df

        # Find all conjunct tokens
        sentence_df["conjunct_tokens"] = sentence_df['spacy_doc'].apply(lambda doc: [y for x in doc for y in list(x.conjuncts) if y])
        # Find the relative parts to the dependentially marked conjuncts
        sentence_df['coordinatives'] = sentence_df.apply(lambda x: [Ellipses_Resolver.ellipsis_split (y, x['sentence_id']) for y in x['conjunct_tokens'] if y], axis = 1)
        # Make the combinations, if there are multiple coordinative bindings in the sentence
        def powersetfrom1(iterable):
            "powerset([1,2,3]) --> (1) (2) (3) (1,2) (1,3) (2,3) (1,2,3)"
            s = list(iterable)
            rng = range(1,len(s)+1)
            return itertools.chain.from_iterable(itertools.combinations(s, r) for r in rng if rng)

        sentence_df['all_coordinative_combinations'] = sentence_df.apply(lambda x: list(powersetfrom1(x['coordinatives'])), axis = 1) 
        def doublereversetuples(iterable):
            def f1 (x):
                return x[0],x[1]
            def f2 (x):
                return x[1],x[0]
            return [[f(y) for y in x for f in (f1, f2)] for x in iterable if x]
        sentence_df['all_coordinative_combinations'] = sentence_df.apply(lambda x: doublereversetuples(x['all_coordinative_combinations']), axis = 1) 
        print (sentence_df)

        # Split the dataframe again and now the combinations are the possible alternatives

        sentence_df = split_data_frame_list_beware(sentence_df, 'all_coordinative_combinations')
        sentence_df = sentence_df.reset_index()
        
        # Build the sentences with these resolved coordinations
        sentence_df['new_sentences'] = sentence_df.apply(lambda x: Ellipses_Resolver.newsentences(x['spacy_doc'], x['all_coordinative_combinations']), axis=1)




        # Im einfachen Fall geht das so:
        # Alle children von dem ancestor von der Konjunktion ('cc') gehören zum ersten Teil,
        # und alle children des Konjunkts ('conj') plus die Konjunktion bilden
        # den zweiten, wenn man diesen zweiten Teil noch von der Menge von tokens aus der ersteren
        # Menge setmäßig abzieht.

        # Es gibt aber noch den komplizierteren Fall:
        # Denn es gibt es noch verschränkende Konjunktionen, wie in dem Satz vor diesem Satz.
        # Dort gilt das finite Verb('ist'), das im ersten Teil erscheint, sowie auch  das 
        # "Prädikatsnomen" ('Teil'), eben auch für den zweiten Teil. 
        # Dann wird nicht nur der übergeordnete Kontext des Satzes von beiden "geteilt", sondern
        # auch noch andere children vom ersten Teil gehören aufgrund von sonstiger grammatischer 
        # Inkonzinnität zum zweiten Teil.
        return sentence_df



def main():
    logging.basicConfig(filename='./ellipses_resolver.log', filemode='w', level=logging.DEBUG)
    logging.info('Started')

    img_folder = '../parse_webann/printedphraphs/'

    webanno_parsed = Webanno_Parser("../corpus/aristotle-categories-edghill-spell-checked.tsv")

    grammarian = GrammarAnnotator(import_dir='/home/bingobongo/code/sokrates_drei/corpus/import_conll/')
    grammarian.annotate(webanno_parsed)

    ellipticator = Ellipses_Resolver()
    ellipticator.annotate(grammarian)

    logging.info('Finished')


    #print (df[['token', 'swid', 'sentence_id','word_id']])
    # """+ now.strftime("%Y-%m-%d %H:%M:%S")"""

if __name__ == '__main__':
    main()