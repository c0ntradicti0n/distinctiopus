# S0krates

S0krates is a tool, that mines distinctions in texts. 

Distinctions are the decisive points in theoretical text, and even more. They are a phenomenon in language that either helps to shape theories, to propose the things we have to decide, or to contrast results from the state of the art in papers, expressing the step forward in the `history of spirit`. Or they take a basic role in the field of aethetics, to know and dicuss, what are these features, you like and which you not like (See [Pierre Bourdieu - Distinction](https://en.wikipedia.org/wiki/Distinction_(book)) It's a fundamental epistemological thing, that shapes our thoughts, theories. You can even read about it as a more basic feature of thinking than the [Table of Categories](https://en.wikipedia.org/wiki/Category_(Kant)) in this german book: [Wilhelm Köller: Perspektivität und Sprache](https://books.google.de/books/about/Perspektivit%C3%A4t_und_Sprache.html?id=sAlCkpcceNwC&redir_esc=y)

This Programm S0krates focusses on some part of the theme of distinctions, namely, how to mine that, what distinctions authors put on the table in their text. I say that, because there can be a lot of misunderstandigs with this, as you can see every classification task or decision-making under the perspective of distinguishign. But I focus on distinctions as a text mining task. 

So if you write a scientific essay, you are advised to define your concepts and make fundamental distinctions, that are at work in your theory.

Another time distinctions appear at the point, when you present the results of these essays. You have to contrast them with the results of before and of other scientists. You say something like: "Before it was so and so. In contrast, afert my results it ist something else, namely so and so."

With ``distinction`` here I mean if a writer proposes two kinds or perspectives of something and gives a little explanation, in what they differ. 

Let's take for the sake of explanation these two sentences of Aristotle:

>  Forms of speech are either simple or composite. 
>   Examples of the latter are such expressions as 'the man runs', 'the man wins'; of the former 'man', 'ox', 'runs', 'wins'.

The pictures you see in this explanation are not build from scratch for the sake of this essay, they are automatically built by the algorithm. That means, if you find there some errors, that is noise in the data and with the algorithm I try to get around it. In data science all your work would be done, if the data would be clean ;)  


In short, how the following works. You are going to think think tl;dr? Just glance through and pick cherries. There are many pictures.
 
1. [Extract predicative structures and their arguments, resolving elliptic expressions](#predicative-chunks)

2. [Search for pairs of predicates, that have a strong contrast, either by anonyms or by negations.](#finding-contrasts)
3. [Search for another pair of expressions that is related to this constrasting pair](#find-correlative-pairs-to-the-contrasting-pairs)
4. [Find the subjects under the arguments of the predicats, that is talked about, and some word, what is the distinctive criterium.](#finding-themes-and-aspects)
5. [If there are contrast, the correlation and a common subject and distinctive criterium together, retrieve that as a ``distinction``.](#matching-results-in-neo4j)

## Predicative Chunks

First the text is cut into chunks by rules based on grammar annotations. (They are taken from `spacy`, a tool for annotating grammar, named entities and so on. Here the dependence structure of the sentences is used). These chunks are also annotated with embeddings (they are embeddings taken and vectors of importance per vector (the tf-idf-value for every word in the sentence). 

The idea of predicates is since early days of western philosophy of Plato and Aristotle, had a decisive discussion about `methexis`, how the things of the real world relate the concepts, that humans have of them, and how this is rendered in natural language. This partitioning of language structure into subject (ὑποκείμενον, subjectum) and predicate (κατηγορόυμενον, predicatum) we have until now in formal logic, when we write `Fa`, when the predicate `F` is said of the individual `a`.

In a sentence there apperear normally multiple such predications. If you look at this sentence (similar to Chomskies famous sentence):

    The brown fox jumps over the lazy dog.
    
Here are three predications implied:
    
    The brown fox
    The (brown) fox jumps over the (lazy) dog
    The lazy dog

  My algo algorithm cuts on some relations of the grammar tree. There are two kinds of such structures: 
  
  * Some are are triggered by verbs:
    * These start at the verbal core of these expressions, everything else are dependents.
    * Nouns and Pronouns are the expressions, that represent the `subjects` of the verbal `predicate`
  * And some are triggered by adjectives.
    * These start at the nouns end at the next paratactical (subclausal) level. 
       Here these adjectives (and participles) are the expressions are the predicative expressions. The noun, they belong to (`brown`--> `fox`), is the argument for that predicate.
     
There are some additional rules (how to deal with the adjectives, that appear in predicative structures: `The (brown) fox jumps over the (lazy) dog` -- Should one remove them here?), but thats it mainly. A little problem arises with coordinative bindings (if you say 'and', 'or' and so on):

    The brown fox jumps over the lazy dog and {the brown fox} fell into the mouth of a white shark
    
   
![Image](dependency.png)   
   
Here we get then 6 predicates:

    The brown fox                               
    The (brown) fox jumps over the (lazy) dog
    The lazy dog
    The (brown) fox fell into the mouuth of a white shark
    the mouth of a white shark
    a white shark

My algorithm puts these into a tree, which expressions appear in others.

![Image](predicate_chunks.svg)

That is done to determine, where negations acutally stand:

    The brown fox jumps over the NOT AT ALL lazy dog.

They stand in the expression on the deepest level of the tree, where they appear. 

This happens in the module [`Predication`](../predicatrix.py).

## Finding Contrasts

These chunks are paired by the similarity of the words that appear in them and by embeddings. For such pairing, that this programm does really often there is specific module [`Simmix`](../simmix.py), that gets two lists of annotated expressions and pairs them according to feature metrics, that can be adjusted in each case. 

By adjusting these feature metrices one can achieve both:
 
 1. Find the similar pairs (according to embeddings and some weighted verbal equality), as well as 

 2. Contasts between these similar pairs are in place, if there is an antonym pair appears in both or a negation appears in one of them.

 The solution in the programm is, that each sentence has a logical formula like `a ∧ b ∧ ¬c ∧ d`. for each of these letters there is a dictionary, for what the variables stand here. This formula then is fitted to another formula, and not fitting names are renamend. If a antonym pair is present in both, a negation is inserted in to one. After all, a contradiction is derived from this with the help of [`pyprover`](https://github.com/evhub/pyprover).  
 
The following contradictions are found in the two sentences: 

![Image](contradiction.svg)

This happens in the module [`Contradiction`](../contradictrix.py)

## Find correlative pairs to the contrasting pairs

With a quite similar process and using the same functionality of  [`Simmix`](../simmix.py) there are pairs of correlated predicates found.

With `correlated` I mean a pair of expressions in `distinctions, that, explains, what holds the two sides of disctinctions apart from another.

>  Forms of speech are either simple or composite. 
>   **Examples of the latter are such expressions as 'the man runs', 'the man wins'**; *of the former 'man', 'ox', 'runs', 'wins'.*

The bold expression tells you something, what means `composite forms of speach` the expression in italics tells you, what are `simple forms of speech`

By intuition you learn two things about these `correlated` predicates:

* They are also very similar to each other and have just a little part that makes them different (like in the formula for the contradiction `a and NOT a`, this `a` is despite the contradiction very the same).
 
* And they are distinct from what they should explain.

* They are related by some grammatical or coreferential binding. Here you have this coreference with `the former` and `the latter`. The alogorithm follows these signs.

And these criteria are the things the algorithm uses. In [`correlatrix`](../correlatrix.py) there are two filters defined, that make these constraints:

            self.correlative = \
                Simmix([(3, Simmix.common_words_sim, 0.35, 1),
                        (1, Simmix.dep_sim, 0.65, 1),
                        (1, Simmix.pos_sim, 0.65, 1),
                        (1, Simmix.elmo_sim(), 0.35,1),
                        (1,Simmix.fuzzystr_sim, 0.3,1),
                        (-1000, Simmix.boolean_subsame_sim, 0, 0.1)
                        ],
                       )
    
            # That's a distinctive criterium, that the correlative keys can't be too similar to the contradicting pair
            self.distinct = \
                Simmix([(1, Simmix.multi_sim(fun=Simmix.common_words_sim, n=7), 0, 0.95),
                        (1, Simmix.multi_sim(fun=Simmix.elmo_sim(), n=7), 0, 0.95),
                        (1, Simmix.multi_sim(fun=Simmix.dep_sim, n=7), 0.0, 1),
                        (1, Simmix.multi_sim(fun=Simmix.pos_sim,n=7), 0.0, 1)
                        ],
                       n=None)
                       
And so the algorithm computes for every found constrasting pair of predicates a set of possibly correlated other predicates, that are as similar to each other and as distinct from the contrasting pair as possible, but grammatically related.
![Image](correlation.svg)

## Finding themes and aspects

![Image](subjects_aspects.svg)

## Matching results in neo4j

![Image](./img/neo4j jpg.svg)
 