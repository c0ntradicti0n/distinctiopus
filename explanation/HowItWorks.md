# S0krates

## Chunking the text into predicative expressions

First the text is cut into chunks by rules based on grammar annotations (they are taken from spacy, that annotates the dependence structure of hte sentence). These chunks are also annotated with embeddings (they are embeddings taken and vectors of importance per vector (the tf-idf-value for every word in the sentence). 

The idea of predicates is since very old days of philosophy of Plato and Aristotle, who set up the discussion about `methexis`, how the things of the real world relate the concepts, that humans have of them, and how this is rendered in natural language. This partitioning of language structure into subject (ὑποκείμενον, subjectum) and predicate (κατηγορόυμενον, predicatum) we have until now in formal logic, when we write `Fa`, when the predicate `F` is said of the individual `a`.

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

My algorithm puts these into a tree, which expressions appear in others

![Image](predicate_chunks.svg)

That is done to determine, where negations acutally stand:

    The brown fox jumps over the NOT AT ALL lazy dog.

They stand in the expression on the deepest level of the tree, where they appear. 

## Finding the contradictions

Then S0krates fits these chunks into very similar pairs and tries to find there contradictions, if an anonym pair appears in both or a a negation appears in one of them.

The following contradictions are 

![Image](contradiction.svg)

## Finding correlating predicate pairs to the contradictions

![Image](correlation.svg)

## Finding themes and aspects

![Image](subjects_aspects.svg)

## Matching results in neo4j

![Image](./img/neo4j jpg.svg)
 