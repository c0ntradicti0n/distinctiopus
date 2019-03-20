# S0krates

A tool for mining distinctions based on textlinguistic and rhetoric features of a text.

This means, that this tool, when e.g. reading the Microsoft License Agreement and being informed, that in some cases you have
limited warranty and in other cases you have no warranty, it looks for the concret cases, when you have warranty.

To see, [how it works](./explanation/HowItWorks.md)

The output are subgraphs, that represent distinctions. To have a picture for better imagination:

![Image](./logo.png){:height="50%" width="50%"}
<img src="./logo.png" width="100" height="100">




## Getting Started

First there are some Natural Language Processing tools to install with the AI-models, that belong to them.

### Prerequisites

You are going to need:

* [Spacy](https://spacy.io)

It is important, that you install the dependencies in a `virtualenv`, because of an incompatability with the version of 
`spacy`. For the preprocessing with the Prepr0cessor, you need `spacy==2.0.12`, for S0krates you need 'spacy==2.1.0a4 '
(= `spacy-nightly`), that must be silently installed over `spacy 2.0.18`. because AllenAI again uses the sentence 
segmentation of spacy, but checks for the installed version (this works, if you install `spacy-nightly` after `spacy==2.0.18`, to override it)

   * [NeuralCoref](https://github.com/huggingface/neuralcoref)

Use the model `en_coref_sm` in Prepr0cess0r.
 
   * [AllenAI's](https://allennlp.org) [Elmo Embedder](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md)

For this you can fetch the models with `wget` from the S0krates's home directory:

    mkdir './others_models'
    cd ./others_models
    wget "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    wget "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"


  * [NLTK](https://www.nltk.org)

It it used because of wordnet for fetching antonyms and abstractness.

  * [Neo4J](https://neo4j.com/)

If you didn't install Neo4J with autostart, start it:

     sudo service neo4j start
 
  * [Prepr0cess0r](https://github.com/c0ntradicti0n/Prepr0cess0r)

If you want to mine distinctions of your choice, you have also to preprocess the text, to obtain a folder of conll-files for your text.
This is necessary combine the best features of each of these tools, because in other dimensions they may have not so nice results.
This is possible with the text-preprocessor tool, I build, watch out here for using this before:

### Installing

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

To see the docs, go along here:

* [Module Documentation](http://htmlpreview.github.com/?https://github.com/c0ntradicti0n/S0krates3/blob/master/docs/_build/html/index.html)

## Built With

## Contributing

## Versioning

## Authors
*** Stefan Werner *** 
See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

## Acknowledgments


