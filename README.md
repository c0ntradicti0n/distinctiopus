# S0krates

A tool for mining distinctions based on textlinguistic and rhetoric features of a text.

![Image](/home/stefan/PycharmProjects/Sokrates3.1/img/contradiction -- new correlation36 -- 37.svg)
<img src="img/contradiction -- new correlation36 -- 37.svg">


This means, that this tool, when e.g. reading the Microsoft License Agreement and being informed, that in some cases you have
limited warranty and in other cases you have no warranty, it looks for the concret cases, when you have warranty.

To see, [how it works](explanation.html)


## Getting Started

To get all dependencies, you have to get some Natural Language Processing tools with their models.

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

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

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

To see the docs, go here:

*[Module Documentation](./docs/index.html)

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags).

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

