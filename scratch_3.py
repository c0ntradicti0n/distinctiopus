from gensim.models import Word2Vec
from predicatrix import collect_all_simple_predicates, nlp
import numpy as np
from corpus_reader import read_snli_testset, read_aristotle, read_milton

corpus_text = read_aristotle()
#corpus_text = read_milton()

#corpus_text = read_snli_testset(just_text=True, out = "text", max=2000)
print (corpus_text)

def text2predicates (text):
    return collect_all_simple_predicates (text)

doc = nlp(corpus_text)
predicates = text2predicates(doc)
sentences = []
for i, pred in enumerate(predicates):
    sentences.append(pred["lemma_"])

#for i, sent in enumerate(corpus_text.split(".")):
#    sentences.append([str(x) for x in sent.split(" ") if x.isalpha() ])

print (sentences)
"""
sentences = [['this', 'is', 'the', 'good', 'machine', 'learning', 'book'],
            ['this', 'is',  'another', 'book'],
            ['one', 'more', 'book'],
            ['this', 'is', 'the', 'new', 'post'],
                        ['this', 'is', 'about', 'machine', 'learning', 'post'],
            ['and', 'this', 'is', 'the', 'last', 'post']]
"""

text_args = \
    {
        'size': 300,
        'window': 10,
        'min_count': 1,
    }

model = Word2Vec(sentences, **text_args, iter=150, seed=1)

#print (model.similarity('this', 'is'))
#print (model.similarity('equivocally', 'univocally'))
#output -0.0198180344218
#output -0.079446731287
print (model.most_similar(positive=['affirm'], negative=[], topn=2))
print (model.most_similar(positive=['deny'], negative=[], topn=2))
print (model.wv.most_similar(positive=['old'], negative=[], topn=2))
print (model.wv.most_similar(positive=['old'], negative=[], topn=2))
ql = model.wv.most_similar(positive=['quality'], negative=[], topn=150)
qn = model.wv.most_similar(positive=['quantity'], negative=[], topn=150)
print (list(x for (x,w1) in ql for y in qn if x==y))

print (model.similarity('quality', 'quantity'))

qn = model.wv.most_similar(positive=['quantity','quality'], negative=[], topn=15)

print (qn)
qn = model.wv.most_similar(positive=['category'], negative=[], topn=15)
print (qn)

print ("speak, white, large")
qn = model.wv.most_similar(positive=['speak'], negative=[], topn=15)
print (qn)
qn = model.wv.most_similar(positive=['white'], negative=[], topn=15)
print (qn)
qn = model.wv.most_similar(positive=['large'], negative=[], topn=15)
print (qn)

print (model.wv.most_similar(positive=['generation'], negative=[], topn=2))

#output: [('new', 0.24608060717582703), ('is', 0.06899910420179367)]
#print (model['the'])
#output [-0.00217354 -0.00237131  0.00296396 ...,  0.00138597  0.00291924  0.00409528]

#print (list(model.wv.vocab))
#print (len(list(model.wv.vocab)))

X = model[model.wv.vocab]
print (X)
from nltk.cluster import KMeansClusterer

import nltk

from sklearn.preprocessing import QuantileTransformer

X = QuantileTransformer(output_distribution='uniform').fit_transform(X)
X = QuantileTransformer(output_distribution='uniform').fit_transform(X)
X = QuantileTransformer(output_distribution='uniform').fit_transform(X)

from sklearn import cluster
from sklearn import metrics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA
pca3 = PCA(n_components=3,random_state=0)
pca2 = PCA(n_components=2, random_state=0)
kpca2 = KernelPCA(n_components=2, random_state=0)
from dict_tools import invert_dict
import pprint
tsne3 = TSNE(n_components=3, random_state=0)
tsne2 = TSNE(n_components=3, random_state=0)

from scipy.spatial.distance import cosine
paths = []

triple_dict = {}

for x in list(range(500,1000,20)):

    NUM_CLUSTERS=x
    """
    kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=cosine, repeats=6, avoid_empty_clusters=True)
    assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
    print (assigned_clusters)
    # output: [0, 2, 1, 2, 2, 1, 2, 2, 0, 1, 0, 1, 2, 1, 2]


    for cl in list(set(assigned_clusters)):
        i_s_cl =
    words = list(model.wv.vocab)
    for i, word in enumerate(words):
        print (word + ":" + str(assigned_clusters[i]))
    """


    # -------------------------------



    #kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS, precompute_distances=True, random_state=655)
    #kmeans.fit(X)

    k2means = cluster.KMeans(n_clusters=NUM_CLUSTERS, precompute_distances=True, random_state=655)
    kmeans.fit(X)

    assigned_clusters = labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    print (assigned_clusters)

    words = list(model.wv.vocab)
    dict  = {}
    for i, word in enumerate(words):
        print (word + ":" + str(assigned_clusters[i]))
        dict[word] = [assigned_clusters[i]]

    d = invert_dict(dict)

    triple_dict[x] = [v for k, v in d.items() if len(v) == 2 ]

    pprint.pprint(d)

    print("Cluster id labels for inputted data")
    print(labels)
    print("Centroids data")
    for centroid in centroids:

        print ("center of centroid represented by word: %s" % model.most_similar(positive=[centroid], topn=1))


    print(
        "Score (Opposite of the value of X on the K-means objective which is Sum of distances of samples to their closest cluster center):")
    print(kmeans.score(X))

    silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')

    print("Silhouette_score: ")
    print(silhouette_score)
    

    np.set_printoptions(suppress=True)

    #Y = tsne2.fit_transform(X)
    #Y = tsne3.fit_transform(X)

    Y = kpca2.fit_transform(X)
    #Y = pca3.fit_transform(X)

    # 3D
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(Y[:, 0], Y[:, 1],Y[:,2], c=assigned_clusters, cmap='plasma',  alpha=1.)


    plt.scatter(Y[:, 0], Y[:, 1], c=assigned_clusters, cmap='plasma',  alpha=1.)

    path = "gensimplot" +str(x) +".png"
    paths.append(path)
    plt.savefig(path)
    plt.clf()
    """
    
    for j in range(len(sentences)):
        plt.annotate(assigned_clusters[j], xy=(Y[j][0], Y[j][1]), xytext=(0, 0), textcoords='offset points')
        print("%s %s" % (assigned_clusters[j], sentences[j]))

    plt.show()


    import numpy as np
    from sklearn.manifold import TSNE
    X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    X_embedded = TSNE(n_components=2).fit_transform(X)
    print (X_embedded.shape)
    print (X_embedded)
    """

    pprint.pprint (triple_dict)

    #https://scikit-learn.org/stable/auto_examples/cluster/plot_mini_batch_kmeans.html#sphx-glr-auto-examples-cluster-plot-mini-batch-kmeans-py

import imageio

with imageio.get_writer('movie.gif', mode='I') as writer:
    for filename in paths:
        image = imageio.imread(filename)
        writer.append_data(image)