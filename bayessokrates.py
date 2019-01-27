from tdfidf_tool import tdfidf
import networkx as nx
from predicatrix import collect_all_simple_predicates, nlp
import itertools
from multiprocessing import Pool
import matplotlib.pyplot as plt
import pylab

from sklearn import cluster
import numpy as np

class ClusterConceptRelations:
    def n_cluster(self, sparse_matr, n):
        dist_matr = 1 - sparse_matr / sparse_matr.max()

        model = cluster.AgglomerativeClustering(n_clusters=n, affinity='euclidean', linkage='ward')
        clustered = model.fit_predict(dist_matr)
        print (clustered)
        new_order = np.argsort(model.labels_)
        ordered_dist = sparse_matr[new_order] # can be your original matrix instead of dist[]
        #ordered_dist = ordered_dist[:,new_order]
        return clustered #ordered_dist


    def __init__(self, text):
        self.text = text
        self.doc = nlp(self.text)
        self.predicates = self.text2predicates(self.doc)
        self.tdfidf = tdfidf(text)
        G = nx.MultiDiGraph()
        for i, pred in enumerate(self.predicates):
            relevant_concepts = self.tdfidf.sentence2relevantwords(pred["text"], 0.9,1)
            conept_tuples = itertools.combinations(relevant_concepts,2)
            G.add_edges_from(conept_tuples)
            print (i)

        X = nx.google_matrix(G)

        #spring_pos = nx.spring_layout(G)
        #plt.axis("off")
        #nx.draw_networkx_edge_labels(G, spring_pos)
        #nx.draw(G, spring_pos, with_labels=False, node_size=5)
        #pylab.show()
        #pylab.savefig('1.png')

        Y = self.n_cluster(X,2)

        import seaborn as sns

        #clustermap = sns.clustermap(X, method ='single')
        #pylab.savefig("1single.png")

        pr = nx.pagerank_scipy(G, alpha=0.9)

        clustermap = sns.clustermap(X, method = 'complete' ,robust=True,  standard_scale=1,  metric=pr)
        pylab.savefig("1complete.png")

        clustermap = sns.clustermap(X, method = 'average',robust=True ,  standard_scale=1,  metric="correlation")
        pylab.savefig("1average.png")

        clustermap = sns.clustermap(X, method = 'weighted' ,robust=True,  standard_scale=1,  metric=pr)
        pylab.savefig("1weighted.png")

        #clustermap = sns.clustermap(X, method = 'centroid'robust=True) )
        #pylab.savefig("1centroid.png")

        #clustermap = sns.clustermap(X, method = 'median' )
        #pylab.savefig("1median.png")

        clustermap = sns.clustermap(X, method = 'ward',  standard_scale=1 ,  metric="correlation")
        pylab.savefig("1ward.png")





        pylab.savefig("1.png")

        print (list(Y))
        plt.scatter(X[0].A1, X[0].A1, c=Y.T, cmap='nipy_spectral')#, cmap='rainbow')
        pylab.savefig("2.png")
        return None


    def text2predicates (self, text):
        return collect_all_simple_predicates (text)

    def drawgraph(self):
        print ("not implemented")


with open('./corpus/aristotle-categories-edghill-spell-checked.txt', 'r') as cf:
    corpus = cf.read()

CCL = ClusterConceptRelations(corpus)