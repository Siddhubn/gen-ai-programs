import gensim.downloader as api
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

word_vectors=api.load("word2vec-google-news-300")
words=["#10similiar words to a particular field"]
vectors=np.array([word_vectors[word] for word in words])

def plot_embeddings(vectors,words,method='PCA'):
    if method=='PCA':
        reduced=PCA(n_components=2).fit_transform(vectors)
    else:
        reduced=TSNE(n_components=2,perplexity=5,random_state=42).fit_transform(vectors)
    
    plt.figure(figsize=(8,6))
    plt.scatter(reduced[:,0],reduced[:,1])

    for i,word in enumerate(words):
        plt.annotate(word,reduced[i,0],reduced[i,1],fontsize=12)
    plt.title(f"Word embeddings method used: {method}")
    plt.show()

plot_embeddings(vectors,words,method="PCA")
plot_embeddings(vectors,words,method="t-SNE")