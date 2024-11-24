import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

def plot_feature_distribution(features):
    tsne = TSNE(n_components=2)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(features_2d[:, 0], features_2d[:, 1])
    plt.title('Feature Distribution (t-SNE)')
    plt.savefig('evaluation/feature_distribution.png')
    
def compute_similarity_matrix(features):
    return np.dot(features, np.array(features).T)
    
def plot_similarity_heatmap(similarity_matrix):
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix[:50, :50])
    plt.title('Similarity Matrix Heatmap (First 50 samples)')
    plt.savefig('evaluation/similarity_heatmap.png')
    
def calculate_clustering_quality(features):
    return silhouette_score(features, labels=None)