import random
import numpy as np
import scipy as sp

random.seed(77)
np.random.seed(77)

from sklearn.manifold import Isomap
from sklearn.cluster  import KMeans

from .otfunctions import (wasserstein_distance, wasserstein_barycenter,
                         gromov_distance_from_matrices, gromov_barycenter,
                         gromov_distance_from_pcd_euclidean, gromov_distance_from_pcd_geodesic,
                         procrustes_distance, procrustes_barycenter)


# =============================================================================
# K-means custom distance
# =============================================================================

def sk_kmeans_centroid(X, num_points):
    k_means = KMeans(n_clusters=num_points, n_init='auto')
    k_means.fit(X)
    return k_means.cluster_centers_


def custom_kmeans_plusplus(data, num_clusters, num_points, compute_distance='wasserstein'):
    '''
    Initialisation method, inspired to kmeans++, that samples an optimal
    initial configuration for the centroids.
    
    Parameters
    ----------
    data : (numpy arrays)
        List of point clouds.
    num_clusters : (int)
        Number of clusters, which corresponds to the number of centroids.
    num_points : (int)
        Number of points of the centroids.

    Returns
    -------
    initial_centroids: Initial centroids configuration for kmean
    '''
    
    initial_centroids = []
    list_of_candidates = []
    
    initial_index = random.choices(range(len(data)))
    candidate = data[initial_index[0]]
    list_of_candidates.append(candidate)
    if candidate.shape[0] > num_points:
        centroids = sk_kmeans_centroid(candidate, num_points)
    else:
        centroids = candidate
    initial_centroids.append(centroids)
    
    M = np.zeros((num_clusters, len(data)))
    
    for i in range(num_clusters-1):
        distance = np.array([compute_distance(initial_centroids[i], pc) for pc in data])
        M[i,:] = distance
            
        # Calculate the distance of each point from its nearest centroid
        min_dist = np.min(M[:i+1, :], axis=0)
        candidate = data[np.argmax(min_dist)]
        list_of_candidates.append(candidate)
        if candidate.shape[0] > num_points:
            centroids = sk_kmeans_centroid(candidate, num_points)
        else:
            centroids = candidate
        initial_centroids.append(centroids)
    
    return initial_centroids



def custom_kmeans(data, num_clusters, 
                  max_iter=100, num_points=256, 
                  ot_metric='wasserstein', 
                  feature_metric='euclidean'):
    """
    Perform K-means clustering using a custom distance.
    Args:
        data: List of point clouds (numpy arrays).
        num_clusters: Number of clusters.
        max_iter: Maximum number of iterations.
        num_points: Number of points in barycenters.
    Returns:
        centroids: Final cluster centroids (barycenters).
        labels: Cluster assignments for each point cloud.
    """
    
    if ot_metric == 'wasserstein':
        compute_distance = wasserstein_distance
        compute_barycenter = wasserstein_barycenter
    elif ot_metric == 'gromov-wasserstein' and feature_metric=='euclidean':
        compute_distance = gromov_distance_from_pcd_euclidean
        compute_barycenter = gromov_barycenter
    elif ot_metric == 'gromov-wasserstein' and feature_metric=='geodesic':
        compute_distance = gromov_distance_from_pcd_geodesic
        compute_barycenter = gromov_barycenter
    elif ot_metric == 'procrustes-wasserstein':
        compute_distance = procrustes_distance
        compute_barycenter = procrustes_barycenter
    else:
        print('Non valid metric')
        
    
    # Initialize the centroids via kmeans++ strategy
    centroids = custom_kmeans_plusplus(data, num_clusters, num_points, compute_distance)
    
    if ot_metric == 'gromov-wasserstein':
        compute_distance = gromov_distance_from_matrices
        if feature_metric == 'euclidean':
            data = [sp.spatial.distance.cdist(x, x) for x in data]
            data = [di / di.max() for di in data]
            centroids = [sp.spatial.distance.cdist(x, x) for x in centroids]
            centroids = [ci / ci.max() for ci in centroids]
        elif feature_metric == 'geodesic':
            embedding = Isomap()
            data = [embedding.fit(x).dist_matrix_ for x in data]
            data = [di / di.max() for di in data]
            centroids = [embedding.fit(x).dist_matrix_ for x in centroids]
            centroids = [ci / ci.max() for ci in centroids]
        
    
    
    predictions = np.zeros(len(data))
    cluster_loss = []
    for iteration in range(max_iter):
        
        '''
        1) Assignment Steps
        '''
        wcss_list = []
        clusters = {i: [] for i in range(num_clusters)}
        for i, shape in enumerate(data):
            distances = [compute_distance(shape, centroid) for centroid in centroids]
            cluster_idx = np.argmin(distances).astype(int)
            clusters[cluster_idx].append(shape)
            predictions[i] = cluster_idx
            wcss_list.append(np.min(distances)**2)
        
        print('\n- iteration:', iteration)
        print(predictions)
        print('loss value:', np.sum(wcss_list))
        cluster_loss.append(np.sum(wcss_list))
        
        '''
        2) Update Steps
        '''
        new_centroids = []
        dictionary = []
        for i in range(num_clusters):
            if clusters[i]:
                bar, loss_bar = compute_barycenter(clusters[i], num_points)
                new_centroids.append(bar)
                if ot_metric == 'procrustes-wasserstein':
                    dictionary.append(loss_bar['procrustes'])
            else:
                new_centroids.append(centroids[i])  # Handle empty clusters
            print('Updated centroid', i)
        
        
        
        '''
        3) Convergence Check
        '''
        count = 0
        for i in range(num_clusters):
            delta = compute_distance(centroids[i], new_centroids[i])
            print(delta)
            if delta < 1e-5:
                count += 1
        print(count)
        if count == num_clusters:
            break
    
        centroids = new_centroids

    
    
    return centroids, predictions, cluster_loss, dictionary