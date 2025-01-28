import ot
import numpy as np
import scipy as sp

from sklearn.manifold import Isomap
from .pwfunction import procrustes_wasserstein, pw_barycenter



# =============================================================================
# Wasserstein functions
# =============================================================================

def wasserstein_distance(pc1, pc2):
    """
    Compute the Wasserstein distance between two point clouds.
    Args:
        pc1, pc2: Point clouds as (N, 3) numpy arrays.
    Returns:
        float: Wasserstein distance.
    """
    cost_matrix = ot.dist(pc1, pc2)
    weights1 = np.ones(pc1.shape[0]) / pc1.shape[0]  # Uniform weights for pcd1
    weights2 = np.ones(pc2.shape[0]) / pc2.shape[0]  # Uniform weights for pcd2
    distance = ot.emd2(weights1, weights2, cost_matrix)
    
    return distance
    

def wasserstein_barycenter(point_clouds, num_points=256):
    """
    Compute the Wasserstein barycenter of a set of point clouds.
    Args:
        point_clouds: List of point clouds as (N, 3) numpy arrays.
        num_points: Number of points in the barycenter.
    Returns:
        numpy array: Barycenter point cloud.
    """
    d = point_clouds[0].shape[1]
    X_init = np.random.normal(0., 1., (num_points, d))
    
    barycenter, log = ot.lp.free_support_barycenter([pc for pc in point_clouds],
                                               [np.ones(pc.shape[0]) / pc.shape[0] for pc in point_clouds],
                                               X_init=X_init, log=True)
    
    
    return barycenter.reshape(-1, d), log



# =============================================================================
# Gromov-Wasserstein functions
# =============================================================================

def gromov_distance_from_matrices(C1, C2):
    """
    Compute the Gromov-Wasserstein distance between two point clouds.
    Args:
        pc1, pc2: Point clouds as (N, d) numpy arrays.
    Returns:
        float: Gromov-Wasserstein distance.
    """
    
    distance = ot.gromov.gromov_wasserstein2(C1, C2)
    return distance

def gromov_distance_from_pcd_euclidean(pc1, pc2):
    """
    Compute the Gromov-Wasserstein distance between two point clouds.
    Args:
        pc1, pc2: Point clouds as (N, d) numpy arrays.
    Returns:
        float: Gromov-Wasserstein distance.
    """
    C1 = sp.spatial.distance.cdist(pc1, pc1)
    C2 = sp.spatial.distance.cdist(pc2, pc2)
    C1 = C1 / C1.max()
    C2 = C2 / C2.max()
    distance = ot.gromov.gromov_wasserstein2(C1, C2)
    
    return distance


def gromov_distance_from_pcd_geodesic(pc1, pc2):
    """
    Compute the Gromov-Wasserstein distance between two point clouds.
    Args:
        pc1, pc2: Point clouds as (N, d) numpy arrays.
    Returns:
        float: Gromov-Wasserstein distance.
    """
    embedding = Isomap()
    dis1 = embedding.fit(pc1)
    C1 = dis1.dist_matrix_
    dis2 = embedding.fit(pc2)
    C2 = dis2.dist_matrix_
    C1 = C1 / C1.max()
    C2 = C2 / C2.max()
    distance = ot.gromov.gromov_wasserstein2(C1, C2)
    
    return distance
    

def gromov_barycenter(C, num_points=256):
    """
    Compute the Gromov-Wasserstein barycenter of a set of point clouds.
    Args:
        point_clouds: List of point clouds as (N, d) numpy arrays.
        num_points: Number of points in the barycenter.
    Returns:
        numpy array: Barycenter point cloud.
    """

    weights = [np.ones(C[i].shape[0]) / C[i].shape[0] for i in range(len(C))]
    
    if len(C)==1:
        C.append(C[0])
        weights.append(weights[0])
        
    barycenter, log = ot.gromov.gromov_barycenters(num_points, C, weights,
                                              max_iter=50,
                                              tol=1e-3,
                                              log=True)
    
    return barycenter, log
    


# =============================================================================
# Procrustes-Wasserstein functions
# =============================================================================

def procrustes_distance(pc1, pc2):
    """
    Compute the Procrustes-Wasserstein distance between two point clouds.
    Args:
        pc1, pc2: Point clouds as (N, d) numpy arrays.
    Returns:
        float: Procrustes-Wasserstein distance.
    """
    weights1 = np.ones(pc1.shape[0]) / pc1.shape[0]  # Uniform weights for pcd1
    weights2 = np.ones(pc2.shape[0]) / pc2.shape[0]  # Uniform weights for pcd2
    
    P, Gamma, log_dict = procrustes_wasserstein(pc1, pc2, weights1, weights2,
                                                #P_init=P_init,
                                                numItermax=30, stopThr=1e-5,
                                                verbose=False, log=True)
    distance = log_dict['dist']
    return distance


def procrustes_barycenter(point_clouds, num_points=256):
    """
    Compute the Procrustes-Wasserstein barycenter of a set of point clouds.
    Args:
        point_clouds: List of point clouds as (N, d) numpy arrays.
        num_points: Number of points in the barycenter.
    Returns:
        numpy array: Barycenter point cloud.
    """
    d = point_clouds[0].shape[1]
    X_init = np.random.normal(0., 1., (num_points, d))

    barycenter, log = pw_barycenter([pc for pc in point_clouds], 
                                [np.ones(pc.shape[0]) / pc.shape[0] for pc in point_clouds],
                                X_init=X_init, 
                                init_method='wasserstein',
                                numItermax=30, stopThr=1e-3,
                                log=True)
    
    # P = log['procrustes'][0]
    # barycenter = barycenter @ P.T
    return barycenter, log

