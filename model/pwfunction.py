import ot
import numpy as np
import scipy as sp
# import networkx as netx

from ot.backend import get_backend
from .utils import fiedler_matching


def procrustes_wasserstein(X, Y, 
                           p, q, 
                           P_init=None, 
                           numItermax=100, 
                           stopThr=1e-10, 
                           verbose=False, 
                           log=None,
                           numThreads=1):    
    a = np.sum(X**2, 1)
    b = np.sum(Y**2, 1)

    if P_init is None:
        P_init = np.identity(X.shape[1])
    P = P_init
    
    
    old = 10
    log_dict = {}
    norm = stopThr + 1.
    iter_count = 0    
    
    while (norm > stopThr and iter_count < numItermax):
        M = ot.dist(X ,Y @ P )
        
        Gamma = ot.emd(p, q, M, numThreads=numThreads)
        
        U, E, Vh = np.linalg.svd(Y.T @ Gamma.T @ X)
        P = U @ Vh
        
        # Compute PW distance
        dist = (a.T @ p) + (b.T @ q) - 2 * np.trace(Y @ P @ X.T @ Gamma)
        
        if verbose:
            print('distance : ', dist)
        
        norm = np.abs(old - dist)
        old = dist
        
        iter_count += 1
        

    if log:
        log_dict['dist'] = dist
        log_dict['err'] = norm
        return P, Gamma, log_dict
    else:
        return P, Gamma



def pw_barycenter(measures_locations, measures_weights, X_init, b=None, 
                  weights=None, init_method='wasserstein', numItermax=100,
                  stopThr=1e-10, verbose=False, log=None, numThreads=1):
    
    r"""
    Solves the (locations of the barycenters are optimized, not the weights) Procrustes-Wasserstein barycenter problem, formally:

    .. math::
        \min_\mathbf{X} \quad \sum_{i=1}^N w_i PW(\mathbf{b}, \mathbf{X}, \mathbf{a}_i, \mathbf{X}_i)

    where :

    - :math:`w \in \mathbb{(0, 1)}^{N}` are the barycenter weights and sum to one
    - `measure_weights` denotes the :math:`\mathbf{a}_i \in \mathbb{R}^{k_i}`: empirical measures weights (on simplex)
    - `measures_locations` denotes the :math:`\mathbf{X}_i \in \mathbb{R}^{k_i, d}`: empirical measures atoms locations
    - :math:`\mathbf{b} \in \mathbb{R}^{k}` is the desired weights vector of the barycenter

    Parameters
    ----------
    measures_locations : list of N (k_i,d) array-like
        The discrete support of a measure supported on :math:`k_i` locations of a `d`-dimensional space
        (:math:`k_i` can be different for each element of the list)
    measures_weights : list of N (k_i,) array-like
        Numpy arrays where each numpy array has :math:`k_i` non-negatives values summing to one
        representing the weights of each discrete input measure

    X_init : (k,d) array-like
        Initialization of the support locations (on `k` atoms) of the barycenter
    b : (k,) array-like
        Initialization of the weights of the barycenter (non-negatives, sum to 1)
    weights : (N,) array-like
        Initialization of the coefficients of the barycenter (non-negatives, sum to 1)

    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    numThreads: int or "max", optional (default=1, i.e. OpenMP is not used)
        If compiled with OpenMP, chooses the number of threads to parallelize.
        "max" selects the highest number possible.


    Returns
    -------
    X : (k,d) array-like
        Support locations (on k atoms) of the barycenter


    References
    ----------
    .. 
    
    """
    
    nx = get_backend(*measures_locations, *measures_weights, X_init)

    N = len(measures_locations)       # n. of distributions

    k = X_init.shape[0]
    d = X_init.shape[1]
    if b is None:
        b = nx.ones((k,), type_as=X_init) / k
    if weights is None:
        weights = nx.ones((N,), type_as=X_init) / N
    
    X = X_init
    log_dict = {}
    displacement_square_norms = []
    displacement_square_norm = stopThr + 1.
    iter_count = 0

    while (displacement_square_norm > stopThr and iter_count < numItermax):
        
        T_sum = nx.zeros((k, d), type_as=X_init)
        
        store_P = []
        store_T = []

        for (measure_locations_i, measure_weights_i, weight_i) in zip(measures_locations, measures_weights, weights):
            
            '''
            1)  Initialize the PW problem
            '''
            if init_method == 'wasserstein':
                P_init = np.identity(d)
            elif init_method == 'gromov-wasserstein':
                C1 = sp.spatial.distance.cdist(X, X)
                C2 = sp.spatial.distance.cdist(measure_locations_i, measure_locations_i)
                gw0 = ot.gromov.gromov_wasserstein(C1, C2, 
                                                   b, measure_weights_i,
                                                   max_iter=100, 
                                                   tol_rel=1e-3, 
                                                   tol_abs=1e-3
                                                   )
                
                U, E, Vh = np.linalg.svd(measure_locations_i.T @ gw0.T @ X)
                P_init = U @ Vh
            elif init_method == 'fiedler':
                P_init = fiedler_matching(X, measure_locations_i)

            '''
            2)  Solve pair-wised PW problem
            '''
            P_i, T_i = procrustes_wasserstein(X, measure_locations_i,
                                              b, measure_weights_i, 
                                              P_init,
                                              verbose=verbose,
                                              numThreads=numThreads)
            
            '''
            3) Update barycenter
            '''
            T_sum = T_sum + weight_i * 1. / b[:, None] * nx.dot(nx.dot(T_i, measure_locations_i), P_i)
            
            
            store_P.append(P_i)
            store_T.append(T_i)

        displacement_square_norm = nx.sum((T_sum - X) ** 2)
        
        if log:
            displacement_square_norms.append(displacement_square_norm)

        X = T_sum

        if verbose:
            print('\n- Iter. ', iter_count,', displacement_square_norm = ', displacement_square_norm)

        iter_count += 1

    if log:
        log_dict['displacement_square_norms'] = displacement_square_norms
        log_dict['procrustes'] = store_P
        log_dict['transport_plans'] = store_T
        return X, log_dict
    else:
        return X
    















