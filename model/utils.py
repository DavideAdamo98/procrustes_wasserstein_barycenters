import ot
import random
import numpy as np

from sklearn import manifold
from scipy.sparse import diags
from scipy.stats import ortho_group
from scipy.sparse.linalg import eigs
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph


def normalize(data):
    
    centroid = np.mean(data, axis=0)
    data -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(data) ** 2, axis=-1)))
    data /= furthest_distance
    
    return data


def fiedler(X, method='kng', param=50):
    """
    Returns the fiedler vector of a point cloud
    
    Parameters
    ----------
    X : ndarray, shape (n, d)
        point cloud
    method : str, optional
        method for constructing the graph, in can be 'radius' or 'kng'. The default is 'kng'.
    param : int, optional
        parameter associated to the method (it represents the radius value if 'radius' or the number 
        of nearest neighboors if 'kng'). The default is 50.

    Returns
    -------
    v : ndarray, shape (1, n)
        fiedler vector.

    """    
    if method=='radius':
        A = radius_neighbors_graph(X, param, mode='connectivity',include_self=True)
    elif (method == 'kng'):
        A_ = kneighbors_graph(X, param, mode='connectivity', include_self=True)
        A = (A_ + A_.T) * .5

    N = A.shape[0]
    deg = np.sum(A, axis = 0).A1
    L = diags(deg, 0, shape = (N,N)) - A
    lbd, Q = eigs(L, k=3, which = 'SM')
    pos = np.argsort(lbd)
    V = Q[:,pos]
    Fiedler = V[:,1].real
    ff = Fiedler - np.min(Fiedler)
    v = ff/np.max(ff)
    
    return v


def fiedler_matching(x, y):
    
    fx   = normalize(fiedler(x, method='kng', param=int(x.shape[0]*.2))).reshape(-1,1)
    fy   = normalize(fiedler(y, method='kng', param=int(y.shape[0]*.2))).reshape(-1,1)
    
    F = [fy, -fy]
    min_distance = 1000
    for f in F:
        
        cost_matrix = ot.dist(fx, f)
        weights1 = np.ones(x.shape[0]) / x.shape[0]
        weights2 = np.ones(y.shape[0]) / y.shape[0]
        Gamma, log = ot.emd(weights1, weights2, cost_matrix, log=True)
        distance = log['cost']
        #print('- fiedler distance: ', distance)
        if distance < min_distance:
            min_distance = distance
            Gamma0 = Gamma
    
    U, E, Vh = np.linalg.svd(y.T @ Gamma0.T @ x)
    P = U @ Vh
    return P



def smacof_mds(C, dim, max_iter=3000, eps=1e-9):
    """
    Returns an interpolated point cloud following the dissimilarity matrix C
    using SMACOF multidimensional scaling (MDS) in specific dimensioned
    target space

    Parameters
    ----------
    C : ndarray, shape (ns, ns)
        dissimilarity matrix
    dim : int
          dimension of the targeted space
    max_iter :  int
        Maximum number of iterations of the SMACOF algorithm for a single run
    eps : float
        relative tolerance w.r.t stress to declare converge

    Returns
    -------
    npos : ndarray, shape (R, dim)
           Embedded coordinates of the interpolated point cloud (defined with
           one isometry)
    """

    rng = np.random.RandomState(seed=3)

    mds = manifold.MDS(
        dim,
        max_iter=max_iter,
        eps=1e-9,
        dissimilarity='precomputed',
        n_init=1,
        normalized_stress='auto')
    pos = mds.fit(C).embedding_

    nmds = manifold.MDS(
        2,
        max_iter=max_iter,
        eps=1e-9,
        dissimilarity="precomputed",
        random_state=rng,
        n_init=1,
        normalized_stress='auto')
    npos = nmds.fit_transform(C, init=pos)

    return npos


def uncorrelate(x):
    Sigma = np.cov(x.T)
    eigen_val, eigen_vec = np.linalg.eig(Sigma)
    Q = eigen_vec
    return Q


def pca_wass_2d(x, y):
    
    Qx = uncorrelate(x)
    Qy = uncorrelate(y)
    
    A = Qy.T @ Qx
    x_ = x @ A
    
    s1 = np.eye(2) # no reflection
    s2 = np.array([[-1., 0.],[0., 1.]]) # y-axis reflection
    s3 = np.array([[1., 0.],[0., -1.]]) # x-axis reflection
    s4 = -np.eye(2) # xy-axis reflection
    t = np.pi/2
    R = np.array([[np.cos(t), -np.sin(t)],
                  [np.sin(t), np.cos(t)]])
    S = [s1, s2, s3, s4, s1@R, s2@R, s3@R, s4@R]
    
    min_distance = 1000
    for s in S:
        candidate = x_ @ s
        cost_matrix = ot.dist(candidate, y)
        weights1 = np.ones(candidate.shape[0]) / candidate.shape[0]
        weights2 = np.ones(y.shape[0]) / y.shape[0]
        distance = ot.emd2(weights1, weights2, cost_matrix)
        # print('- pca_wass distance: ', distance)
        if distance < min_distance:
            min_distance = distance
            best_s = s
    
    P = A @ best_s
    return P


def pca_wass_3d(x, y):
    
    Qx = uncorrelate(x)
    Qy = uncorrelate(y)
    
    A = Qy.T @ Qx
    x_ = x @ A
    
    s1 = np.eye(3) # no reflection
    s2 = np.array([[-1, 0, 0],[0, 1, 0], [0, 0, 1]])
    s3 = np.array([[1, 0, 0],[0, -1, 0], [0, 0, 1]])
    s4 = np.array([[1, 0, 0],[0, 1, 0], [0, 0, -1]])
    s5 = np.array([[-1, 0, 0],[0, -1, 0], [0, 0, 1]])
    s6 = np.array([[-1, 0, 0],[0, 1, 0], [0, 0, -1]])
    s7 = np.array([[1, 0, 0],[0, -1, 0], [0, 0, -1]])
    s8 = -np.eye(3) # xyz-axis reflection
    S = [s1, s2, s3, s4, s5, s6, s7, s8]
    
    min_distance = 1000
    for s in S:
        candidate = x_ @ s
        cost_matrix = ot.dist(candidate, y)
        weights1 = np.ones(candidate.shape[0]) / candidate.shape[0]
        weights2 = np.ones(y.shape[0]) / y.shape[0]
        distance = ot.emd2(weights1, weights2, cost_matrix)
        # print('- pca_wass distance: ', distance)
        if distance < min_distance:
            min_distance = distance
            best_s = s
    
    P = A @ best_s
    return P


# =============================================================================
# Create data sets
# =============================================================================

def create_2D_class(pivot, color, num_samples, noise=[.01,.02], rotate=False, seed=None):
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n, d = pivot.shape
    dataset_non_ruotato = []
    dataset_ruotato     = []
    dataset_colori      = []
    
    for i in range(num_samples):
        
        frequency = np.random.choice(a=[1, 2, 3], 
                                     size=n,
                                     p=[.90, .07, .03])
        
        cand = np.concatenate((pivot, pivot[frequency!=1], pivot[frequency==3]))
        color_cand = np.concatenate((color, color[frequency!=1], color[frequency==3]))
        
        if rotate==True:
            t = random.uniform(0, 2*np.pi)
            R = np.array([[np.cos(t), -np.sin(t)],
                          [np.sin(t), np.cos(t)]])
        else:
            R = np.eye(2)
        
        eps = np.random.normal(noise[0], noise[1], [cand.shape[0], d]) 
        new_cloud = cand + eps
        
        idx = np.random.permutation(new_cloud.shape[0])
        new_cloud = new_cloud[idx]
        color_cand = color_cand[idx]
        dataset_non_ruotato.append(new_cloud)
        dataset_colori.append(color_cand)
        
        # Apply random symmetry (flip along the x-axis or y-axis)
        apply_symmetry = np.random.choice([True, False])
        if apply_symmetry:
            S = np.array([[1, 0], [0, -1]])  # Reflection over the y-axis
        else:
            S = np.eye(2)  # No symmetry applied
        
        dataset_ruotato.append(new_cloud @ R @ S)
    return dataset_non_ruotato, dataset_ruotato, dataset_colori



def create_3D_class(pivot, color, num_samples, noise=[.01,.02], rotate=False, seed=None):
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    n, d = pivot.shape
    dataset_non_ruotato = []
    dataset_ruotato     = []
    dataset_colori      = []
    
    for i in range(num_samples):
        
        frequency = np.random.choice(a=[1, 2, 3], 
                                     size=n,
                                     p=[.90, .07, .03])
        
        cand = np.concatenate((pivot, pivot[frequency!=1], pivot[frequency==3]))
        color_cand = np.concatenate((color, color[frequency!=1], color[frequency==3]))
        # 1) Add gaussian noise
        eps = np.random.normal(noise[0], noise[1], [cand.shape[0], d]) 
        new_cloud = cand + eps
        # 2) Random permutation of the vertices
        idx = np.random.permutation(new_cloud.shape[0])
        new_cloud = new_cloud[idx]
        color_cand = color_cand[idx]
        dataset_non_ruotato.append(new_cloud)
        dataset_colori.append(color_cand)
        
        # 3) Apply random rotation/symmetry (flip along the x-axis or y-axis)
        if rotate==True:
            R = ortho_group.rvs(dim=3)
        else:
            R = np.eye(3)
        
        dataset_ruotato.append(new_cloud @ R)
    return dataset_non_ruotato, dataset_ruotato, dataset_colori