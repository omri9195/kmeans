import numpy as np

def get_random_centroids(X, k):

    '''
    Each centroid is a point in RGB space (color) in the image. 
    Uniformly picks `k` centroids from the dataset.
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids. 
    Notice we are flattening the image to a two dimentional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array. 
    '''
    
    centroids = []
    
    rands = np.random.rand(k)
    rands = rands * X.shape[0]
    rands = np.round(rands)
    for i in range(k):
        centroids.append(X[int(rands[i])])
    
    return np.asarray(centroids).astype(np.float) 



def lp_distance(X, centroids, p=2):

    '''
    Inputs: 
    A single image of shape (num_pixels, 3)
    The centroids (k, 3)
    The distance parameter p

    output: numpy array of shape `(k, num_pixels)` thats holds the distances of 
    all points in RGB space from all centroids
    '''
    distances = []
    k = len(centroids)
    
    for i in range(k):
        body = np.sum(np.abs(X - centroids[i])**(p), axis = 1)
        body = body**(1/p)
        distances.append(body)
    distances = np.array(distances)
    
    return distances

def kmeans(X, k, p ,max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = []
    centroids = get_random_centroids(X, k)
    
    flag = False
    i = 0
    while not flag and i < max_iter:
        prev_centroids = centroids.copy()
        distances = lp_distance(X, centroids, p)
        classes = np.argmin(distances, axis = 0)
        for j in range(k):
            current_class = np.where(classes == j)
            centroids[j] = np.mean(X[current_class],axis = 0)
        flag = np.array_equal(centroids,prev_centroids)
        i += 1
    
    return centroids, classes

def kmeans_pp(X, k, p ,max_iter=100):
    """
    Your implenentation of the kmeans++ algorithm.
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = []
    centroids = []
    
    r = int(np.round(np.random.rand()*X.shape[0]))
    centroids.append(X[r])
    centroids = np.array(centroids,dtype = float)
    Y = np.delete(X,r,0)
    for i in range(k-1):
        distances = lp_distance(Y, centroids, p)
        min_dist = np.min(distances, axis = 0)
        min_dist_sq = min_dist**2
        cent_prob = min_dist_sq / np.sum(min_dist_sq)
        arg = np.random.choice(len(cent_prob), 1, p = cent_prob)
        choice = Y[arg]
        centroids = np.vstack((centroids,choice))
        Y = np.delete(Y,arg,0)

    flag = False
    i = 0
    while not flag and i < max_iter:
        prev_centroids = centroids.copy()
        distances = lp_distance(X, centroids, p)
        classes = np.argmin(distances, axis = 0)
        for j in range(k):
            current_class = np.where(classes == j)
            centroids[j] = np.mean(X[current_class],axis = 0)
        flag = np.array_equal(centroids,prev_centroids)
        i += 1
    
    
    return centroids, classes
