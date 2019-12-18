# %%
import numpy as np
import sklearn.neighbors as knn

# %%
def get_filtered_indices(inputs, passed_ind):
    """
    Returns the indices where are not in the passed_ind.

    Parameters
    ----------
    inputs: array_like
        Array of row vectors.

    passed_ind: array
        Array of indices. The index should always smaller than len(inputs).
    
    Returns
    -------
    exclude_ind: array
        Array of indices which are not found in passed_ind.
    """
    if np.max(passed_ind) >= len(inputs):
        return np.array([])
    return np.where(
        np.isin(np.array(range(len(inputs))), passed_ind) == False)[0]

# %%
def unimodal_knn(x, k):
    """
    Returns a kNN model for single class.

    Parameters
    ----------
    x: array_like
        Array of row vectors.

    k: int
        # of neighbours.

    Returns
    -------
    model: sklearn.neighbors.KNeighborsClassifier
        The fitted kNN model.
    """
    model = knn.KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    model.fit(x, np.ones(len(x)))
    return model

# %%
def get_distance_info(knn_model, x, k, seen_in_train_set=False):
    """
    Return the mean and standard deviation of distance for given x.

    Parameters
    ----------
    knn_model: sklearn.neighbors.KNeighborsClassifier
        Fitted kNN model.
    
    k: int
        The # of neighbours will be used for calculating mu and s.d..
    
    seen_in_train_set: bool (default=False)
        Is x already seen in training set?

    Returns
    -------
    mu: float
        mean distance

    sd: float
        standard deviation
    """

    if seen_in_train_set is True:
        n = k + 1
    else:
        n = k
    
    dist, _ = knn_model.kneighbors(x, n_neighbors=n)
    avg_dist = np.sum(dist, axis=1) / float(k)
    mu = np.mean(avg_dist)
    sd = np.std(avg_dist)
    return mu, sd

# %%
def get_rate(numerator, denominator):
    """
    Returns the rate between the length of numerator and the length of 
    denominator.

    Parameters
    ----------
    numerators: array
        A list of values.

    denominators: array
        A list of values.

    Returns
    -------
    rate: float
        The rate between the length of numerator and the length of denominator.
    """
    if len(denominator) == 0:
        return .0
    if len(denominator) <= len(numerator):
        return 1.0
    return len(numerator) / float(len(denominator))