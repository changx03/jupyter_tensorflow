# %%
import numpy as np

# %%
def check_applicability(inputs, x_train, y_train, classes=[0,1]):
    """
    Check the bounding box of the inputs. This tests the max and min
    value for every class. Each input sample must in range of one of the classes
    in training set.
    This function treats the predictions as black-boxed model.
    
    Parameters
    ----------
    inputs: array_like
        Array of row vectors.
    
    x_train: array_like
        Array of row vectors of the training set.
    
    y_train: array
        Array of labels of the training set.
    
    classes: array (default=[0,1])
        Array of all possible categorical labels.
        By default, it is set for binary classification.
    
    Returns
    -------
    outputs: array
        Array of inputs which are passed this test.
    
    input_indices: array
        Array of original indices.
    """

    if len(inputs) == 0:
        print('No sample has passed into this stage.')
        return inputs, np.array([])

    # the input only need to within one of the classes
    idx_set = set()
    
    # looping through classes
    for c in classes:
        x = x_train[np.where(y_train == c)]
        x_max = np.amax(x, axis=0)
        x_min = np.amin(x, axis=0)
        idx = np.where(
            np.all(np.logical_and(inputs <= x_max, inputs >= x_min), axis=1))[0]
        idx_set.update(idx)
    
    input_indices = np.array(list(idx_set))
    outputs = inputs[input_indices]
    
    return outputs, input_indices

# %%
def check_reliability(
        inputs, predictions, models, dist_thresholds, 
        k=9, classes=[0,1], verbose=0):
    """
    Filtering the inputs by the distance of its in-class k nearest neighbours. 
    This function treats the predictions as black-boxed model.
    
    Parameters
    ----------
    inputs: array_like
        Array of row vectors.
    
    predictions: array
        Array of predicted labels for inputs.
    
    models: array of sklearn.neighbors.KNeighborsClassifier
        Array of kNN models. Each class should have its own model.
        It should have same order as classes (default order is [0, 1]).
    
    distance_thresholds: array
        The cut-off distance for each class. 
        It should have same order as classes (default order is [0, 1]).
    
    k: int (default=9)
        The # of neighbours.

    classes: array (default=[0, 1])
        Categorical outputs
    
    verbose: int (default=0)
        Control the verbosity. {0: 'silent', 1: 'print information'}

    Returns
    -------
    outputs: array
        Array of inputs which are passed this test.
    
    input_indices: array
        Array of original indices.
    """
    if len(models) != len(dist_thresholds) or len(classes) != len(models):
        raise Exception('models and dist_thresholds must have same length.')
    
    if len(inputs) != len(predictions):
        raise Exception('inputs and predictions must have same length.')

    if len(inputs) == 0:
        print('No sample has passed into this stage.')
        return inputs, np.array([])
    
    passed_ind = set()
    # placeholder for indices of inputs
    indices = np.array(range(len(inputs)))

    for model, threshold, c in zip(models, dist_thresholds, classes):
        inclass_ind = np.where(predictions==c)
        neigh_dist, neigh_ind = model.kneighbors(
            inputs[inclass_ind], n_neighbors=k, return_distance=True)
        mu = np.mean(neigh_dist, axis=1)
        sub_ind = np.where(mu <= threshold)[0]
        passed_sub_ind = indices[inclass_ind][sub_ind]
        passed_ind.update(passed_sub_ind)
        if verbose == 1:
            print(f'\nIn {c} class:')
            print(f'Threshold = {threshold:.4f}')
            print(f'Average mean = {np.mean(mu):.4f}')
            # for x, dist in zip(inputs[inclass_ind], mu):
            #     print(f'[{x[0]: .4f}, {x[1]: .4f}] mean = {dist:.4f}')
            print('Passed indices:')
            print(*passed_sub_ind, sep=', ')
    
    # convert from set to array
    passed_ind = np.array(list(passed_ind), dtype=np.int)
    return inputs[passed_ind], passed_ind

# %%
def check_decidability(inputs, predictions, knn_model, verbose=0):
    """
    Filtering the inputs by the distance of its in-class k nearest neighbours. 
    This function treats the predictions as black-boxed model.
    
    Parameters
    ----------
    inputs: array_like
        Array of row vectors.
    
    predictions: array
        Array of predicted labelsfor inputs.
    
    models: sklearn.neighbors.KNeighborsClassifier
        a kNN model which fits the entire training set.

    verbose: int (default=0)
        Control the verbosity. {0: 'silent', 1: 'print information'}

    Returns
    -------
    outputs: array
        Array of inputs which are passed this test.
    
    input_indices: array
        Array of original indices.
    """
    if len(inputs) != len(predictions):
        raise Exception('inputs and predictions must have same length.')

    if len(inputs) == 0:
        print('No sample has passed into this stage.')
        return inputs, np.array([])

    pred_knn = knn_model.predict(inputs)
    ind_match = np.where(np.equal(predictions, pred_knn))
    if verbose == 1:
        print('Prediction from kNN:')
        print(*pred_knn, sep=', ')
        print('Expected outputs:')
        print(*predictions, sep=', ')
        print('Matched indices:')
        print(*ind_match[0], sep=', ')
        print(f'# of inputs = {len(inputs)}')
        print(f'# of passed = {len(ind_match[0])}')

    return inputs[ind_match], ind_match

# %%
def get_reliability_threshold(mean, sd, zeta):
    """
    Returns the threshold based on Normal distribution

    Parameters
    ----------
    mean: float
        The mean value for Normal distribution.

    sd: float
        The standard deviation for Normal distribution.
    
    zeta: float
        The standard normal deviate.

    Returns
    -------
    threshold: float
        The cutoff threshold.
    """
    return mean + zeta * sd
