# %%
import numpy as np

# %%
def moving_mean(x, y, targets, means, epsilon, epoch=None, verbose=0):
    """
    Generating Adversarial Examples by moving x toward to another class
    
    Parameters
    ----------
    x: array_like
        Array of n by m row vectors, with n rows and each row has m features.

    y: array
        Array of labels. It should have same length as x.
    
    targets: array
        Array of targetting labels. It should have same length as x.
        TODO: Multi-class implementation is NOT implemented.
        
    means: dictionary
        Means for each class. In shape of {int: array}.
        e.g.: {<class_label_1>: [<mean_x1>, <mean_x2>]}
    
    epsilon: float
        Step size.
    
    verbose: int
        Control the verbosity. {0: 'silent', 1: 'print information'}
        
    Returns
    -------
    outputs: array
        Array of row vectors. Generated from x.
    """
    need_update = np.equal(y, targets)
    # not equal
    ind = np.where(need_update == False)
    
    # only compute the samples which haven't meet the target class
    # convert 1xn labels into nx2
    tar_trans = targets[ind]
    n_features = x.shape[1]
    tar_trans = np.repeat([tar_trans], n_features, axis=0).T
    
    # the opposite of tar_trans
    invert_tar_trans = np.logical_not(tar_trans)
    
    # convert to float64
    tar_trans = tar_trans.astype('float64')
    invert_tar_trans = invert_tar_trans.astype('float64')

    # build target mean map
    t = tar_trans * means[1] + invert_tar_trans * means[0]

    if verbose == 1:
        print(f'Epoch:{epoch:3d} -> # of updated samples = {len(ind[0])}')
        print('Updated:', end=' ')
        ii = ind[0]
        print(*ii, sep=',')

    sign_map = np.sign(t - x[ind])
    x[ind] = x[ind] + epsilon * sign_map

    return x
