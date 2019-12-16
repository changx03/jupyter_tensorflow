# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sklearn.neighbors as knn


# %%
# Repeatable seed
random_state = 2**12
np.random.seed(seed=random_state)


# %%
# The AND logic has 4 possible outcomes: 0^0=0, 0^1=0, 1^0=0, 1^1=1
# Since the state can only be 0 or 1, I populate the sample with Logistic
# Distribution
n = 1000
mean, scale = 0.5, 0.15
x1 = np.random.logistic(mean, scale, n)
x2 = np.random.logistic(mean, scale, n)
x = np.column_stack((x1, x2))


# %%
def get_y(inputs):
    """
    Returns y based on AND logic. If x1 >= 0.5 and x2 >= 0.5 then y = 1, else 
    y = 0.

    Parameters
    ----------
    inputs: array_like
        An array of n by 2 matrix.

    Returns
    -------
    out: darray
        Array of labels.
    """
    return np.array([1 if x[0] >= 0.5 and x[1] >= 0.5 else 0 for x in inputs])


# %%
bins = 100
plt.figure()
plt.hist(x1, bins, density=True, histtype='step', cumulative=True, label='CDF')
plt.show()


# %%
plt.figure()
count, bins, ignored = plt.hist(x1, bins=50)
plt.show()


# %%
y = get_y(x)


# %%
# 1/4 of the samples should output 1
print(y[y == 1].size/n)


# %%
# Increasing the size of the plots
figsize = np.array(plt.rcParams["figure.figsize"]) * 2


# %%
x_min, x_max = -1.0, 2

plt.figure(figsize=figsize.tolist())
plt.scatter(
    x[:, 0], x[:, 1], marker='.', c=y, alpha=0.8, cmap='coolwarm',
    s=8, edgecolor='face')
plt.grid(False)
plt.xlim(x_min, x_max)
plt.ylim(x_min, x_max)
plt.show()


# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)


# %%
# SVM
gamma = 10.0
C = 1000
model_svm1 = svm.SVC(
    kernel='rbf', decision_function_shape='ovo',
    random_state=random_state, gamma=gamma, C=C)
model_svm1.fit(x_train, y_train)


# %%
print(f'With gamma = {gamma} and C = {C}')

y_pred = model_svm1.predict(x_train)
score = accuracy_score(y_train, y_pred)
print(f'Accuracy on train set = {score*100:.4f}%')

y_pred = model_svm1.predict(x_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy on test set = {score*100:.4f}%')


# %%
print(model_svm1.predict([[1., 1.]]))
print(model_svm1.predict([[1., 0.]]))
print(model_svm1.predict([[0., 1.]]))
print(model_svm1.predict([[0., 0.]]))


# %%
h = .01
# by symmetry x and y axis should be in same range
# x_min, x_max = x_test[:, 0].min() - 1, x_test[:, 0].max() + 1
x_min, x_max = -1.0, 2

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(x_min, x_max, h))
Z = model_svm1.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=figsize.tolist())
plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.6)
plt.scatter(
    x_test[:, 0], x_test[:, 1], c=y_test, marker='.', alpha=0.8,
    cmap='coolwarm', s=8, edgecolor='face')
plt.xlim(x_min, x_max)
plt.ylim(x_min, x_max)
plt.show()


# %%
def print_adversarial_examples(model):
    print(f'0.530 ^ 1.420 = {model_svm1.predict([[.53, 1.42]])[0]}')
    print(f'0.540 ^ 1.420 = {model_svm1.predict([[.54, 1.42]])[0]}')
    print(f'0.540 ^ 1.430 = {model_svm1.predict([[.54, 1.43]])[0]}\n')

    print(f'0.530 ^ 1.430 = {model_svm1.predict([[.53, 1.43]])[0]}')
    print(f'0.520 ^ 1.430 = {model_svm1.predict([[.52, 1.43]])[0]}\n')

    print(f'0.500 ^ 0.499 = {model_svm1.predict([[.5, .499]])[0]}')
    print(f'0.501 ^ 0.510 = {model_svm1.predict([[.501, .51]])[0]}')


# %%
print_adversarial_examples(model_svm1)


# %%
model_svm1.get_params()

# %% [markdown]
# ### SVM with overfit parameter
#
# By increasing gamma, kernel function works better in the center.

# %%
gamma = 60.0
C = 1000
model_svm2 = svm.SVC(kernel='rbf', decision_function_shape='ovo',
                     random_state=random_state, gamma=gamma, C=C)
model_svm2.fit(x_train, y_train)


# %%
print(f'With gamma = {gamma} and C = {C}')
y_pred = model_svm2.predict(x_train)
score = accuracy_score(y_train, y_pred)
print(f'Accuracy on train set = {score*100:.4f}%')

y_pred = model_svm2.predict(x_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy on test set = {score*100:.4f}%')


# %%
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(x_min, x_max, h))
Z = model_svm2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=figsize.tolist())
plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.6)
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, marker='.',
            alpha=0.8, cmap='coolwarm', s=8, edgecolor='face')
plt.xlim(x_min, x_max)
plt.ylim(x_min, x_max)
plt.show()


# %%
print_adversarial_examples(model_svm2)


# %%
model_svm2.get_params()

# %% [markdown]
# ## Conclusion
#
# The adversarial examples exist in both models, but in different regions.
# %% [markdown]
# ## Neural Network Model

# %%


# %%
model_nn = keras.Sequential([
    keras.layers.Dense(2,
                       input_shape=(2,),
                       activation='relu',
                       kernel_initializer=tf.initializers.GlorotNormal
                       ),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])


# %%
model_nn.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])


# %%
model_nn.summary()


# %%
epochs = 20
batch_size = 32
model_nn.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
model_nn.evaluate(x_test, y_test, verbose=2)


# %%
model_nn.weights


# %%
print(f'1 ^ 1 = {model_nn.predict([[1.0, 1.0]])[0][0]:.4f}')
print(f'0 ^ 1 = {model_nn.predict([[.0, 1.0]])[0][0]:.4f}')
print(f'1 ^ 0 = {model_nn.predict([[1.0, .0]])[0][0]:.4f}')
print(f'1 ^ 0 = {model_nn.predict([[.0, .0]])[0][0]:.4f}')


# %%
print(f'0.530 ^ 1.420 = {model_nn.predict([[.53, 1.42]])[0][0]:.4f}')
print(f'0.540 ^ 1.420 = {model_nn.predict([[.54, 1.42]])[0][0]:.4f}')
print(f'0.540 ^ 1.430 = {model_nn.predict([[.54, 1.43]])[0][0]:.4f}\n')

print(f'0.530 ^ 1.430 = {model_nn.predict([[.53, 1.43]])[0][0]:.4f}')
print(f'0.520 ^ 1.430 = {model_nn.predict([[.52, 1.43]])[0][0]:.4f}\n')

print(f'0.500 ^ 0.499 = {model_nn.predict([[.5, .499]])[0][0]:.4f}')
print(f'0.501 ^ 0.510 = {model_nn.predict([[.501, .51]])[0][0]:.4f}')

# %% [markdown]
# ## Applicability Domain

# %%
# all positive samples
x_pos = x_train[np.where(y_train == 1)]
x_pos_max = np.amax(x_pos, axis=0)
x_pos_min = np.amin(x_pos, axis=0)
print('Positive range:')
print(f'{x_pos_min[0]:.4f} <= x1 <= {x_pos_max[0]:.4f}')
print(f'{x_pos_min[1]:.4f} <= x2 <= {x_pos_max[1]:.4f}')

# all negative samples
x_neg = x_train[np.where(y_train == 0)]
x_neg_max = np.amax(x_neg, axis=0)
x_neg_min = np.amin(x_neg, axis=0)
print('\nNegative range:')
print(f'{x_neg_min[0]:.4f} <= x1 <= {x_neg_max[0]:.4f}')
print(f'{x_neg_min[1]:.4f} <= x2 <= {x_neg_max[1]:.4f}')

# %% [markdown]
# ### Creating adversarial test set

# %%
# Hand picked samples which contain Adversarial Examples
x_ae = np.array([
    # standard
    [1.0000, 1.0000],  # 1
    [1.0000, 0.0000],  # 0
    [0.0000, 1.0000],  # 0
    [0.0000, 0.0000],  # 0
    # top centre
    [0.5200, 1.4300],  # 1
    [0.5100, 1.4300],  # 1
    [0.4900, 1.4300],  # 0
    [0.4900, 1.4500],  # 0
    # centre
    [0.5020, 0.4990],  # 0
    [0.5010, 0.4990],  # 0
    [0.4990, 0.5010],  # 0
    [0.4980, 0.5010],  # 0
    [0.5010, 0.5010],  # 1
    [0.5020, 0.5010],  # 1
    [0.5010, 0.5020],  # 1
    [0.5020, 0.5020],  # 1
    # out of training range
    # positive
    [0.5003, 1.0000],  # 1
    [1.0000, 0.5001],  # 1
    [1.4584, 1.0000],  # 1
    [1.0000, 1.4010],  # 1
    # negative
    [-0.7756,  0.0000],  # 0
    [1.4500,  0.0000],  # 0
    [0.0000, -0.9839],  # 0
    [0.0000,  1.4690]   # 0
])

y_ae = get_y(x_ae)
print(*y_ae, sep=', ')


# %%
def print_misclassified_samples(x, y, pred):
    ind_mis = np.where(np.logical_xor(y, pred) == True)
    for xx, yy, pp in zip(x[ind_mis], y[ind_mis], pred[ind_mis]):
        print(f'[{xx[0]: .4f}, {xx[1]: .4f}] y = {yy} pred = {pp}')


# %%
# Do NOT change the initial predictions
pred_ae = tuple(model_svm1.predict(x_ae))
print('Misclassified samples:')
print_misclassified_samples(x_ae, y_ae, np.array(pred_ae))


# %%
def get_filtered_indices(inputs, passed_ind):
    """
    Returns an array of indices which are not in the passed_ind.

    Parameters
    ----------
    inputs: array_like
        Array of n by m row vectors, with n rows and each row has m features.

    passed_ind: array
        indices of the inputs.

    Returns
    -------
    exclude_ind: array
        Array of indices which are not in the passed_ind.
    """
    if np.max(passed_ind) >= len(inputs):
        return np.array([])
    return np.where(np.isin(np.array(range(len(inputs))), passed_ind) == False)[0]

# %% [markdown]
# ### Stage 1 - Check Applicability Domain

# %%


def check_applicability(inputs, x_train, y_train, classes=[0, 1]):
    """
    Returning an array of inputs which are passed the bounding box test.

    Parameters
    ----------
    inputs: array_like
        Array of n by m row vectors, with n rows and each row has m features.

    x_train: array_like
        Array of row vectors. Each row has same shape as the inputs.

    y_train: array
        Array of labels. Same length as x_train.

    classes: array
        Array of all possible categorical outputs of y_train.

    Returns
    -------
    outputs: array
        Array of inputs which are passed the test.

    indices: array
        Array of original indices of the outputs which are passed the test.
    """
    # the input only need to within one of the classes
    idx_set = set()

    # looping through classes
    for c in classes:
        x = x_train[np.where(y_train == c)]
        x_max = np.amax(x, axis=0)
        x_min = np.amin(x, axis=0)
        idx = np.where(np.all(np.logical_and(
            inputs <= x_max, inputs >= x_min), axis=1))[0]
        idx_set.update(idx)

    input_indices = np.array(list(idx_set))
    outputs = inputs[input_indices]

    return outputs, input_indices

# %% [markdown]
# ### Testing Applicability Domain


# %%
x_ad, ind_ad = check_applicability(x_ae, x_train, y_train)
print(f'Pass rate = {len(x_ad) / len(x_ae) * 100.0:.4f}%')


# %%
ind_blocked = get_filtered_indices(x_ae, ind_ad)
print('Blocked by Bounding box (Applicability Domain):')
for x in x_ae[ind_blocked]:
    print(f'[{x[0]: .4f}, {x[1]: .4f}]')

# %% [markdown]
# ### Stage 2 - Check Reliability Domain

# %%
# # of neighbours in kNN model
k = 9


# %%
def create_knn_model(x, k):
    """
    Returns a kNN model which is fitted with given x. This function should only
    be used to create single class kNN model. It does NOT take labels as 
    parameter.

    Parameters
    ----------
    x: array_like
        Array of n by m row vectors, with n rows and each row has m features. 
        This is the training set for the kNN model.

    k: int
        # of neighbours

    Returns
    -------
    model: sklearn.neighbors.KNeighborsClassifier
    """
    model = knn.KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    model.fit(x, np.ones(len(x)))
    return model


# %%
# Build one model for each class.
ind_train_pos = np.where(y_train == 1)
model_knn_pos = create_knn_model(x_train[ind_train_pos], k)

ind_train_neg = np.where(y_train == 0)
model_knn_neg = create_knn_model(x_train[ind_train_neg], k)


# %%
def get_dist_info(model, x_train, k):
    """ 
    Returns the mean and standard deviation of distance in the kNN model.

    Parameters
    ----------
    model: sklearn.neighbors.KNeighborsClassifier

    x_train: array_like
        Array of n by m row vectors, with n rows and each row has m features. 
        This is the training set for the kNN model.

    k: int
        # of neighbours.

    Returns
    -------
    mu: float
        Mean distance.

    sd: float
        Standard deviation.
    """
    # k+1: The neighbours will include themselves
    dist, _ = model.kneighbors(x_train, n_neighbors=k+1)
    avg_dist = np.sum(dist, axis=1) / (k + .0)
    mu = np.mean(avg_dist)
    sd = np.std(avg_dist)
    return mu, sd


# %%
mu_pos_dist, sd_pos_dist = get_dist_info(
    model_knn_pos, x_train[ind_train_pos], k)
print('Distance of positive samples in training set:')
print('{:18s} = {:.4f}'.format('Mean', mu_pos_dist))
print('{:18s} = {:.4f}\n'.format('Standard deviation', sd_pos_dist))

mu_neg_dist, sd_neg_dist = get_dist_info(
    model_knn_neg, x_train[ind_train_neg], k)
print('Distance of negative samples in training set:')
print('{:18s} = {:.4f}'.format('Mean', mu_neg_dist))
print('{:18s} = {:.4f}\n'.format('Standard deviation', sd_neg_dist))


# %%
# parameter for proportion
# 95%
zeta = 1.959


# %%
def get_threshold(mean, sd, zeta):
    return mean + zeta * sd


# %%
pos_dist_threshold = get_threshold(mu_pos_dist, sd_pos_dist, zeta)
neg_dist_threshold = get_threshold(mu_neg_dist, sd_neg_dist, zeta)
print(f'Positive distance threshold = {pos_dist_threshold:.4f}')
print(f'Negative distance threshold = {neg_dist_threshold:.4f}')


# %%
def check_reliability(
        inputs, predictions, models, dist_thresholds, k=9, classes=[0, 1],
        verbose=0):
    """
    Returning an array of inputs which are passed the reliability test.
    The reliability test is done by an in-class k nearest neighbours.

    Parameters
    ----------
    inputs: array_like
        Array of n by m row vectors, with n rows and each row has m features. 

    predictions: array
        Array of labels. Same length as this inputs.

    models: array
        Array of sklearn.neighbors.KNeighborsClassifier models

    distance_thresholds: array
        Array of float numbers. The cut-off distance for each class. It must use
        the same order as `classes`.

    k: int
        # of neighbours.

    classes: array
        Array of categorical outputs.

    verbose: int
        Control the verbosity. {0: 'silent', 1: 'print information'}

    Returns
    -------
    outputs: array
        Array of inputs which are passed the test.

    indices: array
        Array of original indices of the outputs which are passed the test.
    """
    if len(models) != len(dist_thresholds) or len(classes) != len(models):
        raise Exception('models and dist_thresholds must have same length.')

    if len(inputs) != len(predictions):
        raise Exception('inputs and predictions must have same length.')

    passed_ind = set()
    # placeholder for indices of inputs
    indices = np.array(range(len(inputs)))

    for model, threshold, c in zip(models, dist_thresholds, classes):
        inclass_ind = np.where(predictions == c)
        neigh_dist, _ = model.kneighbors(
            inputs[inclass_ind], n_neighbors=k, return_distance=True)
        mu = np.mean(neigh_dist, axis=1)
        sub_ind = np.where(mu <= threshold)[0]
        passed_sub_ind = indices[inclass_ind][sub_ind]
        passed_ind.update(passed_sub_ind)
        if verbose == 1:
            print(f'\nIn {c} class:')
            print(f'Threshold = {threshold:.4f}')
            for x, dist in zip(inputs[inclass_ind], mu):
                print(f'[{x[0]: .4f}, {x[1]: .4f}] mean = {dist:.4f}')
            print('Passed indices:')
            print(*passed_sub_ind, sep=', ')

    # convert from set to array
    passed_ind = np.array(list(passed_ind))
    return inputs[passed_ind], passed_ind

# %% [markdown]
# ### Testing reliability domain


# %%
# x_passed_ad, idx_within_ad = check_applicability(test_x, x_train, y_train)
x_passed_rd, idx_within_rd = check_reliability(
    x_ae, np.array(pred_ae), [model_knn_neg, model_knn_pos],
    [neg_dist_threshold, pos_dist_threshold], verbose=1)
print(f'\nPass rate = {len(x_passed_rd) / len(x_ae) * 100.0:.4f}%')


# %%
blocked_ind = get_filtered_indices(x_ae, idx_within_rd)
print(f'# of blocked samples = {len(blocked_ind)}')
print('Blocked by in-class kNN (Reliability Domain):')
for x in x_ae[blocked_ind]:
    print(f'[{x[0]: .4f}, {x[1]: .4f}]')


# %%
# Check
print('Script for testing [1.4584, 1.0000]:')

chec_pred = model_svm1.predict([[1.4584, 1.0000]])
print(f'Prediction from SVM model = {chec_pred[0]}\n')

chec_dist, _ = model_knn_pos.kneighbors(
    [[1.4584, 1.0000]], n_neighbors=k, return_distance=True)
print('Using positive kNN model:')
print(chec_dist[0])
print(np.mean(chec_dist, axis=1)[0])

chec_dist, _ = model_knn_neg.kneighbors(
    [[1.4584, 1.0000]], n_neighbors=k, return_distance=True)
print('\nUsing negative kNN model:')
print(chec_dist[0])
print(np.mean(chec_dist, axis=1)[0])

# %% [markdown]
# ### Conclusion - Stage 2
# When the predictions are different between kNN and original model (SVM in this case), the average distance from the misclassified kNN model will greater than  the correct kNN model.
# As the result, this sample is more likely to be blocked by reliability domain.
# %% [markdown]
# ### Stage 3 - Decidability Domian

# %%
# Decidability Domian uses the kNN model which is trained by the entire training
# set.
# Using the same value from Reliability Domain.
print(f'k = {k}')
model_knn = knn.KNeighborsClassifier(
    n_neighbors=k,
    n_jobs=-1,
    weights='distance'
)
model_knn.fit(x_train, y_train)


# %%
def check_decidability(inputs, predictions, knn_model, verbose=0):
    """
    Returning an array of inputs which are passed the decidability test.
    Filtering the inputs, block it when the prediction from the kNN model does 
    not match the input prediction.

    Parameters
    ----------
    inputs: array_like
        Array of n by m row vectors, with n rows and each row has m features. 

    predictions: array
        Array of labels. Same length as the inputs.

    knn_model: sklearn.neighbors.KNeighborsClassifier

    verbose: int
        Control the verbosity. {0: 'silent', 1: 'print information'}

    Returns
    -------
    outputs: array
        Array of inputs which are passed the test.

    indices: array
        Array of original indices of the outputs which are passed the test.
    """
    if len(inputs) != len(predictions):
        raise Exception('inputs and predictions must have same length.')

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

# %% [markdown]
# ### Testing Decidability Domain


# %%
x_passed_dd, idx_within_dd = check_decidability(
    x_ae, np.array(pred_ae), model_knn, verbose=1)
print(f'\nPass rate = {len(x_passed_dd) / len(x_ae) * 100.0:.4f}%')


# %%
# Utility functions
def get_pass_rate(x, x_passed):
    """Computing passing rate
    """
    return len(x_passed) / len(x) * 1.0


# %%
def print_blocked_samples(x, ind_passed):
    """
    Printing blocked samples.

    Parameters
    ----------
    x: array_like
        Array of n by m row vectors, with n rows and each row has m features. 

    ind_passed: array
        Array of indices.
    """
    ind_blocked = get_filtered_indices(x, ind_passed)
    for x_i in x[ind_blocked]:
        print(f'[{x_i[0]: .4f}, {x_i[1]: .4f}]')

# %% [markdown]
# ## Full pipeline
# The entire pipeline for AD


# %%
# Preparing the test set
x_new = x_ae
y_new = get_y(x_new)

# Prediction from the initial model
model = model_svm1
pred_new = model.predict(x_new)

score = accuracy_score(y_new, pred_new)
print(f'Accuracy on the given set = {score*100:.4f}%')

print('Misclassified samples:')
print_misclassified_samples(x_new, y_new, np.array(pred_new))

# Applicability Domain
# Stage 1 - Applicability
print('\n---------- Applicability ---------------')
x_passed_s1, ind_passed_s1 = check_applicability(x_new, x_train, y_train)
y_passed_s1 = y_new[ind_passed_s1]

# Print infomation

pass_rate_ad = get_pass_rate(x_new, x_passed_s1)
print(f'Pass rate = {pass_rate_ad * 100:.4f}%')
print('Blocked by Applicability Domain:')
print_blocked_samples(x_new, ind_passed_s1)

# State 2 - Reliability
print('\n---------- Reliability -----------------')
# Parameters:
k = 9
zeta = 1.959

# Creating kNN models for each class
ind_train_c1 = np.where(y_train == 1)
model_knn_c1 = create_knn_model(x_train[ind_train_c1], k)

ind_train_c0 = np.where(y_train == 0)
model_knn_c0 = create_knn_model(x_train[ind_train_c0], k)

# Computing mean, standard deviation and threshold
mu_c1, sd_c1 = get_dist_info(model_knn_c1, x_train[ind_train_c1], k)
threshold_c1 = get_threshold(mu_c1, sd_c1, zeta)

mu_c0, sd_c0 = get_dist_info(model_knn_c0, x_train[ind_train_c0], k)
threshold_c0 = get_threshold(mu_c0, sd_c0, zeta)

x_passed_s2, ind_passed_s2 = check_reliability(
    x_passed_s1,
    predictions=y_passed_s1,
    models=[model_knn_c0, model_knn_c1],
    dist_thresholds=[threshold_c0, threshold_c1],
    classes=[0, 1]
)
y_passed_s2 = y_passed_s1[ind_passed_s2]

# Print infomation
print('Distance of c1 in training set:')
print('{:18s} = {:.4f}'.format('Mean', mu_c1))
print('{:18s} = {:.4f}'.format('Standard deviation', sd_c1))
print('{:18s} = {:.4f}\n'.format('Threshold', threshold_c1))

print('Distance of c0 in training set:')
print('{:18s} = {:.4f}'.format('Mean', mu_c0))
print('{:18s} = {:.4f}'.format('Standard deviation', sd_c0))
print('{:18s} = {:.4f}\n'.format('Threshold', threshold_c0))

pass_rate_rd = get_pass_rate(x_passed_s1, x_passed_s2)
print(f'Pass rate = {pass_rate_rd * 100:.4f}%')
print('Blocked by Reliability Domain:')
print_blocked_samples(x_passed_s1, ind_passed_s2)

# Stage 3 - Decidability
print('\n---------- Decidability ----------------')
model_knn = knn.KNeighborsClassifier(
    n_neighbors=k, n_jobs=-1, weights='distance')
model_knn.fit(x_train, y_train)

x_passed_s3, ind_passed_s3 = check_decidability(
    x_passed_s2, y_passed_s2, model_knn)
y_passed_s3 = y_passed_s2[ind_passed_s3]

# Print infomation
pass_rate_dd = get_pass_rate(x_passed_s2, x_passed_s3)
print(f'Pass rate = {pass_rate_dd * 100:.4f}%')
print('Blocked by Decidability Domain:')
print_blocked_samples(x_passed_s2, ind_passed_s3)

# %% [markdown]
# ## Results

# %%
pass_rate = len(x_passed_s3) / len(x_new) * 1.0
print(f'\nOverall pass rate = {pass_rate * 100:.4f}%')
print('Passed samples:')
for x_i in x_passed_s3:
    print(f'[{x_i[0]: .4f}, {x_i[1]: .4f}]')

pred_after_ad = model.predict(x_passed_s3)

score = accuracy_score(y_passed_s3, pred_after_ad)
print(f'Accuracy on the given set = {score*100:.4f}%')
