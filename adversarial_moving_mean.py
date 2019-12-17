# %%
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sklearn.neighbors as knn

# %%
# Repeatable seed
random_state = 2**12
np.random.seed(seed=random_state)

# %%
# Creating samples
n = 1000
mean, scale = 0.5, 0.15

x = np.random.logistic(mean, scale, n * 2)
x = x.reshape((n, 2))
print(x.shape)
x[:5]
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
y = get_y(x)

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# %%
import matplotlib.pyplot as plt
figsize = np.array(plt.rcParams["figure.figsize"]) * 2
print(figsize)

x_min, x_max = -1.0, 2

plt.figure(figsize=figsize.tolist())
plt.scatter(
    x_train[:, 0], x_train[:, 1], marker='.', c=y_train, alpha=0.8, 
    cmap='coolwarm', s=8, edgecolor='face')
plt.grid(False)
plt.xlim(x_min, x_max)
plt.ylim(x_min, x_max)
plt.show()

# %%
# SVM
gamma, c = 10.0, 1000

model_svm = svm.SVC(
    kernel='rbf', random_state=random_state, gamma=gamma, C=c)
model_svm.fit(x_train, y_train)

# %%
y_train_pred = model_svm.predict(x_train)
score = accuracy_score(y_train, y_train_pred)
print(f'Accuracy on train set = {score*100:.4f}%')

y_test_pred = model_svm.predict(x_test)
score = accuracy_score(y_test, y_test_pred)
print(f'Accuracy on test set  = {score*100:.4f}%')

# %%
h = .01

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(x_min, x_max, h))
Z = model_svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=figsize.tolist())
plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.6)
plt.scatter(
    x_test[:, 0], x_test[:, 1], c=y_test_pred, marker='.', alpha=0.8,
    cmap='coolwarm', s=8, edgecolor='face')
plt.xlim(x_min, x_max)
plt.ylim(x_min, x_max)
plt.show()

# %%
model_svm.get_params()

# %%
ind_train_c1 = np.where(y_train == 1)
x_train_c1 = x_train[ind_train_c1]
y_train_c1 = np.ones(len(x_train_c1))
mu_train_c1 = np.mean(x_train_c1, axis=0)

ind_train_c0 = np.where(y_train == 0)
x_train_c0 = x_train[ind_train_c0]
y_train_c0 = np.zeros(len(x_train_c0))
mu_train_c0 = np.mean(x_train_c0, axis=0)

print(f'Positive mean = [{mu_train_c1[0]:.4f}, {mu_train_c1[1]:.4f}]')
print(f'Negative mean = [{mu_train_c0[0]:.4f}, {mu_train_c0[1]:.4f}]')

# %%
def get_not_y(y):
    """
    Apply logical not on y and return it

    Parameters
    ----------
    y: array
        Array of binary labels
    
    Returns
    -------
    outputs: array
        Array of binary labels after logical not applied on y
    """
    return np.array(np.logical_not(y), dtype=np.int)

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
    tar_trans = np.transpose([targets[ind]])
    tar_trans = np.hstack((tar_trans, tar_trans))
    
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

# %%
# Generating Adversarial Examples from test set
# This implementation uses multiple iterations to update x, until all of them
# match the target classes.
epsilon = 0.0006

adversarial_examples = np.copy(x_test)  # make a clone
pred = model_svm.predict(x_test)
targets = get_not_y(y_test)

epoch = 1
while np.array_equal(pred, targets) == False:
    adversarial_examples = moving_mean(
        x=adversarial_examples,
        y=pred,
        targets=targets,
        means={0: mu_train_c0, 1: mu_train_c1},
        epsilon=epsilon,
        verbose=0,
        epoch=epoch)
    pred = model_svm.predict(adversarial_examples)
    epoch += 1

print(f'Completed after {epoch} epoch.')

# %%
# Results
original_pred = model_svm.predict(x_test)
pred_ae = model_svm.predict(adversarial_examples)
y_ae = get_y(adversarial_examples)

for xx, ae, p, p_ae, yy_ae in zip(
        x_test, 
        adversarial_examples, 
        original_pred, 
        pred_ae, 
        y_ae):
    if p != p_ae:
        print(f'from [{xx[0]: .4f}, {xx[1]: .4f}] = {p} to [{ae[0]: .4f}, {ae[1]: .4f}] = {p_ae}; True y = {yy_ae}')

matches = np.equal(y_ae, pred)
count = len(matches[matches==False])
print(f'\nFound {count} Adversarial Examples out of {len(y_ae)}. {count / len(y_ae) * 100.0:.4f}% successful rate')

# %%
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(x_min, x_max, h))
Z = model_svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=figsize.tolist())
plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.6)
plt.scatter(
    adversarial_examples[:, 0], adversarial_examples[:, 1], 
    c=pred_ae, marker='.', alpha=0.8, cmap='coolwarm', s=8, edgecolor='face')
plt.xlim(x_min, x_max)
plt.ylim(x_min, x_max)
plt.show()

# %%
