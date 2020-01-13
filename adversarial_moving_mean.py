# %%
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sklearn.neighbors as knn
import and_logic_generator as and_gen
import utils
import applicability_domain as ad
import adversarial_generator as adversarial

# %%
# Repeatable seed
random_state = 2**12
np.random.seed(seed=random_state)

# %%
# Creating samples
n = 1000
x, y = and_gen.generate_logistic_samples(1000)

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
# Generating Adversarial Examples from test set
# This implementation uses multiple iterations to update x, until all of them
# match the target classes.
epsilon = 0.0006

adversarial_examples = np.copy(x_test)  # make a clone
pred = model_svm.predict(x_test)
targets = and_gen.get_not_y(y_test)

epoch = 1
while np.array_equal(pred, targets) == False:
    adversarial_examples = adversarial.moving_mean(
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
y_ae = and_gen.get_y(adversarial_examples)

for xx, ae, p, p_ae, yy_ae in zip(
        x_test, 
        adversarial_examples, 
        original_pred, 
        pred_ae, 
        y_ae):
    if p != p_ae:
        print(f'from [{xx[0]: .4f}, {xx[1]: .4f}] = {p} to ' +
        f'[{ae[0]: .4f}, {ae[1]: .4f}] = {p_ae}; True y = {yy_ae}')

matches = np.equal(y_ae, pred)
count = len(matches[matches==False])
print(f'\nFound {count} Adversarial Examples out of ' +
    f'{len(y_ae)}. {count / len(y_ae) * 100.0:.4f}% successful rate')

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
