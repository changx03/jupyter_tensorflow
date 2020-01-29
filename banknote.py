# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import sklearn.neighbors as knn

import and_gate_pipeline as pipeline
from and_logic_generator import get_not_y
import adversarial_generator as adversarial
import applicability_domain as ad
import utils

# %load_ext autoreload
# %autoreload 2

# %%
# Repeatable random generator
seed = 2**12
np.random.seed(seed=seed)

# %%
# Step 1: Load data
file_name = 'data_banknote_authentication.csv'
data = np.genfromtxt(file_name, delimiter=',')
data.shape

# %%
x = data[:, :4]
y = np.array(data[:, 4], dtype=np.int)
print(x.shape)
print(y.shape)
print(x[:5,:])
print(y[:5])
print(len(y[y==1]))

# %%
# Step 2: Preprocessing
# zero mean
x_mean = np.mean(x, axis=0)
x = x - x_mean

# scale to [-1, 1] range
x = preprocessing.minmax_scale(
    x, feature_range=(-1, 1), axis=0, copy=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2)

print(len(y_train))
print(len(y_test))
print(len(y_test[y_test==1]))

# %%
df = pd.DataFrame(data=np.column_stack((x, y)), 
    columns=['variance', 'skewness', 'curtosis', 'entropy', 'y'])
g = sns.PairGrid(df)
g.map(plt.scatter)

# %%
# Step 3: Train model
gamma = 'scale'
C = 100

model = svm.SVC(C=C, kernel='rbf', gamma=gamma, random_state=seed)
model.fit(x_train, y_train)
print(model.get_params())
pred_train = model.predict(x_train)
pred_test = model.predict(x_test)

gamma = model._gamma
score_train = metrics.accuracy_score(y_train, pred_train)
score_test = metrics.accuracy_score(y_test, pred_test)

print(f'\nWith gamma = {gamma} and C = {C}')
print(f'Accuracy on train set = {score_train*100:.4f}%')
print(f'Accuracy on test set  = {score_test*100:.4f}%')
conf_mat = metrics.confusion_matrix(y_test, pred_test, labels=[0, 1])
print('\nConfusion matrix')
print('Actual classes')
print('     0   1')
for clas, row in zip([0, 1], conf_mat):
    print(clas,row)


# %%
# Step 4: Generate adversarial examples
epsilon = 0.006 # the range of x is [-1, 1]
max_epoch = 2000

ind_train_c0 = np.where(y_train == 0)
x_train_c0 = x_train[ind_train_c0]
mu_train_c0 = np.mean(x_train_c0, axis=0)

ind_train_c1 = np.where(y_train == 1)
x_train_c1 = x_train[ind_train_c1]
mu_train_c1 = np.mean(x_train_c1, axis=0)

print('Negative mean =', mu_train_c0)
print('Positive mean =', mu_train_c1)

x_ae = np.copy(x_test)
pred_ae = np.copy(pred_test)
targets = get_not_y(y_test)

epoch = 1
while np.array_equal(pred_ae, targets) == False and epoch <= max_epoch:
    x_ae = adversarial.moving_mean(
        x=x_ae,
        y=pred_ae,
        targets=targets,
        means={0: mu_train_c0, 1: mu_train_c1},
        epsilon=epsilon,
        verbose=0,
        epoch=epoch)
    pred_ae = model.predict(x_ae)
    epoch += 1

print(f'Completed after {epoch} epoch...')


# %%
df_ae = pd.DataFrame(data=np.column_stack((x_ae, y_test)), 
    columns=['variance', 'skewness', 'curtosis', 'entropy', 'y-prime'])
g = sns.PairGrid(df_ae)
g.map(plt.scatter)

# %%
# Step 5: Test Applicability Domain
# Xi are rescaled to [-1, 1], so the same zeta value should suit all Xi
zeta = 0.4 
k = 9

# Stage 1 - Applicability
print('\n---------- Applicability ---------------')
x_passed_s1, ind_passed_s1 = ad.check_applicability(
    x_ae, x_train, y_train)
pred_passed_s1 = pred_ae[ind_passed_s1]

# Print
pass_rate = utils.get_rate(x_passed_s1, x_ae)
print(f'Pass rate = {pass_rate * 100:.4f}%')
if pass_rate == 0:
    raise Exception('All samples are blocked by Applicability check')

# Stage 2 - Reliability
print('\n---------- Reliability -----------------')
# Creating kNN models for each class
ind_train_c0 = np.where(y_train == 0)
model_knn_c0 = utils.unimodal_knn(x_train[ind_train_c0], k)

ind_train_c1 = np.where(y_train == 1)
model_knn_c1 = utils.unimodal_knn(x_train[ind_train_c1], k)

# Computing mean, standard deviation and threshold
mu_c0, sd_c0 = utils.get_distance_info(
    model_knn_c0, x_train[ind_train_c0], k, seen_in_train_set=True)
threshold_c0 = ad.get_reliability_threshold(mu_c0, sd_c0, zeta)

mu_c1, sd_c1 = utils.get_distance_info(
    model_knn_c1, x_train[ind_train_c1], k, seen_in_train_set=True)
threshold_c1 = ad.get_reliability_threshold(mu_c1, sd_c1, zeta)

x_passed_s2, ind_passed_s2 = ad.check_reliability(
    x_passed_s1,
    predictions=pred_passed_s1,
    models=[model_knn_c0, model_knn_c1],
    dist_thresholds=[threshold_c0, threshold_c1],
    classes=[0, 1],
    verbose=1
)
pred_passed_s2 = pred_passed_s1[ind_passed_s2]

# Print
print('Distance of c0 in training set:')
print('{:18s} = {:.4f}'.format('Mean', mu_c0))
print('{:18s} = {:.4f}'.format('Standard deviation', sd_c0))
print('{:18s} = {:.4f}\n'.format('Threshold', threshold_c0))

print('Distance of c1 in training set:')
print('{:18s} = {:.4f}'.format('Mean', mu_c1))
print('{:18s} = {:.4f}'.format('Standard deviation', sd_c1))
print('{:18s} = {:.4f}\n'.format('Threshold', threshold_c1))

pass_rate = utils.get_rate(x_passed_s2, x_passed_s1)
print(f'Pass rate = {pass_rate * 100:.4f}%')

if pass_rate == 0:
    raise Exception('All samples are blocked by Reliability check')

# Stage 3 - Decidability
print('\n---------- Decidability ----------------')
model_knn = knn.KNeighborsClassifier(
    n_neighbors=k, n_jobs=-1, weights='distance')
model_knn.fit(x_train, y_train)

x_passed_s3, ind_passed_s3 = ad.check_decidability(
    x_passed_s2, pred_passed_s2, model_knn)

# Print
pass_rate = utils.get_rate(x_passed_s3, x_passed_s2)
print(f'Pass rate = {pass_rate * 100:.4f}%')

if pass_rate == 0:
    raise Exception('All samples are blocked by Decidability check')

x_passed_ad = x_passed_s3


# %%
