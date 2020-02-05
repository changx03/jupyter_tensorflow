# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# %%
seed = 2**12
np.random.seed(seed=seed)

# %%
data = pd.read_csv('./dataset/BreastCancerCoimbra.csv')
data.head()

# %%
features = ['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 'Leptin', 'Adiponectin',
    'Resistin', 'MCP.1']

pd_X = data.loc[:,features]
pd_y = data.loc[:, 'Classification']
# %%
pd_y = pd_y.apply(lambda x: x - 1)
pd_y.head()

# %%
X = pd_X.to_numpy(dtype=np.float32)
y = pd_y.to_numpy(dtype=np.int)

print(X[:5])
print(y[:5])
# %%
