# %%
import numpy as np
import and_logic_generator as and_gen
import and_gate_pipeline as pipeline
import matplotlib.pyplot as plt

# reload modules every 2 seconds
%load_ext autoreload
%autoreload 2

# %%
# Repeatable seed
random_state = 2**12
np.random.seed(seed=random_state)

# %%
n = 2000
x, y = and_gen.generate_uniform_samples(
    n=n, 
    threshold=0, 
    radius=1.0,
    logic='xor')

# %%
and_gate_pipeline = pipeline.LogicGatePipeline(x, y)
and_gate_pipeline.random_state = random_state

# Parameters for figures
figsize = np.array(plt.rcParams["figure.figsize"]) * 2
x_max = np.amax(x, axis=0) * 1.1
x_min = np.amin(x, axis=0) * 1.1

and_gate_pipeline.plot_data(
    figsize=figsize, xlim=[x_min[0], x_max[0]], ylim=[x_min[1], x_max[1]])

# %%
