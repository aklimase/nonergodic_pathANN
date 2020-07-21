#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 13:37:26 2020

@author: aklimasewski
"""

#GPFlow
# Demo 2: grouped noise variances
import gpflow
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import power_transform
from sklearn.preprocessing import PowerTransformer
import keras
import tensorflow as tf

from gpflow.ci_utils import ci_niter
from gpflow.optimizers import NaturalGradient
from gpflow import set_trainable



np.random.seed(1)

def generate_data(N=100):
    X = np.random.rand(N)[:, None] * 10 - 5  # Inputs, shape N x 1
    F = 2.5 * np.sin(6 * X) + np.cos(3 * X)  # Mean function values
    groups = np.where(X > 0, 0, 1)
    NoiseVar = np.array([0.02, 0.5])[groups]  # Different variances for the two groups
    Y = F + np.random.randn(N, 1) * np.sqrt(NoiseVar)  # Noisy data
    return X, Y, groups


X, Y, groups = generate_data()
# here's a plot of the raw data.
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
_ = ax.plot(X, Y, "kx")

natgrad = NaturalGradient(gamma=1.0)
adam = tf.optimizers.Adam()


Y_data = np.hstack([Y, groups])


likelihood = gpflow.likelihoods.SwitchedLikelihood(
    [gpflow.likelihoods.Gaussian(variance=1.0), gpflow.likelihoods.Gaussian(variance=1.0)]
)

# model construction (notice that num_latent_gps is 1)
kernel = gpflow.kernels.Matern52(lengthscales=0.5)
model = gpflow.models.VGP((X, Y_data), kernel=kernel, likelihood=likelihood, num_latent_gps=1)

for _ in range(ci_niter(1000)):
    natgrad.minimize(model.training_loss, [(model.q_mu, model.q_sqrt)])
    
# let's do some plotting!
xx = np.linspace(-5, 5, 200)[:, None]

mu, var = model.predict_f(xx)

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.plot(xx, mu, "C0")
ax.plot(xx, mu + 2 * np.sqrt(var), "C0", lw=0.5)
ax.plot(xx, mu - 2 * np.sqrt(var), "C0", lw=0.5)

ax.plot(X, Y, "C1x", mew=2)
_ = ax.set_xlim(-5, 5)



likelihood = gpflow.likelihoods.SwitchedLikelihood(
    [gpflow.likelihoods.Gaussian(variance=1.0), gpflow.likelihoods.Gaussian(variance=1.0)]
)
kernel = gpflow.kernels.Matern52(lengthscales=0.5)
model = gpflow.models.VGP((X, Y_data), kernel=kernel, likelihood=likelihood, num_latent_gps=1)

set_trainable(model.q_mu, False)
set_trainable(model.q_sqrt, False)

for _ in range(ci_niter(1000)):
    natgrad.minimize(model.training_loss, [(model.q_mu, model.q_sqrt)])
    adam.minimize(model.training_loss, model.trainable_variables)


xx = np.linspace(-5, 5, 200)[:, None]

mu, var = model.predict_f(xx)

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.plot(xx, mu, "C0")
ax.plot(xx, mu + 2 * np.sqrt(var), "C0", lw=0.5)
ax.plot(xx, mu - 2 * np.sqrt(var), "C0", lw=0.5)

ax.plot(X, Y, "C1x", mew=2)
ax.set_xlim(-5, 5)
_ = ax.plot(xx, 2.5 * np.sin(6 * xx) + np.cos(3 * xx), "C2--")



