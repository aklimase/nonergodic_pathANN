#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 14:46:53 2020

@author: aklimase
"""

import gpflow
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from gpflow.utilities import print_summary
import sys
import os
sys.path.append(os.path.abspath('/Users/aklimase/Documents/USGS/nonergodic_ANN'))
from preprocessing import transform_dip, readindata, transform_data
from model_plots import gridded_plots, obs_pre, plot_resid, plot_outputs
import random

#start with 1d in and out
X = np.asarray([0,1,3,3.5,6,6.2,8,8.4,8.5]).reshape(-1, 1)
Y = np.asarray([1,1.1,1.4,5,5.4,4,3.4,6.4,4]).reshape(-1, 1)

_ = plt.plot(X, Y, "kx", mew=2)

k = gpflow.kernels.Matern52()

print_summary(k)

#meanf = gpflow.mean_functions.Linear()

m = gpflow.models.GPR(data=(X, Y), kernel=k, mean_function=None)

print_summary(m)

#optimize model parameters (variance and length scale)
opt = gpflow.optimizers.Scipy()

opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))
print_summary(m)

#generate test points
xx = np.linspace(-0.1, 10.0, 100).reshape(100, 1)  # test points must be of shape (N, D)

## predict mean and variance of latent GP at test points
mean, var = m.predict_f(xx)

## generate 10 samples from posterior
tf.random.set_seed(1)  # for reproducibility
samples = m.predict_f_samples(xx, 10)  # shape (10, 100, 1)

## plot
plt.figure(figsize=(12, 6))
plt.plot(X, Y, "kx", mew=2)
plt.plot(xx, mean, "C0", lw=2)
plt.fill_between(
    xx[:, 0],
    mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
    mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
    color="C0",
    alpha=0.2,
)

plt.plot(xx, samples[:, :, 0].numpy().T, "C0", linewidth=0.5)

#%%
n=13
train_data1, test_data1, train_targets1, test_targets1, feature_names = readindata(nametrain='/Users/aklimase/Documents/USGS/data/cybertrainyeti10_residfeb.csv', nametest='/Users/aklimase/Documents/USGS/data/cybertestyeti10_residfeb.csv', n = n)

nsamples = 1000
randindex = random.sample(range(0, len(train_data1)), nsamples)
    
#try in higher dimensions
X = train_data1[:,0:3][randindex]
Y = train_targets1[:,0:3][randindex]

#active dimes is input dimensions
k = gpflow.kernels.Matern52(active_dims=[0], lengthscales=0.5) + gpflow.kernels.Matern52(
    active_dims=[1], lengthscales=0.5)+ gpflow.kernels.Matern52(active_dims=[2], lengthscales=0.5)

print_summary(k)

#meanf = gpflow.mean_functions.Linear()

m = gpflow.models.GPR(data=(X, Y), kernel=k, mean_function=None)

print_summary(m)

#optimize model parameters (variance and length scale)
opt = gpflow.optimizers.Scipy()

opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))
print_summary(m)

#generate test points
# xx = np.asarray([np.linspace(min(X[:,0]),max(X[:,0]), 100),np.linspace(min(X[:,1]),max(X[:,1]), 100)]).reshape(100, 2)  # test points must be of shape (N, D)
xx = np.asarray([np.linspace(min(X[:,i]),max(X[:,i]), 100),np.linspace(min(X[:,1]),max(X[:,1]), 100)]).reshape(100, 2)  # test points must be of shape (N, D)

mean, var = m.predict_f(xx)

## generate 10 samples from posterior
tf.random.set_seed(1)  # for reproducibility
samples = m.predict_f_samples(xx, 10)  # shape (10, 100, 1)

## plot
#first period vs distance
plt.figure(figsize=(12, 6))
plt.plot(X[:,1], Y[:,0], "kx", mew=2, label = 'data')
plt.plot(xx[:,1], mean[:,0], "C0", lw=2, label = 'mean prediction')
plt.xlabel('Rhypo')
plt.ylabel('target T=10s')
plt.fill_between(
    xx[:, 1],
    mean[:,0] - 1.96 * np.sqrt(var[:, 0]),
    mean[:,0] + 1.96 * np.sqrt(var[:, 0]),
    color="C0",
    alpha=0.2,
)
plt.legend()
# plt.plot(xx[:,1], samples[0, :, 0].numpy().T, "C0", linewidth=0.5)



#%%
#VGP model
from gpflow.ci_utils import ci_niter

#number of inducing points
M=40
Zinit = np.asarray([[min(X.T[i]) for i in range(len(X[0]))],[max(X.T[i]) for i in range(len(X[0]))]])



kern_list = [gpflow.kernels.Matern52() + gpflow.kernels.Matern52() for _ in range(2)]
# Create multi-output kernel from kernel list
kernel = gpflow.kernels.SeparateIndependent(kern_list)
# initialization of inducing input locations, one set of locations per output
Zs = [Zinit.copy() for _ in range(2)]
# initialize as list inducing inducing variables
iv_list = [gpflow.inducing_variables.InducingPoints(Z) for Z in Zs]
# create multi-output inducing variables from iv_list
iv = gpflow.inducing_variables.SeparateIndependentInducingVariables(iv_list)

m = gpflow.models.SVGP(kernel, gpflow.likelihoods.Gaussian(), inducing_variable=iv, num_latent_gps=2)

data = X, Y

MAXITER = ci_niter(2000)
def optimize_model_with_scipy(model):
    optimizer = gpflow.optimizers.Scipy()
    optimizer.minimize(
        model.training_loss_closure(data),
        variables=model.trainable_variables,
        method="l-bfgs-b",
        options={"disp": True, "maxiter": MAXITER},
    )


optimize_model_with_scipy(m)

print_summary(m)



#generate test points
xx = np.asarray([np.linspace(min(X[:,0]),max(X[:,0]), 100),np.linspace(min(X[:,1]),max(X[:,1]), 100)]).reshape(100, 2)  # test points must be of shape (N, D)


mean, var = m.predict_f(xx)

## generate 10 samples from posterior
tf.random.set_seed(1)  # for reproducibility
samples = m.predict_f_samples(xx, 10)  # shape (10, 100, 1)

## plot
#first period vs distance
plt.figure(figsize=(12, 6))
plt.plot(X[:,1], Y[:,0], "kx", mew=2, label = 'data')
plt.plot(xx[:,1], mean[:,0], "C0", lw=2, label = 'mean prediction')
plt.xlabel('Rhypo')
plt.ylabel('target T=10s')
plt.fill_between(
    xx[:, 1],
    mean[:,0] - 1.96 * np.sqrt(var[:, 0]),
    mean[:,0] + 1.96 * np.sqrt(var[:, 0]),
    color="C0",
    alpha=0.2,
)
plt.legend()

#%%
P=2
# Create list of kernels for each output
kern_list = [gpflow.kernels.Matern52() for _ in range(P)]
# Create multi-output kernel from kernel list
kernel = gpflow.kernels.SeparateIndependent(kern_list)
# initialization of inducing input locations (M random points from the training inputs)
Z = Zinit.copy()
# create multi-output inducing variables from Z
iv = gpflow.inducing_variables.SharedIndependentInducingVariables(
    gpflow.inducing_variables.InducingPoints(Z)
)

m = gpflow.models.SVGP(kernel, gpflow.likelihoods.Gaussian(), inducing_variable=iv, num_latent_gps=P)
print_summary(m)
optimize_model_with_scipy(m)
print_summary(m)

#generate test points
xx = np.asarray([np.linspace(min(X[:,0]),max(X[:,0]), 100),np.linspace(min(X[:,1]),max(X[:,1]), 100)]).reshape(100, 2)  # test points must be of shape (N, D)

mean, var = m.predict_f(xx)

## plot
#first period vs distance
plt.figure(figsize=(12, 6))
plt.plot(X[:,1], Y[:,0], "kx", mew=2, label = 'data')
plt.plot(xx[:,1], mean[:,0], "C0", lw=2, label = 'mean prediction')
plt.xlabel('Rhypo')
plt.ylabel('target T=10s')
plt.fill_between(
    xx[:, 1],
    mean[:,0] - 1.96 * np.sqrt(var[:, 0]),
    mean[:,0] + 1.96 * np.sqrt(var[:, 0]),
    color="C0",
    alpha=0.2,
)
plt.legend()


#%%
from gpflow import set_trainable
from gpflow.optimizers import NaturalGradient


class HeteroskedasticGaussian(gpflow.likelihoods.Likelihood):
    def __init__(self, **kwargs):
        # this likelihood expects a single latent function F, and two columns in the data matrix Y:
        super().__init__(latent_dim=1, observation_dim=2, **kwargs)

    def _log_prob(self, F, Y):
        # log_prob is used by the quadrature fallback of variational_expectations and predict_log_density.
        # Because variational_expectations is implemented analytically below, this is not actually needed,
        # but is included for pedagogical purposes.
        # Note that currently relying on the quadrature would fail due to https://github.com/GPflow/GPflow/issues/966
        Y, NoiseVar = Y[:, 0], Y[:, 1]
        return gpflow.logdensities.gaussian(Y, F, NoiseVar)

    def _variational_expectations(self, Fmu, Fvar, Y):
        Y, NoiseVar = Y[:, 0], Y[:, 1]
        return (
            -0.5 * np.log(2 * np.pi)
            - 0.5 * tf.math.log(NoiseVar)
            - 0.5 * (tf.math.square(Y - Fmu) + Fvar) / NoiseVar
        )

    # The following two methods are abstract in the base class.
    # They need to be implemented even if not used.

    def _predict_log_density(self, Fmu, Fvar, Y):
        raise NotImplementedError

    def _predict_mean_and_var(self, Fmu, Fvar):
        raise NotImplementedError



def generate_data(N=80):
    X = np.random.rand(N)[:, None] * 10 - 5  # Inputs, shape N x 1
    F = 2.5 * np.sin(6 * X) + np.cos(3 * X)  # Mean function values
    NoiseVar = 2 * np.exp(-((X - 2) ** 2) / 4) + 0.3  # Noise variances
    Y = F + np.random.randn(N, 1) * np.sqrt(NoiseVar)  # Noisy data
    return X, Y, NoiseVar

#known noise

X, Y, NoiseVar = generate_data()
Y_data = np.hstack([Y, NoiseVar])


likelihood = HeteroskedasticGaussian()
kernel = gpflow.kernels.Matern52(lengthscales=0.5)
model = gpflow.models.VGP((X, Y_data), kernel=kernel, likelihood=likelihood, num_latent_gps=1)


natgrad = NaturalGradient(gamma=1.0)
adam = tf.optimizers.Adam()

set_trainable(model.q_mu, False)
set_trainable(model.q_sqrt, False)

for _ in range(ci_niter(1000)):
    natgrad.minimize(model.training_loss, [(model.q_mu, model.q_sqrt)])
    adam.minimize(model.training_loss, model.trainable_variables)
for _ in range(ci_niter(1000)):
    natgrad.minimize(model.training_loss, [(model.q_mu, model.q_sqrt)])
    adam.minimize(model.training_loss, model.trainable_variables)
    
    
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
_ = ax.errorbar(
    X.squeeze(),
    Y.squeeze(),
    yerr=2 * (np.sqrt(NoiseVar)).squeeze(),
    marker="x",
    lw=0,
    elinewidth=1.0,
    color="C1",
)

xx = np.linspace(-5, 5, 200)[:, None]

mu, var = model.predict_f(xx)

plt.figure(figsize=(12, 6))
plt.plot(xx, mu, "C0")
plt.plot(xx, mu + 2 * np.sqrt(var), "C0", lw=0.5)
plt.plot(xx, mu - 2 * np.sqrt(var), "C0", lw=0.5)

plt.errorbar(
    X.squeeze(),
    Y.squeeze(),
    yerr=2 * (np.sqrt(NoiseVar)).squeeze(),
    marker="x",
    lw=0,
    elinewidth=1.0,
    color="C1",
)
_ = plt.xlim(-5, 5)







