#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 10:31:01 2020

@author: aklimasewski

based on demo

https://woogle.dev/michel-kana/my-deep-learning-model-says-sorry-i-dont-know-the-answer-that-s-absolutely-ok/
aleatory and epistemic uncertainty in linear data with two groups of aleatory uncertainty

generates linear data with two populations of noise

"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

tf.enable_v2_behavior()

tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

# We'll use double precision throughout for better numerics.
dtype = np.float64

# # Generate noisy data from a known function.
n = 50
x_func = np.linspace(-4,4,100)
y_func = x_func

x_train = np.random.uniform(-3, -2, n)
y_train = x_train + np.random.randn(*x_train.shape)*0.5

x_train = np.concatenate([x_train, np.random.uniform(2, 3, n)])
y_train = np.concatenate([y_train, x_train[n:] + np.random.randn(*x_train[n:].shape)*0.1])

x_test = np.linspace(-5,5,100)
y_test = x_test
# x_test = x_test.reshape(100,1)

fig, ax = plt.subplots(1,1,figsize=(10,5))
ax.scatter(x_train, y_train, label='training data', s = 3)
ax.plot(x_func, y_func, ls='--', label='real function', color='green')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.set_title('Data with uncertainty')
plt.show()

#%%
# Build model with negattive log likelihood loss function
negloglik = lambda y, p_y: -p_y.log_prob(y)

model = tf.keras.Sequential([
  tf.keras.layers.Dense(1 + 1),
  tfp.layers.DistributionLambda(
      lambda t: tfd.Normal(loc=t[..., :1],
                           scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:]))),
])

# Do inference.
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.05), loss=negloglik)
model.fit(x_train, y_train, epochs=500, verbose=True)

# Make predictions.
yhat = model.predict(x_test)

predict_mean = []
predict_al = []
predict_epistemic = []
for i in range(100):
    p = np.array(model.predict(x_test)) 
    mean = p[:,0]
    predict_mean.append(mean)

mean_x_test = np.mean(predict_mean, axis = 0)
predict_epistemic = np.std(predict_mean, axis = 0)

fig, ax = plt.subplots(1,1,figsize=(10,5))
ax.scatter(x_train,y_train,s=3, label = 'training data')
ax.errorbar(x_test,mean_x_test,yerr = predict_epistemic, c = 'green', label = 'predictions with 1 sigma error')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.set_title('Predictions with epistemic uncertainty')
plt.show()

#%%
# Create kernel with trainable parameters, and trainable observation noise
# variance variable. Each of these is constrained to be positive.
amplitude = tfp.util.TransformedVariable(
    1., tfb.Softplus(), dtype=dtype, name='amplitude')
length_scale = tfp.util.TransformedVariable(
    1., tfb.Softplus(), dtype=dtype, name='length_scale')
kernel = tfk.ExponentiatedQuadratic(
    amplitude=amplitude,
    length_scale=length_scale)

observation_noise_variance = tfp.util.TransformedVariable(
    1., tfb.Softplus(), dtype=dtype, name='observation_noise_variance')

# Create trainable inducing point locations and variational parameters.
num_inducing_points_ = 10

inducing_index_points = tf.Variable(
    np.linspace(-3, 3, num_inducing_points_)[..., np.newaxis],
    dtype=dtype, name='inducing_index_points')

variational_loc, variational_scale = (
    tfd.VariationalGaussianProcess.optimal_variational_posterior(
        kernel=kernel,
        inducing_index_points=inducing_index_points,
        observation_index_points=x_train,
        observations=y_train,
        observation_noise_variance=observation_noise_variance))

# These are the index point locations over which we'll construct the
# (approximate) posterior predictive distribution.
num_predictive_index_points_ = 100
index_points_ = np.linspace(-10., 10.,
                            num_predictive_index_points_,
                            dtype=dtype)[..., np.newaxis]

# Construct our variational GP Distribution instance.
vgp = tfd.VariationalGaussianProcess(
    kernel,
    index_points=index_points_,
    inducing_index_points=inducing_index_points,
    variational_inducing_observations_loc=variational_loc,
    variational_inducing_observations_scale=variational_scale,
    observation_noise_variance=observation_noise_variance)

# For training, we use some simplistic numpy-based minibatching.
batch_size = 64
num_training_points= num_predictive_index_points_

optimizer = tf.optimizers.Adam(learning_rate=.05, beta_1=.5, beta_2=.99)

@tf.function
def optimize(x_train_batch, y_train_batch):
  with tf.GradientTape() as tape:
    # Create the loss function we want to optimize.
    loss = vgp.variational_loss(
        observations=y_train_batch,
        observation_index_points=x_train_batch,
        kl_weight=float(batch_size) / float(num_training_points))
  grads = tape.gradient(loss, vgp.trainable_variables)
  optimizer.apply_gradients(zip(grads, vgp.trainable_variables))
  return loss

num_iters = 500
num_logs = 10
for i in range(num_iters):
  batch_idxs = np.random.randint(num_training_points, size=[batch_size])
  x_train_batch = x_train[batch_idxs, ...]
  y_train_batch = y_train[batch_idxs]

  loss = optimize(x_train_batch, y_train_batch)
  if i % (num_iters / num_logs) == 0 or i + 1 == num_iters:
    print(i, loss.numpy())

# Generate a plot with
#   - the posterior predictive mean
#   - training data
#   - inducing index points (plotted vertically at the mean of the
#     variational posterior over inducing point function values)
#   - 50 posterior predictive samples

num_samples = 50

samples_ = vgp.sample(num_samples).numpy()
mean_ = vgp.mean().numpy()
inducing_index_points_ = inducing_index_points.numpy()
variational_loc_ = variational_loc.numpy()

plt.figure(figsize=(15, 5))
# plt.scatter(inducing_index_points_[..., 0], variational_loc_,
            # marker='x', s=50, color='k', zorder=10)
plt.scatter(x_train[..., 0], y_train, color='#00ff00', alpha=.1, zorder=9)
# plt.plot(np.tile(index_points_, num_samples),
          # samples_.T, color='r', alpha=.1)
plt.errorbar(index_points_, mean_,yerr= np.std(samples_, axis = 0))
# plt.plot(index_points_, mean_, color='k')
# plt.plot(index_points_, f(index_points_), color='b')


plt.show()





