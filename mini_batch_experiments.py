import time
import itertools

import numpy as np
import jax.numpy as jnp
from jax.config import config
from jax import jit, grad, random
from jax import grad, jit, vmap, hessian, jvp
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, LogSoftmax, BatchNorm, Dropout
import matplotlib.pyplot as plt

key = random.PRNGKey(0)


import torch
print(torch.__version__)

import torchvision

import torchvision.datasets as datasets

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

NUM_CLASSES = 10

np_images, np_labels = mnist_trainset.data.cpu().detach().numpy(), mnist_trainset.train_labels.cpu().detach().numpy()
train_images = np_images[:,:,:].reshape(-1, 28*28).astype(jnp.float32)/255.0
train_labels = jnp.eye(NUM_CLASSES)[np_labels]


np_images, np_labels = mnist_testset.data.cpu().detach().numpy(), mnist_testset.train_labels.cpu().detach().numpy()
test_images = np_images[:,:,:].reshape(-1, 28*28).astype(jnp.float32)/255.0
test_labels = jnp.eye(NUM_CLASSES)[np_labels]

# plt.imshow(train_images[1].reshape(28,28))
# print(train_labels[1,:])


step_size   = 0.001
momentum_mass = 0.9
batch_size  = 1024
NUM_EPOCHS  = 150
# Define Architectures

# Define FC Net
init_fun, apply_fun = stax.serial(
    Dense(256),  Relu,
    Dense(256),  Relu, 
    Dense(NUM_CLASSES), LogSoftmax)

# Initialize Parameters (considering only MNIST)
print(train_images.shape[1:])
_, init_params = init_fun(key, (-1,) + train_images.shape[1:])

# define the loss and accuracy functions
def loss(params, batch):
  inputs, targets = batch  
  preds = apply_fun(params, inputs)
  return - jnp.mean(preds * targets)

def accuracy(params, batch):
  inputs, targets = batch
  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(apply_fun(params, inputs), axis=1)
  return jnp.mean(predicted_class == target_class)

# Set optimizer
# opt_init, opt_update, get_params = optimizers.sgd(step_size)

opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=momentum_mass)



# Define generator for batches
num_train = train_images.shape[0]
num_complete_batches, leftover = divmod(num_train, batch_size)
num_batches = num_complete_batches + bool(leftover)

def data_stream():
  rng = np.random.RandomState(0)
  while True:
    perm = rng.permutation(num_train)
    for i in range(num_batches):
      batch_idx = perm[i * batch_size:(i + 1) * batch_size]
      yield train_images[batch_idx], train_labels[batch_idx]

batches = data_stream()

# Define the hvp and update functions
batches_list = [next(batches) for i in range(num_batches)]
def batches_fn():
  for b in batches_list:
    yield b
    

import hessian_computation as hessian_computation
from jax.flatten_util import ravel_pytree


@jit
def jitted_hvp(params, batch, v):
  return hessian_computation.hvp2(loss, params, batch, v)


@jit
def update(i, opt_state, batch):
  params = get_params(opt_state)
  grads  = grad(loss)(params, batch)
  new_opt_state = opt_update(i,grads, opt_state)
  flat_grads, _ = ravel_pytree(grads)
  Hg = jitted_hvp(params, batch, flat_grads)  
  Hg_overlap = jnp.dot(flat_grads, Hg) / (jnp.linalg.norm(Hg) * jnp.linalg.norm(flat_grads))
  return new_opt_state, Hg_overlap


opt_state = opt_init(init_params)
itercount = itertools.count()

Hg_overlap_sum = 0
Hg_overlap_vec = jnp.zeros((NUM_EPOCHS,))

print("\nStarting training...")
for epoch in range(NUM_EPOCHS):
  start_time = time.time()
  for _ in range(num_batches):
    cntr = next(itercount)
    batchz = next(batches)
    # Hg_overlap_vec = Hg_overlap_vec.at[cntr-1].set(Hg_overlap)
    opt_state, Hg_overlap = update(cntr, opt_state, batchz)
    Hg_overlap_sum += Hg_overlap
    # print(Hg_overlap)
    
  # print(Hg_overlap_sum/num_batches)
  Hg_overlap_vec = Hg_overlap_vec.at[epoch].set(Hg_overlap)
  Hg_overlap_sum = 0

  epoch_time = time.time() - start_time

  params = get_params(opt_state)
  train_acc = accuracy(params, (train_images, train_labels))
  test_acc = accuracy(params, (test_images, test_labels))
  print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
  print("Training set accuracy {}".format(train_acc))
  print("Test set accuracy {}".format(test_acc))
  


plt.plot( list(range(1,NUM_EPOCHS+1))   , Hg_overlap_vec, label = 'Batch_size = 1024')
plt.xlabel('Epoch')
plt.title('Hessian Gradient Overlap (Minibatch)')
plt.grid()
plt.legend()
plt.show()



