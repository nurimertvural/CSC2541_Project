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

key = random.PRNGKey(3)


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
batch_size  = 128
NUM_EPOCHS  = 50
momentum_mass = 0.9
# Define Architectures

# Define FC Net
init_fun, apply_fun = stax.serial(
    Dense(128),  Relu,
    Dense(128),  Relu, 
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

import hessian_computation as hessian_computation
from jax.flatten_util import ravel_pytree
import lanczos as lanczos

@jit
def update(i, opt_state, batch):
  params = get_params(opt_state)
  grads  = grad(loss)(params, batch)
  new_opt_state = opt_update(i,grads, opt_state)
  return new_opt_state



opt_state = opt_init(init_params)
itercount = itertools.count()

order = 20
hvp_cl = lambda v: hessian_computation.hvp2(loss, params, (train_images, train_labels), v)
key, split = random.split(key)

Hg_overlap_vec128 = jnp.zeros( shape = (NUM_EPOCHS,) )
Ev_overlap_vec128 = jnp.zeros( shape = (1+ (NUM_EPOCHS//5),) )

print("\nStarting training...")
for epoch in range(NUM_EPOCHS):
  start_time = time.time()
  for _ in range(num_batches):
    cntr = next(itercount)
    batchz = next(batches)
    params = get_params(opt_state)
    opt_state = update(cntr, opt_state, batchz)
    # print(Hg_overlap)

  params = get_params(opt_state)
  grads  = grad(loss)(params, (train_images, train_labels) )  
  flat_grads, _ = ravel_pytree(grads)
  Hg = hvp_cl(flat_grads)
  Hg_overlap = jnp.dot(flat_grads, Hg) / (jnp.linalg.norm(Hg) * jnp.linalg.norm(flat_grads))
  Hg_overlap_vec128 = Hg_overlap_vec128.at[epoch].set(Hg_overlap)
  print("Hg_overlap: ", Hg_overlap)
  
  if(epoch == 0):
    tridiag, pre_vecs = lanczos.lanczos_alg(hvp_cl, flat_grads.shape[0], order, split)
    eigs_tridiag, _ = jnp.linalg.eigh(tridiag)
    eigs_tridiag = eigs_tridiag[-11:]
    Ev_overlap_vec128 = Ev_overlap_vec128.at[0].set(0)
    print(eigs_tridiag)
  elif( (epoch+1)%5 == 0 ):
    tridiag, vecs = lanczos.lanczos_alg(hvp_cl, flat_grads.shape[0], order, split)
    eigs_tridiag, _ = jnp.linalg.eigh(tridiag)
    eigs_tridiag = eigs_tridiag[-11:]
    print(eigs_tridiag)
    Ev_overlap = jnp.sum(vecs[0:10,] * pre_vecs[0:10,]) / jnp.sqrt( jnp.sum(vecs[0:10,]*vecs[0:10,]) * jnp.sum(pre_vecs[0:10,]*pre_vecs[0:10,]) )
    Ev_overlap_vec128 = Ev_overlap_vec128.at[(epoch//5)+1].set(Ev_overlap)
    print("Ev_overlap: ", Ev_overlap)
    pre_vecs = vecs
  
  epoch_time = time.time() - start_time

  # params = get_params(opt_state)
  train_acc = accuracy(params, (train_images, train_labels))
  test_acc = accuracy(params, (test_images, test_labels))
  print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
  print("Training set accuracy {}".format(train_acc))
  print("Test set accuracy {}".format(test_acc))
  

fig, ax = plt.subplots(2,1)
ax[0].plot( list(range(1,NUM_EPOCHS+1))   , Hg_overlap_vec32  , label = 'Batch_size = 32', zorder = 1)
ax[0].plot( list(range(1,NUM_EPOCHS+1))   , Hg_overlap_vec128 , label = 'Batch_size = 128', zorder = 1)
ax[0].plot( list(range(1,NUM_EPOCHS+1))   , Hg_overlap_vec1024, label = 'Batch_size = 1024', zorder = 1)
ax[0].grid(True)
ax[0].set_title('Hessian Gradient Overlap')
ax[0].set_ylabel('Overlap')
ax[0].legend()
ax[1].plot( list(range(0, NUM_EPOCHS+1,5)), Ev_overlap_vec32  , label = 'Batch_size = 32', zorder = 2)
ax[1].plot( list(range(0, NUM_EPOCHS+1,5)), Ev_overlap_vec128 , label = 'Batch_size = 128', zorder = 2)
ax[1].plot( list(range(0, NUM_EPOCHS+1,5)), Ev_overlap_vec1024, label = 'Batch_size = 1024', zorder = 2)
ax[1].grid(True)
ax[1].set_title('Eigenvectors Overlap (k = 10)')
ax[1].set_ylabel('Overlap')
ax[1].legend()


plt.xlabel('Epoch')
plt.tight_layout()
# plt.legend()
plt.show()





