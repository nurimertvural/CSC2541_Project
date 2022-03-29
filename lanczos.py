"""Code for running the Lanczos algorithm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax.numpy as np
import jax.random as random


def lanczos_alg(matrix_vector_product, dim, order, rng_key):
  """Lanczos algorithm for tridiagonalizing a real symmetric matrix.

  This function applies Lanczos algorithm of a given order.  This function
  does full reorthogonalization.

  WARNING: This function may take a long time to jit compile (e.g. ~3min for
  order 90 and dim 1e7).

  Args:
    matrix_vector_product: Maps v -> Hv for a real symmetric matrix H.
      Input/Output must be of shape [dim].
    dim: Matrix H is [dim, dim].
    order: An integer corresponding to the number of Lanczos steps to take.
    rng_key: The jax PRNG key.

  Returns:
    tridiag: A tridiagonal matrix of size (order, order).
    vecs: A numpy array of size (order, dim) corresponding to the Lanczos
      vectors.
  """

  tridiag = np.zeros((order, order))
  vecs = np.zeros((order, dim))

  init_vec = random.normal(rng_key, shape=(dim,))
  init_vec = init_vec / np.linalg.norm(init_vec)
  vecs =  vecs.at[0].set(init_vec) 
  
  beta = 0
  for i in range(order):
    v = vecs[i, :].reshape((dim))
    if i == 0:
      v_old = 0
    else:
      v_old = vecs[i - 1, :].reshape((dim))

    w = matrix_vector_product(v)
    assert (w.shape[0] == dim and len(w.shape) == 1), (
        'Output of matrix_vector_product(v) must be of shape [dim].')

    alpha = np.dot(w, v)
    tridiag = tridiag.at[(i,i)].set(alpha) 
    
    w = w  - alpha * v - beta * v_old 
    w_temp = w

    for j in range(i):
      tau = vecs[j, :].reshape((dim))
      coeff = np.dot(w_temp, tau)
      w = w -coeff * tau

    if(np.linalg.norm(w) < 1e-4):
      tridiag = tridiag[0:i,0:i]
      vecs = vecs[0:i,:]
      # print("Early stop")
      # print("i: ", i, "shape: ", tridiag.shape)
      return (tridiag, vecs)
    else: 
      beta = np.linalg.norm(w)
      if i + 1 < order:
        tridiag = tridiag.at[(i,i+1)].set(beta)
        tridiag = tridiag.at[(i+1,i)].set(beta) 
        vecs =  vecs.at[i+1].set(w/np.linalg.norm(w)) 
      else:  
        return (tridiag, vecs)
