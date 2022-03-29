"""Code to perform Hessian vector products on neural networks.
"""

from jax import grad
from jax import jit
from jax import jvp
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import jax.tree_util as tu




def hvp2(loss, params, batch, v):
  """Computes the hessian vector product Hv.

  This implementation uses forward-over-reverse mode for computing the hvp.

  Args:
    loss: function computing the loss with signature
      loss(params, batch).
    params: pytree for the parameters of the model.
    batch:  A batch of data. Any format is fine as long as it is a valid input
      to loss(params, batch).
    v: An array of shape [num_params]

  Returns:
    hvp: array of shape [num_params] equal to Hv where H is the hessian.
  """
  flat_params, unravel = ravel_pytree(params)
  w = unravel(v)
  loss_fn = lambda x: loss(x, batch)
  res_flat, _ = ravel_pytree(jvp(grad(loss_fn), [params], [w])[1])
  return res_flat


