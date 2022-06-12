from functools import partial

import jax
from jax import random, numpy as jnp
import numpy as np

x = np.arange(5)
w = np.array([2., 3., 4.])


def convolve(x, w):
    output = []
    for i in range(1, len(x)-1):
        output.append(jnp.dot(x[i-1:i+2], w))
    return jnp.array(output)

n_devices = jax.local_device_count()

xs = np.arange(5 * n_devices).reshape(-1, 5)

print(jax.vmap(partial(convolve, w=w))(xs))
print(jax.pmap(partial(convolve, w=w))(xs))
