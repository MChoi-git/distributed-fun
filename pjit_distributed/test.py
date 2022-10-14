from functools import partial

import jax
from jax import numpy as jnp, random
from jax.experimental import maps, PartitionSpec
from jax.experimental.pjit import pjit
import flax.linen as nn
import numpy as np

from layers import ColumnParallelLinear, RowParallelLinear

mesh_shape = (2, 1)
mesh_axis_names = ("tp", "pp")
devices = np.asarray(jax.devices()).reshape(*mesh_shape)
mesh = maps.Mesh(devices, mesh_axis_names)


col_linear = ColumnParallelLinear(
    hidden=256,
    dropout=0.0,
    param_dtype=jnp.float16,
)
row_linear = RowParallelLinear(
    hidden=256,
    dropout=0.0,
    param_dtype=jnp.float16,
)

col_data = random.normal(random.PRNGKey(0), shape=(16, 512, 256), dtype=jnp.float16)
row_data = random.normal(random.PRNGKey(0), shape=(16, 512, 256), dtype=jnp.float16)

key = random.PRNGKey(42069)
sk1, sk2 = random.split(key)

col_init_fn = lambda x: col_linear.init(sk1, x, train=False)
row_init_fn = lambda x: row_linear.init(sk1, x, train=False)

def pjit_init():
    pjit_init_col = pjit(
        lambda x: col_linear.init(sk1, x, train=False),
        in_axis_resources=None,
        out_axis_resources=PartitionSpec(None, "tp"),
    )
    pjit_init_row = pjit(
        lambda x: row_linear.init(sk1, x, train=False),
        in_axis_resources=PartitionSpec(None, None, "tp"),
        out_axis_resources=PartitionSpec("tp", None),
    )

    with maps.Mesh(mesh.devices, mesh.axis_names):
        col_params = pjit_init_col(col_data)
        row_params = pjit_init_row(row_data)
    return col_params, row_params


def pjit_fwd(x, col_params, row_params):
    pjit_fwd_col = pjit(
        lambda p, x: col_linear.apply(p, x, train=False),
        in_axis_resources=[PartitionSpec(None, "tp"), None],
        out_axis_resources=PartitionSpec(None, None, "tp"),
    )
    pjit_fwd_row = pjit(
        lambda p, x: row_linear.apply(p, x, train=False),
        in_axis_resources=[PartitionSpec("tp", None), PartitionSpec(None, None, "tp")],
        out_axis_resources=PartitionSpec(None, None, "tp"),
    )

    with maps.Mesh(mesh.devices, mesh.axis_names):
        col_out = pjit_fwd_col(col_params, x)
        row_out = pjit_fwd_row(row_params, col_out)

        col_jaxpr = jax.make_jaxpr(pjit_fwd_col)(col_params, x)
        row_jaxpr = jax.make_jaxpr(pjit_fwd_row)(row_params, col_out)

    breakpoint()
    return col_out, col_jaxpr, row_out, row_jaxpr


col_params, row_params = pjit_init()
col_out, col_jaxpr, row_out, row_jaxpr = pjit_fwd(col_data, col_params, row_params)

breakpoint()
