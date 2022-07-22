import jax
from jax import numpy as jnp, random
from jax.experimental import maps

import flax

from model_parallel import TransformerInit


def verify_module_metadata(
    forward_key,
    mesh,
    module_metadata,
    atol,
):
    """
    Verifies that a pjit module forward result is the same as the
    non-distributed version.
    """

    def init_both(module_metadata):
        dummy = jnp.ones(module_metadata.data_shape, dtype=module_metadata.dtype)

        single_layer = module_metadata.layer(*module_metadata.module_init_args)
        single_params = single_layer.init(
            module_metadata.rng,
            dummy,
            *module_metadata.init_args,
            **module_metadata.init_kwargs,
        )

        dist_layer = TransformerInit(mesh, 1, [module_metadata])
        dist_params = dist_layer.init_from_pjit_metadata(const_layer_end_idx=1)
        dist_layer.forward_from_pjit_metadata(const_layer_end_idx=1)

        # Make sure that parameters are the same
        for single_p, dist_p in zip(jax.tree_leaves(single_params), jax.tree_leaves(dist_params)):
            assert jnp.allclose(single_p, dist_p)

        return single_params, dist_params, dist_layer

    def forward_both(key, mesh, single_params, dist_layer, dist_params, atol):
        if dist_layer.module_metadata_list[0].dtype == jnp.int32:
            data = random.randint(
                key, dist_layer.module_metadata_list[0].data_shape, minval=2, maxval=40
            )
        else:
            data = random.normal(key, dist_layer.module_metadata_list[0].data_shape)

        single_out = dist_layer.module_metadata_list[0].layer.apply(
            single_params,
            data,
            *dist_layer.module_metadata_list[0].train_args,
            **dist_layer.module_metadata_list[0].train_kwargs,
        )  # Shouldn't need rngs, all dropout/batch norm should be off

        with maps.Mesh(mesh.devices, mesh.axis_names):
            dist_out = dist_layer.module_metadata_list[0].pjit_forward(
                dist_params,
                data,
                None
            )

        return single_out, dist_out, jnp.allclose(single_out, dist_out, atol=atol)

    single_params, dist_params, dist_layer = init_both(
        module_metadata,
    )

    single_out, dist_out, result = forward_both(
        forward_key,
        mesh,
        single_params,
        dist_layer,
        *jax.tree_util.tree_leaves(
            dist_params,
            is_leaf=lambda x: True
            if isinstance(x, flax.core.frozen_dict.FrozenDict)
            else False,
        ),
        atol,
    )

    return single_out, dist_out, result
