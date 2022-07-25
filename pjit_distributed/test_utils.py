import jax
from jax import numpy as jnp, random
from jax.experimental import maps


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

    def init_both(mesh, module_metadata):
        dummy = jnp.ones(
            module_metadata.data_shape,
            dtype=module_metadata.dtype
        )

        # Get params for single GPU layer
        single_params = module_metadata.layer.init(
            module_metadata.rng,
            dummy,
            *module_metadata.init_args,
            **module_metadata.init_kwargs,
        )

        # Get params for distributed layer
        with maps.Mesh(mesh.devices, mesh.axis_names):
            dist_params = module_metadata.pjit_init()

        # Make sure that parameters are the same
        for single_p, dist_p in zip(
            jax.tree_leaves(single_params), jax.tree_leaves(dist_params)
        ):
            assert jnp.allclose(single_p, dist_p)

        return single_params, dist_params

    def forward_both(key, mesh, module_metadata, single_params, dist_params, atol):
        # Different input data types
        if module_metadata.dtype == jnp.int32:
            data = random.randint(key, module_metadata.data_shape, minval=2, maxval=40)
        else:
            data = random.normal(key, module_metadata.data_shape)

        # Get output for single GPU layer
        single_out = module_metadata.layer.apply(
            single_params,
            data,
            *module_metadata.train_args,
            **module_metadata.train_kwargs,
        )  # Shouldn't need rngs, all dropout/batch norm should be off

        # Get output for distributed layer
        with maps.Mesh(mesh.devices, mesh.axis_names):
            dist_out = module_metadata.pjit_forward(dist_params, data, None)

        return single_out, dist_out, jnp.allclose(single_out, dist_out, atol=atol)

    single_params, dist_params = init_both(
        mesh,
        module_metadata,
    )

    single_out, dist_out, result = forward_both(
        forward_key,
        mesh,
        module_metadata,
        single_params,
        dist_params,
        atol,
    )

    return single_out, dist_out, result


def verify_dist_model(key, mesh, module_metadata_manager):
    """
    Verify the correctness of distributed model layers. Note that this
    does not verify the correctness of ragged pjit forward functions, for
    example nn.Embed's attend function.
    Note that this verification utility does not use any existing parameter
    state. All verification of parameter and output correctness is done
    encapsulated within this utility.
    """
    # Check for correctness with non-distributed modules
    single_outs = {}
    dist_outs = {}
    results = {}
    for module_metadata in module_metadata_manager.module_metadata_list:
        key, sk = random.split(key)

        module_metadata.train_kwargs["train"] = False

        single_out, dist_out, result = verify_module_metadata(
            sk, mesh, module_metadata, atol=1e-6
        )

        module_metadata.train_kwargs["train"] = True

        single_outs[module_metadata.name] = single_out
        dist_outs[module_metadata.name] = dist_out
        results[module_metadata.name] = result

    overall = sum(results.values()) == len(results)

    return overall
