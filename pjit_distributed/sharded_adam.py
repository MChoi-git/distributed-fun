from typing import Optional, Any, Union, Callable
from functools import partial

import jax
from jax import numpy as jnp
from jax.experimental import maps, PartitionSpec
from jax.experimental.pjit import pjit
import flax
import optax
from optax._src import base as optax_base
from optax._src import transform as optax_transform
from optax._src import combine as optax_combine
from optax._src import alias as optax_alias
from optax._src import numerics as optax_numerics
from optax._src import update as optax_update
from optax._src import utils as optax_utils

from model_parallel import ModuleMetadataManager


def align_pspecs_for_module_layers(module_metadata_manager):
    """
    Since modules can occur multiple times in a model, this function returns
    a list of partition specs for a module_metadat object, repeated num_layers
    number of times. This function can be tree mapped over a list of
    module_metadat objects to populate the correct PartitionSpecs for each
    layer and its repeats.
    """
    params_pspecs = []

    def repeat_pspecs(meta):
        pspecs = {}
        for i in range(meta.num_layers):
            pspecs[f"{meta.name}_{i}"] = (
                meta.in_optim_init_pspec,
                meta.out_optim_init_pspec,
            )
        return pspecs

    params_pspecs = jax.tree_util.tree_map(
        repeat_pspecs,
        module_metadata_manager.module_metadata_list,
    )
    params_pspecs = {k: v for layer in params_pspecs for k, v in layer.items()}
    return params_pspecs


def apply_sharded_fn_to_params(mesh, pspecs, fn, *tree_map_args):
    """
    Converts some function taking params-pytree-like args into a function
    which operates on a pytree of sharded params arrays, according to the
    corresponding PartitionSpecs for each parameter FrozenDict group
    """
    def eval_shard_fn(mesh, pspec_tuple, fn, *tree_map_args):
        in_pspec, out_pspec = pspec_tuple

        with maps.Mesh(
            mesh.devices,
            mesh.axis_names,
        ):
            out = pjit(
                fn,
                in_axis_resources=[in_pspec]
                * len(
                    tree_map_args
                ),  # tree_map_args could have > 1 params-like args
                out_axis_resources=out_pspec,
            )(*tree_map_args)

        return out

    out = jax.tree_util.tree_map(
        lambda ps, *params_like: eval_shard_fn(mesh, ps, fn, *params_like),
        pspecs,
        *tree_map_args,
        is_leaf=lambda x: x is None or isinstance(x, tuple),
    )
    return out


def scale_by_adam_dist(
    module_metadata_manager: ModuleMetadataManager,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
) -> optax_base.GradientTransformation:
    """
    Re-implementation of scale_by_adam from original Optax source code. Here
    the init_fn and update_fn will be re-implemented to support multi-GPU
    sharding. Returns an initial ScaleByAdamState containing the metadata and
    mean and stddev zero arrays.
    """
    mu_dtype = optax_utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        """Init optim state for Adam, ie. mean and stddev arrays of zeros."""
        # Align each parameter FrozenDict with it's respective pspecs given
        # by the corresponding ModuleMetadata
        aligned_pspecs = align_pspecs_for_module_layers(module_metadata_manager)

        def sharded_zeros_like(pspec_tuple, p, **kwargs):
            """
            Equivalent to `jax.numpy.zeros_like` but shards the tensors
            over specified pjit in and out axes.
            """
            in_pspec, out_pspec = pspec_tuple
            with maps.Mesh(
                module_metadata_manager.mesh.devices,
                module_metadata_manager.mesh.axis_names,
            ):
                zeros_like_fn = pjit(
                    lambda p: jnp.zeros_like(p, **kwargs),
                    in_axis_resources=in_pspec,
                    out_axis_resources=out_pspec,
                )
                opt_zeros_like = jax.tree_util.tree_map(
                    zeros_like_fn,
                    p,
                )
            return opt_zeros_like

        # Create mu and nu sharded opt states
        mu = jax.tree_util.tree_map(
            lambda pspecs, p: sharded_zeros_like(pspecs, p, dtype=mu_dtype),
            aligned_pspecs,
            params,
            is_leaf=lambda x: x is None or isinstance(x, tuple),
        )
        nu = jax.tree_util.tree_map(
            sharded_zeros_like,
            aligned_pspecs,
            params,
            is_leaf=lambda x: x is None or isinstance(x, tuple),
        )

        assert (
            jax.tree_util.tree_structure(params)
            == jax.tree_util.tree_structure(mu)
            == jax.tree_util.tree_structure(nu)
        )

        return optax_transform.ScaleByAdamState(
            count=jnp.zeros([], jnp.int32), mu=mu, nu=nu
        )

    def update_fn(updates, state, params=None):
        # TODO: Why is the first line to `del params`?
        """
        Update fn implementing the Adam update rule. Returns a tuple of the
        calculated updates state and an updated ScaleByAdamState metadata
        object with the updates mean and stddev arrays.
        """
        # Align each parameter FrozenDict with it's respective pspecs given
        # by the corresponding ModuleMetadata
        del params  # Not sure why this is here
        aligned_pspecs = align_pspecs_for_module_layers(module_metadata_manager)

        def calculate_updates(mu, nu):
            return jax.tree_util.tree_map(
                lambda m, v: m / (jnp.sqrt(v + eps_root) + eps),
                mu,
                nu,
            )

        mu = apply_sharded_fn_to_params(
            module_metadata_manager.mesh,
            aligned_pspecs,
            partial(optax_transform._update_moment, decay=b1, order=1),
            updates,
            state.mu,
        )
        nu = apply_sharded_fn_to_params(
            module_metadata_manager.mesh,
            aligned_pspecs,
            partial(optax_transform._update_moment_per_elem_norm, decay=b2, order=2),
            updates,
            state.nu,
        )
        count_inc = optax_numerics.safe_int32_increment(state.count)
        mu_hat = apply_sharded_fn_to_params(
            module_metadata_manager.mesh,
            aligned_pspecs,
            partial(optax_transform._bias_correction, decay=b1, count=count_inc),
            mu,
        )
        nu_hat = apply_sharded_fn_to_params(
            module_metadata_manager.mesh,
            aligned_pspecs,
            partial(optax_transform._bias_correction, decay=b2, count=count_inc),
            nu,
        )

        updates = apply_sharded_fn_to_params(
            module_metadata_manager.mesh,
            aligned_pspecs,
            calculate_updates,
            mu_hat,
            nu_hat,
        )

        # Note that if dtype is not partial'd, then this will return None
        # mu values
        mu = apply_sharded_fn_to_params(
            module_metadata_manager.mesh,
            aligned_pspecs,
            partial(optax_utils.cast_tree, dtype=mu_dtype),
            mu_hat,
        )

        return updates, optax_transform.ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

    return optax_base.GradientTransformation(init_fn, update_fn)


def adamw_dist(
    module_metadata_manager: ModuleMetadataManager,
    learning_rate: Union[float, optax_base.Schedule],
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
    weight_decay: float = 1e-4,
    mask: Optional[Union[Any, Callable[[optax_base.Params], Any]]] = None,
) -> optax_base.GradientTransformation:
    return optax_combine.chain(
        scale_by_adam_dist(
            module_metadata_manager=module_metadata_manager,
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            mu_dtype=mu_dtype,
        ),
        optax_transform.add_decayed_weights(weight_decay, mask),
        optax_alias._scale_by_learning_rate(learning_rate),
    )


def apply_updates_dist(params, updates, module_metadata_manager):
    """
    Applies updates to params by adding them together. This is an extension
    of `optax.apply_udpates` helper function for sharded arrays.
    """
    aligned_pspecs = align_pspecs_for_module_layers(module_metadata_manager)
    out = apply_sharded_fn_to_params(
        module_metadata_manager.mesh,
        aligned_pspecs,
        optax_update.apply_updates,
        params,
        updates,
    )
    return out
