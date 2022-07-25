from typing import Optional, Any, Union, Callable

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
        aligned_pspecs = align_pspecs_for_module_layers(module_metadata_manager)

        params_values, params_treedef = jax.tree_util.tree_flatten(
            params, is_leaf=lambda x: isinstance(x, flax.core.frozen_dict.FrozenDict)
        )
        pspecs_values, pspecs_treedef = jax.tree_util.tree_flatten(
            aligned_pspecs, is_leaf=lambda x: x is None or isinstance(x, tuple)
        )

        # Make sure that treedefs for params and their corresponding pspecs are
        # the same
        assert params_treedef == pspecs_treedef

        breakpoint()
        # TODO: Tree map jointly over params and pspecs, using function make_mu
        #       to init the optim state.

        def make_mu(in_pspec, out_pspec, param):
            with maps.Mesh(
                module_metadata_manager.mesh.devices,
                module_metadata_manager.mesh.axis_names,
            ):
                opt_state = pjit(
                    lambda p: jnp.zeros_like(p, dtype=mu_dtype),
                    in_axis_resources=in_pspec,
                    out_axis_resources=out_pspec,
                )(param)
            return opt_state

    def update_fn(updates, state, params=None):
        # TODO: Why is the first line to `del params`?
        """
        Update fn implementing the Adam update rule. Returns a tuple of the
        calculated updates state and an updated ScaleByAdamState metadata
        object with the updates mean and stddev arrays.
        """
        pass

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
