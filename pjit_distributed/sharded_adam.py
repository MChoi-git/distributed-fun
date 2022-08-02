from typing import Optional, Any, Union, NamedTuple
from functools import partial

import chex
import jax
from jax import numpy as jnp
from jax.experimental import maps, PartitionSpec
from jax.experimental.pjit import pjit
import optax
from optax._src import base as optax_base
from optax._src import transform as optax_transform
from optax._src import combine as optax_combine
from optax._src import numerics as optax_numerics
from optax._src import linear_algebra as optax_lin_alg
from optax._src import utils as optax_utils

from model_parallel import ModuleMetadataManager


def align_pspecs_for_module_layers(module_metadata_manager, scalar=False):
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
                meta.in_optim_pspec,
                meta.out_optim_pspec if scalar is False else None,
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

    def eval_shard_fn(mesh, pspec_tuple, fn, *args):
        in_pspec, out_pspec = pspec_tuple

        with maps.Mesh(
            mesh.devices,
            mesh.axis_names,
        ):
            out = pjit(
                fn,
                in_axis_resources=[in_pspec] * len(tree_map_args),
                out_axis_resources=out_pspec,
            )(*args)

        return out

    out = jax.tree_util.tree_map(
        lambda ps, *args: eval_shard_fn(mesh, ps, fn, *args),
        pspecs,
        *tree_map_args,
        is_leaf=lambda x: x is None or isinstance(x, tuple),
    )
    return out


class ScaleByAdamDistState(NamedTuple):
    """
    Primary gradient transformation object containing the Adam state and a
    copy of the parameters.
    """
    count: chex.Array
    mu: optax_base.Updates  # chex.ArrayTree
    nu: optax_base.Updates
    params_copy: optax_base.Updates


def adamw_dist(
    module_metadata_manager: ModuleMetadataManager,
    learning_rate: Union[float, optax_base.Schedule],
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
    nu_dtype: Optional[Any] = None,
    params_copy_dtype: Optional[Any] = None,
    weight_decay: Optional[float] = 1e-4,
    clipping: Optional[float] = 1.0,
    loss_scaling: Optional[int] = None,
) -> optax_base.GradientTransformation:
    """
    Takes the adam optimizer hyperparameters and composes the necessary
    extra transformations for gradient clipping, weight decay, etc.
    """
    transforms = []
    transforms.append(
        scale_by_adam_dist(
            module_metadata_manager=module_metadata_manager,
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            mu_dtype=mu_dtype,
            nu_dtype=nu_dtype,
            params_copy_dtype=params_copy_dtype,
        )
    )
    if loss_scaling is not None:
        transforms.append(loss_scaling_dist(
            loss_scaling,
            module_metadata_manager
        ))

    if clipping is not None:
        transforms.append(clip_by_global_norm_dist(
            clipping,
            module_metadata_manager
        ))

    transforms.append(scale_dist(-1 * learning_rate, module_metadata_manager))

    if weight_decay is not None:
        transforms.append(
            add_decayed_weights_dist(weight_decay, module_metadata_manager)
        )

    return optax_combine.chain(*transforms)


def scale_by_adam_dist(
    module_metadata_manager: ModuleMetadataManager,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
    nu_dtype: Optional[Any] = None,
    params_copy_dtype: Optional[Any] = None,
) -> optax_base.GradientTransformation:
    """
    Re-implementation of scale_by_adam from original Optax source code. Here
    the init_fn and update_fn will be re-implemented to support multi-GPU
    sharding. Returns an initial ScaleByAdamState containing the metadata and
    mean and stddev zero arrays.
    """
    mu_dtype = optax_utils.canonicalize_dtype(mu_dtype)
    nu_dtype = optax_utils.canonicalize_dtype(nu_dtype)
    params_copy_dtype = optax_utils.canonicalize_dtype(params_copy_dtype)

    def init_fn(params):
        """Init optim state for Adam, ie. mean and stddev arrays of zeros."""
        # Align each parameter FrozenDict with it's respective pspecs given
        # by the corresponding ModuleMetadata
        aligned_pspecs = align_pspecs_for_module_layers(
            module_metadata_manager
        )

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
            lambda pspecs, p: sharded_zeros_like(pspecs, p, dtype=nu_dtype),
            aligned_pspecs,
            params,
            is_leaf=lambda x: x is None or isinstance(x, tuple),
        )
        params_copy = apply_sharded_fn_to_params(
            module_metadata_manager.mesh,
            aligned_pspecs,
            partial(optax_utils.cast_tree, dtype=params_copy_dtype),
            params,
        )

        return ScaleByAdamDistState(
            count=jnp.zeros([], jnp.int32),
            mu=mu,
            nu=nu,
            params_copy=params_copy,
        )

    def update_fn(updates, state, params):
        # TODO: Why is the first line to `del params`?
        """
        Update fn implementing the Adam update rule. Returns a tuple of the
        calculated updates state and an updated ScaleByAdamState metadata
        object with the updates mean and stddev arrays.
        """
        # Align each parameter FrozenDict with it's respective pspecs given
        # by the corresponding ModuleMetadata
        aligned_pspecs = align_pspecs_for_module_layers(
            module_metadata_manager
        )

        def calculate_updates(mu, nu):
            return jax.tree_util.tree_map(
                lambda m, v: m / (jnp.sqrt(v + eps_root) + eps),
                mu,
                nu,
            )

        # Promote grads to fp32 for calculations
        promoted_updates = apply_sharded_fn_to_params(
            module_metadata_manager.mesh,
            aligned_pspecs,
            partial(optax_utils.cast_tree, dtype=params_copy_dtype),
            updates,
        )

        mu = apply_sharded_fn_to_params(
            module_metadata_manager.mesh,
            aligned_pspecs,
            partial(optax_transform.update_moment, decay=b1, order=1),
            promoted_updates,
            state.mu,
        )
        # TODO: This function blows up large grad values which are in fp16
        nu = apply_sharded_fn_to_params(
            module_metadata_manager.mesh,
            aligned_pspecs,
            partial(
                optax_transform.update_moment_per_elem_norm,
                decay=b2,
                order=2
            ),
            promoted_updates,
            state.nu,
        )

        count_inc = optax_numerics.safe_int32_increment(state.count)

        mu_hat = apply_sharded_fn_to_params(
            module_metadata_manager.mesh,
            aligned_pspecs,
            partial(
                optax_transform.bias_correction,
                decay=b1,
                count=count_inc
            ),
            mu,
        )
        nu_hat = apply_sharded_fn_to_params(
            module_metadata_manager.mesh,
            aligned_pspecs,
            partial(
                optax_transform.bias_correction,
                decay=b2,
                count=count_inc
            ),
            nu,
        )

        updates_new = apply_sharded_fn_to_params(
            module_metadata_manager.mesh,
            aligned_pspecs,
            calculate_updates,
            mu_hat,
            nu_hat,
        )

        # Note that if dtype is not partial'd, then this will return None
        # mu updates=values
        mu_new = apply_sharded_fn_to_params(
            module_metadata_manager.mesh,
            aligned_pspecs,
            partial(optax_utils.cast_tree, dtype=mu_dtype),
            mu,
        )
        nu_new = apply_sharded_fn_to_params(
            module_metadata_manager.mesh,
            aligned_pspecs,
            partial(optax_utils.cast_tree, dtype=nu_dtype),
            nu,
        )
        promoted_params = apply_sharded_fn_to_params(
            module_metadata_manager.mesh,
            aligned_pspecs,
            partial(optax_utils.cast_tree, dtype=params_copy_dtype),
            params,
        )

        return updates_new, ScaleByAdamDistState(
            count=count_inc, mu=mu_new, nu=nu_new, params_copy=promoted_params
        )

    return optax_base.GradientTransformation(init_fn, update_fn)


def apply_updates_dist(half_params, state, updates, module_metadata_manager):
    """
    Applies updates to the single precision copy of the model parameters held
    in state, and returns the new parameters with the same dtypes given by the
    half precision parameters.
    """
    aligned_pspecs = align_pspecs_for_module_layers(module_metadata_manager)

    def apply_updates(single_p, half_p, single_u):
        return jax.tree_util.tree_map(
            lambda s, h, u: jnp.asarray(s + u).astype(jnp.asarray(h).dtype),
            single_p,
            half_p,
            single_u,
        )

    out = apply_sharded_fn_to_params(
        module_metadata_manager.mesh,
        aligned_pspecs,
        apply_updates,
        state[0].params_copy,
        half_params,
        updates,
    )
    return out


def clip_by_global_norm_dist(
    max_norm: float,
    module_metadata_manager: ModuleMetadataManager,
):
    """
    Gradient transformation which clips the gradient to the max norm
    specified
    """
    def init_fn(params):
        del params
        return optax_transform.ClipByGlobalNormState()

    def update_fn(updates, state, params=None):
        del params

        aligned_pspecs = align_pspecs_for_module_layers(
            module_metadata_manager
        )
        # This does an all-reduce
        g_norm = optax_lin_alg.global_norm(updates)

        trigger = jnp.squeeze(g_norm < max_norm)

        def clip_fn(t):
            return jax.lax.select(
                trigger,
                t,
                (t / g_norm.astype(t.dtype)) * max_norm
            )

        def update_tree(updates):
            return jax.tree_util.tree_map(
                clip_fn,
                updates,
            )

        out = apply_sharded_fn_to_params(
            module_metadata_manager.mesh,
            aligned_pspecs,
            update_tree,
            updates,
        )
        return out, state

    return optax_base.GradientTransformation(init_fn, update_fn)


def add_decayed_weights_dist(
    weight_decay: float,
    module_metadata_manager: ModuleMetadataManager,
):
    """
    Transformation which scales the current parameters by a specified amount
    """
    def init_fn(params):
        del params
        return optax_transform.AddDecayedWeightsState()

    def update_fn(updates, state, params):
        aligned_pspecs = align_pspecs_for_module_layers(
            module_metadata_manager
        )

        def update_tree(updates, params):
            return jax.tree_util.tree_map(
                lambda g, p: g + weight_decay * p,
                updates,
                params,
            )

        out = apply_sharded_fn_to_params(
            module_metadata_manager.mesh,
            aligned_pspecs,
            update_tree,
            updates,
            params,
        )

        return out, state

    return optax_base.GradientTransformation(init_fn, update_fn)


def scale_dist(
    step_size: float,
    module_metadata_manager: ModuleMetadataManager,
):
    """
    Transformation which scales the weights by an arbitrary specified
    amount
    """
    def init_fn(params):
        del params
        return optax_transform.ScaleState()

    def update_fn(updates, state, params=None):
        del params
        aligned_pspecs = align_pspecs_for_module_layers(
            module_metadata_manager
        )

        def update_tree(params_like):
            return jax.tree_util.tree_map(lambda g: step_size * g, params_like)

        updates = apply_sharded_fn_to_params(
            module_metadata_manager.mesh,
            aligned_pspecs,
            update_tree,
            updates,
        )
        return updates, state

    return optax_base.GradientTransformation(init_fn, update_fn)


def loss_scaling_dist(
    loss_scaling: int, module_metadata_manager: ModuleMetadataManager
):
    """Transformation which reverses loss scaling by unscaling the gradients"""
    def init_fn(params):
        del params
        return optax_transform.ScaleState()

    def update_fn(updates, state, params=None):
        del params
        aligned_pspecs = align_pspecs_for_module_layers(
            module_metadata_manager
        )

        def unscale_grads(grads):
            return jax.tree_util.tree_map(
                lambda g: 1 / loss_scaling * g,
                grads
            )

        updates = apply_sharded_fn_to_params(
            module_metadata_manager.mesh,
            aligned_pspecs,
            unscale_grads,
            updates,
        )
        return updates, state

    return optax_base.GradientTransformation(init_fn, update_fn)


def dynamic_loss_unscaling(
    grads: chex.Array,
    dynamic_loss_scale: int,
    module_metadata_manager: ModuleMetadataManager,
):
    aligned_pspecs = align_pspecs_for_module_layers(
        module_metadata_manager,
    )

    def unscale_grads(grads):
        return jax.tree_util.tree_map(
            lambda g: 1 / dynamic_loss_scale * g,
            grads,
        )

    unscaled_grads = apply_sharded_fn_to_params(
        module_metadata_manager.mesh,
        aligned_pspecs,
        unscale_grads,
        grads,
    )

    return unscaled_grads


def dynamic_loss_nan_check(
    grads: chex.Array,
    module_metadata_manager: ModuleMetadataManager,
):
    aligned_pspecs = align_pspecs_for_module_layers(
        module_metadata_manager,
        scalar=True,
    )

    def accumulate_nans(grads):
        n = jax.tree_util.tree_map(
            lambda g: (~jnp.isfinite(g)).sum(),
            jax.tree_leaves(grads),
        )
        return sum(n)

    nans = apply_sharded_fn_to_params(
        module_metadata_manager.mesh,
        aligned_pspecs,
        accumulate_nans,
        grads,
    )

    return sum(jax.tree_leaves(nans))
