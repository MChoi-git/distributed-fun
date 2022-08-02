from typing import Union, Callable, Tuple, Any
from dataclasses import dataclass
from functools import partial

import jax
from jax import numpy as jnp
from jax.experimental import maps, PartitionSpec
from jax.experimental.pjit import pjit
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict
import numpy as np


def get_mesh(mesh_shape, axis_names):
    devices = np.asarray(jax.devices()).reshape(*mesh_shape)
    mesh = maps.Mesh(devices, axis_names)
    return mesh


def generic_pjit_init_fn(module_metadata):
    """Generic flax nn.Module init function for params sharded using pjit"""
    fn = pjit(
        lambda: module_metadata.layer.init(
            module_metadata.rng,
            jnp.ones(module_metadata.data_shape, dtype=module_metadata.dtype),
            *module_metadata.init_args,
            **module_metadata.init_kwargs,
        ),
        in_axis_resources=module_metadata.in_init_pspec,
        out_axis_resources=module_metadata.out_init_pspec,
    )
    return fn


def generic_pjit_forward_fn(module_metadata):
    """Generic flax nn.Module apply function for params sharded using pjit"""
    # fn needs to accept params and data, since the in and out PartitionSpecs
    # need to specify both params and data sharding
    # TODO: Clean this up by removing train args and kwargs, since they are
    #       always empty or only contain train=True for dropout.
    if module_metadata.checkpoint_activations:
        apply_fn = jax.checkpoint(partial(module_metadata.layer.apply, train=True))
    else:
        apply_fn = partial(module_metadata.layer.apply, train=True)

    fn = pjit(
        lambda params, data, rngs: apply_fn(
            params,
            data,
            *module_metadata.train_args,
            #**module_metadata.train_kwargs,
            rngs=rngs,
        ),
        in_axis_resources=module_metadata.in_train_pspec,
        out_axis_resources=module_metadata.out_train_pspec,
    )

    return fn


def generic_pjit_inference_fn(module_metadata):
    """
    Generic flax nn.Module inference function for params sharded using
    pjit.
    """
    fn = pjit(
        lambda params, data: module_metadata.layer.apply(
            params,
            data,
            *module_metadata.inference_args,
            **module_metadata.inference_kwargs,
        ),
        in_axis_resources=module_metadata.in_inference_pspec,
        out_axis_resources=module_metadata.out_inference_pspec,
    )
    return fn


@dataclass
class ModuleMetadata:
    """
    Metadata object that holds all necessary information for each
    megatron transformer layer.
    """
    rng: jax._src.prng.PRNGKeyArray
    name: str
    num_layers: int  # Number of instances of specific module in whole model
    checkpoint_activations: bool

    # Partition specs for module.init and module.apply
    in_init_pspec: Union[PartitionSpec, None]
    out_init_pspec: Union[PartitionSpec, None]
    in_train_pspec: Union[PartitionSpec, None]
    out_train_pspec: Union[PartitionSpec, None]

    # Layer expected input metadata
    layer: nn.Module
    data_shape: Tuple[int]
    dtype: jax._src.numpy.lax_numpy._ScalarMeta

    # Args for initializing the module and training the the module
    module_init_args: Union[Tuple, dict, None] = None
    module_init_kwargs: Union[FrozenDict[str, Any], None] = None
    init_args: Union[Tuple, dict, None] = None
    init_kwargs: Union[FrozenDict[str, Any], None] = None
    train_args: Union[Tuple, None] = None
    train_kwargs: Union[FrozenDict[str, Any], None] = None
    inference_args: Union[Tuple, None] = None
    inference_kwargs: Union[FrozenDict[str, Any], None] = None

    # pjit functions for initialization and forward
    pjit_init: Union[Callable, None] = None
    pjit_forward: Union[Callable, None] = None
    pjit_inference: Union[Callable, None] = None

    # Partition specs for optax optimizer functions TransformInitFn,
    # TransformUpdateFn, and apply_updates
    in_optim_pspec: Union[PartitionSpec, None] = None
    out_optim_pspec: Union[PartitionSpec, None] = None

    def __post_init__(self):
        # For optim update_fns, input is updates (grads), state, and params
        # which are all the same dimensionality and treedef. Therefore we
        # simply repeat the PartitionSpec of the parameters. The output does
        # not include the params, so the PartitionSpec is only repeated twice.
        if self.in_optim_pspec is None:
            self.in_optim_pspec = self.out_init_pspec
        if self.out_optim_pspec is None:
            self.out_optim_pspec = self.out_init_pspec

        self.in_inference_pspec = self.in_train_pspec
        self.out_inference_pspec = self.out_train_pspec
        self.inference_kwargs = self.train_kwargs.copy({"train": False})

    def __hash__(self):
        return hash(
            (
                self.name,
                self.num_layers,
                self.in_init_pspec,
                self.out_init_pspec,
                self.in_train_pspec,
                self.out_train_pspec,
                self.layer,
                self.data_shape,
                self.dtype,
                self.module_init_args,
                self.init_args,
                self.init_kwargs,
                self.train_args,
                self.train_kwargs,
                self.inference_args,
                self.inference_kwargs,
                self.pjit_init,
                self.pjit_forward,
                self.pjit_inference,
                self.in_optim_pspec,
                self.out_optim_pspec,
                self.in_inference_pspec,
                self.out_inference_pspec,
            )
        )

    def _tree_flatten(self):
        raise NotImplementedError

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        raise NotImplementedError


@dataclass
class ModuleMetadataManager:
    """
    Class which handles metadata allocation for each layer in the
    transformer. Creates both the pjit functions to init parameters, and
    the pjit functions to do forward passes.
    """

    mesh: maps.Mesh
    num_layers: int  # Number of core layer repeats in model
    module_metadata_list: Tuple[ModuleMetadata]

    def __post_init__(self):
        """
        Initializes the flax layer for each ModuleMetadata object in the
        list
        """
        jax.tree_util.tree_map(
            lambda meta: setattr(
                meta,
                "layer",
                meta.layer(*meta.module_init_args, **meta.module_init_kwargs),
            ),
            self.module_metadata_list,
        )

    def __hash__(self):
        return hash(
            (
                self.mesh,
                self.num_layers,
                *self.module_metadata_list,
            )
        )

    def bind_pjit_fns(self, pjit_fns, attribute):
        """
        Bind the given pjit_fns to the specified attribute in the
        ModuleMetadata object.
        """
        for meta, fn in zip(self.module_metadata_list, pjit_fns):
            setattr(meta, attribute, fn)

    def init_from_pjit_metadata(self):
        """
        Initialize the collection of ModuleMetadata objects using their
        initialization arguments. The resulting parameters are sharded
        according to their initialization PartitionSpecs, and returned in a
        separate dictionary. The pjit_init functions are also saved in each
        respective ModuleMetadata object.
        """
        # Generate pjit functions for each layer type
        pjit_fns = jax.tree_util.tree_map(
            generic_pjit_init_fn, self.module_metadata_list
        )

        # Bind pjit functions to each ModuleMetadata object
        self.bind_pjit_fns(pjit_fns, "pjit_init")

        def init_according_to_num_layers(meta):
            """
            Helper fn which simply calls the pjit_init function for the
            given ModuleMetadata object the number of times specified by the
            num_layers attribute.
            """
            meta_params = {}
            for i in range(meta.num_layers):
                meta_params[f"{meta.name}_{i}"] = meta.pjit_init()
            return meta_params

        # Create list of layer parameter dicts
        with maps.Mesh(self.mesh.devices, self.mesh.axis_names):
            model_params = jax.tree_util.tree_map(
                init_according_to_num_layers,
                self.module_metadata_list,
            )

        # Combine the list of dicts into one dict
        full_params_dict = {k: v for layer in model_params for k, v in layer.items()}

        return full_params_dict

    def init_forward_from_pjit_metadata(self):
        """
        Create the forward pjit functions for each ModuleMetadata object in
        the collection, according to its specific training arguments and
        PartitionSpecs. Save these pjit forward functions in each respective
        ModuleMetadata object.
        """
        # Core forward functions, ie. __call__ methods
        pjit_fns = jax.tree_util.tree_map(
            generic_pjit_forward_fn, self.module_metadata_list
        )

        self.bind_pjit_fns(pjit_fns, "pjit_forward")

        # Ragged forward functions, ie. other methods like nn.Embed's attend
        # TODO: Expand the ModuleMetata object to take an arbitrary number of
        #       module functions, as well as matchin pspecs.
        attend_fn = pjit(
            lambda params, x: self.module_metadata_list[0].layer.apply(
                params,
                x,
                method=self.module_metadata_list[0].layer.attend,
            ),
            in_axis_resources=None,
            out_axis_resources=PartitionSpec(None, None, "tp"),
        )
        setattr(
            self.module_metadata_list[0],
            "pjit_attend",
            attend_fn,
        )

        fused_softmax_ce_loss_fn = pjit(
            lambda logits, targets, label_smoothing: self.module_metadata_list[
                0
            ].layer.fused_softmax_ce_loss(
                logits,
                targets,
                label_smoothing,
            ),
            in_axis_resources=[
                PartitionSpec(None, None, "tp"),  # Same sharding as embed.__call__
                None,
                None,  # Find out way to fix label smoothing
            ],
            out_axis_resources=PartitionSpec(None, "tp"),
        )
        setattr(
            self.module_metadata_list[0],
            "pjit_fused_softmax_ce_loss",
            fused_softmax_ce_loss_fn,
        )

    def init_inference_from_pjit_metadata(self):
        """
        Create the forward pjit functions for each ModuleMetadata object in
        the collection, according to its specific training arguments and
        PartitionSpecs. Save these pjit forward functions in each respective
        ModuleMetadata object.
        """
        # Core forward functions, ie. __call__ methods
        pjit_fns = jax.tree_util.tree_map(
            generic_pjit_inference_fn, self.module_metadata_list
        )

        # Ragged functions can be reused from init_forward_from_pjit_metadata
        self.bind_pjit_fns(pjit_fns, "pjit_inference")

    @staticmethod
    def inspect_params(param_tree):
        """
        Return a pytree where the leaves are the shape of the sharded
        parameters.
        """
        tree_shape = jax.tree_util.tree_map(
            lambda param: jax.tree_util.tree_map(
                lambda p: p.shape,
                param.device_buffers,
            ),
            param_tree,
        )
        return tree_shape
