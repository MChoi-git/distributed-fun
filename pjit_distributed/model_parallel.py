from typing import Union, Callable, Tuple, Dict, Any, List
from dataclasses import dataclass

import jax
from jax import numpy as jnp, random
from jax.experimental import maps, PartitionSpec
from jax.experimental.pjit import pjit
import flax.linen as nn


class ModelParallelMaskedMSA(nn.Module):
    hidden: int
    heads: int
    qkv_dropout: int

    kernel_init: Callable = jax.nn.initializers.lecun_normal()
    bias_init: Callable = jax.nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs, train, *args, **kwargs):
        B, S, E = inputs.shape

        x = inputs

        qkv_kernel = self.param(
            "qkv_kernel",
            self.kernel_init,
            (self.hidden, self.hidden * 3),
        )
        qkv_bias = self.param(
            "qkv_bias",
            self.bias_init,
            (1, self.hidden * 3),
        )

        qkv = jnp.einsum("bse,ef->bsf", x, qkv_kernel) + qkv_bias

        qkv_heads = qkv.reshape(B, self.heads, S, -1)

        q, k, v = jnp.split(qkv_heads, 3, axis=-1)

        qk = jnp.einsum("bhse,bhte->bhst", q, k) / (2 * self.hidden) ** 0.5

        mask = jnp.triu(jnp.zeros((S, S)) - jnp.inf, k=1)
        attn = jax.nn.softmax(qk + mask, axis=-1)

        attn = nn.Dropout(rate=self.qkv_dropout)(attn, deterministic=not train)

        self_attn = jnp.einsum("bhst,bhte->bhse", attn, v).reshape(B, S, E)

        return self_attn


class RowParallelLinear(nn.Module):
    hidden: int
    dropout: int

    kernel_init: Callable = jax.nn.initializers.lecun_normal()
    bias_init: Callable = jax.nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs, train, *args, **kwargs):
        out_kernel = self.param(
            "row_kernel",
            self.kernel_init,
            (self.hidden, self.hidden),
        )
        out_bias = self.param("row_bias", self.bias_init, (self.hidden, 1))

        x = inputs

        # E is full features dimensionality
        out = jnp.einsum("bse,eE->bsE", x, out_kernel) + out_bias.transpose(1, 0)

        out = nn.Dropout(rate=self.dropout)(out, deterministic=not train)

        return out


class ColumnParallelLinear(nn.Module):
    hidden: int
    dropout: int

    nonlin: Callable = jax.nn.gelu

    kernel_init: Callable = jax.nn.initializers.lecun_normal()
    bias_init: Callable = jax.nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs, train, *args, **kwargs):
        out_kernel = self.param(
            "col_kernel",
            self.kernel_init,
            (inputs.shape[-1], self.hidden),
        )
        out_bias = self.param("col_bias", self.bias_init, (1, self.hidden))

        x = inputs

        out = jnp.einsum("bse,eE->bsE", x, out_kernel) + out_bias

        out = self.nonlin(out)

        out = nn.Dropout(rate=self.dropout)(out, deterministic=not train)

        return out


class VocabParallelEmbed(nn.Module):
    vocab_size: int
    hidden: int

    def setup(self):
        self.embed = nn.Embed(self.vocab_size, self.hidden)

    def __call__(self, inputs, *args, **kwargs):
        x = inputs

        out = self.embed(x)

        return out

    def attend(self, inputs):
        x = inputs

        out = self.embed.apply(
            self.embed.variables,
            x,
            method=self.embed.attend,
        )
        return out

    def fused_softmax_ce_loss(self, logits, targets, label_smoothing):
        x = logits  # (B, S, V/tp)

        softmax_embeds = jax.nn.softmax(x, axis=-1)

        def ce_loss(x, y):
            smooth_label = jnp.where(
                jax.nn.one_hot(y, self.vocab_size) == 0,
                label_smoothing / (self.vocab_size - 1),
                1 - label_smoothing,
            )
            return -jnp.sum(smooth_label * jnp.clip(jnp.log(x), a_min=-100))

        partial_losses = jax.vmap(jax.vmap(ce_loss))(softmax_embeds, targets)

        return partial_losses


class PositionEmbed(nn.Module):
    seq_len: int
    hidden: int

    kernel_init: Callable = jax.nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, inputs, *args, **kwargs):
        x = inputs

        embeds = self.param(
            "pos_embed",
            self.kernel_init,
            (self.seq_len, self.hidden),
        )

        return x + embeds


class Layernorm(nn.Module):
    @nn.compact
    def __call__(self, inputs, *args, **kwargs):
        x = inputs

        out = nn.LayerNorm()(x)

        return out


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
    fn = pjit(
        lambda params, data, rngs: module_metadata.layer.apply(
            params,
            data,
            *module_metadata.train_args,
            **module_metadata.train_kwargs,
            rngs=rngs,
        ),
        in_axis_resources=module_metadata.in_train_pspec,
        out_axis_resources=module_metadata.out_train_pspec,
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
    module_init_args: Union[Tuple, dict, None]
    init_args: Union[Tuple, dict, None] = None
    init_kwargs: Union[Dict[str, Any], None] = None
    train_args: Union[Tuple, None] = None
    train_kwargs: Union[Dict[str, Any], None] = None

    # pjit functions for initialization and forward
    pjit_init: Union[Callable, None] = None
    pjit_forward: Union[Callable, None] = None

    # Partition specs for optax optimizer functions TransformInitFn,
    # TransformUpdateFn, and apply_updates
    in_optim_init_pspec: Union[PartitionSpec, None] = None
    out_optim_init_pspec: Union[PartitionSpec, None] = None
    in_optim_update_pspec: Union[PartitionSpec, None] = None
    out_optim_update_pspec: Union[PartitionSpec, None] = None
    in_optim_apply_updates_pspec: Union[PartitionSpec, None] = None
    out_optim_apply_updates_pspec: Union[PartitionSpec, None] = None

    def __post_init__(self):
        # Maintain the sharding of the parameters, since the output is the same
        # dimensionality and treedef as the params.
        if self.in_optim_init_pspec is None:
            self.in_optim_init_pspec = self.out_init_pspec
        if self.out_optim_init_pspec is None:
            self.out_optim_init_pspec = self.out_init_pspec

        # For optim update_fns, input is updates (grads), state, and params
        # which are all the same dimensionality and treedef. Therefore we
        # simply repeat the PartitionSpec of the parameters. The output does
        # not include the params, so the PartitionSpec is only repeated twice.
        if self.in_optim_update_pspec is None:
            self.in_optim_update_pspec = [self.out_init_pspec] * 3
        if self.out_optim_update_pspec is None:
            self.out_optim_update_pspec = [self.out_init_pspec] * 2

        # Applying updates takes params and updates, and just returns the
        # updated params.
        if self.in_optim_apply_updates_pspec is None:
            self.in_optim_apply_updates_pspec = [self.out_init_pspec] * 2
        if self.out_optim_apply_updates_pspec is None:
            self.out_optim_apply_updates_pspec = self.out_init_pspec


@dataclass
class ModuleMetadataManager:
    """
    Class which handles metadata allocation for each layer in the
    transformer. Creates both the pjit functions to init parameters, and
    the pjit functions to do forward passes.
    """

    mesh: maps.Mesh
    num_layers: int  # Number of core layer repeats in model
    module_metadata_list: List[ModuleMetadata]

    def __post_init__(self):
        jax.tree_util.tree_map(
            lambda meta: setattr(meta, "layer", meta.layer(*meta.module_init_args)),
            self.module_metadata_list,
        )

    def bind_pjit_fns(self, pjit_fns, attribute):
        """
        Attach the given pjit_fns to the specified attribute in the
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

    def forward_from_pjit_metadata(self):
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
            out_axis_resources=PartitionSpec(None, None, "tp")
        )
        setattr(
            self.module_metadata_list[0], "pjit_attend", attend_fn,
        )
        fused_softmax_ce_loss_fn = pjit(
            lambda logits, targets, label_smoothing: self.module_metadata_list[0].layer.fused_softmax_ce_loss(
                logits,
                targets,
                label_smoothing,
            ),
            in_axis_resources=[
                PartitionSpec(None, None, "tp"),  # Same sharding as embed.__call__
                None,
                None,   # Find out way to fix label smoothing
            ],
            out_axis_resources=PartitionSpec(None, "tp"),
        )
        setattr(
            self.module_metadata_list[0], "pjit_fused_softmax_ce_loss", fused_softmax_ce_loss_fn,
        )

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


def forward(all_params, module_metadata_manager, inputs, targets, mesh, dropout_rng_key, label_smoothing):
    """
    Forward pass for transformer. Uses binded params and pjit functions from
    module_metadata_list container.
    """
    # Quick alias
    meta_list = module_metadata_manager.module_metadata_list

    with maps.Mesh(mesh.devices, mesh.axis_names):
        x = inputs

        embeds = meta_list[0].pjit_forward(all_params["embed_0"], x, None)

        core_input = meta_list[1].pjit_forward(all_params["pos_embed_0"], embeds, None)

        for i in range(module_metadata_manager.num_layers):
            dropout_rng_key, qkv_dropout, msa_dropout, mlp_dropout = random.split(
                dropout_rng_key, num=4
            )

            ln_msa = meta_list[2].pjit_forward(
                all_params[f"layernorm_msa_{i}"], core_input, None
            )
            self_attn = meta_list[3].pjit_forward(
                all_params[f"msa_attn_{i}"], ln_msa, {"dropout": qkv_dropout}
            )
            msa_out = meta_list[4].pjit_forward(
                all_params[f"msa_mlp_{i}"], self_attn, {"dropout": msa_dropout}
            )
            msa_res_out = msa_out + core_input

            ln_mlp = meta_list[5].pjit_forward(
                all_params[f"layernorm_mlp_{i}"], msa_res_out, None
            )
            mlp_col_out = meta_list[6].pjit_forward(
                all_params[f"mlp_col_{i}"], ln_mlp, None
            )
            mlp_row_out = meta_list[7].pjit_forward(
                all_params[f"mlp_row_{i}"],
                mlp_col_out,
                {"dropout": mlp_dropout},
            )
            core_input = mlp_row_out + msa_res_out

        logits = meta_list[0].pjit_attend(
            all_params["embed_0"],
            core_input,
        )
        out = meta_list[0].pjit_fused_softmax_ce_loss(
            logits,
            targets,
            label_smoothing,
        )

    return out


def softmax_cross_entropy_loss(
    all_params,
    module_metadata_manager,
    x_batched,
    labels,
    mesh,
    dropout_rng_key,
    vocab_size,
    label_smoothing,
):
    preds_batched = forward(
        all_params, module_metadata_manager, x_batched, labels, mesh, dropout_rng_key, label_smoothing
    )
    # TODO: Outputs NaN when training on the same batch
    return preds_batched.mean()
