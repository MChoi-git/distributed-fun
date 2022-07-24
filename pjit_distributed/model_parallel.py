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

    def fused_softmax_ce_loss_output_embedding(self, inputs, targets):
        x = inputs

        output_embeds = self.embed.apply(
            self.embed.variables, x, method=self.embed.attend
        )

        softmax_embeds = jax.nn.softmax(output_embeds, axis=-1)

        def ce_loss(x, y):
            y_vec = jax.nn.one_hot(y, num_classes=self.vocab_size)
            return -jnp.sum(y_vec * jnp.log(x))

        partial_losses = jax.vmap(ce_loss)(softmax_embeds, targets)

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


@dataclass
class ModuleMetadataManager:
    """
    Class which handles metadata allocation for each layer in the
    transformer. Creates both the pjit functions to init parameters, and
    the pjit functions to do forward passes.
    """

    mesh: maps.Mesh
    num_layers: int
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

    def init_from_pjit_metadata(self, const_layer_end_idx):
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

        # Save for later if needed
        self.bind_pjit_fns(pjit_fns, "pjit_init")

        # Separate layers which aren't repeated
        core_pjit_fns = pjit_fns[:const_layer_end_idx]
        multilayer_pjit_fns = pjit_fns[const_layer_end_idx:]

        # Create sharded parameters for each layer
        with maps.Mesh(self.mesh.devices, self.mesh.axis_names):
            # Core layers exist once in model
            core_params = jax.tree_util.tree_map(
                lambda pjit_init_fn: pjit_init_fn(),
                core_pjit_fns,
            )

            # Multilayer layers exist num_layers times in model
            multilayer_params = {}
            for i in range(self.num_layers):
                params = jax.tree_util.tree_map(
                    lambda pjit_init_fn: pjit_init_fn(),
                    multilayer_pjit_fns,
                )
                multilayer_params[f"layer_{i}"] = params

        # Assign parameters to each metadata module
        core_params_dict = {}
        multilayer_params_dict = {}
        for meta, p in zip(
            self.module_metadata_list[:const_layer_end_idx], core_params
        ):
            core_params_dict[meta.name] = p

        for idx, meta in enumerate(self.module_metadata_list[const_layer_end_idx:]):
            params_list = []
            for p in multilayer_params.values():
                params_list.append(p[idx])
            multilayer_params_dict[meta.name] = params_list

        full_params_dict = {
            "core_params": core_params_dict,
            "multilayer_params": multilayer_params_dict,
        }
        return full_params_dict

    def forward_from_pjit_metadata(self, const_layer_end_idx):
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
        # Need to handle these by hand
        output_embed_attend_fn = pjit(
            lambda params, x: self.module_metadata_list[0].layer.apply(
                params, x, method=self.module_metadata_list[0].layer.attend
            ),
            in_axis_resources=[
                PartitionSpec(None, "tp"),
                PartitionSpec(None, None, "tp"),
            ],
            out_axis_resources=PartitionSpec(None, None, "tp"),
        )
        setattr(
            self.module_metadata_list[0], "pjit_forward_attend", output_embed_attend_fn
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


def forward(all_params, module_metadata_manager, inputs, mesh, dropout_rng_key):
    """
    Forward pass for transformer. Uses binded params and pjit functions from
    module_metadata_list container.
    """
    qkv_dropout, msa_dropout, mlp_dropout = random.split(dropout_rng_key, num=3)

    # Quick alias
    module_metadata_list = module_metadata_manager.module_metadata_list

    # Forward-pass logic
    # TODO: 1. Check for correctness against regular transformer forward
    #       2. Alias layers better
    #       3. Do efficient multilayer loops
    #       4. Check for all-reduces and all-gathers between pjit calls
    with maps.Mesh(mesh.devices, mesh.axis_names):
        x = inputs

        # Core params forward
        core_params = all_params["core_params"]

        embeds = module_metadata_list[0].pjit_forward(core_params["embed"], x, None)

        core_input = module_metadata_list[1].pjit_forward(
            core_params["pos_embed"], embeds, None
        )

        # Multilayer params forward
        multilayer_params = all_params["multilayer_params"]

        for i in range(module_metadata_manager.num_layers):
            ln_msa = module_metadata_list[2].pjit_forward(
                multilayer_params["layernorm_msa"][i], core_input, None
            )
            self_attn = module_metadata_list[3].pjit_forward(
                multilayer_params["msa_attn"][i], ln_msa, {"dropout": qkv_dropout}
            )
            msa_out = module_metadata_list[4].pjit_forward(
                multilayer_params["msa_mlp"][i], self_attn, {"dropout": msa_dropout}
            )
            msa_res_out = msa_out + core_input

            ln_mlp = module_metadata_list[5].pjit_forward(
                multilayer_params["layernorm_mlp"][i], msa_res_out, None
            )
            mlp_col_out = module_metadata_list[6].pjit_forward(
                multilayer_params["mlp_col"][i], ln_mlp, None
            )
            mlp_row_out = module_metadata_list[7].pjit_forward(
                multilayer_params["mlp_row"][i],
                mlp_col_out,
                {"dropout": mlp_dropout},
            )
            core_input = mlp_row_out + msa_res_out

        out = module_metadata_list[0].pjit_forward_attend(
            core_params["embed"], core_input
        )

    return out


def softmax_cross_entropy_loss(
    dropout_rng_key,
    all_params,
    module_metadata_manager,
    x_batched,
    labels,
    mesh,
    vocab_size,
    label_smoothing,
):
    def cross_entropy(x, y):
        smooth_label = jnp.where(
            jax.nn.one_hot(y, vocab_size) == 0,
            label_smoothing / (vocab_size - 1),
            1 - label_smoothing,
        )
        return -jnp.sum(smooth_label * jnp.clip(jnp.log(x), a_min=-100))

    preds_batched = forward(
        all_params, module_metadata_manager, x_batched, mesh, dropout_rng_key
    )

    softmax_preds_batched = jax.nn.softmax(preds_batched, axis=-1)

    return jnp.mean(jax.vmap(cross_entropy)(softmax_preds_batched, labels))
