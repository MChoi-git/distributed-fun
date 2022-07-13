from typing import Union, Callable, Tuple, Dict, Any
from dataclasses import dataclass
from functools import partial

import jax
from jax import numpy as jnp, random
from jax.experimental import maps, PartitionSpec
from jax.experimental.pjit import pjit
import flax
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
        out_bias = self.param(
            "row_bias",
            self.bias_init,
            (self.hidden, 1)
        )

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
        out_bias = self.param(
            "col_bias",
            self.bias_init,
            (1, self.hidden)
        )

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
            self.embed.variables,
            x,
            method=self.embed.attend
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


@dataclass
class ModuleMetadata:
    rng: jax._src.prng.PRNGKeyArray
    name: str
    in_init_pspec: Union[PartitionSpec, None]
    out_init_pspec: Union[PartitionSpec, None]
    in_train_pspec: Union[PartitionSpec, None]
    out_train_pspec: Union[PartitionSpec, None]
    layer: nn.Module
    data_shape: Tuple[int]
    dtype: jax._src.numpy.lax_numpy._ScalarMeta
    module_init_args: Union[Tuple, dict, None]
    init_args: Union[Tuple, dict, None] = None
    train_args: Union[Tuple, None] = None
    train_kwargs: Union[Dict[str, Any], None] = None
    params: Union[Dict[str, flax.core.FrozenDict], flax.core.FrozenDict, None] = None
    pjit_init: Callable = None
    pjit_forward: Callable = None


def generic_pjit_init_fn(module_metadata):
    """Generic flax nn.Module init function for params sharded using pjit"""
    fn = pjit(
        lambda dummy_data: module_metadata.layer.init(
            module_metadata.rng,
            dummy_data,
            *module_metadata.init_args,
        ),
        in_axis_resources=module_metadata.in_init_pspec,
        out_axis_resources=module_metadata.out_init_pspec,
    )
    out_tuple = (module_metadata.data_shape, module_metadata.dtype, fn)
    return out_tuple


def generic_pjit_forward_fn(module_metadata):
    """Generic flax nn.Module apply function for params sharded using pjit"""
    # fn needs to accept params and data, since the in and out PartitionSpecs
    # need to specify both params and data sharding
    fn = pjit(
        lambda params, data, rngs: module_metadata.layer.apply(
            params,
            data,
            train=module_metadata.train_args,
            rngs=rngs,
        ),
        in_axis_resources=module_metadata.in_train_pspec,
        out_axis_resources=module_metadata.out_train_pspec,
    )
    return fn


@dataclass
class Transformer:
    hidden: int
    heads: int
    seq_len: int
    qkv_dropout: float
    msa_dropout: float
    mlp_dropout: float
    vocab_size: int
    mesh: maps.Mesh

    def __post_init__(self):
        self.module_metadata_list = []

    def prepare_layer_pjit_metadata(
        self,
        rng,
        name,
        in_init_pspec,
        out_init_pspec,
        in_train_pspec,
        out_train_pspec,
        layer,
        data_shape,
        dtype,
        module_init_args=None,
        init_args=None,
        train_args=None,
        train_kwargs=None,
    ):
        if module_init_args is None:
            module_init_args = ()

        if init_args is None:
            init_args = ()

        if train_args is None:
            train_args = ()

        if train_kwargs is None:
            train_kwargs = {}

        model_metadata = ModuleMetadata(
            rng=rng,
            name=name,
            in_init_pspec=in_init_pspec,
            out_init_pspec=out_init_pspec,
            in_train_pspec=in_train_pspec,
            out_train_pspec=out_train_pspec,
            layer=layer(*module_init_args),
            data_shape=data_shape,
            dtype=dtype,
            module_init_args=module_init_args,
            init_args=init_args,
            train_args=train_args,
            train_kwargs=train_kwargs,
            params={},
        )

        self.module_metadata_list.append(model_metadata)

    def init_from_pjit_metadata(self, num_layers, const_layer_end_idx):
        # Generate pjit functions for each layer type
        pjit_fns = jax.tree_util.tree_map(generic_pjit_init_fn, self.module_metadata_list)

        # Save for later if needed
        self.bind_pjit_fns(pjit_fns, "pjit_init")

        # Separate layers which aren't repeated
        core_pjit_fns = pjit_fns[:const_layer_end_idx]
        multilayer_pjit_fns = pjit_fns[const_layer_end_idx:]

        # Create sharded parameters for each layer
        with maps.Mesh(self.mesh.devices, self.mesh.axis_names):
            core_params = jax.tree_util.tree_map(
                lambda meta: meta[2](jnp.ones(meta[0], dtype=meta[1])),
                core_pjit_fns,
                is_leaf=lambda x: isinstance(x, tuple) and len(x) == 3,
            )
            multilayer_params = {}
            for i in range(num_layers):
                params = jax.tree_util.tree_map(
                    lambda meta: meta[2](jnp.ones(meta[0], dtype=meta[1])),
                    multilayer_pjit_fns,
                    is_leaf=lambda x: isinstance(x, tuple) and len(x) == 3,
                )
                multilayer_params[f"layer_{i}"] = params

        for meta, p in zip(self.module_metadata_list[:const_layer_end_idx], core_params):
            meta.params[meta.name] = p

        for layer, params_list in multilayer_params.items():
            for meta, p in zip(self.module_metadata_list[const_layer_end_idx:], params_list):
                meta.params[f"{meta.name}_{layer}"] = p

    def inspect_module_metadata_list(self):
        param_tree_shapes = jax.tree_util.tree_map(
            lambda x: jax.tree_util.tree_map(
                lambda d: jax.tree_util.tree_map(
                    lambda e: e.shape,
                    d.device_buffers,
                ),
                x.params,
            ),
            self.module_metadata_list,
        )
        return param_tree_shapes

    def bind_pjit_fns(self, pjit_fns, attribute):
        for meta, fn in zip(self.module_metadata_list, pjit_fns):
            setattr(meta, attribute, fn)

    def forward(self, inputs, dropout_rng_key):
        pjit_fns = jax.tree_util.tree_map(
            generic_pjit_forward_fn,
            self.module_metadata_list
        )

        output_embed_attend_fn = pjit(
            lambda params, x: self.module_metadata_list[0].layer.apply(
                params, x, method=self.module_metadata_list[0].layer.attend
            ),
            in_axis_resources=[PartitionSpec(None, "tp"), PartitionSpec(None, None, "tp")],
            out_axis_resources=PartitionSpec(None, None, "tp"),
        )

        qkv_dropout, msa_dropout, mlp_dropout = random.split(dropout_rng_key, num=3)

        # Save for later if needed
        self.bind_pjit_fns(pjit_fns, "pjit_forward")

        # Forward-pass logic
        with maps.Mesh(self.mesh.devices, self.mesh.axis_names):
            x = inputs

            init_embeds = self.module_metadata_list[0].pjit_forward(
                self.module_metadata_list[0].params["embed"], x, None
            )

            embeds = self.module_metadata_list[1].pjit_forward(
                self.module_metadata_list[1].params["pos_embed"], init_embeds, None
            )

            self_attn = self.module_metadata_list[2].pjit_forward(
                self.module_metadata_list[2].params["msa_attn_layer_0"], embeds, {"dropout": qkv_dropout}
            )

            msa_out = self.module_metadata_list[3].pjit_forward(
                self.module_metadata_list[3].params["msa_mlp_layer_0"], self_attn, {"dropout": msa_dropout}
            )

            msa_out = msa_out + embeds

            mlp_col_out = self.module_metadata_list[4].pjit_forward(
                self.module_metadata_list[4].params["mlp_col_layer_0"], msa_out, None
            )

            mlp_row_out = self.module_metadata_list[5].pjit_forward(
                self.module_metadata_list[5].params["mlp_row_layer_0"], mlp_col_out, {"dropout": mlp_dropout}
            )

            mlp_out = mlp_row_out + msa_out

            out = output_embed_attend_fn(self.module_metadata_list[0].params["embed"], mlp_out)

        return out
