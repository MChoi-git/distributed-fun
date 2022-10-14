from typing import Callable, Optional, Any

import jax
from jax import numpy as jnp
from jax.experimental import maps, PartitionSpec
from jax.experimental.pjit import with_sharding_constraint
import flax.linen as nn
from flax.linen.dtypes import promote_dtype


class ModelParallelMaskedMSA(nn.Module):
    hidden: int
    heads: int
    qkv_dropout: int

    dtype: Optional[Any] = None
    param_dtype: Any = jnp.float32

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
            self.param_dtype,
        )
        qkv_bias = self.param(
            "qkv_bias",
            self.bias_init,
            (1, self.hidden * 3),
            self.param_dtype,
        )

        x, qkv_kernel, qkv_bias = promote_dtype(
            x, qkv_kernel, qkv_bias, dtype=self.param_dtype
        )

        qkv = jnp.einsum("bse,ef->bsf", x, qkv_kernel) + qkv_bias

        qkv_heads = qkv.reshape(B, self.heads, S, -1)

        q, k, v = jnp.split(qkv_heads, 3, axis=-1)
        q = jax.experimental.pjit.with_sharding_constraint(q, PartitionSpec(None, "tp", None))

        qk = jnp.einsum("bhse,bhte->bhst", q, k) / (2 * self.hidden) ** 0.5

        mask = jnp.triu(jnp.zeros((S, S), dtype=self.param_dtype) - jnp.inf, k=1)
        attn = jax.nn.softmax(qk + mask, axis=-1)

        attn = nn.Dropout(rate=self.qkv_dropout)(attn, deterministic=not train)

        self_attn = jnp.einsum("bhst,bhte->bhse", attn, v).reshape(B, S, E)

        return self_attn


class RowParallelLinear(nn.Module):
    hidden: int
    dropout: float

    dtype: Optional[Any] = None
    param_dtype: Any = jnp.float32

    kernel_init: Callable = jax.nn.initializers.lecun_normal()
    bias_init: Callable = jax.nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs, train, *args, **kwargs):
        out_kernel = self.param(
            "row_kernel",
            self.kernel_init,
            (inputs.shape[-1], self.hidden),
            self.param_dtype,
        )
        out_bias = self.param(
            "row_bias",
            self.bias_init,
            (self.hidden, 1),
            self.param_dtype,
        )

        x = inputs

        x, out_kernel, out_bias = promote_dtype(
            x, out_kernel, out_bias, dtype=self.param_dtype
        )

        # E is full features dimensionality
        out = jnp.einsum("...se,eE->...sE", x, out_kernel) + out_bias.transpose(1, 0)

        out = nn.Dropout(rate=self.dropout)(out, deterministic=not train)

        return out


class ColumnParallelLinear(nn.Module):
    hidden: int
    dropout: float

    dtype: Optional[Any] = None
    param_dtype: Any = jnp.float32

    nonlin: Callable = jax.nn.gelu

    kernel_init: Callable = jax.nn.initializers.lecun_normal()
    bias_init: Callable = jax.nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs, train, *args, **kwargs):
        out_kernel = self.param(
            "col_kernel",
            self.kernel_init,
            (inputs.shape[-1], self.hidden),
            self.param_dtype,
        )

        out_bias = self.param(
            "col_bias", self.bias_init, (1, self.hidden), self.param_dtype
        )

        x = inputs

        x, out_kernel, out_bias = promote_dtype(
            x, out_kernel, out_bias, dtype=self.param_dtype
        )

        out = jnp.einsum("bse,eE->bsE", x, out_kernel)
        out = with_sharding_constraint(out, PartitionSpec(None, None, "tp"))

        out = out + out_bias
        #out = with_sharding_constraint(out, PartitionSpec(None, None, "tp"))

        out = self.nonlin(out)
        #out = with_sharding_constraint(out, PartitionSpec(None, None, "tp"))

        out = nn.Dropout(rate=self.dropout)(out, deterministic=not train)
        #out = with_sharding_constraint(out, PartitionSpec(None, None, "tp"))

        return out


class VocabParallelEmbed(nn.Module):
    vocab_size: int
    hidden: int

    dtype: Optional[Any] = None
    param_dtype: Any = jnp.float32

    def setup(self):
        self.embed = nn.Embed(
            self.vocab_size, self.hidden, param_dtype=self.param_dtype
        )

    def __call__(self, inputs, *args, **kwargs):
        x = inputs

        out = self.embed(x)

        return out

    def attend(self, inputs):
        x = inputs

        x = promote_dtype(x, dtype=self.param_dtype)

        out = self.embed.apply(
            self.embed.variables,
            *x,
            method=self.embed.attend,
        )
        return out

    def fused_softmax_ce_loss(self, inputs, targets, label_smoothing, tp):
        # TODO: Try to reduce memory usage here
        logits = inputs

        target_one_hots = jax.nn.one_hot(targets, self.vocab_size)

        max_val = jnp.max(logits, axis=-1, keepdims=True)

        logits = logits - jax.lax.stop_gradient(max_val)

        sum_exp_logits = jnp.exp(logits).sum(axis=-1, keepdims=True)
        log_z = jnp.log(sum_exp_logits)

        log_softmax = logits - log_z
        loss = -jnp.sum(target_one_hots * log_softmax, axis=-1)

        return loss


class PositionEmbed(nn.Module):
    seq_len: int
    hidden: int

    dtype: Optional[Any] = None
    param_dtype: Any = jnp.float32

    kernel_init: Callable = jax.nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, inputs, *args, **kwargs):
        x = inputs

        embeds = self.param(
            "pos_embed",
            self.kernel_init,
            (self.seq_len, self.hidden),
            self.param_dtype,
        )

        x, embeds = promote_dtype(x, embeds, dtype=self.param_dtype)

        return x + embeds


class Layernorm(nn.Module):
    dtype: Optional[Any] = None
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, inputs, *args, **kwargs):
        x = inputs

        x = promote_dtype(x, dtype=self.param_dtype)

        out = nn.LayerNorm(param_dtype=self.param_dtype)(*x)

        return out


class ResidualConnection(nn.Module):
    dtype: Optional[Any] = None
    param_dtype: Any = jnp.float32
