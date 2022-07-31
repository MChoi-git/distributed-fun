from typing import Callable, Optional, Any

import jax
from jax import numpy as jnp
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

        x, qkv_kernel, qkv_bias = promote_dtype(x, qkv_kernel, qkv_bias, dtype=self.dtype)

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

        x, out_kernel, out_bias = promote_dtype(x, out_kernel, out_bias, dtype=self.dtype)

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
            "col_bias",
            self.bias_init,
            (1, self.hidden),
            self.param_dtype
        )

        x = inputs

        x, out_kernel, out_bias = promote_dtype(x, out_kernel, out_bias, dtype=self.dtype)

        out = jnp.einsum("...se,eE->...sE", x, out_kernel) + out_bias

        out = self.nonlin(out)

        out = nn.Dropout(rate=self.dropout)(out, deterministic=not train)

        return out


class VocabParallelEmbed(nn.Module):
    vocab_size: int
    hidden: int

    dtype: Optional[Any] = None
    param_dtype: Any = jnp.float32

    def setup(self):
        self.embed = nn.Embed(self.vocab_size, self.hidden, param_dtype=self.param_dtype)

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

    def fused_softmax_ce_loss(self, inputs, targets, label_smoothing):
        logits = inputs  # (B, S, V/tp)

        # Do operations in FP32 to avoid underflow in softmax
        logits = promote_dtype(logits, dtype=jnp.float32)

        high = 1.0 - label_smoothing
        low = label_smoothing / (self.vocab_size - 1)
        norm = -(
            high * jnp.log(high) + (self.vocab_size - 1) * low * jnp.log(low + 1e-20)
        )

        def ce_loss(x, y):
            smooth_label = jnp.where(
                jax.nn.one_hot(y, self.vocab_size, dtype=jnp.float32) == 0,
                label_smoothing / (self.vocab_size - 1),
                1 - label_smoothing,
            )

            loss = -jnp.sum(smooth_label * jax.nn.log_softmax(x))
            return loss

        partial_losses = jax.vmap(jax.vmap(ce_loss))(*logits, targets)

        return partial_losses - norm


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

        x, embeds = promote_dtype(x, embeds, dtype=self.dtype)

        return x + embeds


class Layernorm(nn.Module):
    dtype: Optional[Any] = None
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, inputs, *args, **kwargs):
        x = inputs

        x = promote_dtype(x, dtype=self.dtype)

        out = nn.LayerNorm(param_dtype=self.param_dtype)(*x)

        return out
