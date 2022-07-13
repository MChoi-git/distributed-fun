from typing import Callable
from functools import partial

import jax
from jax import numpy as jnp
import flax.linen as nn

from utils import masked_softmax


class MaskedMSA(nn.Module):
    hidden: int
    heads: int
    qkv_dropout: float
    msa_dropout: float

    @nn.compact
    def __call__(self, inputs, train):
        B, S, E = inputs.shape

        x = nn.Dense(features=3 * self.hidden)(inputs)
        x = x.reshape(B, self.heads, S, -1)

        q, k, v = x.split(3, axis=-1)

        qk = jnp.einsum("bhse, bhte -> bhst", q, k) * (2 * self.hidden) ** -0.5

        mask = jnp.triu(jnp.zeros((S, S)) + -jnp.inf, k=1)
        attn = masked_softmax(qk, mask)

        attn = nn.Dropout(rate=self.qkv_dropout)(attn, deterministic=not train)

        self_attn = jnp.einsum("bhst, bhte -> bhse", attn, v).reshape(B, S, -1)

        out = nn.Dense(features=self.hidden)(self_attn)
        out = nn.Dropout(rate=self.msa_dropout)(out, deterministic=not train)

        return out


class MLP(nn.Module):
    hidden: int
    mlp_dropout: float
    nonlin: Callable = jax.nn.gelu

    @nn.compact
    def __call__(self, inputs):
        x = inputs

        x = nn.Dense(features=self.hidden * 4)(x)
        x = self.nonlin(x)

        out = nn.Dense(features=self.hidden)(x)
        out = nn.Dropout(self.mlp_dropout, deterministic=True)(out)

        return out


class TransformerLayer(nn.Module):
    hidden: int
    heads: int
    qkv_dropout: float
    msa_dropout: float
    mlp_dropout: float
    nonlin: Callable = jax.nn.gelu

    @nn.compact
    def __call__(self, inputs, train):
        x = inputs

        x = nn.LayerNorm()(x)
        msa = MaskedMSA(
            hidden=self.hidden,
            heads=self.heads,
            qkv_dropout=self.qkv_dropout,
            msa_dropout=self.msa_dropout,
        )(x, train=train)
        msa = msa + inputs

        mlp = nn.LayerNorm()(msa)
        mlp = MLP(
            hidden=self.hidden,
            mlp_dropout=self.mlp_dropout,
        )(mlp)
        mlp = mlp + msa

        return mlp


class TransformerLM(nn.Module):
    hidden: int
    heads: int
    qkv_dropout: float
    msa_dropout: float
    mlp_dropout: float
    num_layers: int
    seq_len: int
    vocab_size: int
    nonlin: Callable = jax.nn.gelu
    pos_embeds_init: Callable = jax.nn.initializers.normal()

    def setup(self):
        self.position_embeds = self.param(
            "pos_embeds",
            self.pos_embeds_init,
            (self.seq_len, self.hidden)
        )

        self.layers = [
            TransformerLayer(
                self.hidden,
                self.heads,
                self.qkv_dropout,
                self.msa_dropout,
                self.mlp_dropout,
            )
            for _ in range(self.num_layers)
        ]

        self.embed = nn.Embed(self.vocab_size, self.hidden)

    def __call__(self, inputs, train):
        x = inputs

        x = self.embed(x)

        x = x + self.position_embeds

        for layer in self.layers:
            x = layer(x, train=train)

        out = self.embed.apply(
            self.embed.variables,
            x,
            method=self.embed.attend
        )

        return out
