import time

import chex
import jax
from jax import numpy as jnp
import numpy as np


def get_number_of_params(params):
    values, _ = jax.tree_util.tree_flatten(params)
    total_params = sum([p.size for p in values])
    return total_params


def masked_softmax(x, mask):
    x_shape = x.shape
    x = x.reshape(-1, x_shape[-1], x_shape[-1])

    out = jax.nn.softmax(x + mask, axis=-1)

    return out.reshape(x_shape)


def softmax_cross_entropy_loss(params, model, x_batched, labels, dropout_rng, vocab_size, train, label_smoothing):
    def cross_entropy(x, y):
        # Only works for one-hot idxs y
        smooth_label = jnp.where(
            jax.nn.one_hot(y, vocab_size) == 0,
            label_smoothing / (vocab_size - 1),
            1 - label_smoothing,
        )
        return -jnp.sum(smooth_label * jnp.clip(jnp.log(x), a_min=-100))

    preds_batched = model.apply(params, inputs=x_batched, train=train, rngs={'dropout': dropout_rng})

    softmax_preds_batched = jax.nn.softmax(preds_batched, axis=-1)

    return jnp.mean(jax.vmap(cross_entropy)(softmax_preds_batched, labels))


def model_inference(tokenizer, prompts, model, model_params, seq_len):
    tokenizer.padding_side = "left"

    tokenized_prompts = tokenizer(
        prompts,
        return_tensors="np",
        padding="max_length",
        return_token_type_ids=False,
        return_attention_mask=False,
    )["input_ids"]
    tokenized_prompts = jnp.array(tokenized_prompts)

    apply_fn = jax.jit(lambda p: model.apply(model_params, p, train=False))

    def predict_next_token(batch):
        results = apply_fn(batch)
        argmax = jnp.argmax(jax.nn.softmax(results, axis=-1), axis=-1)
        new_tokens = argmax[:, -1]
        return new_tokens

    def prune_pads(prompts):
        out = []
        for p in prompts:
            out.append(jnp.delete(p, (p == 2)))
        return out

    start = time.time()
    for i in range(seq_len):
        # Slide window
        right_idx = i
        left_idx = seq_len + i
        prompt_window = tokenized_prompts[:, right_idx: left_idx]

        # Get next row of predicted tokens
        new_tokens = predict_next_token(jnp.array(prompt_window))

        # Concatenate them with prompts buffer
        tokenized_prompts = jnp.concatenate((tokenized_prompts, jnp.expand_dims(new_tokens, axis=-1)), axis=-1)
    end = time.time()

    results = prune_pads(tokenized_prompts)
    final = tokenizer.batch_decode(results)
    for p in final:
        print("=" * 50)
        print(p)
    print(f"Took {end - start}s to generate")
