from dataclasses import dataclass
import logging
from functools import partial

import jax
from jax import random, numpy as jnp
from jax.experimental import maps
from datasets.arrow_dataset import Dataset
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from model_parallel import ModuleMetadataManager
from sharded_adam import apply_updates_dist, adamw_dist, dynamic_loss_unscaling, dynamic_loss_nan_check


def inspect_params_like(fn, params_like):
    """
    Short-hand helper function to inspect pytree of sharded arrays with the
    same tree structure as model parameters.
    """
    return jax.tree_util.tree_map(
        fn,
        params_like,
    )


@dataclass
class TransformerInterface:
    rng: jnp.ndarray
    module_metadata_manager: ModuleMetadataManager
    train_dset: Dataset
    val_dset: Dataset
    tokenizer: PreTrainedTokenizerFast
    logger: logging.Logger

    def encode(self, batch):
        """Encode a batch of text using the tokenizer."""
        return jnp.array(
            self.tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
            )["input_ids"]
        )

    def left_shift_batch(self, batch, pad_id=2):
        """
        Left shift the batch of examples and replace the wrapped
        column with more pad tokens. This is used to generate a batch of
        labels for each batch.
        """
        pad_col = jnp.ones((batch.shape[0], 1), dtype=jnp.int32) * pad_id
        shifted_labels = jnp.roll(batch, shift=-1)
        labels = jnp.concatenate((shifted_labels[:, :-1], pad_col), axis=-1)
        return labels

    def get_data(self, idx, batch_size, dset):
        """
        Given a batch index, return the corresponding batch from the
        dataset.
        """
        batch_slice = slice(idx * batch_size, idx * batch_size + batch_size)
        batch = self.encode(dset[batch_slice])
        labels = self.left_shift_batch(batch)
        return batch, labels

    def validate(self, params, loss_fn, batch_size, label_smoothing):
        """Validate the model with params on the entire validation split"""
        def val_step(params, x_batched, y_batched, vocab_size, label_smoothing, rng):
            loss_value = loss_fn(
                params,
                self.module_metadata_manager,
                x_batched,
                y_batched,
                vocab_size,
                label_smoothing,
                rng,
                train=False,
            )
            return loss_value

        num_batches = len(self.val_dset) // batch_size

        val_loss = []
        for b in range(num_batches):
            batch, labels = self.get_data(b, batch_size, self.val_dset)

            loss = val_step(
                params,
                batch,
                labels,
                self.tokenizer.vocab_size,
                label_smoothing,
                random.PRNGKey(0),
            )

            val_loss.append(loss)

        val_loss = sum(val_loss) // num_batches

        self.logger.info(f"Val loss: {val_loss}")

        return val_loss

    def train(
        self,
        params,
        loss_fn,
        num_epochs,
        batch_size,
        lr,
        wd,
        clipping,
        label_smoothing,
        dynamic_loss_scaling,
        dynamic_loss_decrease_scale,
        dynamic_loss_increase_scale,
        dynamic_loss_increase_window,
        optim_eps,
    ):
        """
        Training function for the model, using an externally defined forward
        function contained within the loss function.
        """
        optim = adamw_dist(
            module_metadata_manager=self.module_metadata_manager,
            learning_rate=lr,
            weight_decay=wd,
            clipping=clipping,
            eps=optim_eps,
            mu_dtype=jnp.float32,
            nu_dtype=jnp.float32,
            params_copy_dtype=jnp.float32,
        )

        def dynamic_loss_train_step(
            params,
            opt_state,
            meta,
            x_batched,
            y_batched,
            vocab_size,
            label_smoothing,
            dropout_rng,
            loss_scaling,
        ):
            nans = 1
            while nans > 0:
                loss_value, scaled_grads = jax.value_and_grad(loss_fn)(
                    params,
                    meta,
                    x_batched,
                    y_batched,
                    vocab_size,
                    label_smoothing,
                    dropout_rng,
                    loss_scaling=loss_scaling,
                    train=True,
                )

                nans = dynamic_loss_nan_check(
                    scaled_grads,
                    self.module_metadata_manager,
                )

                if nans > 0:
                    loss_scaling /= dynamic_loss_decrease_scale
                    self.logger.info(f"Detected {nans} NaNs or infs. Decreasing dynamic loss scale to {loss_scaling}")

            self.logger.info(f"Successful dynamic loss scaling at: {loss_scaling}")

            unscaled_grads = dynamic_loss_unscaling(
                scaled_grads,
                loss_scaling,
                self.module_metadata_manager,
            )

            def update_opt(grads, opt_state, params):
                updates, new_opt_state = optim.update(grads, opt_state, params)
                params = apply_updates_dist(
                    params, new_opt_state, updates, self.module_metadata_manager
                )
                return params, new_opt_state

            params, opt_state = update_opt(unscaled_grads, opt_state, params)

            return params, opt_state, loss_scaling, loss_value / loss_scaling

        opt_state = optim.init(params)

        num_batches = len(self.train_dset) // batch_size

        dynamic_loss_iters = 0
        for e in range(num_epochs):
            train_dset = self.train_dset.shuffle(seed=e)

            for b in range(num_batches):
                self.rng, dropout_key = random.split(self.rng)

                batch, labels = self.get_data(b, batch_size, train_dset)

                params, opt_state, dynamic_loss_scaling, loss_value  = dynamic_loss_train_step(
                    params,
                    opt_state,
                    self.module_metadata_manager,
                    batch,
                    labels,
                    self.tokenizer.vocab_size,
                    label_smoothing,
                    dropout_key,
                    dynamic_loss_scaling,
                )

                dynamic_loss_iters += 1

                self.logger.info(
                    f"Loss epoch {e} batch {b}: {loss_value}"
                )

                if dynamic_loss_iters == dynamic_loss_increase_window:
                    dynamic_loss_scaling *= dynamic_loss_increase_scale
                    self.logger.info(f"Increasing dynamic loss scale to {dynamic_loss_scaling}")

            self.validate(params, loss_fn, batch_size, label_smoothing)

        return params


def forward(
    all_params,
    module_metadata_manager,
    batch,
    labels,
    dropout_rng_key,
    label_smoothing,
    train=True,
):
    """
    Forward pass for transformer. Uses binded params and pjit functions from
    module_metadata_list container.
    """
    # Quick alias
    meta_list = module_metadata_manager.module_metadata_list

    pjit_fn_type = "pjit_forward" if train is True else "pjit_inference"

    def pjit_fn(m):
        return getattr(m, pjit_fn_type)

    with maps.Mesh(
        module_metadata_manager.mesh.devices, module_metadata_manager.mesh.axis_names
    ):
        x = batch

        embeds = pjit_fn(meta_list[0])(all_params["embed_0"], x, None)

        core_input = pjit_fn(meta_list[1])(all_params["pos_embed_0"], embeds, None)

        def forward_core_transformer(key, inputs):
            for i in range(module_metadata_manager.num_layers):
                dropout_rng_key, qkv_dropout, msa_dropout, mlp_dropout = random.split(
                    key, num=4
                )

                ln_msa = pjit_fn(meta_list[2])(
                    all_params[f"layernorm_msa_{i}"], inputs, None
                )
                self_attn = pjit_fn(meta_list[3])(
                    all_params[f"msa_attn_{i}"], ln_msa, {"dropout": qkv_dropout}
                )
                msa_out = pjit_fn(meta_list[4])(
                    all_params[f"msa_mlp_{i}"], self_attn, {"dropout": msa_dropout}
                )
                msa_res_out = msa_out + core_input

                ln_mlp = pjit_fn(meta_list[5])(
                    all_params[f"layernorm_mlp_{i}"], msa_res_out, None
                )
                mlp_col_out = pjit_fn(meta_list[6])(
                    all_params[f"mlp_col_{i}"], ln_mlp, None
                )
                mlp_row_out = pjit_fn(meta_list[7])(
                    all_params[f"mlp_row_{i}"],
                    mlp_col_out,
                    {"dropout": mlp_dropout},
                )
                inputs = mlp_row_out + msa_res_out
            return inputs

        core_output = forward_core_transformer(dropout_rng_key, core_input)

        logits = meta_list[0].pjit_attend(
            all_params["embed_0"],
            core_output,
        )
        out = meta_list[0].pjit_fused_softmax_ce_loss(
            logits,
            labels,
            label_smoothing,
        )

    return out


def softmax_cross_entropy_loss(
    all_params,
    module_metadata_manager,
    x_batched,
    labels,
    vocab_size,
    label_smoothing,
    dropout_rng_key,
    loss_scaling,
    train=True,
):
    """
    Softmax cross entropy loss is defined partially in models last layer,
    but the final reduction is done here.
    """
    preds_batched = forward(
        all_params,
        module_metadata_manager,
        x_batched,
        labels,
        dropout_rng_key,
        label_smoothing,
        train,
    )
    return preds_batched.sum() / labels.size * loss_scaling
