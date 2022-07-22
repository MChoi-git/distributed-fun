import argparse
import os
import logging

import chex
import jax
from jax import numpy as jnp, random
import optax
from flax.training import checkpoints

from layers import TransformerLM
from dataset import (
    tokenize_shakespeare,
    make_shakespeare_dataset,
    save_tokenizer,
    get_tokenizer,
)
from utils import (
    softmax_cross_entropy_loss,
    get_number_of_params,
    model_inference,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Transformer LM args")
    parser.add_argument("--seed", type=int, default=42069)
    parser.add_argument("--max_vocab_size", type=int, default=1000)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2.4e-4)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/h/mchoi/distributed/jax_transformer/checkpoints",
    )
    parser.add_argument(
        "--exp_id",
        type=str,
        default="transformerlm_shakespeare"
    )
    parser.add_argument(
        "--dset_path",
        type=str,
        default="/h/mchoi/distributed/jax_transformer"
    )
    parser.add_argument(
        "--corpus_save_path",
        type=str,
        default="/h/mchoi/distributed/jax_transformer"
    )
    parser.add_argument(
        "--tokenizer_save_path",
        type=str,
        default="/h/mchoi/distributed/jax_transformer/tokenizer_shakespeare",
    )
    parser.add_argument("--inference", type=int, default=0)
    args = parser.parse_args()
    return args


def main(args):
    # Configure logger
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("transformer_logger")
    logger.setLevel(logging.INFO)

    main_key = random.PRNGKey(args.seed)

    # Make/load tokenizer logic
    if os.path.isdir(args.tokenizer_save_path):
        tokenizer = get_tokenizer(args.tokenizer_save_path)
        logger.info(f"Loaded tokenizer from {args.tokenizer_save_path}")
    else:
        tokenizer = tokenize_shakespeare(
            args.dset_path, args.corpus_save_path, args.max_vocab_size, args.seq_len
        )
        save_tokenizer(tokenizer, args.tokenizer_save_path)
        logger.info(
            f"Created pretrained tokenizer, saved at: {args.tokenizer_save_path}"
        )

    # Create model
    model = TransformerLM(
        hidden=args.hidden,
        heads=args.num_heads,
        qkv_dropout=0.1,
        msa_dropout=0.1,
        mlp_dropout=0.1,
        num_layers=args.num_layers,
        seq_len=args.seq_len,
        vocab_size=args.max_vocab_size,
    )

    # Init model
    main_key, params_key, dropout_key = random.split(main_key, num=3)

    if args.inference == 1:
        prompts = [
            "We are accounted poor citizens",
            "Her wondrous qualities and mild behavior",
            "I know him well: you are very welcome",
            "Wherefore art thou",
            "Greetings, I am Matt",
            "Show me your toilet",
        ]

        model_params = checkpoints.restore_checkpoint(
            args.checkpoint_dir, target=model, prefix=args.exp_id
        )

        model_inference(
            tokenizer,
            prompts,
            model,
            model_params,
            args.seq_len,
        )

        # Terminate after inference
        quit()

    if checkpoints.latest_checkpoint(args.checkpoint_dir, prefix=args.exp_id) is not None:
        model_params = checkpoints.restore_checkpoint(
            args.checkpoint_dir, target=model, prefix=args.exp_id
        )
        logging.info(f"Resuming training from found checkpoint: {args.checkpoint_dir}")

    else:
        init_rngs = {
            "params": params_key,
        }
        dummy = jnp.ones((args.batch_size, args.seq_len)).astype(jnp.int32)
        model_params = model.init(init_rngs, dummy, train=False)
        logging.info(f"No checkpoint found at {args.checkpoint_dir}. Starting from random init.")

    # Get the Shakespeare dataset
    train_dset, val_dset, test_dset = make_shakespeare_dataset(
        tokenizer,
        args.dset_path,
        args.seq_len,
        args.batch_size,
    )

    logger.info(f"{get_number_of_params(model_params)} total parameters")

    # Loss function
    loss_fn = softmax_cross_entropy_loss

    # Create and init optim
    optim = optax.adamw(learning_rate=args.lr, weight_decay=args.wd)
    opt_state = optim.init(model_params)

    # Checkpoint save helper
    def checkpoint_model(checkpoint_dir, model_params, step, exp_id):
        checkpoints.save_checkpoint(
            target=model_params, ckpt_dir=checkpoint_dir, step=step, prefix=args.exp_id
        )

    # Dataset doesn't have start token, so shift and add it
    def retrofit_batch_and_labels(example):
        labels = example["input_ids"]

        zero_column = jnp.zeros((labels.shape[0], 1), dtype=jnp.int32)
        shifted_labels = jnp.roll(labels, shift=1)
        batch = jnp.concatenate((zero_column, shifted_labels[:, 1:]), axis=-1)
        return batch, labels

    @jax.jit
    @chex.assert_max_traces(n=1)
    def train_step(params, opt_state, batch, labels, dropout_rng):
        loss_value, grads = jax.value_and_grad(loss_fn)(
            params,
            model,
            batch,
            labels,
            dropout_rng,
            args.max_vocab_size,
            train=True,
            label_smoothing=args.label_smoothing,
        )
        updates, opt_state = optim.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    def train_epoch(key, train_dset, model_params, opt_state):
        loss_values = []
        for i, ex in enumerate(train_dset):
            key, dropout_rng = random.split(key)

            batch, labels = retrofit_batch_and_labels(ex)

            model_params, opt_state, loss_value = train_step(
                model_params, opt_state, batch, labels, dropout_rng
            )

            if i % 50 == 0:
                logger.info(f"Loss at iteration {i}: {loss_value}")

            loss_values.append(loss_value)
        return model_params, sum(loss_values) / len(loss_values)

    @jax.jit
    @chex.assert_max_traces(n=1)
    def val_step(params, batch, labels, dropout_rng):
        loss_value = loss_fn(
            params,
            model,
            batch,
            labels,
            dropout_rng,
            args.max_vocab_size,
            train=False,
            label_smoothing=0,
        )
        return loss_value

    def validate(key, val_dset, model_params, model, max_vocab_size):
        loss = []
        for ex in val_dset:
            key, dropout_rng = random.split(key)

            batch, labels = retrofit_batch_and_labels(ex)

            loss_value = val_step(model_params, batch, labels, dropout_rng)
            loss.append(loss_value)
        return sum(loss) / len(loss)

    # Training loop
    best_val_loss = 10000  # Just some large number
    for i in range(args.num_epochs):
        main_key, train_key, val_key = random.split(main_key, num=3)

        model_params, epoch_loss = train_epoch(
            train_key, train_dset, model_params, opt_state
        )
        logger.info(f"Average training loss for epoch {i} loss: {epoch_loss}")

        val_loss = validate(
            val_key,
            val_dset,
            model_params,
            model,
            args.max_vocab_size
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_model(
                args.checkpoint_dir, model_params, step=i, exp_id=args.exp_id
            )
            logger.info(f"Saved best model checkpoint at {args.checkpoint_dir}")

        logger.info(f"Average validation loss for epoch {i}: {val_loss}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
