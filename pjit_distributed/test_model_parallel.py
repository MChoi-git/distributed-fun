import argparse
from pathlib import Path
import logging

import jax
from jax import numpy as jnp, random
from jax.experimental import maps, PartitionSpec
import flax.linen as nn
import numpy as np
import optax

from wikitext_dataset import (
    make_wikitext_dataset,
    make_wikitext_tokenizer,
    setup_wikitext_dataset_and_tokenizer,
)
from model_parallel import (
    ModelParallelMaskedMSA,
    RowParallelLinear,
    ColumnParallelLinear,
    VocabParallelEmbed,
    PositionEmbed,
    Layernorm,
    ModuleMetadata,
    ModuleMetadataManager,
    forward,
    softmax_cross_entropy_loss,
    generic_pjit_init_fn,
    generic_pjit_forward_fn,
)
from sharded_adam import adamw_dist, apply_updates_dist
from test_utils import verify_dist_model


def parse_args():
    parser = argparse.ArgumentParser(description="Transformer LM args")
    parser.add_argument("--seed", type=int, default=42069)
    parser.add_argument("--max_vocab_size", type=int, default=30000)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/h/mchoi/distributed/jax_distributed/checkpoints",
    )
    parser.add_argument("--exp_id", type=str, default="DEFAULT_MP")
    parser.add_argument(
        "--wikitext_script_path",
        type=str,
        default="/h/mchoi/distributed/jax_distributed/wikitext.py",
    )
    parser.add_argument("--wikitext_name", type=str, default="wikitext-103-v1")
    parser.add_argument(
        "--tokenizer_save_dir",
        type=str,
        default="tokenizer_wikitext-103",
    )
    parser.add_argument(
        "--json_save_dir",
        type=str,
        default="/dset_json_temp",
    )
    parser.add_argument("--mesh_x", type=int, default=2)
    parser.add_argument("--mesh_y", type=int, default=1)
    parser.add_argument("--inference", type=int, default=0)
    args = parser.parse_args()
    return args


def get_mesh(x, y):
    mesh_shape = (x, y)  # x: DP, y: MP
    assert len(jax.devices()) == x * y

    available_devices = jax.devices()
    devices = np.asarray(available_devices).reshape(*mesh_shape)
    mesh = maps.Mesh(devices, ("tp", "pp"))

    return mesh


def main(args):
    # Configure logger
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("transformer_logger")
    logger.setLevel(logging.INFO)

    """
    # Create dataset and tokenizer
    tokenizer, test_dset, train_dset, val_dset = setup_wikitext_dataset_and_tokenizer(
        args.json_save_dir,
        args.tokenizer_save_dir,
        args.wikitext_script_path,
        args.wikitext_name,
        args.max_vocab_size,
        args.seq_len,
    )
    """

    # Configure devices and mesh
    mesh = get_mesh(args.mesh_x, args.mesh_y)

    # Make initial RNG
    main_key = random.PRNGKey(args.seed)

    # Construct module to test
    main_key, sk1, sk2 = random.split(main_key, num=3)

    dist_col = ModuleMetadata(
        rng=sk1,
        name="mlp_col",
        num_layers=args.num_layers,
        in_init_pspec=None,
        out_init_pspec=PartitionSpec(None, "tp"),
        in_train_pspec=[PartitionSpec(None, "tp"), None, None],
        out_train_pspec=PartitionSpec(None, "tp"),
        layer=ColumnParallelLinear,
        data_shape=(args.batch_size, args.seq_len, args.hidden),
        dtype=jnp.float32,
        module_init_args=(args.hidden * 4, 0),
        init_args=(),
        init_kwargs={"train": False},
        train_args=(),
        train_kwargs={"train": True},
    )
    dist_row = ModuleMetadata(
        rng=sk2,
        name="mlp_row",
        num_layers=args.num_layers,
        in_init_pspec=PartitionSpec(None, None, "tp"),
        out_init_pspec=PartitionSpec("tp", None),
        in_train_pspec=[PartitionSpec("tp", None), PartitionSpec(None, "tp"), None],
        out_train_pspec=PartitionSpec(None, "tp"),
        layer=RowParallelLinear,
        data_shape=(args.batch_size, args.seq_len, args.hidden * 4),
        dtype=jnp.float32,
        module_init_args=(args.hidden, 0),
        init_args=(),
        init_kwargs={"train": False},
        train_args=(),
        train_kwargs={"train": True},
    )

    meta = ModuleMetadataManager(mesh, args.num_layers, [dist_col, dist_row])

    params = meta.init_from_pjit_metadata()
    meta.forward_from_pjit_metadata()

    main_key, verify_key = random.split(main_key)

    results, overall = verify_dist_model(
        verify_key,
        mesh,
        meta,
    )

    assert overall.item() is True

    def mse_batched(x_batched, y_batched):
        def mse(x, y):
            return 0.5 * (x - y) ** 2

        return jnp.mean(jax.vmap(mse)(x_batched, y_batched))

    def forward_mlp(
        params, meta, inputs, targets, mesh, dropout_rng_key, label_smoothing
    ):
        meta_list = meta.module_metadata_list

        with maps.Mesh(mesh.devices, mesh.axis_names):
            x = inputs

            mlp_col = meta_list[0].pjit_forward(params["mlp_col_0"], x, None)
            mlp_row = meta_list[1].pjit_forward(
                params["mlp_row_0"], mlp_col, {"dropout": dropout_rng_key}
            )

        return mlp_row

    main_key, sk1, sk2, sk3 = random.split(main_key, num=4)
    true_params = random.normal(sk1, shape=(args.hidden, args.hidden))
    true_bias = random.uniform(sk2, shape=(args.hidden,))

    true_fn = lambda x: jnp.einsum("bh,hi->bi", x, true_params) + true_bias

    dset = random.uniform(sk3, shape=(10 * args.batch_size, args.hidden))
    labels = true_fn(dset)

    optim = adamw_dist(
        module_metadata_manager=meta,
        learning_rate=args.lr,
        weight_decay=None,
        clipping=None,
    )
    opt_state = optim.init(params)

    def train_step(key, opt_state, optim, params, meta, batch, labels, label_smoothing):
        key, dropout_rng_key = random.split(key)

        def step(p, x, y):
            preds = forward_mlp(p, meta, x, y, mesh, dropout_rng_key, label_smoothing)
            loss = mse_batched(preds, y)
            return loss

        loss_value, grads = jax.value_and_grad(step)(params, batch, labels)
        # loss_value, grads = step(params, batch, labels)
        updates, opt_state = optim.update(grads, opt_state, params)
        params = apply_updates_dist(params, updates, meta)
        return params, opt_state, loss_value

    class DenseModel(nn.Module):
        hidden: int

        @nn.compact
        def __call__(self, inputs):
            x = inputs

            mlp_col = ColumnParallelLinear(hidden=args.hidden, dropout=0.0)(
                x, train=False
            )
            out = RowParallelLinear(hidden=args.hidden, dropout=0.0)(
                mlp_col, train=False
            )
            return out

    main_key, k1, k2 = random.split(main_key, num=3)
    dense_model = DenseModel(hidden=args.hidden)
    dense_params = dense_model.init(
        k1, random.normal(k2, shape=(args.batch_size, args.hidden))
    )

    dense_optim = optax.adam(learning_rate=args.lr)
    dense_opt_state = dense_optim.init(dense_params)

    def dense_train_step(opt_state, optim, params, batch, labels):
        def step(p, x, y):
            preds = dense_model.apply(p, batch)
            return mse_batched(preds, labels)

        loss_value, grads = jax.value_and_grad(step)(params, batch, labels)
        updates, opt_State = optim.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    for i in range(1000):
        for j in range(10):
            main_key, k = random.split(main_key)

            window = slice(j * args.batch_size, j * args.batch_size + args.batch_size)
            xs = dset[window]
            ys = labels[window]

            params, opt_state, loss_value = train_step(
                k, opt_state, optim, params, meta, xs, ys, args.label_smoothing
            )
            dense_params, dense_opt_state, dense_loss_value = dense_train_step(
                dense_opt_state, dense_optim, dense_params, xs, ys
            )
            print(f"loss at {i * j}:        {loss_value}")
            print(f"dense loss at {i * j}:  {dense_loss_value}")

    """
    dist_module = ModuleMetadata(
        rng=sk1,
        name="embed",
        in_init_pspec=None,
        out_init_pspec=PartitionSpec("tp", None),
        in_train_pspec=[PartitionSpec("tp", None), None, None],
        out_train_pspec=None,
        layer=VocabParallelEmbed,
        data_shape=(args.batch_size, args.seq_len),
        dtype=jnp.int32,
        module_init_args=(args.max_vocab_size, args.hidden),
        init_args=None,
        init_kwargs=None,
        train_args=None,
        train_kwargs=None,
    )
    """
    """
    # Make the dummy init/forward data
    main_key, dummy_key, verify_key = random.split(main_key, num=3)

    single_out, dist_out, result = verify_module_metadata(
        verify_key, mesh, dist_module, "core_params", "embed"
    )

    breakpoint()
    """


if __name__ == "__main__":
    args = parse_args()
    main(args)
