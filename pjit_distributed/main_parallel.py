import argparse
import logging

import jax
from jax import numpy as jnp, random
from jax.experimental import maps, PartitionSpec
import numpy as np
import optax
import chex

from wikitext_dataset import (
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
)
from sharded_adam import adamw_dist
from test_utils import verify_module_metadata, verify_dist_model


def parse_args():
    parser = argparse.ArgumentParser(description="Transformer LM args")
    parser.add_argument("--seed", type=int, default=42069)
    parser.add_argument("--max_vocab_size", type=int, default=30000)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2.4e-4)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/h/mchoi/distributed/pjit_distributed/checkpoints",
    )
    parser.add_argument("--exp_id", type=str, default="DEFAULT_MP")
    parser.add_argument(
        "--wikitext_script_path",
        type=str,
        default="/h/mchoi/distributed/pjit_distributed/wikitext.py",
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
    parser.add_argument("--mesh_x", type=str, default=2)
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

    # Create dataset and tokenizer
    tokenizer, test_dset, train_dset, val_dset = setup_wikitext_dataset_and_tokenizer(
        args.json_save_dir,
        args.tokenizer_save_dir,
        args.wikitext_script_path,
        args.wikitext_name,
        args.max_vocab_size,
        args.seq_len,
    )

    # Configure devices and mesh
    mesh = get_mesh(args.mesh_x, args.mesh_y)

    # Make initial RNG
    main_key = random.PRNGKey(args.seed)

    # TODO: Can probably get rid of args.batch_size in data_shape, since
    #       modules are agnostic of batch size
    main_key, sk1, sk2, sk3, sk4, sk5, sk6, sk7, sk8 = random.split(main_key, num=9)
    embed_metadata = ModuleMetadata(
        rng=sk1,
        name="embed",
        num_layers=1,
        in_init_pspec=None,
        out_init_pspec=PartitionSpec("tp", None),
        in_train_pspec=[PartitionSpec("tp", None), None, None],
        out_train_pspec=None,
        layer=VocabParallelEmbed,
        data_shape=(args.batch_size, args.seq_len),
        dtype=jnp.int32,
        module_init_args=(args.max_vocab_size, args.hidden),
        init_args=(),
        init_kwargs={},
        train_args=(),
        train_kwargs={},
    )
    pos_embed_metadata = ModuleMetadata(
        rng=sk2,
        name="pos_embed",
        num_layers=1,
        in_init_pspec=None,
        out_init_pspec=PartitionSpec(None, "tp"),
        in_train_pspec=[PartitionSpec("tp", None), None, None],
        out_train_pspec=None,
        layer=PositionEmbed,
        data_shape=(args.batch_size, args.seq_len, args.hidden),
        dtype=jnp.float32,
        module_init_args=(args.seq_len, args.hidden),
        init_args=(),
        init_kwargs={},
        train_args=(),
        train_kwargs={},
    )
    layernorm_msa_metadata = ModuleMetadata(
        rng=sk3,
        name="layernorm_msa",
        num_layers=args.num_layers,
        in_init_pspec=None,
        out_init_pspec=None,
        in_train_pspec=[None, None, None],
        out_train_pspec=None,
        layer=Layernorm,
        data_shape=(args.batch_size, args.seq_len, args.hidden),
        dtype=jnp.float32,
        module_init_args=(),
        init_args=(),
        init_kwargs={},
        train_args=(),
        train_kwargs={},
    )
    msa_attn_metadata = ModuleMetadata(
        rng=sk4,
        name="msa_attn",
        num_layers=args.num_layers,
        in_init_pspec=None,
        out_init_pspec=PartitionSpec(None, "tp"),
        in_train_pspec=[PartitionSpec(None, "tp"), None, None],
        out_train_pspec=PartitionSpec(None, None, "tp"),
        layer=ModelParallelMaskedMSA,
        data_shape=(args.batch_size, args.seq_len, args.hidden),
        dtype=jnp.float32,
        module_init_args=(args.hidden, args.num_heads, 0.1),
        init_args=(),
        init_kwargs={"train": False},
        train_args=(),
        train_kwargs={"train": True},
    )
    msa_mlp_metadata = ModuleMetadata(
        rng=sk5,
        name="msa_mlp",
        num_layers=args.num_layers,
        in_init_pspec=PartitionSpec(None, None, "tp"),
        out_init_pspec=PartitionSpec("tp", None),
        in_train_pspec=[
            PartitionSpec("tp", None),
            PartitionSpec(None, None, "tp"),
            None,
        ],
        out_train_pspec=PartitionSpec(None, None, "tp"),
        layer=RowParallelLinear,
        data_shape=(args.batch_size, args.seq_len, args.hidden),
        dtype=jnp.float32,
        module_init_args=(args.hidden, 0.1),
        init_args=(),
        init_kwargs={"train": False},
        train_args=(),
        train_kwargs={"train": True},
    )
    layernorm_mlp_metadata = ModuleMetadata(
        rng=sk6,
        name="layernorm_mlp",
        num_layers=args.num_layers,
        in_init_pspec=None,
        out_init_pspec=None,
        in_train_pspec=[None, None, None],
        out_train_pspec=None,
        layer=Layernorm,
        data_shape=(args.batch_size, args.seq_len, args.hidden),
        dtype=jnp.float32,
        module_init_args=(),
        init_args=(),
        init_kwargs={},
        train_args=(),
        train_kwargs={},
    )
    mlp_col_metadata = ModuleMetadata(
        rng=sk7,
        name="mlp_col",
        num_layers=args.num_layers,
        in_init_pspec=None,
        out_init_pspec=PartitionSpec(None, "tp"),
        in_train_pspec=[PartitionSpec(None, "tp"), None, None],
        out_train_pspec=PartitionSpec(None, None, "tp"),
        layer=ColumnParallelLinear,
        data_shape=(args.batch_size, args.seq_len, args.hidden),
        dtype=jnp.float32,
        module_init_args=(args.hidden, 0),
        init_args=(),
        init_kwargs={"train": False},
        train_args=(),
        train_kwargs={"train": True},
    )
    mlp_row_metadata = ModuleMetadata(
        rng=sk8,
        name="mlp_row",
        num_layers=args.num_layers,
        in_init_pspec=PartitionSpec(None, None, "tp"),
        out_init_pspec=PartitionSpec("tp", None),
        in_train_pspec=[
            PartitionSpec("tp", None),
            PartitionSpec(None, None, "tp"),
            None,
        ],
        out_train_pspec=PartitionSpec(None, None, "tp"),
        layer=RowParallelLinear,
        data_shape=(args.batch_size, args.seq_len, args.hidden),
        dtype=jnp.float32,
        module_init_args=(args.hidden, 0.1),
        init_args=(),
        init_kwargs={"train": False},
        train_args=(),
        train_kwargs={"train": True},
    )

    module_metadata_list = [
        embed_metadata,
        pos_embed_metadata,
        layernorm_msa_metadata,
        msa_attn_metadata,
        msa_mlp_metadata,
        layernorm_mlp_metadata,
        mlp_col_metadata,
        mlp_row_metadata,
    ]

    # Create transformer model metadata manager
    main_key, verification_key = random.split(main_key)
    transformer = ModuleMetadataManager(mesh, args.num_layers, module_metadata_list)
    params = transformer.init_from_pjit_metadata()
    transformer.forward_from_pjit_metadata()

    # Verify that the pjit layer function is identical to running the layer on
    # one GPU
    dist_verification = verify_dist_model(
        verification_key,
        mesh,
        transformer,
    )
    assert dist_verification.item() is True, "Dist layer(s) have different" "outputs"

    def encode(batch, tokenizer):
        """Helper function using tokenizer to encode dataset batch"""
        encoded_batch = jnp.array(
            tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
            )
        )
        return encoded_batch

    # Start training loop
    num_batches = len(train_dset) // args.batch_size

    # Make optim
    optim = adamw_dist(module_metadata_manager=transformer, learning_rate=args.lr, weight_decay=args.wd)
    opt_state = optim.init(params)

    # TODO: Finish distributed adam optimizer impl

    def train_step(
        key, opt_state, optim, params, module_metadata_manager, batch, labels, vocab_size, label_smoothing
    ):
        """Calculates loss and applies gradient for one training step"""
        loss_value, grads = jax.value_and_grad(softmax_cross_entropy_loss)(
            subkey,
            params,
            transformer,
            batch,
            batch,
            mesh,
            args.num_layers,
            args.max_vocab_size,
            args.label_smoothing,
        )
        updates, opt_state = optim.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    main_key, sk = random.split(main_key)
    for i in range(args.num_epochs):
        # Shuffle training dataset
        train_dset = train_dset.shuffle(seed=i)

        for batch_idx in range(num_batches):
            main_key, subkey = random.split(main_key)

            # Get batch
            batch_idx = 0  # TODO: Remove after testing
            batch_slice = slice(
                batch_idx * args.batch_size,
                batch_idx * args.batch_size + args.batch_size,
            )
            batch = encode(train_dset[batch_slice], tokenizer)
            batch = jnp.array(batch["input_ids"])

            # TODO: Insert loss and optim/params update logic

    breakpoint()


if __name__ == "__main__":
    args = parse_args()
    main(args)
