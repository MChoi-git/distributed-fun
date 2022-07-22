import argparse
import os
import logging
from pathlib import Path

import jax
from jax import numpy as jnp, random
from jax.experimental import maps, PartitionSpec
import numpy as np

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
    TransformerInit,
    forward,
    softmax_cross_entropy_loss
)
from test_utils import verify_module_metadata


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
    pos_embed_metadata = ModuleMetadata(
        rng=sk2,
        name="pos_embed",
        in_init_pspec=None,
        out_init_pspec=PartitionSpec(None, "tp"),
        in_train_pspec=[PartitionSpec("tp", None), None, None],
        out_train_pspec=None,
        layer=PositionEmbed,
        data_shape=(args.batch_size, args.seq_len, args.hidden),
        dtype=jnp.float32,
        module_init_args=(args.seq_len, args.hidden),
        init_args=None,
        init_kwargs=None,
        train_args=None,
        train_kwargs=None,
    )
    layernorm_msa_metadata = ModuleMetadata(
        rng=sk3,
        name="layernorm_msa",
        in_init_pspec=None,
        out_init_pspec=None,
        in_train_pspec=[None, None, None],
        out_train_pspec=None,
        layer=Layernorm,
        data_shape=(args.batch_size, args.seq_len, args.hidden),
        dtype=jnp.float32,
        module_init_args=None,
        init_args=None,
        init_kwargs=None,
        train_args=None,
        train_kwargs=None,
    )
    msa_attn_metadata = ModuleMetadata(
        rng=sk4,
        name="msa_attn",
        in_init_pspec=None,
        out_init_pspec=PartitionSpec(None, "tp"),
        in_train_pspec=[PartitionSpec(None, "tp"), None, None],
        out_train_pspec=PartitionSpec(None, None, "tp"),
        layer=ModelParallelMaskedMSA,
        data_shape=(args.batch_size, args.seq_len, args.hidden),
        dtype=jnp.float32,
        module_init_args=(args.hidden, args.num_heads, 0.1),
        init_args=None,
        init_kwargs={"train": False},
        train_args=None,
        train_kwargs={"train": True},
    )
    msa_mlp_metadata = ModuleMetadata(
        rng=sk5,
        name="msa_mlp",
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
        init_args=None,
        init_kwargs={"train": False},
        train_args=None,
        train_kwargs={"train": True},
    )
    layernorm_mlp_metadata = ModuleMetadata(
        rng=sk6,
        name="layernorm_mlp",
        in_init_pspec=None,
        out_init_pspec=None,
        in_train_pspec=[None, None, None],
        out_train_pspec=None,
        layer=Layernorm,
        data_shape=(args.batch_size, args.seq_len, args.hidden),
        dtype=jnp.float32,
        module_init_args=None,
        init_args=None,
        init_kwargs=None,
        train_args=None,
        train_kwargs=None,
    )
    mlp_col_metadata = ModuleMetadata(
        rng=sk7,
        name="mlp_col",
        in_init_pspec=None,
        out_init_pspec=PartitionSpec(None, "tp"),
        in_train_pspec=[PartitionSpec(None, "tp"), None, None],
        out_train_pspec=PartitionSpec(None, None, "tp"),
        layer=ColumnParallelLinear,
        data_shape=(args.batch_size, args.seq_len, args.hidden),
        dtype=jnp.float32,
        module_init_args=(args.hidden, 0),
        init_args=None,
        init_kwargs={"train": False},
        train_args=None,
        train_kwargs={"train": True},
    )
    mlp_row_metadata = ModuleMetadata(
        rng=sk8,
        name="mlp_row",
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
        init_args=None,
        init_kwargs={"train": False},
        train_args=None,
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

    # Check for correctness with non-distributed modules
    single_outs = {}
    dist_outs = {}
    results = {}
    for module_metadata in module_metadata_list:
        main_key, sk = random.split(main_key)

        module_metadata.train_kwargs["train"] = False

        single_out, dist_out, result = verify_module_metadata(
            sk,
            mesh,
            module_metadata,
            atol=1e-6
        )

        module_metadata.train_kwargs["train"] = True

        single_outs[module_metadata.name] = single_out
        dist_outs[module_metadata.name] = dist_out
        results[module_metadata.name] = result

    overall = (sum(results.values()) == len(results))

    if overall.item() is False:
        raise Exception("Some layer(s) do not have correct outputs")

    breakpoint()

    transformer = TransformerInit(mesh, args.num_layers, module_metadata_list)

    # Constuct pjit functions for initializing params and layer forward passes
    params = transformer.init_from_pjit_metadata(const_layer_end_idx=2)
    transformer.forward_from_pjit_metadata(const_layer_end_idx=2)

    def encode(batch, tokenizer):
        return tokenizer(
            batch["text"], padding="max_length", truncation=True
        )

    main_key, sk = random.split(main_key)
    def get_batch_indices(key, num_batches, train_dset, batch_size):
        dset_indices = jnp.arange(len(train_dset))
        batch_indices_dropped = dset_indices[:num_batches * batch_size]    # drop last
        return batch_indices_dropped

    num_batches = len(train_dset) // args.batch_size
    
    for i in range(args.num_epochs):
        train_dset = train_dset.shuffle(seed=i)
        val_dset = val_dset.shuffle(seed=i)
        test_dset = test_dset.shuffle(seed=i)

        batch_indices = get_batch_indices(sk, num_batches, train_dset, args.batch_size)

        for j in range(num_batches):
            main_key, subkey = random.split(main_key)

            # Get batch
            low = j
            high = j + args.batch_size
            batch_slice = batch_indices[low: high]
            batch = encode(train_dset[batch_slice], tokenizer)
            batch = jnp.array(batch["input_ids"])

            # Get loss
            batch_loss = softmax_cross_entropy_loss(
                params,
                transformer.module_metadata_list,
                batch,
                batch,
                mesh,
                args.num_layers,
                args.max_vocab_size,
                subkey,
                args.label_smoothing,
            )
            print(f"Loss batch {j} epoch {i}: {batch_loss}")

    breakpoint()


if __name__ == "__main__":
    args = parse_args()
    main(args)
