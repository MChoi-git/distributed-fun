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
)
from model_parallel import (
    ModelParallelMaskedMSA,
    RowParallelLinear,
    ColumnParallelLinear,
    VocabParallelEmbed,
    PositionEmbed,
    Transformer
)


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
    parser.add_argument(
        "--exp_id",
        type=str,
        default="DEFAULT_MP"
    )
    parser.add_argument(
        "--wikitext_script_path",
        type=str,
        default="/h/mchoi/distributed/jax_distributed/wikitext.py"
    )
    parser.add_argument(
        "--wikitext_name",
        type=str,
        default="wikitext-103-v1"
    )
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
    parser.add_argument(
        "--mesh_x",
        type=str,
        default=2
    )
    parser.add_argument(
        "--mesh_y",
        type=int,
        default=1
    )
    parser.add_argument("--inference", type=int, default=0)
    args = parser.parse_args()
    return args


def main(args):
    # Configure logger
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("transformer_logger")
    logger.setLevel(logging.INFO)

    # Configure paths for datasets/tokenizer
    working_dir = Path(os.getcwd())
    json_save_dir = working_dir / args.json_save_dir
    tokenizer_save_dir = working_dir / args.tokenizer_save_dir

    dset_tuple = make_wikitext_dataset(
        args.wikitext_script_path,
        args.wikitext_name
    )
    tokenizer = make_wikitext_tokenizer(
        dset_tuple,
        args.max_vocab_size,
        args.seq_len,
        json_save_dir,
        tokenizer_save_dir,
    )
    test_dset, train_dset, val_dset = dset_tuple

    # Configure devices and mesh
    mesh_shape = (args.mesh_x, args.mesh_y)     # x: DP, y: MP
    assert len(jax.devices()) == args.mesh_x * args.mesh_y

    available_devices = jax.devices()
    devices = np.asarray(available_devices).reshape(*mesh_shape)
    mesh = maps.Mesh(devices, ("tp", "pp"))

    # Make initial RNG
    main_key = random.PRNGKey(args.seed)

    """
    ## TESTING ##
    x1 = train_dset[23589]["text"]
    x2 = train_dset[123]["text"]
    tokens1 = tokenizer(x1, padding="max_length")
    tokens2 = tokenizer(x2, padding="max_length")
    data1 = jnp.expand_dims(jnp.array(tokens1["input_ids"]), axis=(0,))
    data2 = jnp.expand_dims(jnp.array(tokens2["input_ids"]), axis=(0,))
    data = jnp.concatenate((data1, data2), axis=0)

    embed_apply_fn = jax.experimental.pjit.pjit(
        lambda params, x: embed.apply(params, x),
        in_axis_resources=[PartitionSpec("tp", None), None],
        out_axis_resources=PartitionSpec(None, None, "tp"),
    )
    msa_apply_fn = jax.experimental.pjit.pjit(
        lambda params, x: msa_attn.apply(params, x, train=False),
        in_axis_resources=[PartitionSpec(None, "tp"), None],
        out_axis_resources=PartitionSpec(None, None, "tp"),
    )
    msa_mlp_apply_fn = jax.experimental.pjit.pjit(
        lambda params, x: msa_mlp.apply(params, x, train=False),
        in_axis_resources=[PartitionSpec("tp", None), PartitionSpec(None, None, "tp")],
        out_axis_resources=PartitionSpec(None, None, "tp"),
    )

    with maps.Mesh(mesh.devices, mesh.axis_names):
        embed_out = embed_apply_fn(embed_params, data)
        msa_out = msa_apply_fn(msa_attn_params, embed_out)
        out = msa_mlp_apply_fn(msa_mlp_params, msa_out)
    """

    transformer = Transformer(
        hidden=args.hidden,
        heads=args.num_heads,
        seq_len=args.seq_len,
        qkv_dropout=0.1,
        msa_dropout=0.1,
        mlp_dropout=0.1,
        vocab_size=args.max_vocab_size,
        mesh=mesh,
    )

    # TODO: Can probably get rid of args.batch_size in data_shape, since
    #       modules are agnostic of batch size

    main_key, sk1, sk2, sk3, sk4, sk5 = random.split(main_key, num=6)
    transformer.prepare_layer_pjit_metadata(
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
        train_args=None,
    )
    transformer.prepare_layer_pjit_metadata(
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
        train_args=None,
    )
    transformer.prepare_layer_pjit_metadata(
        rng=sk2,
        name="msa_attn",
        in_init_pspec=None,
        out_init_pspec=PartitionSpec(None, "tp"),
        in_train_pspec=[PartitionSpec(None, "tp"), None, None],
        out_train_pspec=PartitionSpec(None, None, "tp"),
        layer=ModelParallelMaskedMSA,
        data_shape=(args.batch_size, args.seq_len, args.hidden),
        dtype=jnp.float32,
        module_init_args=(args.hidden, args.num_heads, 0.1),
        init_args=(False,),
        train_args=(True,),
    )
    transformer.prepare_layer_pjit_metadata(
        rng=sk3,
        name="msa_mlp",
        in_init_pspec=PartitionSpec(None, None, "tp"),
        out_init_pspec=PartitionSpec("tp", None),
        in_train_pspec=[PartitionSpec("tp", None), PartitionSpec(None, None, "tp"), None],
        out_train_pspec=PartitionSpec(None, None, "tp"),
        layer=RowParallelLinear,
        data_shape=(args.batch_size, args.seq_len, args.hidden),
        dtype=jnp.float32,
        module_init_args=(args.hidden, 0.1),
        init_args=(False,),
        train_args=(True,),
    )
    transformer.prepare_layer_pjit_metadata(
        rng=sk4,
        name="mlp_col",
        in_init_pspec=None,
        out_init_pspec=PartitionSpec(None, "tp"),
        in_train_pspec=[PartitionSpec(None, "tp"), None, None],
        out_train_pspec=PartitionSpec(None, None, "tp"),
        layer=ColumnParallelLinear,
        data_shape=(args.batch_size, args.seq_len, args.hidden),
        dtype=jnp.float32,
        module_init_args=(args.hidden, 0),
        init_args=(False,),
        train_args=(True,),
    )
    transformer.prepare_layer_pjit_metadata(
        rng=sk5,
        name="mlp_row",
        in_init_pspec=PartitionSpec(None, None, "tp"),
        out_init_pspec=PartitionSpec("tp", None),
        in_train_pspec=[PartitionSpec("tp", None), PartitionSpec(None, None, "tp"), None],
        out_train_pspec=PartitionSpec(None, None, "tp"),
        layer=RowParallelLinear,
        data_shape=(args.batch_size, args.seq_len, args.hidden),
        dtype=jnp.float32,
        module_init_args=(args.hidden, 0.1),
        init_args=(False,),
        train_args=(True,),
    )

    transformer.init_from_pjit_metadata(args.num_layers, const_layer_end_idx=2)
    param_tree = transformer.inspect_module_metadata_list()
    print(param_tree)

    x1 = train_dset[23589]["text"]
    x2 = train_dset[123]["text"]
    tokens1 = tokenizer(x1, padding="max_length")
    tokens2 = tokenizer(x2, padding="max_length")
    data1 = jnp.expand_dims(jnp.array(tokens1["input_ids"]), axis=(0,))
    data2 = jnp.expand_dims(jnp.array(tokens2["input_ids"]), axis=(0,))
    data = jnp.concatenate((data1, data2), axis=0)

    out = transformer.forward(data, random.PRNGKey(42069))

    breakpoint()


if __name__ == "__main__":
    args = parse_args()
    main(args)
