import argparse
from pathlib import Path
import logging

import jax
from jax import numpy as jnp, random
from jax.experimental import maps, PartitionSpec
import flax.linen as nn
import numpy as np

from wikitext_dataset import(
    make_wikitext_dataset,
    make_wikitext_tokenizer,
    setup_wikitext_dataset_and_tokenizer,
)
from model_parallel import(
    ModelParallelMaskedMSA,
    RowParallelLinear,
    ColumnParallelLinear,
    VocabParallelEmbed,
    PositionEmbed,
    Layernorm,
    ModuleMetadata,
    TransformerInit,
    forward,
    softmax_cross_entropy_loss,
    generic_pjit_init_fn,
    generic_pjit_forward_fn,
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

    # Construct module to test
    main_key, sk1 = random.split(main_key)
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

    # Make the dummy init/forward data
    main_key, dummy_key, verify_key = random.split(main_key, num=3)

    single_out, dist_out, result = verify_module_metadata(verify_key, mesh, dist_module, "core_params", "embed")

    breakpoint()


if __name__ == "__main__":
    args = parse_args()
    main(args)
