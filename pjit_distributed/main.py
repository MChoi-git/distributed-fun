import argparse
import logging

from jax import numpy as jnp, random
from jax.experimental import PartitionSpec
from flax.core.frozen_dict import FrozenDict

from wikitext_dataset import (
    setup_wikitext_dataset_and_tokenizer,
)
from layers import (
    ModelParallelMaskedMSA,
    RowParallelLinear,
    ColumnParallelLinear,
    VocabParallelEmbed,
    PositionEmbed,
    Layernorm,
)
from model_parallel import (
    get_mesh,
    ModuleMetadata,
    ModuleMetadataManager,
)
# from verification_utils import verify_dist_model
from transformer_utils import (
    TransformerInterface,
    softmax_cross_entropy_loss,
)


# TODO:
#   - Dynamic loss scaling (done)
#   - Validation between epochs
#   - Abstract away PartitionSpecs
#   - Inference
#   - Activation rematerialization (done)
#   - Lax scan
#   - Pipeline parallel
#   - jax.experimental.pjit.with_sharding_constraint
# Notes:
#   - Max batch size on 4 GPU is 18
#   - Max batch size on 2 GPU is 16


def parse_args():
    parser = argparse.ArgumentParser(description="Transformer LM args")
    parser.add_argument("--seed", type=int, default=42069)
    parser.add_argument("--max_vocab_size", type=int, default=30000)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--clipping", type=float, default=None)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--dynamic_loss_scaling", type=float, default=2048.0)
    parser.add_argument("--dynamic_loss_decrease_scale", type=float, default=2.0)
    parser.add_argument("--dynamic_loss_increase_scale", type=float, default=1.5)
    parser.add_argument("--dynamic_loss_increase_window", type=int, default=5)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument(
        "--optim_eps", type=float, default=1e-8
    )  # use this when training fp16
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
        default="dset_json_temp",
    )
    parser.add_argument("--mesh_x", type=int, default=2)
    parser.add_argument("--mesh_y", type=int, default=1)
    parser.add_argument("--inference", type=int, default=0)
    parser.add_argument("--debug", type=int, default=0)
    args = parser.parse_args()
    return args


def main(args):
    # Configure logger
    logging.basicConfig(format="%(asctime)s | "
                        "%(levelname)s | "
                        "%(filename)s | "
                        "%(funcName)s | "
                        "%(message)s")
    logger = logging.getLogger("transformer_logger")
    logger.setLevel(logging.INFO)

    if args.precision == 32:
        int_dtype = jnp.int32
        float_dtype = jnp.float32
    elif args.precision == 16:
        int_dtype = jnp.int16
        float_dtype = jnp.float16
    else:
        raise Exception("No valid precision specified")

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
    mesh_shape = (args.mesh_x, args.mesh_y)
    mesh_axis_names = ("tp", "pp")
    mesh = get_mesh(mesh_shape, mesh_axis_names)

    # Construct hparams and metadata for every layer type in transformer
    main_key = random.PRNGKey(args.seed)
    main_key, sk1, sk2, sk3, sk4, sk5, sk6, sk7, sk8 = random.split(main_key, num=9)
    embed_metadata = ModuleMetadata(
        rng=sk1,
        name="embed",
        num_layers=1,
        checkpoint_activations=False,
        in_init_pspec=None,
        out_init_pspec=PartitionSpec("tp", None),
        in_train_pspec=(PartitionSpec("tp", None), None, None),
        out_train_pspec=None,
        layer=VocabParallelEmbed,
        data_shape=(args.batch_size, args.seq_len),
        dtype=int_dtype,
        module_init_args=(args.max_vocab_size, args.hidden),
        module_init_kwargs=FrozenDict({"param_dtype": float_dtype}),
        init_args=(),
        init_kwargs=FrozenDict({}),
        train_args=(),
        train_kwargs=FrozenDict({}),
    )
    pos_embed_metadata = ModuleMetadata(
        rng=sk2,
        name="pos_embed",
        num_layers=1,
        checkpoint_activations=False,
        in_init_pspec=None,
        out_init_pspec=PartitionSpec(None, "tp"),
        in_train_pspec=(PartitionSpec("tp", None), None, None),
        out_train_pspec=None,
        layer=PositionEmbed,
        data_shape=(args.batch_size, args.seq_len, args.hidden),
        dtype=float_dtype,
        module_init_args=(args.seq_len, args.hidden),
        module_init_kwargs=FrozenDict({"param_dtype": float_dtype}),
        init_args=(),
        init_kwargs=FrozenDict({}),
        train_args=(),
        train_kwargs=FrozenDict({}),
    )
    layernorm_msa_metadata = ModuleMetadata(
        rng=sk3,
        name="layernorm_msa",
        num_layers=args.num_layers,
        checkpoint_activations=True,
        in_init_pspec=None,
        out_init_pspec=None,
        in_train_pspec=(None, None, None),
        out_train_pspec=None,
        layer=Layernorm,
        data_shape=(args.batch_size, args.seq_len, args.hidden),
        dtype=float_dtype,  # Input will be fp16, but will be promoted to fp32
        module_init_args=(),
        module_init_kwargs=FrozenDict({"param_dtype": jnp.float32}),
        init_args=(),
        init_kwargs=FrozenDict({}),
        train_args=(),
        train_kwargs=FrozenDict({}),
    )
    msa_attn_metadata = ModuleMetadata(
        rng=sk4,
        name="msa_attn",
        num_layers=args.num_layers,
        checkpoint_activations=True,
        in_init_pspec=None,
        out_init_pspec=PartitionSpec(None, "tp"),
        in_train_pspec=(PartitionSpec(None, "tp"), None, None),
        out_train_pspec=PartitionSpec(None, None, "tp"),
        layer=ModelParallelMaskedMSA,
        data_shape=(args.batch_size, args.seq_len, args.hidden),
        dtype=float_dtype,
        module_init_args=(args.hidden, args.num_heads, 0.1),
        module_init_kwargs=FrozenDict({"param_dtype": float_dtype}),
        init_args=(),
        init_kwargs=FrozenDict({"train": False}),
        train_args=(),
        train_kwargs=FrozenDict({"train": True}),
    )
    msa_mlp_metadata = ModuleMetadata(
        rng=sk5,
        name="msa_mlp",
        num_layers=args.num_layers,
        checkpoint_activations=True,
        in_init_pspec=PartitionSpec(None, None, "tp"),
        out_init_pspec=PartitionSpec("tp", None),
        in_train_pspec=(
            PartitionSpec("tp", None),
            PartitionSpec(None, None, "tp"),
            None,
        ),
        out_train_pspec=PartitionSpec(None, None, "tp"),
        layer=RowParallelLinear,
        data_shape=(args.batch_size, args.seq_len, args.hidden),
        dtype=float_dtype,
        module_init_args=(args.hidden, 0.1),
        module_init_kwargs=FrozenDict({"param_dtype": float_dtype}),
        init_args=(),
        init_kwargs=FrozenDict({"train": False}),
        train_args=(),
        train_kwargs=FrozenDict({"train": True}),
    )
    layernorm_mlp_metadata = ModuleMetadata(
        rng=sk6,
        name="layernorm_mlp",
        num_layers=args.num_layers,
        checkpoint_activations=True,
        in_init_pspec=None,
        out_init_pspec=None,
        in_train_pspec=(None, None, None),
        out_train_pspec=None,
        layer=Layernorm,
        data_shape=(args.batch_size, args.seq_len, args.hidden),
        dtype=float_dtype,
        module_init_args=(),
        module_init_kwargs=FrozenDict({"param_dtype": jnp.float32}),
        init_args=(),
        init_kwargs=FrozenDict({}),
        train_args=(),
        train_kwargs=FrozenDict({}),
    )
    mlp_col_metadata = ModuleMetadata(
        rng=sk7,
        name="mlp_col",
        num_layers=args.num_layers,
        checkpoint_activations=True,
        in_init_pspec=None,
        out_init_pspec=PartitionSpec(None, "tp"),
        in_train_pspec=(PartitionSpec(None, "tp"), None, None),
        out_train_pspec=PartitionSpec(None, None, "tp"),
        layer=ColumnParallelLinear,
        data_shape=(args.batch_size, args.seq_len, args.hidden),
        dtype=float_dtype,
        module_init_args=(args.hidden * 4, 0),
        module_init_kwargs=FrozenDict({"param_dtype": float_dtype}),
        init_args=(),
        init_kwargs=FrozenDict({"train": False}),
        train_args=(),
        train_kwargs=FrozenDict({"train": True}),
    )
    mlp_row_metadata = ModuleMetadata(
        rng=sk8,
        name="mlp_row",
        num_layers=args.num_layers,
        checkpoint_activations=True,
        in_init_pspec=PartitionSpec(None, None, "tp"),
        out_init_pspec=PartitionSpec("tp", None),
        in_train_pspec=(
            PartitionSpec("tp", None),
            PartitionSpec(None, None, "tp"),
            None,
        ),
        out_train_pspec=PartitionSpec(None, None, "tp"),
        layer=RowParallelLinear,
        data_shape=(args.batch_size, args.seq_len, args.hidden * 4),
        dtype=float_dtype,
        module_init_args=(args.hidden, 0.1),
        module_init_kwargs=FrozenDict({"param_dtype": float_dtype}),
        init_args=(),
        init_kwargs=FrozenDict({"train": False}),
        train_args=(),
        train_kwargs=FrozenDict({"train": True}),
    )

    # Create transformer model metadata manager
    meta = ModuleMetadataManager(
        mesh,
        args.num_layers,
        (
            embed_metadata,
            pos_embed_metadata,
            layernorm_msa_metadata,
            msa_attn_metadata,
            msa_mlp_metadata,
            layernorm_mlp_metadata,
            mlp_col_metadata,
            mlp_row_metadata,
        ),
    )
    params = meta.init_from_pjit_metadata()
    meta.init_forward_from_pjit_metadata()
    meta.init_inference_from_pjit_metadata()

    # Verify that the pjit layer function is identical to running the layer on
    # one GPU
    # TODO: Train kwargs set manually in here, but FrozenDict immutable
    """
    main_key, verification_key = random.split(main_key)
    results, overall = verify_dist_model(
        verification_key,
        mesh,
        meta,
    )
    assert overall.item() is True, "Dist layer(s) have different" "outputs"
    """

    # Create transformer interface for training, inference, and checkpointing
    # handling
    main_key, t_inter_key = random.split(main_key)
    transformer_interface = TransformerInterface(
        rng=t_inter_key,
        module_metadata_manager=meta,
        train_dset=train_dset,
        val_dset=val_dset,
        tokenizer=tokenizer,
        float_dtype=float_dtype,
        int_dtype=int_dtype,
        logger=logger,
        debug=True if args.debug == 1 else False,
    )
    transformer_interface.train(
        params,
        softmax_cross_entropy_loss,
        args.num_epochs,
        args.batch_size,
        args.lr,
        args.wd,
        args.clipping,
        args.label_smoothing,
        args.dynamic_loss_scaling,
        args.dynamic_loss_decrease_scale,
        args.dynamic_loss_increase_scale,
        args.dynamic_loss_increase_window,
        args.optim_eps,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
