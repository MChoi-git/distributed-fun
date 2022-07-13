import os
from pathlib import Path

from jax import numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tokenizers import (
    models,
    Tokenizer,
    pre_tokenizers,
    trainers,
    processors,
    decoders,
)
from transformers import PreTrainedTokenizerFast


# Prevent TF allocating mem
tf.config.experimental.set_visible_devices([], "GPU")


def tfds_to_corpus(text_dset):
    corpus = []
    for ex in text_dset:
        corpus.append(ex["text"].numpy().decode("utf-8"))
    return corpus


def make_shakespeare_dataset(tokenizer, dset_path, seq_len, batch_size):
    dset = tfds.load(name="tiny_shakespeare", data_dir=dset_path, download=False)

    # Need to peel string from 1-element list
    corpus_split_data = [tfds_to_corpus(d)[0] for d in dset.values()]

    train_data_encoded = tokenizer.encode_plus(corpus_split_data[0])
    val_data_encoded = tokenizer.encode_plus(corpus_split_data[1])
    test_data_encoded = tokenizer.encode_plus(corpus_split_data[2])

    def make_data_features(split_data_encoded, seq_len):
        num_ragged_tokens = len(split_data_encoded["input_ids"]) % seq_len
        num_examples = len(split_data_encoded["input_ids"]) // seq_len
        # Drop ragged tokens, then split examples by seq len, and pack into
        # unbatched array
        split_feats = {
            k: np.array(np.split(np.array(v)[:-num_ragged_tokens], num_examples))
            for k, v in split_data_encoded.items()
        }
        return split_feats

    train_feats = make_data_features(train_data_encoded, seq_len)
    val_feats = make_data_features(val_data_encoded, seq_len)
    test_feats = make_data_features(test_data_encoded, seq_len)

    train_tfdset = tf.data.Dataset.from_tensor_slices(train_feats).batch(
        batch_size, drop_remainder=True
    )
    train_tfdset = train_tfdset.shuffle(
        len(train_tfdset), reshuffle_each_iteration=True
    )
    val_tfdset = tf.data.Dataset.from_tensor_slices(val_feats).batch(
        batch_size, drop_remainder=True
    )
    test_tfdset = tf.data.Dataset.from_tensor_slices(test_feats).batch(
        batch_size, drop_remainder=True
    )

    train_tfdset = tfds.as_numpy(train_tfdset)
    val_tfdset = tfds.as_numpy(val_tfdset)
    test_tfdset = tfds.as_numpy(test_tfdset)

    return train_tfdset, val_tfdset, test_tfdset


def tokenize_shakespeare(dset_path, corpus_save_path, max_vocab_size, seq_len):
    dset = tfds.load(name="tiny_shakespeare", data_dir=dset_path, download=False)

    corpus_split_data = [tfds_to_corpus(d) for d in dset.values()]

    # Re-write dataset corpus since original TF format is annoying
    corpus_save_path = Path(corpus_save_path)
    train_path = corpus_save_path / "shakespeare_train_corpus.txt"
    val_path = corpus_save_path / "shakespeare_val_corpus.txt"
    test_path = corpus_save_path / "shakespeare_test_corpus.txt"

    corpus_split_paths = [
        train_path,
        val_path,
        test_path,
    ]
    for corpus_path, corpus_data in zip(corpus_split_paths, corpus_split_data):
        if not os.path.isfile(corpus_path):
            print(f"Saving corpus path {corpus_path}")
            with open(corpus_path, "w", encoding="utf-8") as f:
                f.write(corpus_data[0])  # Data is alone in list

        else:
            print(f"File present, skipping {corpus_path}")

    tokenizer = Tokenizer(models.BPE())

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=max_vocab_size, special_tokens=["[START]", "[END]", "[PAD]"]
    )

    tokenizer.train([str(p) for p in corpus_split_paths], trainer=trainer)

    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    tokenizer.decoder = decoders.ByteLevel()

    model_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        model_max_length=seq_len,
        pad_token="[PAD]",
        bos_token="[START]",
        eos_token="[END]",
    )

    return model_tokenizer


def save_tokenizer(tokenizer, tokenizer_file):
    tokenizer.save_pretrained(tokenizer_file)


def get_tokenizer(tokenizer_file):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=tokenizer_file
    )
    return tokenizer
