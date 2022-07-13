import os

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers import (
    models,
    trainers,
    pre_tokenizers,
    processors,
    decoders,
)
from transformers import PreTrainedTokenizerFast


def make_wikitext_dataset(path, name):
    dataset = load_dataset(
        path=path,
        name=name,
    )
    test_dset, train_dset, val_dset = dataset.values()
    return test_dset, train_dset, val_dset


def make_wikitext_tokenizer(dsets, max_vocab_size, seq_len, temp_save_dir, tokenizer_save_dir):
    if os.path.isdir(tokenizer_save_dir):
        return PreTrainedTokenizerFast.from_pretrained(
            pretrained_model_name_or_path=tokenizer_save_dir
        )

    test_dset, train_dset, val_dset = dsets

    tokenizer = Tokenizer(models.BPE())

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    dset_temp_files = {
        temp_save_dir + "/test_dset.json": test_dset,
        temp_save_dir + "/train_dset.json": train_dset,
        temp_save_dir + "/val_dset.json": val_dset,
    }

    if not os.path.isdir(temp_save_dir):
        os.mkdir(temp_save_dir)

    bytes_written = 0
    files_written = 0
    for temp_file, dset in dset_temp_files.items():
        if not os.path.isfile(temp_file):
            written = dset.to_json(temp_file)
            bytes_written.append(written)
            files_written += 1

    if files_written != 0 and bytes_written == 0:
        raise Exception("Nothing was written!")

    trainer = trainers.BpeTrainer(
        special_tokens=[
            "[BOS]",
            "[EOS]",
            "[PAD]",
        ],
        vocab_size=max_vocab_size,
    )

    tokenizer.train(
        files=list(dset_temp_files.keys()),
        trainer=trainer,
    )

    tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[
            ("[BOS]", tokenizer.token_to_id("[BOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ]
    )

    tokenizer.decoder = decoders.ByteLevel()

    model_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        model_max_length=seq_len,
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )

    model_tokenizer.save_pretrained(tokenizer_save_dir)

    return model_tokenizer
