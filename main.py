"""
Author: Jyothi Vishnu Vardhan Kolla.

This is the main file which exectutes the entire system.
"""

# Import necessary libraries.
import sys
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import datasets
from datasets import load_dataset

from config import get_config
from dataset import get_data_loaders


def get_pairs(raw_data: datasets.arrow_dataset.Dataset, lang: str):
    """Yields each pair in the dataset as an iterator.

    Args:
        raw_data (datasets.arrow_dataset.Dataset): Dataset loaded from hugging face datasets.
        lang (str): language to be tokenized.
    """
    for pair in raw_data:
        yield pair["translation"][lang]


def perform_tokenization(raw_data: datasets.arrow_dataset.Dataset, lang: str, config: dict):
    """Tokenizes the given language by loading if there is an existing tokenizer,
    else creates a new one from scratch.

    Args:
        raw_data (datasets.arrow_dataset.Dataset): Dataset loaded from hugging face datasets.
        lang (str): language to be tokenized.
        config: A dictionary containing config parameters.
    """
    tokenizer_path = Path(config["tokenizer_file_path"].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_pairs(raw_data, lang), trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def check_max_sequence_length(raw_data: datasets.arrow_dataset.Dataset, src_lang: str, tgt_lang: str,
                              src_tokenizer: Tokenizer, tgt_tokenizer: Tokenizer):
    """Calculates and displays the max seq_length in src and tgt language in the
    dataset.

    Args:
        raw_data (datasets.arrow_dataset.Dataset): _description_
        src_lang (str): _description_
        tgt_lang (str): _description_
        config (dict): _description_
    """
    src_max_length, tgt_max_length = 0, 0
    for pair in raw_data:
        src_text = pair["translation"][src_lang]
        tgt_text = pair["translation"][tgt_lang]

        src_ids, tgt_ids = src_tokenizer.encode(
            src_text), tgt_tokenizer.encode(tgt_text)
        src_max_length, tgt_max_length = max(src_max_length, len(
            src_ids)), max(tgt_max_length, len(tgt_ids))

    print(f"max seq_ln of src in dataset is {src_max_length}")
    print(f"max seq_ln of tgt in dataset is {tgt_max_length}")


def main(argv):
    """Main function which takes command line arguments and
    completes training the model.

    Args:
        argv (str): source language.
        argv (str): tgt language.
    """
    src_lang, tgt_lang, config = argv[1], argv[2], get_config()
    raw_data = load_dataset(
        "opus_books", f"{src_lang}-{tgt_lang}", split="train")
    # source tokenizer.
    src_tokenizer = perform_tokenization(raw_data, src_lang, config)
    # target tokenizer.
    tgt_tokenizer = perform_tokenization(raw_data, tgt_lang, config)
    #check_max_sequence_length(
     #   raw_data, src_lang, tgt_lang, src_tokenizer, tgt_tokenizer)
    train_dataloader, val_dataloader = get_data_loaders(
        raw_data=raw_data, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer,
        src_lang=src_lang, tgt_lang=tgt_lang, config=config
    )


if __name__ == "__main__":
    main(sys.argv)
