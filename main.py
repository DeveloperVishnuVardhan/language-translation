"""
Author: Jyothi Vishnu Vardhan Kolla.

This is the mail file which exectutes the entire system.
"""

# Import necessary libraries.
import sys
from pathlib import Path

import tokenizers
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import datasets
from datasets import load_dataset

from config import get_config


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
            special_tokens=["[UNK]", "[PAD]", "SOS", "EOS"], min_frequency=2)
        tokenizer.train_from_iterator(get_pairs(raw_data, lang), trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


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
    src_tokenizer = perform_tokenization(raw_data, src_lang, config)
    tgt_tokeizer = perform_tokenization(raw_data, tgt_lang, config)


if __name__ == "__main__":
    main(sys.argv)
