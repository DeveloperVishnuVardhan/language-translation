"""
Authors: Jyothi Vishnu Vardhan Kolla.

This files contains the code to prepare the dataset
to train the model.
"""
import datasets
from tokenizers import Tokenizer

import torch
from torch.utils.data import Dataset, DataLoader, random_split


class CustomTranslationDataset(Dataset):
    def __init__(self, data, src_tokenizer: Tokenizer, tgt_tokenizer: Tokenizer, src_lang: str, tgt_lang: str, seq_len: int):
        self.data = data
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor(
            [tgt_tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor(
            [tgt_tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor(
            [tgt_tokenizer.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        src_text = self.data[index]['translation'][self.src_lang]
        tgt_text = self.data[index]['translation'][self.tgt_lang]

        encoder_token_ids = self.src_tokenizer.encode(src_text).ids
        decoder_token_ids = self.tgt_tokenizer.encode(tgt_text).ids

        # We add SOS and EOS.
        encoder_pading_tokens = self.seq_len - len(encoder_token_ids) - 2
        # We only add SOS.
        decoder_padding_tokens = self.seq_len - len(decoder_token_ids) - 1

        # See if no sentence is too long.
        if encoder_pading_tokens < 0 or decoder_padding_tokens < 0:
            raise ValueError("Sentence too Long")

        # Prepare the encoder input (add SOS and EOS).
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(encoder_token_ids, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] *
                             encoder_pading_tokens, dtype=torch.int64),
            ],
            dim=0,
        )
        # Prepare the decoder input (add only SOS).
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(decoder_token_ids, dtype=torch.int64),
                torch.tensor([self.pad_token] * decoder_padding_tokens)
            ],
            dim=0
        )

        # Prepare the label (add only EOS).
        label = torch.cat(
            [
                torch.tensor(decoder_token_ids, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * decoder_padding_tokens)
            ],
            dim=0
        )

        print(encoder_input.shape)
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0)
        }


def get_data_loaders(raw_data: datasets.arrow_dataset.Dataset, src_tokenizer: Tokenizer, tgt_tokenizer: Tokenizer,
                     src_lang: str, tgt_lang: str, config: dict):
    """Creates and returns the batched dataloader to train the model.

    Args:
        raw_data (datasets.arrow_dataset.Dataset): Raw hugging face dataset.
        src_tokenizer (Tokenizer): The tokenizer of the source language.
        tgt_tokenizer (Tokenizer): The tokenizer of the target language.
        src_lang (str): Source langauge.
        tgt_lang (str): Target language.
    """
    train_ds_size = int(0.9 * (len(raw_data)))
    val_ds_size = len(raw_data) - train_ds_size
    train_data, val_data = random_split(raw_data, [train_ds_size, val_ds_size])

    train_dataset = CustomTranslationDataset(train_data, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer,
                                             src_lang=src_lang, tgt_lang=tgt_lang, seq_len=config["max_sequence_length"])
    val_dataset = CustomTranslationDataset(val_data, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer,
                                           src_lang=src_lang, tgt_lang=tgt_lang, seq_len=config["max_sequence_length"])
    for i in range(1):
        train_dataset[i]

    return None, None
