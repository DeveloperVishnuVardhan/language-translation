"""
Authors: Jyothi Vishnu Vardhan Kolla.

This files contains config to train the model.
"""


def get_config():
    return {
        "tokenizer_file_path": "tokenizer_{0}.json",
        "max_sequence_length": 350
    }
