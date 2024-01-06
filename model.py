"""
Authors: Jyothi Vishnu Vardhan Kolla.

This files contains the code to build the transformer model.
"""
import torch
import torch.nn as nn
from torch.autograd import Variable

import math


class InputEmbeddingLayer(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        """creates Embeddings by taking input batch tokens.

        Args:
            vocab_size (int): Total number of words in the vocabulary.
            embedding_dim (int): Dimension of Embedding.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim)

    def forward(self, batched_tokens):
        # (batch, vocab_size (or) seq_len) -> (batch, seq_len, d_model).
        return self.embedding(batched_tokens) * math.sqrt(self.embedding_dim)


class PositionalEncodinds(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, dropout: float):
        """Creates Postional encodings to add to the output of
        Embedding layer.

        Args:
            vocab_size (int): Total number of words in the vocabulary.
            embedding_dim (int): Dimension of Embedding.
            dropout (float): dropout rate to apply.
        """
        self.dropout = nn.Dropout(p=dropout)
        # Create a Positional-encoding matrix.
        pe = torch.ones(vocab_size, embedding_dim)
        # create the numerator vector.
        # we add a dimension to make it compatible with broadcasting.
        pos = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        # Div term: pow(1000, 2i / d_model) can be written as:
        # 1 / torch.exp(-(math.log(1000.0) / d_model) * 2i)
        div_term = torch.exp((-math.log(1000.0) / embedding_dim)
                             * torch.arange(0, embedding_dim, 2, dtype=torch.float))

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model).
        self.register_buffer('pe', pe)

    def forward(self, embedded_tensor):
        # (batch, seq_len, d_model) + (batch, vocab_size[:seq_len], d_model)
        x = embedded_tensor + \
            Variable(self.pe[:, :x.shape[1], :], requires_grad=False)
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, heads: int, dropout: float):
        """This class implements the Multi-head Attention block.

        Args:
            embedding_dim (int): Dimension of Embedding.
            heads (int): Number of Attention heads.
            dropout (float): Dropout rate to apply.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.heads = heads
        self.d_k = embedding_dim // heads

        assert embedding_dim // heads == 0, "embedding-dim not divisible by heads"

        # Initialize the query, key, value, concat linear layers.
        self.w_q = nn.Linear(in_features=embedding_dim,
                             out_features=embedding_dim)
        self.w_k = nn.Linear(in_features=embedding_dim,
                             out_features=embedding_dim)
        self.w_v = nn.Linear(in_features=embedding_dim,
                             out_features=embedding_dim)
        self.w_o = nn.Linear(in_features=embedding_dim,
                             out_features=embedding_dim)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        # (batch, h, seq_len, d_k) * (batch, h, seq_len, d_k ->
        # batch, h, d_k, seq_len) = (batch, h, seq_len, seq_len).
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

    def forward(self, query, key, value, mask):
        # (batch, seq_len, d_model) -> (batch, seq_len, d_model).
        query = self.w_q(query)
        # (batch, seq_len, d_model) -> (batch, seq_len, d_model).
        key = self.w_k(query)
        # (batch, seq_len, d_model) -> (batch, seq_len, d_model).
        value = self.w_v(query)

        # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) ->(batch, h, seq_len, d_k)
        query = query.view(
            query.shape[0], query.shape[1], self.heads, self.d_k).transpose(1, 2)
        key = key.view(
            key.shape[0], key.shape[1], self.heads, self.d_k
        ).transpose(1, 2)
        value = value.view(
            value.shape[0], value.shape[1], self.heads, self.d_k
        ).transpose(1, 2)

        attention, attention_scores = MultiHeadSelfAttention.attention(
            query=query, key=key, value=value, mask=mask)


class FeedForwardLayer(nn.Module):
    pass


class ResidualConnection(nn.Module):
    pass


class LayerNormalization(nn.Module):
    pass


class LinearProjectionLayer(nn.Module):
    pass
