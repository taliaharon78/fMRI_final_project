import torch
import torchmetrics
import wandb
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
from sklearn.metrics import confusion_matrix
from torchmetrics import ConfusionMatrix
import random
import math

seq_len = 1
embedding_dim = 512

# Positionwise feed-forward network
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# Positional encoding for transformer
class PositionalEncoding(nn.Module):
    def __init__(self, dropout: float = 0.1, max_seq_len: int = 5000, d_model: int = 512, batch_first: bool = False):
        super().__init__()

        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        self.x_dim = 1 if batch_first else 0
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(50, max_seq_len, d_model)

        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(self.x_dim)]
        x = self.dropout(x)
        return x

# Attention block
class AttentionBlock(nn.Module):
    def __init__(self, num_heads=1, head_size=128, ff_dim=None, dropout=0):
        super(AttentionBlock, self).__init__()

        if ff_dim is None:
            ff_dim = head_size

        self.attention = nn.MultiheadAttention(embed_dim=head_size, num_heads=num_heads, dropout=dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(head_size, eps=1e-6)

        self.ffn = PositionwiseFeedForward(d_model=head_size, hidden=128, drop_prob=0.1)
        self.ff_dropout = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(ff_dim, eps=1e-6)

    def forward(self, inputs):
        x, attention_scores = self.attention(inputs, inputs, inputs)
        x = self.attention_dropout(x)
        x = self.attention_norm(inputs + x)

        x = self.ffn(x)
        x = self.ff_dropout(x)
        x = self.ff_norm(inputs + x)
        return x, attention_scores

# Transformer encoder
class TransformerEncoder(nn.Module):
    def __init__(self, num_voxels, classes, time2vec_dim, num_heads=1, head_size=128, ff_dim=None, num_layers=1,
                 dropout=0):
        super(TransformerEncoder, self).__init__()
        self.encoder_input_layer = nn.Linear(in_features=num_voxels, out_features=ff_dim)
        if ff_dim is None:
            ff_dim = head_size
        self.dropout = dropout
        self.classes = classes
        self.positional_encoding = PositionalEncoding(dropout=0.1, max_seq_len=seq_len, d_model=embedding_dim,  batch_first=False)
        self.attention_layers = nn.ModuleList(
            [AttentionBlock(num_heads=num_heads, head_size=head_size, ff_dim=ff_dim, dropout=dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(ff_dim)
        self.final_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ff_dim * seq_len, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, classes)
        )

    def forward(self, inputs):
        x = self.encoder_input_layer(inputs)
        x = self.positional_encoding(x)
        for attention_layer in self.attention_layers:
            x, attention_scores = attention_layer(x)
        x = self.norm(x)
        x = self.final_layers(x)
        return x
