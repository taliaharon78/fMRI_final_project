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

seq_len = 1 # set to 1 if averaging TRs, else - seq_len = len(TRs)
embedding_dim = 512

# Positionwise Feed-Forward Network
# This network is used within the Transformer model to process the output of the attention layer.
# It applies two linear transformations with a ReLU activation in between and a dropout for regularization.
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)  # First linear layer
        self.linear2 = nn.Linear(hidden, d_model)  # Second linear layer
        self.relu = nn.ReLU()  # ReLU activation function
        self.dropout = nn.Dropout(p=drop_prob)  # Dropout layer for regularization

    def forward(self, x):
        x = self.linear1(x)  # Apply first linear layer
        x = self.relu(x)  # Apply ReLU activation
        x = self.dropout(x)  # Apply dropout
        x = self.linear2(x)  # Apply second linear layer
        return x

# Positional Encoding
# This class implements the positional encoding to provide information about the position of elements in the sequence.
# It uses sine and cosine functions to encode the positions and adds dropout for regularization.
class PositionalEncoding(nn.Module):
    def __init__(self, dropout: float = 0.1, max_seq_len: int = 5000, d_model: int = 512, batch_first: bool = False):
        super().__init__()

        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        self.x_dim = 1 if batch_first else 0
        position = torch.arange(max_seq_len).unsqueeze(1)  # Position tensor
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # Divisor term
        pe = torch.zeros(50, max_seq_len, d_model)  # Positional encoding tensor

        pe[:, :, 0::2] = torch.sin(position * div_term)  # Apply sine function
        pe[:, :, 1::2] = torch.cos(position * div_term)  # Apply cosine function

        self.register_buffer('pe', pe)  # Register positional encoding tensor

    def forward(self, x):
        x = x + self.pe[:x.size(self.x_dim)]  # Add positional encoding to input
        x = self.dropout(x)  # Apply dropout
        return x

# Attention Block
# This class implements the attention mechanism used in the Transformer model.
# It includes multi-head attention, dropout, and layer normalization. It also applies a feed-forward network.
class AttentionBlock(nn.Module):
    def __init__(self, num_heads=1, head_size=128, ff_dim=None, dropout=0):
        super(AttentionBlock, self).__init__()

        if ff_dim is None:
            ff_dim = head_size

        self.attention = nn.MultiheadAttention(embed_dim=head_size, num_heads=num_heads, dropout=dropout)  # Multi-head attention
        self.attention_dropout = nn.Dropout(dropout)  # Dropout for attention layer
        self.attention_norm = nn.LayerNorm(head_size, eps=1e-6)  # Layer normalization after attention

        self.ffn = PositionwiseFeedForward(d_model=head_size, hidden=128, drop_prob=0.1)  # Feed-forward network
        self.ff_dropout = nn.Dropout(dropout)  # Dropout for feed-forward network
        self.ff_norm = nn.LayerNorm(ff_dim, eps=1e-6)  # Layer normalization after feed-forward network

    def forward(self, inputs):
        x, attention_scores = self.attention(inputs, inputs, inputs)  # Apply multi-head attention
        x = self.attention_dropout(x)  # Apply dropout
        x = self.attention_norm(inputs + x)  # Apply layer normalization

        x = self.ffn(x)  # Apply feed-forward network
        x = self.ff_dropout(x)  # Apply dropout
        x = self.ff_norm(inputs + x)  # Apply layer normalization
        return x, attention_scores

# Transformer Encoder
# This class implements the encoder part of the Transformer model.
# It includes an input linear layer, positional encoding, multiple attention blocks, and final linear layers for classification.
class TransformerEncoder(nn.Module):
    def __init__(self, num_voxels, classes, time2vec_dim, num_heads=1, head_size=128, ff_dim=None, num_layers=1,
                 dropout=0):
        super(TransformerEncoder, self).__init__()
        self.encoder_input_layer = nn.Linear(in_features=num_voxels, out_features=ff_dim)  # Input linear layer
        if ff_dim is None:
            ff_dim = head_size
        self.dropout = dropout
        self.classes = classes
        self.positional_encoding = PositionalEncoding(dropout=0.1, max_seq_len=seq_len, d_model=embedding_dim, batch_first=False)  # Positional encoding
        self.attention_layers = nn.ModuleList(
            [AttentionBlock(num_heads=num_heads, head_size=head_size, ff_dim=ff_dim, dropout=dropout) for _ in range(num_layers)])  # Attention blocks
        self.norm = nn.LayerNorm(ff_dim)  # Layer normalization
        self.final_layers = nn.Sequential(
            nn.Flatten(),  # Flatten the input
            nn.Linear(ff_dim * seq_len, 512),  # Linear layer
            nn.ReLU(),  # ReLU activation
            nn.Linear(512, 256),  # Linear layer
            nn.ReLU(),  # ReLU activation
            nn.Linear(256, 128),  # Linear layer
            nn.ReLU(),  # ReLU activation
            nn.Linear(128, classes)  # Output layer for classification
        )

    def forward(self, inputs):
        x = self.encoder_input_layer(inputs)  # Apply input linear layer
        x = self.positional_encoding(x)  # Apply positional encoding
        for attention_layer in self.attention_layers:
            x, attention_scores = attention_layer(x)  # Apply attention blocks
        x = self.norm(x)  # Apply layer normalization
        x = self.final_layers(x)  # Apply final layers for classification
        return x
