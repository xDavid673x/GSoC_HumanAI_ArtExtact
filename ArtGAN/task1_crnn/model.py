from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


class AttentionPool(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1),
        )

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.attention(tokens).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class ConvRecurrentWikiArtClassifier(nn.Module):
    def __init__(
        self,
        num_classes: Dict[str, int],
        grid_size: tuple[int, int] = (6, 6),
        recurrent_hidden_dim: int = 256,
        recurrent_layers: int = 2,
        embedding_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if not {"style", "genre", "artist"}.issubset(num_classes):
            raise ValueError("num_classes must include style, genre, and artist")
        self.grid_size = grid_size

        backbone = resnet18(weights=None)
        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        token_dim = 512
        self.token_projection = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, recurrent_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.recurrent = nn.GRU(
            input_size=recurrent_hidden_dim,
            hidden_size=recurrent_hidden_dim,
            num_layers=recurrent_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if recurrent_layers > 1 else 0.0,
        )
        recurrent_output_dim = recurrent_hidden_dim * 2
        self.sequence_pool = AttentionPool(recurrent_output_dim)
        self.embedding = nn.Sequential(
            nn.LayerNorm(recurrent_output_dim),
            nn.Linear(recurrent_output_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.style_head = MLPHead(embedding_dim, num_classes["style"], dropout)
        self.genre_head = MLPHead(embedding_dim, num_classes["genre"], dropout)
        artist_input_dim = embedding_dim + num_classes["style"] + num_classes["genre"]
        self.artist_head = MLPHead(artist_input_dim, num_classes["artist"], dropout)

    def encode(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        feature_map = self.encoder(images)
        if feature_map.shape[-2:] == self.grid_size:
            pooled_grid = feature_map
        else:
            # `adaptive_avg_pool2d` hits an MPS limitation on non-divisible sizes.
            # Bilinear resizing keeps a fixed token grid without forcing a CPU fallback.
            pooled_grid = F.interpolate(
                feature_map,
                size=self.grid_size,
                mode="bilinear",
                align_corners=False,
            )
        batch_size, channels, height, width = pooled_grid.shape
        tokens = pooled_grid.view(batch_size, channels, height * width).transpose(1, 2).contiguous()
        tokens = self.token_projection(tokens)
        recurrent_tokens, _ = self.recurrent(tokens)
        pooled_sequence, attention = self.sequence_pool(recurrent_tokens)
        embedding = self.embedding(pooled_sequence)
        return {
            "feature_map": feature_map,
            "tokens": recurrent_tokens,
            "attention": attention,
            "embedding": embedding,
        }

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encode(images)
        embedding = encoded["embedding"]

        style_logits = self.style_head(embedding)
        genre_logits = self.genre_head(embedding)
        artist_features = torch.cat(
            [
                embedding,
                torch.softmax(style_logits, dim=-1),
                torch.softmax(genre_logits, dim=-1),
            ],
            dim=-1,
        )
        artist_logits = self.artist_head(artist_features)

        encoded.update(
            {
                "style": style_logits,
                "genre": genre_logits,
                "artist": artist_logits,
            }
        )
        return encoded
