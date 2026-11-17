"""Baseline: o bloco GRU compartilhado + uma camada linear."""

import torch
from comum.gru import BlocoGRU
from torch import nn


class GRUBaseline(nn.Module):
    def __init__(self, n_entradas: int, camadas: list[int]):
        super().__init__()
        self.gru = BlocoGRU(n_entradas, camadas)
        self.linear = nn.Linear(self.gru.saida, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, ultimo = self.gru(x)
        return self.linear(ultimo).squeeze(-1)
