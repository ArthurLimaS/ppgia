"""Bloco GRU compartilhado pelos quatro modelos do estudo."""

import itertools

import torch
from torch import nn


class BlocoGRU(nn.Module):
    """GRUs empilhadas, uma por tamanho em `camadas` (ex.: [128, 64, 32]).

    Cada camada é um `nn.GRU` próprio porque o `num_layers` do PyTorch exige o mesmo
    tamanho em todas. Entrada [lote, tempo, features]; devolve a sequência de estados da
    última camada [lote, tempo, camadas[-1]] e o último estado [lote, camadas[-1]].
    """

    def __init__(self, n_entradas: int, camadas: list[int]):
        super().__init__()
        tamanhos = [n_entradas, *camadas]
        self.camadas = nn.ModuleList(
            nn.GRU(a, b, batch_first=True) for a, b in itertools.pairwise(tamanhos)
        )
        self.saida = camadas[-1]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for gru in self.camadas:
            x, _ = gru(x)
        return x, x[:, -1]
