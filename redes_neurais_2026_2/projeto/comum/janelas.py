"""Série diária → amostras (janela de N pregões → pregão seguinte), com divisão cronológica.

Normalização por janela (D-20, opção c): cada janela é expressa como variação relativa ao seu
último fechamento, e o alvo é a variação relativa do pregão seguinte. Nenhuma estatística é
ajustada nos dados, então não há o que vazar do teste para o treino.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

FECHAMENTO = "Adj Close"  # D-04: alvo e entrada usam o fechamento ajustado
ENTRADAS = ("univariado",)  # "ohlcv" entra depois da D-21 (normalização do volume)
FRACOES = (0.8, 0.1)  # D-02: treino, validação; o resto é teste


@dataclass
class Conjunto:
    x: (
        torch.Tensor
    )  # [amostras, janela, features]: variação relativa ao último fechamento da janela
    y: torch.Tensor  # [amostras]: fechamento(alvo) / fechamento(anterior) - 1
    anterior: np.ndarray  # fechamento do último pregão da janela (base da normalização)
    real: np.ndarray  # fechamento do pregão-alvo
    datas: pd.DatetimeIndex  # data do pregão-alvo

    def __len__(self) -> int:
        return len(self.y)


def limites(n: int) -> tuple[int, int]:
    """Posições onde começam os alvos de validação e de teste."""
    return int(n * FRACOES[0]), int(n * (FRACOES[0] + FRACOES[1]))


def montar(
    df: pd.DataFrame, entrada: str, janela: int, primeiro_alvo_treino: int | None = None
) -> dict[str, Conjunto]:
    """Monta treino, validação e teste. A divisão é pela posição do pregão-alvo.

    `primeiro_alvo_treino` fixa onde o treino começa (D-17: no estudo da GRU, todas as janelas
    treinam nos alvos que a maior janela alcança). O padrão é o primeiro alvo possível.
    """
    if entrada not in ENTRADAS:
        raise ValueError(f"entrada '{entrada}' indisponível; opções: {ENTRADAS}")
    primeiro = janela if primeiro_alvo_treino is None else primeiro_alvo_treino
    if primeiro < janela:
        raise ValueError(
            f"primeiro alvo ({primeiro}) antes do fim da primeira janela ({janela})"
        )

    fechamento = df[FECHAMENTO].to_numpy(dtype=np.float64)
    ini_val, ini_teste = limites(len(df))
    if primeiro >= ini_val:
        raise ValueError("a janela não deixa nenhum alvo no treino")

    def conjunto(inicio: int, fim: int) -> Conjunto:
        alvos = np.arange(inicio, fim)
        indices = (
            alvos[:, None] + np.arange(-janela, 0)[None, :]
        )  # só posições anteriores ao alvo
        anterior = fechamento[alvos - 1]
        x = fechamento[indices] / anterior[:, None] - 1.0
        y = fechamento[alvos] / anterior - 1.0
        return Conjunto(
            x=torch.tensor(x[:, :, None], dtype=torch.float32),
            y=torch.tensor(y, dtype=torch.float32),
            anterior=anterior,
            real=fechamento[alvos],
            datas=df.index[alvos],
        )

    return {
        "treino": conjunto(primeiro, ini_val),
        "validacao": conjunto(ini_val, ini_teste),
        "teste": conjunto(ini_teste, len(df)),
    }


def para_preco(variacao: np.ndarray, anterior: np.ndarray) -> np.ndarray:
    """Desfaz a normalização: variação relativa prevista → fechamento previsto."""
    return anterior * (1.0 + variacao)
