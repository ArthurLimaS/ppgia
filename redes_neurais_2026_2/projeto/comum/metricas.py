"""Métricas de preço e de direção, calculadas sobre o fechamento (não sobre a variação)."""

import numpy as np


def calcular(real: np.ndarray, previsto: np.ndarray, anterior: np.ndarray) -> dict[str, float]:
    """`anterior` é o fechamento do pregão antes do alvo: define a direção (TEGRU, Eq. 9-10)."""
    erro = previsto - real
    mape = float(np.mean(np.abs(erro / real)) * 100)
    acuracia = float(np.mean((real - anterior >= 0) == (previsto - anterior >= 0)))
    return {
        "rmse": float(np.sqrt(np.mean(erro**2))),
        "mae": float(np.mean(np.abs(erro))),
        "mape": mape,
        "r2": float(1 - np.sum(erro**2) / np.sum((real - real.mean()) ** 2)),
        "acuracia_direcao": acuracia,
        "acmape": (1 - acuracia) * 2 + mape / 100,  # TEGRU, Eq. 13
    }


def ingenuo(real: np.ndarray, anterior: np.ndarray) -> dict[str, float]:
    """Referência "amanhã = hoje". A direção dela é sempre "alta" (diferença zero conta como alta)."""
    return calcular(real, anterior, anterior)
