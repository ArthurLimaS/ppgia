"""
Métricas de preço e de direção, calculadas sobre o fechamento (não sobre a variação).
"""

import numpy as np


def calcular(
    real: np.ndarray, previsto: np.ndarray, anterior: np.ndarray
) -> dict[str, float]:
    """
    `anterior` é o fechamento do pregão antes do alvo: define a direção (TEGRU, Eq.
    9-10).
    """
    erro = previsto - real
    mape = float(np.mean(np.abs(erro / real)) * 100)
    acuracia = acuracia_direcao(real, anterior, previsto - anterior >= 0)
    return {
        "rmse": float(np.sqrt(np.mean(erro**2))),
        "mae": float(np.mean(np.abs(erro))),
        "mape": mape,
        "r2": float(1 - np.sum(erro**2) / np.sum((real - real.mean()) ** 2)),
        "acuracia_direcao": acuracia,
        "acmape": (1 - acuracia) * 2 + mape / 100,  # TEGRU, Eq. 13
    }


def acuracia_direcao(
    real: np.ndarray, anterior: np.ndarray, alta_prevista: np.ndarray
) -> float:
    """O rótulo real é o mesmo para todos os modelos: alta se real - anterior ≥ 0."""
    return float(np.mean((real - anterior >= 0) == alta_prevista))


def wilson(proporcao: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de confiança de Wilson (95 % com z = 1,96) para uma proporção
    binomial: Wilson (1927); recomendado por Brown, Cai e DasGupta (2001).
    """
    den = 1 + z**2 / n
    centro = (proporcao + z**2 / (2 * n)) / den
    meia = z / den * np.sqrt(proporcao * (1 - proporcao) / n + z**2 / (4 * n**2))
    return float(centro - meia), float(centro + meia)


def ingenuo(real: np.ndarray, anterior: np.ndarray) -> dict[str, float]:
    """
    Referência "amanhã = hoje". A direção dela é sempre "alta" (diferença zero conta
    como alta).
    """
    return calcular(real, anterior, anterior)
