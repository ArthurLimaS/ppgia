"""Testes de fumaça do HDFM (Etapa 4): entropia amostral, decomposição, normalização e
fusão, vazamento temporal da decomposição, treino e retomada, com dados sintéticos.

A CEEMDAN usa 5 realizações aqui (100 no estudo), para os testes caberem em minutos.

Uso (a partir da raiz do repositório):
    uv run python -m unittest discover -s redes_neurais_2026_2/projeto/testes -v
"""

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from comum.janelas import FECHAMENTO, limites
from modelos import hdfm
from rodar import executar

N, JANELA = (
    300,
    40,
)  # a 1ª decomposição causal tem JANELA pontos: precisa de 3+ componentes
P = hdfm.PARAMS | {"trials": 5}
_tmp = tempfile.TemporaryDirectory()
CACHE = Path(_tmp.name)


def setUpModule() -> None:
    for alvo, valor in (("PARAMS", P), ("CACHE", CACHE)):
        p = mock.patch.object(hdfm, alvo, valor)
        p.start()
        unittest.addModuleCleanup(p.stop)
    unittest.addModuleCleanup(_tmp.cleanup)


def serie(n: int = N) -> pd.DataFrame:
    """Tendência + dois ciclos + ruído: dá componentes de entropias diferentes."""
    t = np.arange(n)
    ruido = np.random.default_rng(0).normal(0, 0.5, n)
    x = 50 + 0.05 * t + 3 * np.sin(2 * np.pi * t / 20) + 1.5 * np.sin(2 * np.pi * t / 7)
    return pd.DataFrame(
        {FECHAMENTO: x + ruido}, index=pd.bdate_range("2020-01-01", periods=n)
    )


def cfg(**mudancas: Any) -> dict[str, Any]:
    base = {
        "modelo": "hdfm_diferenca",
        "entrada": None,
        "decomposicao": "inteira",
        "janela": JANELA,
        "camadas": [8],
        "epocas": 3,
        "lote": 64,
        "lr": 1e-2,
        "semente": 0,
        "dispositivo": "cpu",
        "primeiro_alvo_treino": None,
    }
    return base | mudancas


class TestEntropia(unittest.TestCase):
    def test_igual_a_forca_bruta(self):
        x = np.random.default_rng(1).normal(size=150)
        m, tol = 2, 0.2 * x.std()

        def contar(k: int) -> int:
            t = [x[i : i + k] for i in range(len(x) - m)]
            return sum(
                np.abs(t[i] - t[j]).max() <= tol
                for i in range(len(t))
                for j in range(i + 1, len(t))
            )

        esperado = np.log(contar(m) / contar(m + 1))
        self.assertAlmostEqual(hdfm.entropia_amostral(x, m, 0.2), esperado)

    def test_seno_e_mais_regular_que_ruido(self):
        t = np.arange(300)
        seno = np.sin(2 * np.pi * t / 25)
        ruido = np.random.default_rng(2).normal(size=300)
        self.assertLess(
            hdfm.entropia_amostral(seno, 2, 0.2), hdfm.entropia_amostral(ruido, 2, 0.2)
        )


class TestDecomposicao(unittest.TestCase):
    def test_12_subseries_que_somam_a_serie(self):
        for n in (N, N - 1):  # par e ímpar (a vmdpy descarta um ponto de série ímpar)
            x = serie(n)[FECHAMENTO].to_numpy()
            d = hdfm.decompor(x, P)
            self.assertEqual(d.shape, (hdfm.N_SUB, n))
            np.testing.assert_allclose(d.sum(axis=0), x, atol=1e-9)

    def test_deterministica(self):
        x = serie()[FECHAMENTO].to_numpy()
        np.testing.assert_array_equal(hdfm.decompor(x, P), hdfm.decompor(x, P))


class TestMontar(unittest.TestCase):
    def test_fusao_desfaz_a_normalizacao(self):
        df = serie()
        x = df[FECHAMENTO].to_numpy()
        for norm in ("A", "B"):
            for causal in (False, True):
                dados, desfazer, _ = hdfm.montar(
                    x, pd.DatetimeIndex(df.index), norm, causal, JANELA
                )
                for parte, c in dados.items():
                    np.testing.assert_allclose(
                        desfazer[parte](c.y.numpy().astype(np.float64)),
                        c.real,
                        atol=1e-3,
                    )
                    self.assertEqual(tuple(c.x.shape[1:]), (JANELA, hdfm.N_SUB))

    def test_vazamento_so_na_decomposicao_inteira(self):
        df = serie()
        x = df[FECHAMENTO].to_numpy()
        futuro = x.copy()
        futuro[limites(N)[1] :] *= 3  # preços do trecho de teste
        datas = pd.DatetimeIndex(df.index)
        for causal in (False, True):
            a, _, _ = hdfm.montar(x, datas, "B", causal, JANELA)
            b, _, _ = hdfm.montar(futuro, datas, "B", causal, JANELA)
            iguais = all(
                torch.equal(a[p].x, b[p].x) and torch.equal(a[p].y, b[p].y)
                for p in ("treino", "validacao")
            )
            # causal: o futuro não muda nada; inteira: muda o treino (é o vazamento)
            self.assertEqual(iguais, causal)


class TestTreino(unittest.TestCase):
    def test_tres_epocas_e_retomada(self):
        df = serie()
        for c in (
            cfg(),
            cfg(modelo="hdfm_minmax"),
            cfg(decomposicao="causal"),
        ):
            with (
                self.subTest(c["modelo"] + "_" + c["decomposicao"]),
                tempfile.TemporaryDirectory() as tmp,
            ):
                direto, retomado = Path(tmp, "direto"), Path(tmp, "retomado")
                r = executar(c, df, direto)
                self.assertLess(r["perda_treino_ultima"], r["perda_treino_primeira"])
                self.assertTrue(np.isfinite(r["validacao"]["rmse"]))
                executar(c | {"epocas": 2}, df, retomado)
                executar(c, df, retomado)
                pd.testing.assert_frame_equal(
                    pd.read_csv(direto / "previsoes_teste.csv"),
                    pd.read_csv(retomado / "previsoes_teste.csv"),
                )


if __name__ == "__main__":
    unittest.main()
