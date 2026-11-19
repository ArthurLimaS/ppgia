"""Testes de fumaça com dados sintéticos: formas, vazamento temporal, treino e retomada.

Uso (a partir da raiz do repositório):
    uv run python -m unittest discover -s redes_neurais_2026_2/projeto/testes -v
"""

import itertools
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from comum import janelas, metricas
from comum.gru import BlocoGRU
from rodar import executar

FATOR_DE_AJUSTE = 0.5  # Adj Close / Close da série sintética


def serie(n: int = 600) -> pd.DataFrame:
    """Senoide sobre um nível positivo: previsível, então uma GRU tem de aprendê-la."""
    t = np.arange(n)
    datas = pd.bdate_range("2020-01-01", periods=n)
    ajustado = 50 + 10 * np.sin(2 * np.pi * t / 20)
    bruto = ajustado / FATOR_DE_AJUSTE  # como na PETR4: o bruto é maior que o ajustado
    return pd.DataFrame(
        {
            "Open": 1.01 * bruto,
            "High": 1.02 * bruto,
            "Low": 0.98 * bruto,
            "Close": bruto,
            janelas.FECHAMENTO: ajustado,
            "Volume": 1e6 * (2 + np.cos(2 * np.pi * t / 7)),
        },
        index=datas,
    )


def cfg(**mudancas: Any) -> dict[str, Any]:
    base = {
        "modelo": "gru",
        "entrada": "univariado",
        "janela": 15,
        "camadas": [16],
        "epocas": 4,
        "lote": 64,
        "lr": 1e-2,
        "semente": 0,
        "dispositivo": "cpu",
        "primeiro_alvo_treino": None,
    }
    return base | mudancas


class Janelas(unittest.TestCase):
    def test_formas_e_valores(self):
        df = serie()
        f = df[janelas.FECHAMENTO].to_numpy()
        d = janelas.montar(df, "univariado", 15)
        self.assertEqual([len(d[p]) for p in d], [480 - 15, 60, 60])
        tr = d["treino"]
        self.assertEqual(tuple(tr.x.shape), (465, 15, 1))
        # primeira amostra: alvo na posição 15, janela = posições 0..14
        np.testing.assert_allclose(
            tr.x[0, :, 0].numpy(), f[0:15] / f[14] - 1, rtol=1e-5, atol=1e-6
        )
        self.assertAlmostEqual(tr.y[0].item(), f[15] / f[14] - 1, places=6)
        self.assertEqual(tr.real[0], f[15])
        self.assertEqual(tr.anterior[0], f[14])

    def test_ohlcv_formas_e_valores(self):
        """D-21: preços ajustados e normalizados como o fechamento; volume pela média"""
        df = serie()
        f = df[janelas.FECHAMENTO].to_numpy()
        v = df["Volume"].to_numpy()
        tr = janelas.montar(df, "ohlcv", 15)["treino"]
        self.assertEqual(tuple(tr.x.shape), (465, 15, 5))
        x = tr.x[0].numpy()  # alvo na posição 15, janela = posições 0..14
        for coluna, fator in enumerate((1.01, 1.02, 0.98, 1.0)):
            np.testing.assert_allclose(
                x[:, coluna], fator * f[0:15] / f[14] - 1, rtol=1e-5, atol=1e-6
            )
        np.testing.assert_allclose(
            x[:, 4], v[0:15] / v[0:15].mean() - 1, rtol=1e-5, atol=1e-6
        )
        self.assertAlmostEqual(
            tr.y[0].item(), f[15] / f[14] - 1, places=6
        )  # alvo igual

    def test_ohlcv_volume_com_media_um_em_toda_janela(self):
        x = janelas.montar(serie(), "ohlcv", 30)["validacao"].x[:, :, 4]
        np.testing.assert_allclose(x.mean(dim=1).numpy(), 0.0, atol=1e-5)

    def test_divisao_cronologica_sem_sobreposicao_de_alvos(self):
        d = janelas.montar(serie(), "univariado", 15)
        self.assertLess(d["treino"].datas[-1], d["validacao"].datas[0])
        self.assertLess(d["validacao"].datas[-1], d["teste"].datas[0])

    def test_sem_vazamento_temporal(self):
        """Alterar os pregões de teste não muda amostras de treino nem de validação."""
        df = serie()
        alterado = df.copy()
        ini_teste = janelas.limites(len(df))[1]
        alterado.iloc[ini_teste:] *= 3.0
        for entrada, janela in itertools.product(janelas.ENTRADAS, (5, 200)):
            a = janelas.montar(df, entrada, janela)
            b = janelas.montar(alterado, entrada, janela)
            chave = (entrada, janela)
            for parte in ("treino", "validacao"):
                self.assertTrue(torch.equal(a[parte].x, b[parte].x), (*chave, parte))
                self.assertTrue(torch.equal(a[parte].y, b[parte].y), (*chave, parte))
            self.assertFalse(torch.equal(a["teste"].y, b["teste"].y))

    def test_mesmos_alvos_para_todas_as_janelas(self):
        """D-17: com o primeiro alvo fixado, todas as janelas usam os mesmos dias."""
        df = serie()
        a = janelas.montar(df, "univariado", 5, primeiro_alvo_treino=200)
        b = janelas.montar(df, "univariado", 200, primeiro_alvo_treino=200)
        for parte in a:
            self.assertTrue(a[parte].datas.equals(b[parte].datas), parte)
            np.testing.assert_array_equal(a[parte].real, b[parte].real)

    def test_previsao_perfeita_reconstroi_o_preco(self):
        c = janelas.montar(serie(), "univariado", 15)["validacao"]
        previsto = janelas.para_preco(c.y.numpy().astype("float64"), c.anterior)
        np.testing.assert_allclose(previsto, c.real, rtol=1e-6)


class Metricas(unittest.TestCase):
    def test_valores_conhecidos(self):
        real, anterior = (
            np.array([10.0, 12.0, 11.0, 11.0]),
            np.array([9.0, 10.0, 12.0, 11.0]),
        )
        previsto = np.array(
            [11.0, 9.0, 11.5, 11.0]
        )  # direção: certa, errada, certa, certa
        m = metricas.calcular(real, previsto, anterior)
        self.assertAlmostEqual(m["mae"], (1 + 3 + 0.5 + 0) / 4)
        self.assertAlmostEqual(m["rmse"], np.sqrt((1 + 9 + 0.25 + 0) / 4))
        self.assertAlmostEqual(m["mape"], 100 * (1 / 10 + 3 / 12 + 0.5 / 11 + 0) / 4)
        self.assertAlmostEqual(m["acuracia_direcao"], 0.75)
        self.assertAlmostEqual(m["acmape"], 0.25 * 2 + m["mape"] / 100)
        self.assertAlmostEqual(metricas.calcular(real, real, anterior)["r2"], 1.0)

    def test_wilson(self):
        # 0 acertos em 10: [0; 0,2775], valor de tabela do intervalo de Wilson
        baixo, alto = metricas.wilson(0.0, 10)
        self.assertAlmostEqual(baixo, 0.0)
        self.assertAlmostEqual(alto, 0.2775, places=4)
        # simétrico em torno de 1/2, e contém a proporção
        b1, a1 = metricas.wilson(0.3, 125)
        _, a2 = metricas.wilson(0.7, 125)
        self.assertAlmostEqual(b1, 1 - a2)
        self.assertLess(b1, 0.3)
        self.assertLess(0.3, a1)


class Gru(unittest.TestCase):
    def test_formas_do_bloco(self):
        x = torch.zeros(7, 30, 1)
        for camadas in ([128, 64, 32], [16]):
            seq, ultimo = BlocoGRU(1, camadas)(x)
            self.assertEqual(tuple(seq.shape), (7, 30, camadas[-1]))
            self.assertEqual(tuple(ultimo.shape), (7, camadas[-1]))


class Treino(unittest.TestCase):
    def test_aprende_a_senoide(self):
        """O laço de treino funciona: tem de ganhar da referência "amanhã = hoje"."""
        casos = (
            ("univariado", [16], 15),
            ("univariado", [128, 64, 32], 30),
            ("ohlcv", [16], 15),
            ("ohlcv", [128, 64, 32], 30),
        )
        for entrada, camadas, janela in casos:
            with tempfile.TemporaryDirectory() as tmp:
                r = executar(
                    cfg(entrada=entrada, camadas=camadas, janela=janela, epocas=30),
                    serie(),
                    Path(tmp),
                )
            self.assertLess(r["perda_treino_ultima"], r["perda_treino_primeira"])
            self.assertLess(
                r["validacao"]["rmse"],
                0.5 * r["validacao_ingenuo"]["rmse"],
                (entrada, camadas),
            )

    def test_retomar_da_o_mesmo_resultado_de_rodar_direto(self):
        df = serie()
        with tempfile.TemporaryDirectory() as tmp:
            direto, retomado = Path(tmp) / "direto", Path(tmp) / "retomado"
            executar(cfg(epocas=4), df, direto)
            executar(cfg(epocas=2), df, retomado)  # "interrompida" na época 2
            executar(cfg(epocas=4), df, retomado)
            a = pd.read_csv(direto / "previsoes_teste.csv")
            b = pd.read_csv(retomado / "previsoes_teste.csv")
            pd.testing.assert_frame_equal(a, b)
            self.assertEqual(len(pd.read_csv(retomado / "log.csv")), 4)

    def test_configuracao_diferente_na_mesma_pasta_e_recusada(self):
        with tempfile.TemporaryDirectory() as tmp:
            executar(cfg(epocas=1), serie(), Path(tmp))
            with self.assertRaises(RuntimeError):
                executar(cfg(epocas=1, janela=30), serie(), Path(tmp))


if __name__ == "__main__":
    unittest.main()
