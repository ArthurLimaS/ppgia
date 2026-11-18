"""Testes de fumaça do FinGAT (Etapa 3): universo, formas, máscara do grafo,
vazamento temporal, treino e retomada, com dados sintéticos.

Uso (a partir da raiz do repositório):
    uv run python -m unittest discover -s redes_neurais_2026_2/projeto/testes -v
"""

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from comum.janelas import limites
from comum.treino import prever, treinar
from modelos import custom_fingat
from rodar import executar
from test_fumaca import serie

N = 400
SETORES = ("A", "B")


def universo() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Dois setores com 7 candidatas completas cada (a de menor volume sai) e uma
    candidata incompleta. Cada ação é a senoide com outra fase, nível e volume.
    """
    dfs, linhas = {}, []
    for s, setor in enumerate(SETORES):
        for k in range(custom_fingat.POR_SETOR + 1):
            t = custom_fingat.PETR4 if (s, k) == (0, 0) else f"{setor}{k}.SA"
            df = serie(N + 3 * k + s)  # outra fase: começa mais tarde na senoide
            df = df.iloc[-N:].set_axis(serie(N).index)
            df[["Open", "High", "Low", "Close", "Adj Close"]] *= 1 + 0.1 * k
            df["Volume"] *= 10.0 - k  # volume decresce com k
            dfs[t] = df
            linhas.append({"ticker": t, "setor": setor})
    dfs["CURTA.SA"] = serie(N).iloc[10:]
    linhas.append({"ticker": "CURTA.SA", "setor": "A"})
    return dfs, pd.DataFrame(linhas)


def cfg(**mudancas: Any) -> dict[str, Any]:
    base = {
        "modelo": "fingat_artigo",
        "entrada": None,
        "janela": 15,
        "camadas": [8],
        "epocas": 3,
        "semente": 0,
        "dispositivo": "cpu",
        "primeiro_alvo_treino": None,
    }
    v = custom_fingat.VARIANTES[mudancas.get("modelo", base["modelo"])]
    return base | {"lote": v["lote"], "lr": v["lr"]} | mudancas


class TestUniverso(unittest.TestCase):
    def test_selecao(self):
        dfs, candidatas = universo()
        tickers, setores, excluidas = custom_fingat.selecionar(candidatas, dfs)
        self.assertEqual(len(tickers), 2 * custom_fingat.POR_SETOR)
        self.assertEqual(setores, [0] * 6 + [1] * 6)
        self.assertEqual(tickers[0], custom_fingat.PETR4)  # maior volume do setor A
        self.assertIn("incompleta", excluidas["CURTA.SA"])
        self.assertIn("volume", excluidas["A6.SA"])
        self.assertIn("volume", excluidas["B6.SA"])


class TestFormas(unittest.TestCase):
    def setUp(self):
        dfs, candidatas = universo()
        self.dados = {}
        for m in custom_fingat.VARIANTES:
            torch.manual_seed(0)
            d, modelo, _, _ = custom_fingat.preparar(cfg(modelo=m), dfs, candidatas)
            self.dados[m] = (d, modelo)

    def test_entrada_e_saida(self):
        for m, (d, modelo) in self.dados.items():
            x = d["treino"].x
            # repo: nível + retorno + 3 razões + 6 médias + setor em one-hot (2)
            # artigo: 5 níveis + retorno + 3 razões + 6 médias
            n_feat = 13 if m == "fingat_repo" else 15
            self.assertEqual(tuple(x.shape[1:]), (12, 3, 5, n_feat))
            self.assertEqual(tuple(modelo(x[:4]).shape), (4, 12, 2))

    def test_camadas_internas_do_artigo(self):
        d, modelo = self.dados["fingat_artigo"]
        formas: dict[str, tuple[int, ...]] = {}
        for nome in ("att_curto", "gat_intra", "att_a", "att_g", "gat_setor"):

            def guardar(_m: Any, _e: Any, s: torch.Tensor, nome: str = nome) -> None:
                formas[nome] = tuple(s.shape)

            getattr(modelo, nome).register_forward_hook(guardar)
        modelo(d["treino"].x[:4])
        self.assertEqual(formas["att_curto"], (4 * 12 * 3, 8))  # a: por ação e semana
        self.assertEqual(formas["gat_intra"], (4, 3, 12, 8))  # g: por semana
        self.assertEqual(formas["att_a"], (4 * 12, 8))  # τA
        self.assertEqual(formas["att_g"], (4 * 12, 8))  # τG
        self.assertEqual(formas["gat_setor"], (4, 2, 8))  # τ(π)


class TestGrafo(unittest.TestCase):
    def test_acao_nao_recebe_atencao_de_outro_setor(self):
        setores = [0] * 3 + [1] * 3
        for laco in (True, False):
            intra, _, _ = custom_fingat.grafos(setores, laco)
            gat = custom_fingat.GAT(4, artigo=not laco)
            x = torch.randn(2, 6, 4)
            y = x.clone()
            y[:, 3:] = torch.randn(2, 3, 4)  # muda só o setor 1
            torch.testing.assert_close(gat(x, intra)[:, :3], gat(y, intra)[:, :3])
            self.assertFalse(torch.allclose(gat(x, intra)[:, 3:], gat(y, intra)[:, 3:]))

    def test_laco_proprio(self):
        intra, entre, k = custom_fingat.grafos([0, 0, 1, 1], laco=False)
        self.assertEqual(k, 2)
        self.assertFalse(intra.diagonal().any() or entre.diagonal().any())
        self.assertTrue(
            custom_fingat.grafos([0, 0, 1, 1], laco=True)[0].diagonal().all()
        )


class TestVazamento(unittest.TestCase):
    def test_teste_nao_altera_treino_nem_validacao(self):
        dfs, candidatas = universo()
        tickers, setores, _ = custom_fingat.selecionar(candidatas, dfs)
        ini_teste = limites(N)[1]
        futuro = {}
        for t, df in dfs.items():
            df = df.copy()
            df.iloc[ini_teste:, :5] *= 3  # preços do trecho de teste
            futuro[t] = df
        for repo in (True, False):
            a, b = (
                custom_fingat.montar([d[t] for t in tickers], setores, repo, 15, 0)
                for d in (dfs, futuro)
            )
            for parte in ("treino", "validacao"):
                torch.testing.assert_close(a[parte].x, b[parte].x)
                torch.testing.assert_close(a[parte].y, b[parte].y)
            self.assertFalse(torch.allclose(a["teste"].y, b["teste"].y))


class TestTreino(unittest.TestCase):
    def test_sublote_da_o_mesmo_passo_que_o_lote_inteiro(self):
        dfs, candidatas = universo()
        previsoes = []
        for sublote in (None, 3):
            torch.manual_seed(0)
            d, modelo, extras, _ = custom_fingat.preparar(cfg(), dfs, candidatas)
            with tempfile.TemporaryDirectory() as tmp:
                extras["sublote"] = sublote
                treinar(
                    modelo, d["treino"], d["validacao"], Path(tmp), epocas=1, lote=8,
                    lr=1e-3, semente=0, dispositivo="cpu", **extras,
                )  # fmt: skip
            previsoes.append(prever(modelo, d["validacao"].x))
        torch.testing.assert_close(*previsoes, atol=1e-5, rtol=1e-4)

    def test_tres_epocas_e_retomada(self):
        dfs, candidatas = universo()
        for m in custom_fingat.VARIANTES:
            with self.subTest(m), tempfile.TemporaryDirectory() as tmp:
                direto, retomado = Path(tmp, "direto"), Path(tmp, "retomado")
                r = executar(cfg(modelo=m), dfs, direto, candidatas)
                self.assertLess(r["perda_treino_ultima"], r["perda_treino_primeira"])
                for f in (
                    "config.json",
                    "log.csv",
                    "ultimo.pt",
                    "melhor.pt",
                    "previsoes_teste.csv",
                    "previsoes_teste_todas.csv",
                ):
                    self.assertTrue((direto / f).exists(), f)
                executar(cfg(modelo=m, epocas=2), dfs, retomado, candidatas)
                executar(cfg(modelo=m), dfs, retomado, candidatas)
                pd.testing.assert_frame_equal(
                    pd.read_csv(direto / "previsoes_teste_todas.csv"),
                    pd.read_csv(retomado / "previsoes_teste_todas.csv"),
                )
                self.assertTrue(np.isfinite(r["validacao"]["acuracia_direcao"]))


if __name__ == "__main__":
    unittest.main()
