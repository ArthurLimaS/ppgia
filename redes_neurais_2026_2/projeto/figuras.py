"""Figuras do relatório (`layout-relatorio.md` §4.3 e §5), regeneradas do zero a partir
de `execucoes/` e do snapshot.

- `previsto_real.png` (§5.1, §5.4): fechamento real × previsto no teste, para a GRU e
  os HDFM causais; a semente de cada modelo é a de RMSE de **validação** mediano.
- `vazamento.png` (§5.2): a mesma subsérie do HDFM decomposta só com o passado até `t`
  e com a série inteira: o que a segunda mostra antes de `t` depende do futuro.
- `faixa_treino.png` (§5.4): a subsérie de tendência (Co-IMF3) e a faixa que o treino
  cobre; a validação e o teste passam dela.
- `estudo_gru.png` (§4.3): RMSE de validação por entrada, janela e arquitetura no
  estudo da Etapa 2, contra o ingênuo.

Uso (a partir da raiz do repositório):
    uv run python redes_neurais_2026_2/projeto/figuras.py           # execução longa
    uv run python redes_neurais_2026_2/projeto/figuras.py --ensaio  # dados do ensaio
"""

import argparse
import json
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from comum.janelas import FECHAMENTO, limites
from dados.baixar import carregar
from fila import itens
from modelos import hdfm
from rodar import EXECUCOES, nome

# paleta categórica de referência (skill dataviz, validada: CVD ΔE 9,2) e tinta neutra
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
TINTA, TINTA_2, MUDA, GRADE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": TINTA_2,
        "axes.titlecolor": TINTA,
        "axes.titlesize": 11,
        "axes.grid": True,
        "grid.color": GRADE,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUDA,
        "ytick.color": MUDA,
        "legend.frameon": False,
        "font.size": 9,
    }
)


def semente_mediana(base: Path, modelo: str, decomposicao: str | None) -> Path | None:
    """Pasta da semente de RMSE de validação mediano (o teste não escolhe nada)."""
    pastas = []
    for cfg, _ in itens(None):
        if cfg["modelo"] == modelo and cfg["decomposicao"] == decomposicao:
            r = base / nome(cfg) / "metricas.json"
            if r.exists():
                rmse = json.loads(r.read_text(encoding="utf-8"))["validacao"]["rmse"]
                pastas.append((rmse, base / nome(cfg)))
    if not pastas:
        return None
    pastas.sort()
    return pastas[(len(pastas) - 1) // 2][1]


def previsto_real(base: Path, saida: Path) -> None:
    grupos = [
        (
            "GRU e HDFM B (causal)",
            [("gru", None, "GRU"), ("hdfm_diferenca", "causal", "HDFM B")],
        ),
        (
            "HDFM A (causal): a normalização literal",
            [("hdfm_minmax", "causal", "HDFM A")],
        ),
    ]
    fig, eixos = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    for ax, (titulo, modelos) in zip(eixos, grupos, strict=True):
        real = None
        for i, (m, d, rot) in enumerate(modelos):
            pasta = semente_mediana(base, m, d)
            if pasta is None:
                continue
            p = pd.read_csv(pasta / "previsoes_teste.csv", parse_dates=["data"])
            real = p
            cor = SERIES[i]
            ax.plot(
                p["data"],
                p["previsto"],
                color=cor,
                lw=1.4,
                ls=["-", "--"][i],
                label=rot,
            )
        if real is not None:
            ax.plot(
                real["data"], real["real"], color=TINTA, lw=1.6, label="Real", zorder=3
            )
        ax.set_title(titulo, loc="left")
        ax.set_ylabel("Fechamento ajustado (R$)")
        ax.legend(loc="upper left", ncols=3)
    fig.suptitle(
        "PETR4 no teste: real e previsto (semente de RMSE de validação mediano)",
        x=0.01,
        ha="left",
        fontsize=12,
        color=TINTA,
    )
    fig.tight_layout()
    fig.savefig(saida / "previsto_real.png", dpi=200)
    plt.close(fig)


def vazamento(saida: Path) -> None:
    x = carregar()[FECHAMENTO].to_numpy(np.float64)
    t = limites(len(x))[1] - 1  # último dia da validação
    inteira = hdfm.decompor_cacheado(x, hdfm.PARAMS, hdfm.CACHE)
    passado = hdfm.decompor_cacheado(x[: t + 1], hdfm.PARAMS, hdfm.CACHE)
    dias = np.arange(t - 59, t + 1)
    fig, eixos = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, k, titulo in zip(
        eixos, (10, 11), ("Co-IMF2", "Co-IMF3 (tendência)"), strict=True
    ):
        ax.plot(
            dias - t,
            passado[k, dias],
            color=SERIES[0],
            lw=1.6,
            label="só o passado (causal)",
        )
        ax.plot(
            dias - t,
            inteira[k, dias],
            color=SERIES[1],
            lw=1.6,
            ls="--",
            label="série inteira (como no artigo)",
        )
        ax.axvline(0, color=MUDA, lw=0.8)
        ax.set_title(titulo, loc="left")
        ax.set_xlabel("pregões em relação a t (último dia da validação)")
    eixos[0].set_ylabel("valor da subsérie (R$)")
    eixos[1].legend(loc="lower right")  # canto vazio; à esquerda cobria a linha
    fig.suptitle(
        "A mesma subsérie até o dia t: com a série inteira, ela já conhece o futuro",
        x=0.01,
        ha="left",
        fontsize=12,
        color=TINTA,
    )
    fig.tight_layout()
    fig.savefig(saida / "vazamento.png", dpi=200)
    plt.close(fig)


def faixa_treino(saida: Path) -> None:
    df = carregar()
    x = df[FECHAMENTO].to_numpy(np.float64)
    ini_val, ini_teste = limites(len(x))
    tendencia = hdfm.decompor_cacheado(x, hdfm.PARAMS, hdfm.CACHE)[11]
    datas = pd.DatetimeIndex(df.index)
    lo, hi = tendencia[:ini_val].min(), tendencia[:ini_val].max()
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.axhspan(lo, hi, color=SERIES[0], alpha=0.10, lw=0)
    ax.annotate(
        "faixa do treino (o min-max da normalização A)",
        (datas[30], hi),
        xytext=(0, 4),
        textcoords="offset points",
        fontsize=8,
        color=TINTA_2,
    )
    ax.plot(datas, tendencia, color=SERIES[0], lw=1.6)
    for pos, rot in ((ini_val, "validação"), (ini_teste, "teste")):
        ax.axvline(cast(float, datas[pos]), color=MUDA, lw=0.8, ls=":")
        ax.annotate(
            rot,
            (datas[pos], ax.get_ylim()[0]),
            xytext=(3, 4),
            textcoords="offset points",
            fontsize=8,
            color=TINTA_2,
        )
    ax.set_ylabel("Co-IMF3 (R$)")
    ax.set_title(
        "HDFM: a subsérie de tendência sai da faixa que o treino cobre",
        loc="left",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(saida / "faixa_treino.png", dpi=200)
    plt.close(fig)


def estudo_gru(saida: Path) -> None:
    g = pd.read_csv(EXECUCOES / "grade_gru.csv")
    g = g[g["situacao"] == "ok"]
    ingenuo = float(g["rmse_ingenuo"].iloc[0])
    por_comb = cast(
        pd.DataFrame,
        g.groupby(["entrada", "janela", "arquitetura"], as_index=False)["rmse"].mean(),
    )  # média das 3 sementes de cada combinação
    fig, eixos = plt.subplots(
        1, 3, figsize=(10, 3.4), sharey=True, gridspec_kw={"width_ratios": [2, 6, 9]}
    )
    for ax, eixo, titulo in zip(
        eixos,
        ("entrada", "janela", "arquitetura"),
        ("Entrada", "Janela", "Arquitetura"),
        strict=True,
    ):
        m = por_comb.groupby(eixo)["rmse"].agg(["mean", "std"]).reset_index()
        rot = m[eixo].astype(str)
        ax.errorbar(
            rot,
            m["mean"],
            yerr=m["std"],
            fmt="o",
            ms=5,
            color=SERIES[0],
            ecolor=SERIES[0],
            elinewidth=1,
            capsize=3,
        )
        ax.axhline(ingenuo, color=MUDA, lw=1, ls="--")
        ax.set_title(titulo, loc="left")
    eixos[-1].annotate(
        f"ingênuo ({ingenuo:.4f})",
        (1, ingenuo),
        xycoords=("axes fraction", "data"),
        xytext=(-2, 4),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color=TINTA_2,
    )
    eixos[0].set_ylabel("RMSE de validação (R$)")
    fig.suptitle(
        "Estudo da GRU (Etapa 2): média ± desvio entre as combinações dos outros eixos",
        x=0.01,
        ha="left",
        fontsize=12,
        color=TINTA,
    )
    fig.tight_layout()
    fig.savefig(saida / "estudo_gru.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("--ensaio", action="store_true", help="lê execucoes/ensaio/")
    a = p.parse_args()
    base = EXECUCOES / "ensaio" if a.ensaio else EXECUCOES
    saida = base / "figuras" if a.ensaio else EXECUCOES.parents[1] / "figuras"
    saida.mkdir(parents=True, exist_ok=True)
    for f in (
        lambda: previsto_real(base, saida),
        lambda: vazamento(saida),
        lambda: faixa_treino(saida),
        lambda: estudo_gru(saida),
    ):
        f()
    print("figuras em", saida)
