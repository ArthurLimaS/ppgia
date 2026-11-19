"""Figuras da apresentação, geradas a partir de `execucoes/` e `execucoes_relatorio/`.

- `slide_gru_arquiteturas.png`: RMSE de validação das 9 arquiteturas do estudo da
  Etapa 2 na entrada e janela escolhidas (OHLCV, 120), mais o ingênuo.
- `slide_perda_gru_otimizacao.png` e `slide_perda_gru_mais_dados.png`: perda por
  época de uma semente (a de RMSE de validação mediano) das duas tentativas na GRU.
- `curvas_gru_otimizacao.png` e `curvas_gru_mais_dados.png`: todas as sementes (e
  todos os lr), para mostrar na hora se perguntarem; não vão no slide.

Uso (a partir da raiz do repositório):
    uv run python redes_neurais_2026_2/projeto/figuras_slides.py
"""

import json
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from comum.janelas import montar
from dados.baixar import carregar
from figuras import MUDA, SERIES, TINTA, TINTA_2
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from rodar import EXECUCOES

RELATORIO = EXECUCOES.with_name("execucoes_relatorio")
OTIMIZACAO = RELATORIO / "gru_otimizacao"
MAIS_DADOS = RELATORIO / "gru_mais_dados_entrada"
SAIDA = EXECUCOES.parents[1] / "figuras"
CAMADAS = {  # guia, Etapa 2: as 9 arquiteturas
    "a": "128 → 64 → 32",
    "b": "64 → 32 → 16",
    "c": "128 → 64",
    "d": "64 → 32",
    "e": "32 → 16",
    "f": "128",
    "g": "64",
    "h": "32",
    "i": "16",
}


def virgula(x: float, casas: int = 4) -> str:
    return f"{x:.{casas}f}".replace(".", ",")


def tabela_arquiteturas() -> pd.DataFrame:
    g = pd.read_csv(EXECUCOES / "grade_gru.csv")
    g = cast(
        pd.DataFrame,
        g[(g["entrada"] == "ohlcv") & (g["janela"] == 120) & (g["situacao"] == "ok")],
    )
    assert len(g) == 9 * 3, f"esperava 9 arquiteturas × 3 sementes, achei {len(g)}"
    t = g.groupby("arquitetura")["rmse"].agg(["mean", "std"]).reset_index()
    t["arquitetura"] = [f"{a}: {CAMADAS[a]}" for a in t["arquitetura"]]
    ingenuo = float(g["rmse_ingenuo"].to_numpy()[0])
    t = pd.concat(
        [t, pd.DataFrame([{"arquitetura": "Ingênuo (amanhã = hoje)", "mean": ingenuo}])]
    )
    return t.sort_values("mean").reset_index(drop=True)


def figura_tabela(t: pd.DataFrame, saida: Path) -> None:
    linhas = [
        [a, virgula(m) + ("" if np.isnan(s) else f" ± {virgula(s)}")]
        for a, m, s in zip(t["arquitetura"], t["mean"], t["std"], strict=True)
    ]
    fig = plt.figure(figsize=(8, 0.5 * (len(linhas) + 1)))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")
    tab = ax.table(
        cellText=linhas,
        colLabels=["Arquitetura (camadas)", "RMSE de validação (R$)"],
        cellLoc="left",
        colLoc="left",
        bbox=Bbox.from_bounds(0, 0, 1, 1),
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(14)
    for (i, _), cel in tab.get_celld().items():
        cel.set_edgecolor("#ddd8cc")
        if i == 0:
            cel.set_text_props(weight="bold", color=TINTA)
        elif linhas[i - 1][0].startswith("Ingênuo"):
            cel.set_facecolor("#ece7db")
        elif linhas[i - 1][0].startswith("f:"):
            cel.set_text_props(weight="bold")
    fig.savefig(saida, dpi=200)
    plt.close(fig)


def prever_zero(alvos: list[np.ndarray]) -> float:
    """MSE de prever variação zero (o ingênuo, no espaço da perda)."""
    return float(np.mean(np.concatenate(alvos) ** 2))


def semente_mediana(pastas: list[Path]) -> Path:
    """A semente de RMSE de validação mediano (mesmo critério do figuras.py)."""
    rmse = [
        (
            json.loads((p / "metricas.json").read_text(encoding="utf-8"))["validacao"][
                "rmse"
            ],
            p,
        )
        for p in pastas
    ]
    return sorted(rmse)[(len(rmse) - 1) // 2][1]


def painel(ax: Axes, pasta: Path, zero_tr: float, zero_va: float) -> None:
    lg = pd.read_csv(pasta / "log.csv")
    e = json.loads((pasta / "metricas.json").read_text(encoding="utf-8"))[
        "melhor_epoca"
    ]
    ax.plot(lg["epoca"], lg["perda_treino"], color=SERIES[0], lw=1.2)
    ax.plot(lg["epoca"], lg["perda_validacao"], color=SERIES[1], lw=1.2)
    ax.axhline(zero_tr, color=SERIES[0], ls=":", lw=1.4)
    ax.axhline(zero_va, color=SERIES[1], ls=":", lw=1.4)
    ax.plot(e, lg.loc[lg["epoca"] == e, "perda_validacao"], "o", color=TINTA, ms=8)
    ax.set_yscale("log")


def legenda(ax: Axes, fonte: int, altura: float = 1.02) -> None:
    itens = [
        Line2D([], [], color=SERIES[0], lw=2, label="treino"),
        Line2D([], [], color=SERIES[1], lw=2, label="validação (PETR4)"),
        Line2D([], [], color=MUDA, ls=":", lw=2, label="prever variação zero"),
        Line2D([], [], color=TINTA, marker="o", ls="", ms=8, label="época escolhida"),
    ]
    ax.legend(
        handles=itens,
        ncol=4,
        loc="lower left",
        bbox_to_anchor=(0, altura),
        fontsize=fonte,
        handlelength=1.6,
        columnspacing=1.2,
    )


def figura_painel(pasta: Path, zero_tr: float, zero_va: float, saida: Path) -> None:
    plt.rcParams.update({"font.size": 16})
    fig, ax = plt.subplots(figsize=(10, 4.2))
    painel(ax, pasta, zero_tr, zero_va)
    legenda(ax, 14)
    ax.set_xlabel("época", color=TINTA_2)
    ax.set_ylabel("perda (MSE, log)", color=TINTA_2)
    fig.savefig(saida, dpi=160, bbox_inches="tight")
    plt.close(fig)
    plt.rcParams.update({"font.size": 9})


def figura_grade(
    linhas: list[tuple[str, list[Path]]], zero_tr: float, zero_va: float, saida: Path
) -> None:
    n = max(len(p) for _, p in linhas)
    fig, eixos = plt.subplots(
        len(linhas), n, figsize=(4 * n, 3.2 * len(linhas)), sharex=True, squeeze=False
    )
    for i, (rotulo, pastas) in enumerate(linhas):
        for j, pasta in enumerate(pastas):
            painel(eixos[i, j], pasta, zero_tr, zero_va)
            eixos[i, j].set_title(f"{rotulo} · semente {j}", loc="left")
    for ax in eixos[-1]:
        ax.set_xlabel("época")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    legenda(
        eixos[0, 0], 10, altura=1.12
    )  # depois do tight_layout: a legenda fica fora do cálculo
    fig.savefig(saida, dpi=110)
    plt.close(fig)


def main() -> None:
    SAIDA.mkdir(parents=True, exist_ok=True)

    t = tabela_arquiteturas()
    print(t.to_string(index=False))
    figura_tabela(t, SAIDA / "slide_gru_arquiteturas.png")

    petr4 = montar(carregar("PETR4.SA"), "ohlcv", 120)
    zero_va = prever_zero([petr4["validacao"].y.numpy()])

    # 1) mais épocas e outros lr: treino só com a PETR4
    zero_tr = prever_zero([petr4["treino"].y.numpy()])
    por_lr = [
        (f"lr {lr}", sorted(OTIMIZACAO.glob(f"*_lr{lr}_s?")))
        for lr in ("0.001", "0.01", "0.1")
    ]
    escolhida = semente_mediana(por_lr[0][1])
    print("gru_otimizacao, semente mostrada:", escolhida.name)
    figura_painel(escolhida, zero_tr, zero_va, SAIDA / "slide_perda_gru_otimizacao.png")
    figura_grade(por_lr, zero_tr, zero_va, SAIDA / "curvas_gru_otimizacao.png")

    # 2) mais dados: treino com as ações do universo
    pastas = sorted(MAIS_DADOS.glob("*_s?"))
    acoes = json.loads((pastas[0] / "config.json").read_text(encoding="utf-8"))["acoes"]
    zero_tr = prever_zero(
        [montar(carregar(a), "ohlcv", 120)["treino"].y.numpy() for a in acoes]
    )
    escolhida = semente_mediana(pastas)
    print("gru_mais_dados_entrada, semente mostrada:", escolhida.name)
    figura_painel(escolhida, zero_tr, zero_va, SAIDA / "slide_perda_gru_mais_dados.png")
    figura_grade(
        [("66 ações", pastas)], zero_tr, zero_va, SAIDA / "curvas_gru_mais_dados.png"
    )
    print("figuras em", SAIDA)


if __name__ == "__main__":
    main()
