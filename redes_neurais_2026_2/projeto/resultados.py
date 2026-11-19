"""Tabelas de resultados da Etapa 7, regeneradas do zero a partir de `execucoes/`.

Lê as previsões de **teste** gravadas por cada execução da fila (`condicoes.json`),
calcula as métricas do `condicoes-do-estudo.md` §6 e grava:
- `<saída>.md`: comparação principal, vazamento do HDFM, FinGAT nas 48 ações, custo e
  execuções que faltam (seções 5.1 a 5.5 do `layout-relatorio.md`);
- `<saída>_por_execucao.csv`: uma linha por (configuração, semente) (Apêndice C).

Uso (a partir da raiz do repositório):
    uv run python redes_neurais_2026_2/projeto/resultados.py           # execução longa
    uv run python redes_neurais_2026_2/projeto/resultados.py --ensaio  # dados do ensaio
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from comum import metricas
from fila import itens
from rodar import EXECUCOES, nome

PRECO = ("rmse", "mae", "mape", "r2")
ROTULOS = {
    "gru": "GRU",
    "fingat_repo": "FinGAT (código dos autores)",
    "fingat_artigo": "Custom FinGAT (texto do artigo)",
    "hdfm_minmax": "HDFM A (min-max)",
    "hdfm_diferenca": "HDFM B (diferença)",
}


def rotulo(cfg: dict[str, Any]) -> str:
    r = ROTULOS[cfg["modelo"]]
    return f"{r}, {cfg['decomposicao']}" if cfg.get("decomposicao") else r


def por_execucao(base: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Métricas de teste de cada execução que terminou, e a lista das que faltam."""
    papeis = {
        (c["modelo"], c.get("decomposicao")): c["papel"]
        for c in json.loads(
            Path(__file__).with_name("condicoes.json").read_text(encoding="utf-8")
        )["configuracoes"]
    }
    linhas, faltam = [], []
    for cfg, _ in itens(None):
        pasta = base / nome(cfg)
        resumo, prev = pasta / "metricas.json", pasta / "previsoes_teste.csv"
        if not (resumo.exists() and prev.exists()):
            faltam.append({"execucao": pasta.name})
            continue
        r = json.loads(resumo.read_text(encoding="utf-8"))
        p = pd.read_csv(prev)
        real, anterior = p["real"].to_numpy(), p["anterior"].to_numpy()
        linha: dict[str, Any] = {
            "execucao": pasta.name,
            "configuracao": rotulo(cfg),
            "papel": papeis[(cfg["modelo"], cfg.get("decomposicao"))],
            "semente": cfg["semente"],
            "dias_teste": len(p),
        }
        if "prob_alta" in p:  # FinGAT: só direção (D-06)
            linha["acuracia_direcao"] = metricas.acuracia_direcao(
                real, anterior, p["prob_alta"].to_numpy() >= 0.5
            )
            t = pd.read_csv(pasta / "previsoes_teste_todas.csv")
            alta = (t["retorno_real"] >= 0).to_numpy()
            linha["acuracia_48_acoes"] = float(
                np.mean((t["prob_alta"].to_numpy() >= 0.5) == alta)
            )
            linha["acuracia_48_acoes_sinal_retorno"] = float(
                np.mean((t["retorno_previsto"].to_numpy() > 0) == alta)
            )
        else:
            m = metricas.calcular(real, p["previsto"].to_numpy(), anterior)
            linha |= {k: m[k] for k in (*PRECO, "acuracia_direcao")}
        linha |= {
            "parametros": r["parametros"],
            "epocas": r["epocas"],
            "melhor_epoca": r["melhor_epoca"],
            "s_por_epoca": r["demais_epocas_media_s"] or r["primeira_epoca_s"],
            "treino_total_s": r["tempo_total_s"],
            "vram_mb": r["pico_de_memoria"].get("vram_mb"),
        }
        linhas.append(linha)
    return pd.DataFrame(linhas), pd.DataFrame(faltam, columns=["execucao"])


def ingenuo(base: Path, execucoes: pd.DataFrame) -> dict[str, float]:
    """Referência "amanhã = hoje", nos mesmos dias de teste (todos usam os mesmos)."""
    p = pd.read_csv(base / execucoes["execucao"].iloc[0] / "previsoes_teste.csv")
    return metricas.ingenuo(p["real"].to_numpy(), p["anterior"].to_numpy())


def media_dp(serie: pd.Series, casas: int) -> str:
    if serie.isna().all():
        return "—"
    dp = serie.std(ddof=1) if len(serie) > 1 else float("nan")
    return f"{serie.mean():.{casas}f} ± {dp:.{casas}f}"


def com_wilson(acc: pd.Series, n: int) -> str:
    """Média ± desvio entre sementes, e o intervalo de Wilson de 95 % da média."""
    baixo, alto = metricas.wilson(float(acc.mean()), n)
    return f"{media_dp(acc, 3)} [{baixo:.3f}; {alto:.3f}]"


def tabela(df: pd.DataFrame, cols: dict[str, tuple[str, int]], n: int) -> str:
    """Uma linha por configuração; `cols`: coluna -> (título, casas decimais)."""
    cab = ["Configuração", "Sementes", *[t for t, _ in cols.values()]]
    linhas = ["| " + " | ".join(cab) + " |", "|" + "---|" * len(cab)]
    for conf, g in df.groupby("configuracao", sort=False):
        celulas = [str(conf), str(len(g))]
        for c, (_, casas) in cols.items():
            col = g[c] if c in g else None
            if col is None or col.isna().all():
                celulas.append("—")
            elif c == "acuracia_direcao":
                celulas.append(com_wilson(col, n))
            else:
                celulas.append(media_dp(col, casas))
        linhas.append("| " + " | ".join(celulas) + " |")
    return "\n".join(linhas)


def relatorio(base: Path, saida: Path) -> None:
    df, faltam = por_execucao(base)
    df.to_csv(
        saida.with_name(saida.stem + "_por_execucao.csv"),
        index=False,
        lineterminator="\n",
    )
    if df.empty:
        saida.write_text("Nenhuma execução terminada.\n", encoding="utf-8")
        return
    n = int(df["dias_teste"].iloc[0])
    ing = ingenuo(base, df)
    b, a = metricas.wilson(ing["acuracia_direcao"], n)
    metricas_cols = {
        "rmse": ("RMSE (R$)", 4),
        "mae": ("MAE (R$)", 4),
        "mape": ("MAPE (%)", 3),
        "r2": ("R²", 4),
        "acuracia_direcao": ("Direção [IC 95 % Wilson]", 3),
    }

    def filtro(mascara: Any) -> pd.DataFrame:
        return cast(pd.DataFrame, df[mascara])

    comparacao = filtro(df["papel"] == "comparacao")
    hdfm_b = filtro(df["configuracao"].str.startswith(ROTULOS["hdfm_diferenca"]))
    fingat = filtro(df["configuracao"].str.contains("FinGAT"))
    partes = [
        "# Resultados da Etapa 7 (teste)",
        "",
        f"Gerado em {datetime.now():%Y-%m-%d %H:%M} por `projeto/resultados.py`, a"
        f" partir de `{base.relative_to(EXECUCOES.parent)}/`. Métricas no **teste**"
        f" ({n} dias), média ± desvio-padrão entre sementes; na direção, entre"
        " colchetes, o intervalo de Wilson de 95 % para a acurácia média (Wilson,"
        " 1927; Brown, Cai e DasGupta, 2001). Condições: `condicoes-do-estudo.md`.",
        "",
        "## 5.1. Comparação principal",
        "",
        tabela(comparacao, metricas_cols, n),
        f"| Referência ingênua | — | {ing['rmse']:.4f} | {ing['mae']:.4f} | "
        f"{ing['mape']:.3f} | {ing['r2']:.4f} | {ing['acuracia_direcao']:.3f} "
        f"[{b:.3f}; {a:.3f}] |",
        "",
        "## 5.2. Vazamento da decomposição (HDFM B)",
        "",
        tabela(hdfm_b, metricas_cols, n),
        "",
        "## 5.3. FinGAT: código dos autores × texto do artigo",
        "",
        tabela(
            fingat,
            {
                "acuracia_direcao": ("PETR4 [IC 95 % Wilson]", 3),
                "acuracia_48_acoes": ("48 ações (prob. ≥ 0,5)", 3),
                "acuracia_48_acoes_sinal_retorno": ("48 ações (sinal do retorno)", 3),
            },
            n,
        ),
        "",
        "## 5.5. Custo",
        "",
        tabela(
            df,
            {
                "parametros": ("Parâmetros", 0),
                "s_por_epoca": ("s por época", 3),
                "treino_total_s": ("Treino total (s)", 1),
                "melhor_epoca": ("Época escolhida", 1),
                "vram_mb": ("VRAM (MB)", 0),
            },
            n,
        ),
        "",
        "## Execuções que faltam",
        "",
        "nenhuma" if faltam.empty else "\n".join(f"- {e}" for e in faltam["execucao"]),
        "",
    ]
    saida.write_text("\n".join(partes), encoding="utf-8", newline="\n")
    print(f"{len(df)} execuções, {len(faltam)} faltando -> {saida}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("--ensaio", action="store_true", help="lê execucoes/ensaio/")
    a = p.parse_args()
    base = EXECUCOES / "ensaio" if a.ensaio else EXECUCOES
    # a execução longa vai para junto do guia (versionado), como o resultados-2b.md
    saida = (
        base / "resultados.md" if a.ensaio else EXECUCOES.parents[1] / "resultados-7.md"
    )
    relatorio(base, saida)
