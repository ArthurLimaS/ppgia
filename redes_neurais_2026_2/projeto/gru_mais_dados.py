"""Experimento exploratório: a GRU da Etapa 7 treinada com as ações do universo.csv.

Mesma configuração da GRU do estudo (ohlcv, janela 120, [128], lote 64, Adam 0,001),
mas o treino junta as amostras de todas as ações que têm exatamente os pregões da
PETR4. A divisão é a mesma, por data, para todas as ações; a época é escolhida pela
validação da PETR4 (os mesmos 125 dias), para que a única diferença seja o treino.
Grava também as métricas da época escolhida na validação de todas as ações.

Uso (a partir da raiz do repositório):
    uv run python redes_neurais_2026_2/projeto/gru_mais_dados.py --semente 0
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from comum import janelas, metricas
from comum.treino import prever, semear, treinar
from dados.baixar import UNIVERSO, carregar
from modelos.gru_baseline import GRUBaseline

DESTINO = Path(__file__).with_name("execucoes_relatorio") / "gru_mais_dados_entrada"
PETR4 = "PETR4.SA"
ENTRADA, JANELA, CAMADAS, LOTE, LR = "ohlcv", 120, [128], 64, 1e-3
COLUNAS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def acoes() -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Ações do universo com os mesmos pregões da PETR4 e sem dado faltando."""
    ref = carregar(PETR4)
    usadas, excluidas = {PETR4: ref}, {}
    for t in pd.read_csv(UNIVERSO)["ticker"]:
        if t == PETR4:
            continue
        try:
            df = carregar(t)
        except FileNotFoundError:
            excluidas[t] = "sem snapshot"
            continue
        if not df.index.equals(ref.index):
            excluidas[t] = f"pregões diferentes dos da PETR4 ({len(df)} × {len(ref)})"
        elif df[COLUNAS].isna().to_numpy().any():
            excluidas[t] = "dado faltando"
        else:
            usadas[t] = df
    return usadas, excluidas


def juntar(partes: list[janelas.Conjunto]) -> janelas.Conjunto:
    return janelas.Conjunto(
        x=torch.cat([p.x for p in partes]),
        y=torch.cat([p.y for p in partes]),
        anterior=np.concatenate([p.anterior for p in partes]),
        real=np.concatenate([p.real for p in partes]),
        datas=pd.DatetimeIndex(np.concatenate([p.datas.to_numpy() for p in partes])),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("--semente", type=int, required=True)
    p.add_argument("--epocas", type=int, default=1000)
    p.add_argument("--destino", type=Path, default=DESTINO)
    a = p.parse_args()
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"

    dfs, excluidas = acoes()
    conj = {t: janelas.montar(df, ENTRADA, JANELA) for t, df in dfs.items()}
    treino = juntar([c["treino"] for c in conj.values()])
    val_todas = juntar([c["validacao"] for c in conj.values()])
    val, teste = conj[PETR4]["validacao"], conj[PETR4]["teste"]
    # nenhum alvo de treino (de nenhuma ação) cai no período de validação ou de teste
    inicio_val = val.datas.to_numpy().min()
    assert (
        treino.datas.to_numpy().max() < inicio_val == val_todas.datas.to_numpy().min()
    )

    nome = f"gru_{ENTRADA}_j{JANELA}_c{CAMADAS[0]}_e{a.epocas}_{len(dfs)}acoes"
    pasta = a.destino / f"{nome}_s{a.semente}"
    pasta.mkdir(parents=True, exist_ok=True)
    cfg: dict[str, Any] = {
        "modelo": "gru",
        "entrada": ENTRADA,
        "janela": JANELA,
        "camadas": CAMADAS,
        "epocas": a.epocas,
        "lote": LOTE,
        "lr": LR,
        "semente": a.semente,
        "dispositivo": dispositivo,
        "validacao_escolhe_epoca": PETR4,
        "acoes": list(dfs),
        "excluidas": excluidas,
    }
    (pasta / "config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    semear(a.semente)
    modelo = GRUBaseline(janelas.ENTRADAS[ENTRADA], CAMADAS)
    log = treinar(
        modelo,
        treino,
        val,
        pasta,
        epocas=a.epocas,
        lote=LOTE,
        lr=LR,
        semente=a.semente,
        dispositivo=dispositivo,
    )
    melhor = torch.load(
        pasta / "melhor.pt", map_location=dispositivo, weights_only=False
    )
    modelo.load_state_dict(melhor["modelo"])

    def variacao(c: janelas.Conjunto) -> np.ndarray:
        return prever(modelo, c.x.to(dispositivo)).cpu().numpy().astype("float64")

    previsto: dict[str, np.ndarray] = {}
    for parte, c in (("validacao", val), ("teste", teste)):
        previsto[parte] = janelas.para_preco(variacao(c), c.anterior)
        pd.DataFrame(
            {
                "data": c.datas,
                "anterior": c.anterior,
                "real": c.real,
                "previsto": previsto[parte],
            }
        ).to_csv(pasta / f"previsoes_{parte}.csv", index=False, lineterminator="\n")

    # todas as ações: só no espaço da variação (o R$ de ações diferentes não se soma)
    r, y = variacao(val_todas), val_todas.y.numpy().astype("float64")
    segundos = [e["segundos"] for e in log]
    resumo = {
        "acoes": len(dfs),
        "amostras_treino": len(treino),
        "amostras_validacao": len(val),
        "amostras_validacao_todas": len(val_todas),
        "epocas": len(log),
        "melhor_epoca": melhor["epoca"],
        "primeira_epoca_s": segundos[0],
        "demais_epocas_media_s": float(np.mean(segundos[1:]))
        if len(segundos) > 1
        else None,
        "tempo_total_s": sum(segundos),
        "perda_treino_primeira": log[0]["perda_treino"],
        "perda_treino_ultima": log[-1]["perda_treino"],
        "validacao": metricas.calcular(val.real, previsto["validacao"], val.anterior),
        "validacao_ingenuo": metricas.ingenuo(val.real, val.anterior),
        "validacao_todas": {
            "mse_variacao": float(np.mean((r - y) ** 2)),
            "mse_variacao_ingenuo": float(np.mean(y**2)),  # ingênuo: variação zero
            "acuracia_direcao": float(np.mean((r >= 0) == (y >= 0))),
            "acuracia_direcao_ingenuo": float(np.mean(y >= 0)),  # ingênuo: sempre alta
        },
    }
    (pasta / "metricas.json").write_text(
        json.dumps(resumo, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(resumo, indent=2))


if __name__ == "__main__":
    main()
