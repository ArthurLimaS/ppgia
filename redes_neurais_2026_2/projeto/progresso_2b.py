"""Progresso do estudo da GRU (Etapa 2b): o que terminou, o que está pela metade.

Lê só o que as execuções já gravaram em `execucoes/<nome>/`, então vale depois de
uma queda de energia. Grava o resumo em `execucoes/progresso_2b.csv` e imprime o comando
de cada execução que falta, na ordem da fila (semente, janela, entrada, arquitetura).
Rodar o comando de uma execução interrompida retoma do último checkpoint.

Uso (a partir da raiz do repositório):
    uv run python redes_neurais_2026_2/projeto/progresso_2b.py            # resumo
    uv run python redes_neurais_2026_2/projeto/progresso_2b.py --proximos 3
"""

import argparse
import itertools
import json
from typing import Any

import pandas as pd
from comum.janelas import ENTRADAS
from grade_gru import ARQUITETURAS, JANELAS, PRIMEIRO_ALVO_TREINO
from rodar import EXECUCOES, nome

EPOCAS = 100  # D-19
SEMENTES = (0, 1, 2)  # D-19; cada semente completa é um estudo inteiro
CSV = EXECUCOES / "progresso_2b.csv"


def fila() -> list[dict[str, Any]]:
    itens = []
    for semente, janela, entrada, arq in itertools.product(
        SEMENTES, JANELAS, ENTRADAS, ARQUITETURAS
    ):
        cfg = {
            "modelo": "gru",
            "entrada": entrada,
            "janela": janela,
            "camadas": ARQUITETURAS[arq],
            "primeiro_alvo_treino": PRIMEIRO_ALVO_TREINO,
            "semente": semente,
        }
        itens.append({**cfg, "arquitetura": arq, "pasta": EXECUCOES / nome(cfg)})
    return itens


def situacao(item: dict[str, Any]) -> dict[str, Any]:
    pasta = item["pasta"]
    linha: dict[str, Any] = {
        k: item[k] for k in ("semente", "janela", "entrada", "arquitetura")
    }
    resumo = pasta / "metricas.json"
    log = pasta / "log.csv"
    feitas = len(pd.read_csv(log)) if log.exists() else 0
    if resumo.exists():
        r = json.loads(resumo.read_text(encoding="utf-8"))
        feitas = r["epocas"]
        if feitas >= EPOCAS:
            linha |= {
                "situacao": "ok",
                "epocas_feitas": feitas,
                "melhor_epoca": r["melhor_epoca"],
                "rmse_validacao": r["validacao"]["rmse"],
                "rmse_ingenuo": r["validacao_ingenuo"]["rmse"],
                "tempo_total_s": r["tempo_total_s"],
            }
            return linha
    linha["epocas_feitas"] = feitas
    linha["situacao"] = "parcial" if feitas else "pendente"
    return linha


def comando(item: dict[str, Any]) -> str:
    camadas = ",".join(map(str, item["camadas"]))
    return (
        "uv run python redes_neurais_2026_2/projeto/rodar.py"
        f" --entrada {item['entrada']}"
        f" --janela {item['janela']} --camadas {camadas}"
        f" --primeiro-alvo-treino {PRIMEIRO_ALVO_TREINO} --semente {item['semente']}"
        f" --epocas {EPOCAS} --dispositivo cuda"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument(
        "--proximos", type=int, default=0, help="imprime os N próximos comandos"
    )
    a = p.parse_args()

    itens = fila()
    tabela = pd.DataFrame([situacao(i) for i in itens])
    tabela.to_csv(CSV, index=False, lineterminator="\n")

    por_semente = tabela.groupby(["semente", "situacao"]).size().unstack(fill_value=0)
    print(por_semente.to_string())
    ok = tabela[tabela.situacao == "ok"]
    horas = ok["tempo_total_s"].sum() / 3600 if len(ok) else 0.0
    print(f"\n{len(ok)}/{len(tabela)} ok; {horas:.2f} h de treino das ok → {CSV}")

    faltam = [i for i, s in zip(itens, tabela.situacao, strict=True) if s != "ok"]
    for item in faltam[: a.proximos]:
        print(
            f"\n# semente {item['semente']}, janela {item['janela']}, "
            f"{item['entrada']}, arquitetura {item['arquitetura']}"
        )
        print(comando(item))
