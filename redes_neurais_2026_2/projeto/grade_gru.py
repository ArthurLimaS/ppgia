"""Fila do estudo da GRU (Etapa 2b): entrada × janela × arquitetura × semente.

Cada combinação é um processo separado de `rodar.py`, para uma falha não derrubar
as outras. Rodar de novo retoma as interrompidas e pula as que terminaram. No fim
grava `execucoes/grade_gru.csv`, só com métricas de validação.

Uso (a partir da raiz do repositório):
    uv run python redes_neurais_2026_2/projeto/grade_gru.py \\
        --grade reduzida --epocas 100 --sementes 0,1,2
"""

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from comum.janelas import ENTRADAS
from rodar import EXECUCOES, nome

JANELAS = [5, 15, 30, 60, 120, 200]
ARQUITETURAS = {
    "a": [128, 64, 32],  # HDFM
    "b": [64, 32, 16],
    "c": [128, 64],
    "d": [64, 32],
    "e": [32, 16],
    "f": [128],
    "g": [64],
    "h": [32],
    "i": [16],  # FinGAT
}
REDUZIDA = ("a", "d", "i")
PRIMEIRO_ALVO_TREINO = max(JANELAS)  # D-17: todas as janelas treinam nos mesmos alvos

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("--grade", choices=["reduzida", "completa"], default="reduzida")
    p.add_argument("--epocas", type=int, required=True)
    p.add_argument(
        "--sementes", type=lambda s: [int(v) for v in s.split(",")], default=[0]
    )
    p.add_argument("--dispositivo", choices=["auto", "cpu", "cuda"], default="auto")
    a = p.parse_args()

    arquiteturas = REDUZIDA if a.grade == "reduzida" else tuple(ARQUITETURAS)
    sementes: list[int] = a.sementes
    # semente por fora: cada semente completa é um estudo inteiro (mesma ordem do
    # progresso_2b.py)
    fila = list(itertools.product(sementes, JANELAS, ENTRADAS, arquiteturas))
    linhas = []
    for n, (semente, janela, entrada, arq) in enumerate(fila, 1):
        camadas = ARQUITETURAS[arq]
        cfg = {
            "modelo": "gru",
            "entrada": entrada,
            "janela": janela,
            "camadas": camadas,
            "primeiro_alvo_treino": PRIMEIRO_ALVO_TREINO,
            "semente": semente,
        }
        print(f"\n=== {n}/{len(fila)}: {nome(cfg)} ===", flush=True)
        codigo = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("rodar.py")),
                "--entrada",
                entrada,
                "--janela",
                str(janela),
                "--camadas",
                ",".join(map(str, camadas)),
                "--primeiro-alvo-treino",
                str(PRIMEIRO_ALVO_TREINO),
                "--semente",
                str(semente),
                "--epocas",
                str(a.epocas),
                "--dispositivo",
                a.dispositivo,
            ],
            check=False,  # a falha é registrada na tabela, não interrompe a fila
        ).returncode

        linha = {
            "arquitetura": arq,
            "entrada": entrada,
            "janela": janela,
            "semente": semente,
        }
        resumo = EXECUCOES / nome(cfg) / "metricas.json"
        if codigo == 0 and resumo.exists():
            r = json.loads(resumo.read_text(encoding="utf-8"))
            linha |= {"situacao": "ok", "camadas": "-".join(map(str, camadas))}
            linha |= {
                k: r[k]
                for k in ("parametros", "epocas", "melhor_epoca", "amostras_treino")
            }
            linha |= {
                k: r[k]
                for k in ("primeira_epoca_s", "demais_epocas_media_s", "tempo_total_s")
            }
            linha |= r["validacao"]
            linha["rmse_ingenuo"] = r["validacao_ingenuo"]["rmse"]
        else:
            linha["situacao"] = f"falhou (código {codigo})"
            print(f"FALHOU: {nome(cfg)}", flush=True)
        linhas.append(linha)
        pd.DataFrame(linhas).to_csv(
            EXECUCOES / "grade_gru.csv", index=False, lineterminator="\n"
        )

    tabela = pd.DataFrame(linhas)
    ok = (tabela.situacao == "ok").sum()
    print(f"\n{ok}/{len(tabela)} execuções ok → {EXECUCOES / 'grade_gru.csv'}")
