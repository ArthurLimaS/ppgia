"""Etapa 2a: o treino funciona? quanto custa uma época nos extremos da grade?

Roda `rodar.py` nos dois extremos de custo, cada um em CPU e em GPU, e imprime as
linhas da tabela do §6.4 do guia. Só vale na máquina do estudo: o tempo medido em
outra máquina prova que o código funciona, mas não entra no guia.

Cada execução grava em `execucoes/2a_<extremo>_<dispositivo>/`; rodar de novo pula as
que já terminaram.

Uso (a partir da raiz do repositório):
    uv run python redes_neurais_2026_2/projeto/medir_2a.py
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from rodar import EXECUCOES

EPOCAS = 3  # §4: a validação rápida tem no máximo 3 épocas
PRIMEIRO_ALVO_TREINO = 200  # D-17: os mesmos alvos da maior janela da grade
EXTREMOS = {  # entrada, janela, camadas
    "barata": ("univariado", 5, [16]),  # arquitetura `i`
    "cara": ("ohlcv", 200, [128, 64, 32]),  # arquitetura `a`
}


def executar(extremo: str, dispositivo: str) -> dict[str, Any] | None:
    entrada, janela, camadas = EXTREMOS[extremo]
    pasta = EXECUCOES / f"2a_{extremo}_{dispositivo}"
    print(f"\n=== {pasta.name} ===", flush=True)
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
            "--epocas",
            str(EPOCAS),
            "--dispositivo",
            dispositivo,
            "--pasta",
            str(pasta),
        ],
        check=False,
    ).returncode
    resumo = pasta / "metricas.json"
    if codigo != 0 or not resumo.exists():
        print(f"FALHOU: {pasta.name} (código {codigo})", flush=True)
        return None
    return json.loads(resumo.read_text(encoding="utf-8"))


def linha(nome: str, r: dict[str, Any]) -> str:
    """Uma linha da tabela do §6.4; a projeção é 1ª época + 99 demais."""
    projecao_h = (r["primeira_epoca_s"] + 99 * r["demais_epocas_media_s"]) / 3600
    memoria = ", ".join(
        f"{k.removesuffix('_mb')} {v:.0f} MB" for k, v in r["pico_de_memoria"].items()
    )
    return (
        f"| {nome} | 2a | {r['preprocessamento_s']:.2f} s "
        f"| {r['primeira_epoca_s']:.2f} s | {r['demais_epocas_media_s']:.2f} s "
        f"| {r['amostras_treino']} | {memoria} | {projecao_h:.2f} h |"
    )


if __name__ == "__main__":
    dispositivos = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    if len(dispositivos) == 1:
        print("CUDA indisponível: só a medida em CPU (R-02).", flush=True)

    linhas = []
    for extremo in EXTREMOS:
        for dispositivo in dispositivos:
            r = executar(extremo, dispositivo)
            if r is not None:
                nome = f"GRU, mais {extremo}, {dispositivo.upper()}"
                linhas.append(linha(nome, r))
                print(
                    f"    perda de treino: {r['perda_treino_primeira']:.3e} → "
                    f"{r['perda_treino_ultima']:.3e}  ({r['parametros']} parâmetros)"
                )

    print(
        "\n| Modelo | Etapa | Pré-proc. | 1ª época | Demais (média) | Instâncias "
        "| Pico de memória | 100 épocas |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    print("\n".join(linhas))
