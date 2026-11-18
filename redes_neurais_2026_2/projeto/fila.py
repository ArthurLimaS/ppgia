"""Fila da Etapa 7: cada (configuração, semente) de `condicoes.json`, em sequência,
cada uma num processo de `rodar.py`, para uma falha não derrubar as outras.

Rodar de novo pula o que terminou e retoma o que foi interrompido (do último
checkpoint). O log da fila, com data, hora e previsão de término, vai para
`<base>/fila.log`; a saída de cada execução, para `<pasta da execução>/saida.log`; a
situação de cada item, para `<base>/fila_situacao.csv`.

Uso (a partir da raiz do repositório):
    # ensaio geral: tudo com 1 época, em execucoes/ensaio/
    uv run python redes_neurais_2026_2/projeto/fila.py --ensaio
    # execução longa
    uv run python redes_neurais_2026_2/projeto/fila.py
"""

import argparse
import io
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from rodar import EXECUCOES, nome

CONDICOES = Path(__file__).with_name("condicoes.json")
CUSTO_FIXO_S = (
    15.0  # abrir o processo, carregar dados e gravar previsões (medido ~10-16 s)
)


def itens(epocas: int | None) -> list[tuple[dict[str, Any], float]]:
    """(configuração completa, segundos estimados) na ordem do `condicoes.json`."""
    c = json.loads(CONDICOES.read_text(encoding="utf-8"))
    comum = c["comum"] | ({"epocas": epocas} if epocas else {})
    saida = []
    for conf in c["configuracoes"]:
        for semente in c["sementes"]:
            cfg = {
                "modelo": conf["modelo"],
                "entrada": conf.get("entrada"),
                "decomposicao": conf.get("decomposicao"),
                "janela": comum["janela"],
                "camadas": comum["camadas"],
                "epocas": comum["epocas"],
                "lote": conf["lote"],
                "lr": conf["lr"],
                "semente": semente,
                "dispositivo": comum["dispositivo"],
                "primeiro_alvo_treino": None,
            }
            saida.append((cfg, comum["epocas"] * conf["s_por_epoca"] + CUSTO_FIXO_S))
    return saida


def terminou(pasta: Path, epocas: int) -> bool:
    resumo = pasta / "metricas.json"
    return (
        resumo.exists()
        and json.loads(resumo.read_text(encoding="utf-8"))["epocas"] >= epocas
    )


def comando(cfg: dict[str, Any], pasta: Path) -> list[str]:
    args = [sys.executable, "-u", str(Path(__file__).with_name("rodar.py"))]
    for k in ("modelo", "entrada", "decomposicao", "janela", "epocas", "lote", "lr"):
        if cfg[k] is not None:
            args += [f"--{k}", str(cfg[k])]
    args += ["--camadas", ",".join(map(str, cfg["camadas"]))]
    args += ["--semente", str(cfg["semente"]), "--dispositivo", cfg["dispositivo"]]
    return args + ["--pasta", str(pasta)]


def main() -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument(
        "--ensaio", action="store_true", help="1 época, em execucoes/ensaio/"
    )
    a = p.parse_args()
    # Windows: com a saída redirecionada, o console usa cp1252 e quebra num caractere
    # fora dele; a fila não pode cair por causa de um print
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    base = EXECUCOES / "ensaio" if a.ensaio else EXECUCOES
    base.mkdir(parents=True, exist_ok=True)
    fila = itens(1 if a.ensaio else None)

    # duas filas ao mesmo tempo disputariam a GPU (e a memória dela)
    trava = base / "fila.trava"
    try:
        fd = os.open(trava, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"Já há uma fila rodando ({trava}). Se não houver, apague esse arquivo.")
        return 2
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)

    log = open(base / "fila.log", "a", encoding="utf-8")

    def registrar(texto: str) -> None:
        linha = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {texto}"
        print(linha, flush=True)
        log.write(linha + "\n")
        log.flush()

    situacao: list[dict[str, Any]] = []
    try:
        pendentes = [
            (cfg, s) for cfg, s in fila if not terminou(base / nome(cfg), cfg["epocas"])
        ]
        registrar(
            f"fila {'de ensaio ' if a.ensaio else ''}com {len(fila)} execuções; "
            f"{len(fila) - len(pendentes)} já terminadas"
        )
        restante = sum(s for _, s in pendentes)
        for n, (cfg, estimativa) in enumerate(fila, 1):
            pasta = base / nome(cfg)
            if terminou(pasta, cfg["epocas"]):
                situacao.append(
                    {"execucao": pasta.name, "situacao": "ok (já terminada)"}
                )
                continue
            fim = datetime.now() + timedelta(seconds=restante)
            registrar(
                f"{n}/{len(fila)} {pasta.name}: começa "
                f"(previsão de término da fila: {fim:%d/%m %H:%M})"
            )
            pasta.mkdir(parents=True, exist_ok=True)
            with open(pasta / "saida.log", "a", encoding="utf-8") as saida:
                proc = subprocess.Popen(
                    comando(cfg, pasta),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=os.environ | {"PYTHONIOENCODING": "utf-8"},  # filho em UTF-8
                )
                assert proc.stdout is not None
                for linha in proc.stdout:  # na tela e no arquivo
                    print(linha, end="", flush=True)
                    saida.write(linha)
                codigo = proc.wait()
            restante -= estimativa
            ok = codigo == 0 and terminou(pasta, cfg["epocas"])
            situacao.append(
                {
                    "execucao": pasta.name,
                    "situacao": "ok" if ok else f"falhou (código {codigo})",
                }
            )
            registrar(f"{n}/{len(fila)} {pasta.name}: {situacao[-1]['situacao']}")
            pd.DataFrame(situacao).to_csv(
                base / "fila_situacao.csv", index=False, lineterminator="\n"
            )
        pd.DataFrame(situacao).to_csv(
            base / "fila_situacao.csv", index=False, lineterminator="\n"
        )
        falhas = [s for s in situacao if not s["situacao"].startswith("ok")]
        registrar(f"fim: {len(situacao) - len(falhas)}/{len(situacao)} ok")
        for f in falhas:
            registrar(f"FALHOU: {f['execucao']} (ver {f['execucao']}/saida.log)")
        return 1 if falhas else 0
    finally:
        log.close()
        trava.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
