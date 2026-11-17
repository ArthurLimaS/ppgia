"""Laço de treino comum: tempo por época, log, checkpoint atômico e retomada."""

import csv
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from comum.janelas import Conjunto
from torch import nn


def semear(semente: int) -> None:
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)


def _gravar(objeto: dict[str, Any], caminho: Path) -> None:
    """
    Grava num temporário e renomeia: uma queda no meio não corrompe o checkpoint
    anterior.
    """
    tmp = caminho.with_suffix(".tmp")
    torch.save(objeto, tmp)
    os.replace(tmp, caminho)


@torch.no_grad()
def prever(modelo: nn.Module, x: torch.Tensor, lote: int = 1024) -> torch.Tensor:
    modelo.eval()
    return torch.cat([modelo(x[i : i + lote]) for i in range(0, len(x), lote)])


def treinar(
    modelo: nn.Module,
    treino: Conjunto,
    validacao: Conjunto,
    pasta: Path,
    epocas: int,
    lote: int,
    lr: float,
    semente: int,
    dispositivo: str,
) -> list[dict[str, Any]]:
    """
    Treina até `epocas`, retomando de `pasta/ultimo.pt` se existir. Devolve o log por
    época.

    Grava `ultimo.pt` (tudo o que a retomada precisa) a cada época e `melhor.pt` (pesos
    da época de menor perda de validação). Para com erro se a perda deixar de ser
    finita.
    """
    modelo.to(dispositivo)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=lr)
    perda_fn = nn.MSELoss()
    gerador = torch.Generator().manual_seed(semente)  # ordem dos lotes; sempre em CPU
    x_tr, y_tr = treino.x.to(dispositivo), treino.y.to(dispositivo)
    x_va, y_va = validacao.x.to(dispositivo), validacao.y.to(dispositivo)

    ultimo, melhor, log_csv = (
        pasta / "ultimo.pt",
        pasta / "melhor.pt",
        pasta / "log.csv",
    )
    inicio, melhor_val, log = 1, math.inf, []
    if ultimo.exists():
        ck = torch.load(ultimo, map_location=dispositivo, weights_only=False)
        modelo.load_state_dict(ck["modelo"])
        otimizador.load_state_dict(ck["otimizador"])
        gerador.set_state(ck["gerador"])
        inicio, melhor_val, log = ck["epoca"] + 1, ck["melhor_val"], ck["log"]

    for epoca in range(inicio, epocas + 1):
        t0 = time.perf_counter()
        modelo.train()
        soma = 0.0
        ordem = torch.randperm(len(treino), generator=gerador).to(dispositivo)
        for i in range(0, len(ordem), lote):
            idx = ordem[i : i + lote]
            perda = perda_fn(modelo(x_tr[idx]), y_tr[idx])
            otimizador.zero_grad()
            perda.backward()
            otimizador.step()
            soma += perda.item() * len(idx)
        perda_treino = soma / len(treino)
        perda_val = perda_fn(prever(modelo, x_va), y_va).item()
        if dispositivo.startswith("cuda"):
            torch.cuda.synchronize()
        segundos = time.perf_counter() - t0

        if not (math.isfinite(perda_treino) and math.isfinite(perda_val)):
            raise FloatingPointError(
                f"perda não finita na época {epoca} "
                f"(treino={perda_treino}, validação={perda_val}); "
                f"o último checkpoint bom continua em {pasta}"
            )

        log.append(
            {
                "epoca": epoca,
                "perda_treino": perda_treino,
                "perda_validacao": perda_val,
                "segundos": segundos,
                "terminou_em": datetime.now().isoformat(timespec="seconds"),
            }
        )
        if perda_val < melhor_val:
            melhor_val = perda_val
            _gravar({"modelo": modelo.state_dict(), "epoca": epoca}, melhor)
        _gravar(
            {
                "modelo": modelo.state_dict(),
                "otimizador": otimizador.state_dict(),
                "gerador": gerador.get_state(),
                "epoca": epoca,
                "melhor_val": melhor_val,
                "log": log,
            },
            ultimo,
        )
        with open(log_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(log[0]))
            w.writeheader()
            w.writerows(log)

        print(
            f"[{log[-1]['terminou_em']}] época {epoca}/{epocas}  "
            f"treino={perda_treino:.3e}  validação={perda_val:.3e}  {segundos:.2f}s",
            flush=True,
        )
    return log
