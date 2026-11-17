"""Ponto de entrada único: treina um modelo e grava configuração, log, checkpoints
e previsões.

A validação rápida e a execução longa são o mesmo comando com outro `--epocas`.
Rodar de novo o mesmo comando retoma do último checkpoint; se a execução já
terminou, não faz nada.

Uso (a partir da raiz do repositório):
    uv run python redes_neurais_2026_2/projeto/rodar.py \\
        --modelo gru --janela 15 --camadas 64,32 --epocas 3
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from comum import janelas, metricas
from comum.treino import prever, semear, treinar
from modelos.gru_baseline import GRUBaseline

EXECUCOES = Path(__file__).with_name("execucoes")
MODELOS = {"gru": GRUBaseline}
N_ENTRADAS = {"univariado": 1}


def nome(cfg: dict[str, Any]) -> str:
    camadas = "-".join(map(str, cfg["camadas"]))
    alvo = (
        ""
        if cfg["primeiro_alvo_treino"] is None
        else f"_t{cfg['primeiro_alvo_treino']}"
    )
    base = f"{cfg['modelo']}_{cfg['entrada']}_j{cfg['janela']}_c{camadas}"
    return f"{base}{alvo}_s{cfg['semente']}"


def _pico_de_memoria_mb(dispositivo: str) -> dict[str, float]:
    pico: dict[str, float] = {}
    if dispositivo.startswith("cuda"):
        pico["vram_mb"] = torch.cuda.max_memory_allocated() / 2**20
    try:
        import resource  # Linux/macOS

        pico["ram_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**10
    except ImportError:  # Windows
        try:
            import psutil

            # `peak_wset` só existe no Windows; o pyright checa como se fosse Linux
            pico["ram_mb"] = (
                psutil.Process().memory_info().peak_wset  # pyright: ignore[reportAttributeAccessIssue]
                / 2**20
            )
        except ImportError:
            pass
    return pico


def executar(
    cfg: dict[str, Any], df: pd.DataFrame | None = None, pasta: Path | None = None
) -> dict[str, Any]:
    """Treina conforme `cfg` e devolve o resumo (também gravado em `metricas.json`).

    `df` substitui o snapshot (usado pelos testes com dados sintéticos).
    """
    pasta = pasta or EXECUCOES / nome(cfg)
    pasta.mkdir(parents=True, exist_ok=True)
    resumo_json, config_json = pasta / "metricas.json", pasta / "config.json"

    fixos = {
        k: v for k, v in cfg.items() if k != "epocas"
    }  # retomar com mais épocas é permitido
    if config_json.exists():
        anterior = json.loads(config_json.read_text(encoding="utf-8"))
        if {k: anterior.get(k) for k in fixos} != fixos:
            raise RuntimeError(
                f"{pasta} tem uma execução com outra configuração; "
                "apague a pasta ou use outra."
            )
        if resumo_json.exists():
            resumo = json.loads(resumo_json.read_text(encoding="utf-8"))
            if resumo["epocas"] >= cfg["epocas"]:
                print(f"{pasta.name}: já terminou ({resumo['epocas']} épocas).")
                return resumo

    snapshot = None
    if df is None:
        from dados.baixar import MANIFESTO, carregar

        df = carregar()
        snapshot = json.loads(MANIFESTO.read_text(encoding="utf-8"))["PETR4.SA"][
            "sha256"
        ]
    config_json.write_text(
        json.dumps(
            {**cfg, "snapshot_sha256": snapshot, "torch": torch.__version__}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )

    semear(cfg["semente"])
    dados = janelas.montar(
        df, cfg["entrada"], cfg["janela"], cfg["primeiro_alvo_treino"]
    )
    modelo = MODELOS[cfg["modelo"]](N_ENTRADAS[cfg["entrada"]], cfg["camadas"])
    log = treinar(
        modelo,
        dados["treino"],
        dados["validacao"],
        pasta,
        epocas=cfg["epocas"],
        lote=cfg["lote"],
        lr=cfg["lr"],
        semente=cfg["semente"],
        dispositivo=cfg["dispositivo"],
    )

    melhor = torch.load(
        pasta / "melhor.pt", map_location=cfg["dispositivo"], weights_only=False
    )
    modelo.load_state_dict(melhor["modelo"])
    previsoes = {}
    for parte in ("validacao", "teste"):
        c = dados[parte]
        variacao = (
            prever(modelo, c.x.to(cfg["dispositivo"])).cpu().numpy().astype("float64")
        )
        previsoes[parte] = janelas.para_preco(variacao, c.anterior)
        pd.DataFrame(
            {
                "data": c.datas,
                "anterior": c.anterior,
                "real": c.real,
                "previsto": previsoes[parte],
            }
        ).to_csv(pasta / f"previsoes_{parte}.csv", index=False, lineterminator="\n")

    # O resumo só traz a validação: o teste não escolhe nada e só é avaliado na Etapa 7.
    val = dados["validacao"]
    segundos = [e["segundos"] for e in log]
    resumo = {
        **{
            k: cfg[k]
            for k in (
                "modelo",
                "entrada",
                "janela",
                "camadas",
                "semente",
                "dispositivo",
            )
        },
        "epocas": len(log),
        "melhor_epoca": melhor["epoca"],
        "parametros": sum(p.numel() for p in modelo.parameters()),
        "amostras_treino": len(dados["treino"]),
        "amostras_validacao": len(val),
        "primeira_epoca_s": segundos[0],
        "demais_epocas_media_s": sum(segundos[1:]) / len(segundos[1:])
        if len(segundos) > 1
        else None,
        "tempo_total_s": sum(segundos),
        "pico_de_memoria": _pico_de_memoria_mb(cfg["dispositivo"]),
        "perda_treino_primeira": log[0]["perda_treino"],
        "perda_treino_ultima": log[-1]["perda_treino"],
        "validacao": metricas.calcular(val.real, previsoes["validacao"], val.anterior),
        "validacao_ingenuo": metricas.ingenuo(val.real, val.anterior),
    }
    resumo_json.write_text(json.dumps(resumo, indent=2) + "\n", encoding="utf-8")
    return resumo


def argumentos() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("--modelo", choices=MODELOS, default="gru")
    p.add_argument("--entrada", choices=janelas.ENTRADAS, default="univariado")
    p.add_argument("--janela", type=int, default=15)
    p.add_argument(
        "--camadas",
        type=lambda s: [int(v) for v in s.split(",")],
        default=[64, 32],
        help="tamanhos das camadas da GRU, ex.: 128,64,32",
    )
    p.add_argument("--epocas", type=int, default=3)
    p.add_argument("--lote", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--semente", type=int, default=0)
    p.add_argument("--dispositivo", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument(
        "--primeiro-alvo-treino",
        type=int,
        default=None,
        help="posição do primeiro alvo de treino (D-17); "
        "padrão: o primeiro que a janela alcança",
    )
    p.add_argument(
        "--pasta",
        type=Path,
        default=None,
        help="padrão: execucoes/<nome da configuração>",
    )
    return p


if __name__ == "__main__":
    a = vars(argumentos().parse_args())
    pasta = a.pop("pasta")
    if a["dispositivo"] == "auto":
        a["dispositivo"] = "cuda" if torch.cuda.is_available() else "cpu"
    resumo = executar(a, pasta=pasta)
    json.dump(resumo, sys.stdout, indent=2)
    print()
