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
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from comum import janelas, metricas
from comum.treino import prever, semear, treinar
from modelos import custom_fingat, hdfm
from modelos.gru_baseline import GRUBaseline

EXECUCOES = Path(__file__).with_name("execucoes")
# lote, lr (e a decomposição do HDFM) de cada modelo, se a linha de comando não os dá
PADROES: dict[str, dict[str, Any]] = {
    "gru": {"lote": 64, "lr": 1e-3},
    **{
        m: {"lote": v["lote"], "lr": v["lr"]}
        for m, v in custom_fingat.VARIANTES.items()
    },
    # HDFM, Tabela 2: lote 64, Adam 0,001
    **{
        m: {"lote": 64, "lr": 1e-3, "decomposicao": "inteira"}
        for m in hdfm.NORMALIZACOES
    },
}


def nome(cfg: dict[str, Any]) -> str:
    camadas = "-".join(map(str, cfg["camadas"]))
    alvo = (
        ""
        if cfg["primeiro_alvo_treino"] is None
        else f"_t{cfg['primeiro_alvo_treino']}"
    )
    partes = (
        cfg["modelo"],
        cfg["entrada"],
        cfg.get("decomposicao"),
        f"j{cfg['janela']}",
        f"c{camadas}",
    )
    base = "_".join(
        p for p in partes if p
    )  # só a GRU tem `entrada`; só o HDFM, decomposição
    return f"{base}{alvo}_s{cfg['semente']}"


def _pico_de_memoria_mb(dispositivo: str) -> dict[str, float]:
    pico: dict[str, float] = {}
    if dispositivo.startswith("cuda"):
        pico["vram_mb"] = torch.cuda.max_memory_allocated() / 2**20
    # `sys.platform` (e não try/except ImportError): o pyright entende e checa cada ramo
    # só no sistema dele
    if sys.platform != "win32":
        import resource

        pico["ram_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**10
    else:
        try:
            import psutil

            # `peak_wset` só existe no Windows; os stubs do psutil não o declaram
            pico["ram_mb"] = (
                psutil.Process().memory_info().peak_wset  # pyright: ignore[reportAttributeAccessIssue]
                / 2**20
            )
        except ImportError:
            pass
    return pico


def executar(
    cfg: dict[str, Any],
    df: pd.DataFrame | dict[str, pd.DataFrame] | None = None,
    pasta: Path | None = None,
    candidatas: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Treina conforme `cfg` e devolve o resumo (também gravado em `metricas.json`).

    `df` substitui o snapshot e `candidatas` o `universo.csv` (testes com dados
    sintéticos); no FinGAT, `df` é um dicionário ticker -> DataFrame.
    """
    e_fingat = cfg["modelo"] in custom_fingat.VARIANTES
    e_hdfm = cfg["modelo"] in hdfm.NORMALIZACOES
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

    semear(cfg["semente"])
    t0 = time.perf_counter()
    extras: dict[str, Any] = {}
    fusao: dict[str, Any] = {}
    registro: dict[str, Any]
    if e_fingat:
        assert not isinstance(df, pd.DataFrame)
        dados, modelo, extras, registro = custom_fingat.preparar(cfg, df, candidatas)
    elif e_hdfm:
        assert not isinstance(df, dict)
        dados, modelo, fusao, registro = hdfm.preparar(cfg, df)
    else:
        snapshot = None
        if df is None:
            from dados.baixar import MANIFESTO, carregar

            df = carregar()
            snapshot = json.loads(MANIFESTO.read_text(encoding="utf-8"))["PETR4.SA"][
                "sha256"
            ]
        assert isinstance(df, pd.DataFrame)
        dados = janelas.montar(
            df, cfg["entrada"], cfg["janela"], cfg["primeiro_alvo_treino"]
        )
        modelo = GRUBaseline(janelas.ENTRADAS[cfg["entrada"]], cfg["camadas"])
        registro = {"snapshot_sha256": snapshot}
    preprocessamento_s = time.perf_counter() - t0
    config_json.write_text(
        json.dumps(
            {**cfg, **registro, "torch": torch.__version__},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
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
        **extras,
    )

    melhor = torch.load(
        pasta / "melhor.pt", map_location=cfg["dispositivo"], weights_only=False
    )
    modelo.load_state_dict(melhor["modelo"])
    previsoes: dict[str, np.ndarray] = {}
    for parte in ("validacao", "teste"):
        c = dados[parte]
        saida = (
            prever(modelo, c.x.to(cfg["dispositivo"])).cpu().numpy().astype("float64")
        )
        if e_fingat:  # saida [dias, ações, (retorno, prob. de alta)]
            previsoes[parte] = saida
            tickers = registro["universo"]
            coluna, nome_coluna = (
                saida[:, tickers.index(custom_fingat.PETR4), 1],
                "prob_alta",
            )
            pd.DataFrame(
                {
                    "data": np.repeat(c.datas, len(tickers)),
                    "ticker": np.tile(tickers, len(c)),
                    "retorno_real": c.y[..., 0].numpy().ravel(),
                    "retorno_previsto": saida[..., 0].ravel(),
                    "prob_alta": saida[..., 1].ravel(),
                }
            ).to_csv(
                pasta / f"previsoes_{parte}_todas.csv",
                index=False,
                lineterminator="\n",
            )
        elif e_hdfm:  # saida [dias, 12 subséries], normalizada: desfaz e soma
            coluna = previsoes[parte] = fusao[parte](saida)
            nome_coluna = "previsto"
        else:
            coluna = previsoes[parte] = janelas.para_preco(saida, c.anterior)
            nome_coluna = "previsto"
        pd.DataFrame(
            {
                "data": c.datas,
                "anterior": c.anterior,
                "real": c.real,
                nome_coluna: coluna,
            }
        ).to_csv(pasta / f"previsoes_{parte}.csv", index=False, lineterminator="\n")

    # O resumo só traz a validação: o teste não escolhe nada e só é avaliado na Etapa 7.
    val = dados["validacao"]
    if e_fingat:  # só direção (D-06)
        s, alta_real = previsoes["validacao"], val.y[..., 1].numpy() == 1
        alta_petr4 = s[:, registro["universo"].index(custom_fingat.PETR4), 1] >= 0.5
        metricas_val = {
            "acuracia_direcao": metricas.acuracia_direcao(
                val.real, val.anterior, alta_petr4
            ),
            "acuracia_direcao_todas": float(np.mean((s[..., 1] >= 0.5) == alta_real)),
            # a ACC do train.py do repo usa o sinal do retorno previsto, não a prob.
            "acuracia_direcao_todas_sinal_retorno": float(
                np.mean((s[..., 0] > 0) == alta_real)
            ),
        }
        ingenuo = {
            "acuracia_direcao": metricas.ingenuo(val.real, val.anterior)[
                "acuracia_direcao"
            ]
        }
    else:
        metricas_val = metricas.calcular(val.real, previsoes["validacao"], val.anterior)
        ingenuo = metricas.ingenuo(val.real, val.anterior)
    segundos = [e["segundos"] for e in log]
    resumo = {
        **{
            k: cfg.get(k)
            for k in (
                "modelo",
                "entrada",
                "decomposicao",
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
        "preprocessamento_s": preprocessamento_s,
        **{
            k: registro[k] for k in ("decomposicoes", "decomposicao_s") if k in registro
        },
        "primeira_epoca_s": segundos[0],
        "demais_epocas_media_s": sum(segundos[1:]) / len(segundos[1:])
        if len(segundos) > 1
        else None,
        "tempo_total_s": sum(segundos),
        "pico_de_memoria": _pico_de_memoria_mb(cfg["dispositivo"]),
        "perda_treino_primeira": log[0]["perda_treino"],
        "perda_treino_ultima": log[-1]["perda_treino"],
        "validacao": metricas_val,
        "validacao_ingenuo": ingenuo,
    }
    resumo_json.write_text(json.dumps(resumo, indent=2) + "\n", encoding="utf-8")
    return resumo


def argumentos() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("--modelo", choices=PADROES, default="gru")
    p.add_argument(
        "--entrada",
        choices=janelas.ENTRADAS,
        default="univariado",
        help="só a GRU; o FinGAT usa as features dele",
    )
    p.add_argument(
        "--decomposicao",
        choices=hdfm.DECOMPOSICOES,
        default=None,
        help="só o HDFM (D-10): série inteira (vaza) ou causal; padrão: inteira",
    )
    p.add_argument("--janela", type=int, default=15)
    p.add_argument(
        "--camadas",
        type=lambda s: [int(v) for v in s.split(",")],
        default=[64, 32],
        help="tamanhos das camadas da GRU, ex.: 128,64,32",
    )
    p.add_argument("--epocas", type=int, default=3)
    p.add_argument("--lote", type=int, default=None, help="padrão: o do modelo")
    p.add_argument("--lr", type=float, default=None, help="padrão: o do modelo")
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
    a |= {k: v for k, v in PADROES[a["modelo"]].items() if a[k] is None}
    if a["modelo"] != "gru":
        a["entrada"] = None
    if a["modelo"] not in hdfm.NORMALIZACOES:
        a["decomposicao"] = None
    if a["dispositivo"] == "auto":
        a["dispositivo"] = "cuda" if torch.cuda.is_available() else "cpu"
    resumo = executar(a, pasta=pasta)
    json.dump(resumo, sys.stdout, indent=2)
    print()
