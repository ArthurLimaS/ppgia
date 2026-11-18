"""HDFM (Li et al., 2024): CEEMDAN -> entropia amostral -> K-means -> VMD na Co-IMF de
maior entropia -> uma GRU por subsérie -> soma das previsões.

Duas normalizações (D-14), que são dois modelos do estudo:
- `hdfm_minmax` (A): min-max por subsérie ajustado no treino (Eq. 20 ao pé da letra);
- `hdfm_diferenca` (B): por janela, subtrai o último valor de cada subsérie e divide
  pelo desvio-padrão do passo de um dia dela no treino (a D-20 adaptada).

Duas decomposições (D-10 c), para medir o vazamento:
- `inteira`: uma decomposição da série toda, como o artigo aparenta (o valor de uma
  subsérie em t depende de pontos depois de t);
- `causal`: a amostra com alvo t lê a janela da decomposição de [0..t-1]; o alvo de cada
  subsérie vem da decomposição de [0..t]. Nenhuma entrada vê o futuro.

Uso da conferência da decomposição (entropias, cotovelo, reconstrução, figura):
    uv run python redes_neurais_2026_2/projeto/modelos/hdfm.py
"""

import hashlib
import json
import os
import sys
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from comum.janelas import FECHAMENTO, Conjunto, limites  # noqa: E402
from modelos.gru_baseline import GRUBaseline  # noqa: E402

PARAMS: dict[str, Any] = {
    "trials": 100,  # D-11: padrões da EMD-signal (o artigo não informa)
    "epsilon": 0.005,
    "m": 2,  # D-12: entropia amostral
    "r": 0.2,  # x desvio-padrão
    "grupos": 3,  # D-12: K do K-means (o do artigo nos três índices)
    "vmd_k": 10,  # D-13: padrões usuais da vmdpy (o artigo não informa)
    "vmd_alpha": 2000,
    "vmd_tau": 0.0,
    "semente": 0,  # fixa: a decomposição é a mesma para todas as sementes de treino
}
N_SUB = PARAMS["vmd_k"] + PARAMS["grupos"] - 1  # 10 modos + 2 Co-IMFs = 12
CACHE = Path(__file__).resolve().parents[1] / "execucoes" / "cache_hdfm"
NORMALIZACOES = {"hdfm_minmax": "A", "hdfm_diferenca": "B"}
DECOMPOSICOES = ("inteira", "causal")


# ---------------------------------------------------------------- decomposição


def entropia_amostral(x: np.ndarray, m: int, r: float) -> float:
    """SampEn = ln(B/A): B (A) = pares de trechos de m (m+1) pontos a distância de
    Chebyshev <= r·desvio, sem contar o par do trecho com ele mesmo. Os dois contam os
    mesmos N-m inícios, como na definição usual.
    """
    tol = r * x.std()

    def pares(k: int) -> int:
        t = np.lib.stride_tricks.sliding_window_view(x, k)[: len(x) - m]
        d = np.abs(t[:, None, :] - t[None, :, :]).max(axis=-1)
        return int(((d <= tol).sum() - len(t)) // 2)

    a, b = pares(m + 1), pares(m)
    if a == 0:
        raise ValueError("entropia amostral indefinida: nenhum par de m+1 pontos")
    return float(np.log(b / a))


def decompor(serie: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    """[N_SUB, len(serie)]: os modos da VMD (do mais lento ao mais rápido) e as
    Co-IMFs restantes (da maior para a menor entropia). A soma das linhas é a série.
    """
    from PyEMD import CEEMDAN  # pyright: ignore[reportMissingTypeStubs]
    from sklearn.cluster import KMeans  # pyright: ignore[reportMissingTypeStubs]
    from vmdpy import VMD  # pyright: ignore[reportMissingTypeStubs]

    c = CEEMDAN(trials=p["trials"], epsilon=p["epsilon"], parallel=False)
    c.noise_seed(p["semente"])
    imfs = c(serie)  # IMFs + resíduo: a soma é a série
    if len(imfs) < p["grupos"]:
        raise ValueError(
            f"a CEEMDAN deu {len(imfs)} componentes; o K-means pede {p['grupos']}"
        )
    ent = np.array([entropia_amostral(i, p["m"], p["r"]) for i in imfs])
    km = KMeans(p["grupos"], n_init=10, random_state=p["semente"])  # pyright: ignore[reportArgumentType]
    rotulos = km.fit_predict(ent[:, None])
    ordem = sorted(range(p["grupos"]), key=lambda g: -ent[rotulos == g].mean())
    co = np.stack([imfs[rotulos == g].sum(axis=0) for g in ordem])  # Co-IMF1 primeiro

    # A vmdpy descarta o último ponto de uma série ímpar; aqui sai o primeiro (o mais
    # antigo), para o fim da série, que é o que a janela lê, continuar decomposto.
    impar = len(serie) % 2
    modos, _, omega = VMD(
        co[0][impar:], p["vmd_alpha"], p["vmd_tau"], p["vmd_k"], 0, 1, 1e-7
    )
    modos = modos[np.argsort(omega[-1])]  # a vmdpy não ordena; aqui, por frequência
    modos = np.pad(modos, ((0, 0), (impar, 0)))
    modos[-1] += co[0] - modos.sum(
        axis=0
    )  # D-22: o resíduo da VMD vai para o modo mais rápido
    return np.vstack([modos, co[1:]])


def _cacheado(serie: np.ndarray, p: dict[str, Any], pasta: Path) -> np.ndarray:
    chave = hashlib.sha256(serie.tobytes() + json.dumps(p, sort_keys=True).encode())
    arq = pasta / f"{chave.hexdigest()[:24]}.npy"
    if arq.exists():
        return np.load(arq)
    d = decompor(serie, p)
    pasta.mkdir(parents=True, exist_ok=True)
    tmp = arq.with_name(arq.name + f".{os.getpid()}.tmp")
    with open(tmp, "wb") as f:  # np.save(caminho) acrescentaria ".npy" ao temporário
        np.save(f, d)
    os.replace(tmp, arq)
    return d


def _trabalho(args: tuple[np.ndarray, dict[str, Any], Path]) -> np.ndarray:
    return _cacheado(*args)


def decomposicoes(
    serie: np.ndarray, causal: bool, primeiro: int, p: dict[str, Any], pasta: Path
) -> tuple[Callable[[int], np.ndarray], dict[str, Any]]:
    """`D(t)` = decomposição que conhece a série até a posição t (inclusive).

    `inteira`: sempre a da série toda. `causal`: a de `serie[:t+1]`, para t de
    `primeiro - 1` ao fim, calculadas em paralelo (um processo por núcleo) e em cache.
    """
    t0 = time.perf_counter()
    if not causal:
        d = _cacheado(serie, p, pasta)
        return (lambda _t: d), {
            "decomposicoes": 1,
            "decomposicao_s": time.perf_counter() - t0,
        }
    fins = range(primeiro - 1, len(serie))
    tarefas = [(serie[: t + 1], p, pasta) for t in fins]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        ds = dict(zip(fins, ex.map(_trabalho, tarefas, chunksize=4), strict=True))
    return ds.__getitem__, {
        "decomposicoes": len(ds),
        "decomposicao_s": time.perf_counter() - t0,
    }


# ---------------------------------------------------------------- amostras


def montar(
    fechamento: np.ndarray,
    datas: pd.DatetimeIndex,
    normalizacao: str,
    causal: bool,
    janela: int,
    p: dict[str, Any] | None = None,
    pasta: Path | None = None,
) -> tuple[
    dict[str, Conjunto], dict[str, Callable[[np.ndarray], np.ndarray]], dict[str, Any]
]:
    """Treino, validação e teste pela posição do alvo (D-02), as estatísticas da
    normalização só do treino, e, por parte, a função que leva a saída da rede
    [amostras, 12] ao fechamento previsto (desfaz a normalização e soma: a fusão).
    """
    p = p or PARAMS
    n = len(fechamento)
    ini_val, ini_teste = limites(n)
    D, registro = decomposicoes(fechamento, causal, janela, p, pasta or CACHE)

    def bruto(alvos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.stack([D(t - 1)[:, t - janela : t].T for t in alvos])  # [a, janela, 12]
        y = np.stack([D(t)[:, t] for t in alvos])  # [a, 12]
        return x, y

    partes = {
        "treino": np.arange(janela, ini_val),
        "validacao": np.arange(ini_val, ini_teste),
        "teste": np.arange(ini_teste, n),
    }
    brutos = {k: bruto(v) for k, v in partes.items()}
    x_tr, y_tr = brutos["treino"]
    if normalizacao == "A":  # Eq. 20, por subsérie
        lo = np.minimum(x_tr.min(axis=(0, 1)), y_tr.min(axis=0))
        esc = np.maximum(x_tr.max(axis=(0, 1)), y_tr.max(axis=0)) - lo
    else:  # B: passo de um dia de cada subsérie, no treino
        lo = None
        esc = (y_tr - x_tr[:, -1]).std(axis=0)

    dados: dict[str, Conjunto] = {}
    desfazer: dict[str, Callable[[np.ndarray], np.ndarray]] = {}
    for parte, alvos in partes.items():
        x, y = brutos[parte]
        base = lo if lo is not None else x[:, -1]  # [12] na A; [amostras, 12] na B
        base_x = base if lo is not None else base[:, None]
        dados[parte] = Conjunto(
            x=torch.tensor((x - base_x) / esc, dtype=torch.float32),
            y=torch.tensor((y - base) / esc, dtype=torch.float32),
            anterior=fechamento[alvos - 1],
            real=fechamento[alvos],
            datas=pd.DatetimeIndex(datas[alvos]),
        )
        desfazer[parte] = lambda s, base=base: (base + s * esc).sum(axis=1)
    return dados, desfazer, registro


# ---------------------------------------------------------------- modelo


class HDFM(nn.Module):
    """Uma GRU (bloco compartilhado + linear, a mesma da baseline) por subsérie, sem
    parâmetro em comum: treinar o conjunto com a MSE média das 12 dá a cada uma o
    gradiente que ela teria sozinha (a menos da escala 1/12, que o Adam anula).
    """

    def __init__(self, camadas: list[int], n_sub: int = N_SUB):
        super().__init__()
        self.grus = nn.ModuleList(GRUBaseline(1, camadas) for _ in range(n_sub))

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:  # [lote, janela, 12] -> [lote, 12]
        return torch.stack(
            [g(x[..., k : k + 1]) for k, g in enumerate(self.grus)], dim=-1
        )


def preparar(
    cfg: dict[str, Any], df: pd.DataFrame | None = None
) -> tuple[
    dict[str, Conjunto],
    nn.Module,
    dict[str, Callable[[np.ndarray], np.ndarray]],
    dict[str, Any],
]:
    """Dados, modelo, fusão por parte e o registro (snapshot e decomposição)."""
    registro: dict[str, Any] = {"snapshot_sha256": None}
    if df is None:
        from dados.baixar import MANIFESTO, carregar

        df = carregar()
        registro["snapshot_sha256"] = json.loads(MANIFESTO.read_text(encoding="utf-8"))[
            "PETR4.SA"
        ]["sha256"]
    dados, desfazer, decomp = montar(
        df[FECHAMENTO].to_numpy(np.float64),
        pd.DatetimeIndex(df.index),
        NORMALIZACOES[cfg["modelo"]],
        cfg["decomposicao"] == "causal",
        cfg["janela"],
    )
    return dados, HDFM(cfg["camadas"]), desfazer, registro | decomp | {"hdfm": PARAMS}


# ---------------------------------------------------------------- conferência


def conferir() -> None:
    """Decomposição da série inteira da PETR4: entropias, cotovelo do K-means (D-12),
    reconstrução e uma figura em execucoes/hdfm_conferencia.png.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from dados.baixar import carregar
    from PyEMD import CEEMDAN  # pyright: ignore[reportMissingTypeStubs]
    from sklearn.cluster import KMeans  # pyright: ignore[reportMissingTypeStubs]

    x = carregar()[FECHAMENTO].to_numpy(np.float64)
    p = PARAMS
    c = CEEMDAN(trials=p["trials"], epsilon=p["epsilon"], parallel=False)
    c.noise_seed(p["semente"])
    imfs = c(x)
    ent = [entropia_amostral(i, p["m"], p["r"]) for i in imfs]
    print("entropia amostral por componente:", np.round(ent, 3).tolist())
    inercias = [
        KMeans(k, n_init=10, random_state=0)  # pyright: ignore[reportArgumentType]
        .fit(np.array(ent)[:, None])
        .inertia_
        for k in range(1, len(imfs))
    ]
    print("inércia do K-means, K = 1..:", [round(float(i or 0), 4) for i in inercias])
    d = _cacheado(x, p, CACHE)
    print("erro máximo de reconstrução:", float(np.abs(d.sum(axis=0) - x).max()))

    fig, eixos = plt.subplots(
        len(d) + 1, 1, figsize=(10, 1.4 * (len(d) + 1)), sharex=True
    )
    eixos[0].plot(x, lw=0.8)
    eixos[0].set_ylabel("fech.", rotation=0, ha="right")
    for k, (eixo, s) in enumerate(zip(eixos[1:], d, strict=True)):
        eixo.plot(s, lw=0.6)
        rotulo = f"modo {k + 1}" if k < p["vmd_k"] else f"Co-IMF{k - p['vmd_k'] + 2}"
        eixo.set_ylabel(rotulo, rotation=0, ha="right")
    fig.tight_layout()
    saida = CACHE.parent / "hdfm_conferencia.png"
    fig.savefig(saida, dpi=110)
    print("figura:", saida)


if __name__ == "__main__":
    conferir()
