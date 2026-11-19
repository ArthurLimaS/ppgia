"""Verificação da lógica de validação, nas 30 execuções da Etapa 7 e nos dados reais
(explicada em `validacao.md`).

Uso (a partir da raiz do repositório, depois da fila):
    uv run python redes_neurais_2026_2/projeto/verificar_validacao.py

1. Datas de validação e de teste: as mesmas em todas as execuções, e as esperadas.
2. Época escolhida (`melhor_epoca`) = a de menor perda de validação no `log.csv`.
3. `metricas.json` não guarda métrica de teste.
4. Perturbar validação + teste (preços × 3 a partir do 1º dia de validação) não muda
   nenhuma amostra de treino nem estatística de normalização, em cada modelo.
"""

import json
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch


def main() -> None:
    from comum import janelas
    from comum.janelas import FECHAMENTO, limites
    from dados.baixar import UNIVERSO, carregar
    from fila import itens
    from modelos import custom_fingat, hdfm
    from rodar import EXECUCOES, nome

    falhas: list[str] = []

    def conferir(ok: bool, texto: str) -> None:
        print(("ok   " if ok else "FALHA") + " " + texto)
        if not ok:
            falhas.append(texto)

    # 1-3: as 30 execuções
    pastas = [EXECUCOES / nome(cfg) for cfg, _ in itens(None)]
    datas = {
        p: {
            parte: pd.read_csv(p / f"previsoes_{parte}.csv")["data"].tolist()
            for parte in ("validacao", "teste")
        }
        for p in pastas
    }
    ref = datas[pastas[0]]
    conferir(
        all(d == ref for d in datas.values()),
        f"as {len(pastas)} execuções têm os mesmos dias de validação e de teste",
    )
    conferir(
        (ref["validacao"][0], ref["validacao"][-1], len(ref["validacao"]))
        == ("2025-11-11", "2026-05-18", 125),
        "validação: 2025-11-11 a 2026-05-18, 125 dias",
    )
    conferir(
        (ref["teste"][0], ref["teste"][-1], len(ref["teste"]))
        == ("2026-05-19", "2026-11-13", 125),
        "teste: 2026-05-19 a 2026-11-13, 125 dias",
    )
    erradas = []
    for p in pastas:
        log = pd.read_csv(p / "log.csv")
        r = json.loads((p / "metricas.json").read_text(encoding="utf-8"))
        ck = torch.load(p / "melhor.pt", map_location="cpu", weights_only=False)
        argmin = int(
            log["epoca"].iloc[int(np.argmin(log["perda_validacao"].to_numpy()))]
        )
        if not (r["melhor_epoca"] == ck["epoca"] == argmin):
            erradas.append(p.name)
        if any("teste" in k for k in r):
            falhas.append(f"{p.name}: metricas.json com chave de teste")
    conferir(
        not erradas,
        "em todas, época escolhida = menor perda de validação = a do melhor.pt"
        + (f" (erradas: {erradas})" if erradas else ""),
    )
    conferir(
        not any("chave de teste" in f for f in falhas),
        "nenhum metricas.json guarda teste",
    )

    # 4: perturbar validação + teste não muda o treino
    df = carregar()
    n = len(df)
    ini_val, _ = limites(n)
    colunas = ["Open", "High", "Low", "Close", FECHAMENTO]

    def perturbar(d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        d.iloc[ini_val:, [d.columns.get_loc(c) for c in colunas]] *= 3
        return d

    def iguais(
        a: dict[str, Any], b: dict[str, Any], partes: tuple[str, ...] = ("treino",)
    ) -> bool:
        return all(
            torch.equal(a[p].x, b[p].x) and torch.equal(a[p].y, b[p].y) for p in partes
        )

    for entrada in ("univariado", "ohlcv"):
        a = janelas.montar(df, entrada, 120)
        b = janelas.montar(perturbar(df), entrada, 120)
        conferir(iguais(a, b), f"GRU ({entrada}): treino não muda")
        conferir(
            max(a["treino"].datas) < min(a["validacao"].datas),
            f"GRU ({entrada}): alvos de treino antes da validação",
        )

    candidatas = pd.read_csv(UNIVERSO)
    dfs = {
        t: carregar(t)
        for t in candidatas["ticker"]
        if (EXECUCOES.parent / "dados" / "snapshot" / f"{t}.csv").exists()
    }
    dfs_p = {t: perturbar(d) for t, d in dfs.items()}
    for m in custom_fingat.VARIANTES:
        cfg = {"modelo": m, "janela": 120, "camadas": [128]}
        torch.manual_seed(0)
        a, _, _, ua = custom_fingat.preparar(cfg, dfs, candidatas)
        torch.manual_seed(0)
        b, _, _, ub = custom_fingat.preparar(cfg, dfs_p, candidatas)
        conferir(ua["universo"] == ub["universo"], f"{m}: o universo não muda")
        conferir(
            iguais(a, b), f"{m}: treino (features padronizadas e rótulos) não muda"
        )

    x = df[FECHAMENTO].to_numpy(np.float64)
    xp = perturbar(df)[FECHAMENTO].to_numpy(np.float64)
    datas_idx = pd.DatetimeIndex(df.index)
    for norm in ("A", "B"):
        for causal in (True, False):
            a, fa, _ = hdfm.montar(x, datas_idx, norm, causal, 120)
            b, fb, _ = hdfm.montar(xp, datas_idx, norm, causal, 120)
            mesmo_treino = iguais(a, b)
            # a fusão do treino desfaz a normalização com as estatísticas do treino
            ya = a["treino"].y.numpy().astype(np.float64)
            mesma_estat = np.allclose(fa["treino"](ya), fb["treino"](ya))
            nome_dec = "causal" if causal else "inteira"
            if causal:
                conferir(
                    mesmo_treino and mesma_estat,
                    f"HDFM {norm} {nome_dec}: treino e estatísticas não mudam",
                )
            else:
                conferir(
                    not mesmo_treino,
                    f"HDFM {norm} {nome_dec}: treino MUDA (vazamento esperado, D-10)",
                )

    print("\nRESUMO:", "tudo ok" if not falhas else f"{len(falhas)} falha(s)")
    sys.exit(1 if falhas else 0)


if __name__ == "__main__":
    main()
