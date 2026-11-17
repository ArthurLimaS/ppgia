"""Snapshot dos dados do projeto: baixa do Yahoo Finance uma vez e relê do disco.

Todo treino lê o snapshot (`carregar`), nunca a rede: o yfinance reajusta preços
passados a cada provento e limita requisições, então duas execuções só são
comparáveis sobre o mesmo arquivo.

Uso (a partir da raiz do repositório):
    # PETR4.SA, últimos 5 anos
    uv run python redes_neurais_2026_2/projeto/dados/baixar.py
    # refaz um snapshot existente
    uv run python redes_neurais_2026_2/projeto/dados/baixar.py --substituir
"""

import argparse
import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

SNAPSHOT = Path(__file__).with_name("snapshot")
MANIFESTO = SNAPSHOT / "manifesto.json"
COLUNAS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def _sha256(caminho: Path) -> str:
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def _ler_manifesto() -> dict[str, Any]:
    return (
        json.loads(MANIFESTO.read_text(encoding="utf-8")) if MANIFESTO.exists() else {}
    )


def baixar(
    ticker: str, inicio: date, fim: date, substituir: bool = False
) -> dict[str, Any]:
    """Baixa `ticker` de `inicio` a `fim` (exclusivo), grava o CSV e o manifesto."""
    # só aqui: `carregar` não pode depender de rede nem do yfinance
    import yfinance as yf  # pyright: ignore[reportMissingTypeStubs]

    csv = SNAPSHOT / f"{ticker}.csv"
    if csv.exists() and not substituir:
        raise FileExistsError(
            f"{csv} já existe. O snapshot é a base de todas as execuções; "
            "use --substituir só se for refazer o estudo inteiro sobre dados novos."
        )

    d = yf.download(
        ticker,
        start=inicio.isoformat(),
        end=fim.isoformat(),
        auto_adjust=False,
        progress=False,
    )
    if d is None or d.empty:
        raise RuntimeError(
            f"O Yahoo Finance não devolveu dados de {ticker} de {inicio} a {fim}."
        )
    d.columns = [c[0] for c in d.columns]
    d = d[COLUNAS]
    d.index = pd.to_datetime(d.index).tz_localize(None)
    d.index.name = "Date"

    baixadas = len(d)
    d = d.dropna()
    if not d.index.is_monotonic_increasing or d.index.has_duplicates:
        raise RuntimeError(
            f"{ticker}: datas fora de ordem ou repetidas no retorno do Yahoo Finance."
        )

    SNAPSHOT.mkdir(exist_ok=True)
    d.to_csv(csv, lineterminator="\n")

    datas = pd.DatetimeIndex(d.index).strftime("%Y-%m-%d")
    registro = {
        "inicio_pedido": inicio.isoformat(),
        "fim_pedido_exclusivo": fim.isoformat(),
        "primeiro_pregao": datas[0],
        "ultimo_pregao": datas[-1],
        "pregoes": len(d),
        "linhas_removidas_por_valor_ausente": baixadas - len(d),
        "pregoes_com_volume_zero": int((d["Volume"] == 0).sum()),
        "sha256": _sha256(csv),
        "baixado_em_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "yfinance": yf.__version__,
        "auto_adjust": False,
    }
    manifesto = _ler_manifesto()
    manifesto[ticker] = registro
    MANIFESTO.write_text(
        json.dumps(manifesto, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return registro


def carregar(ticker: str = "PETR4.SA") -> pd.DataFrame:
    """Relê o snapshot de `ticker`, conferindo que é o arquivo do manifesto."""
    csv = SNAPSHOT / f"{ticker}.csv"
    registro = _ler_manifesto().get(ticker)
    if registro is None or not csv.exists():
        raise FileNotFoundError(
            f"Sem snapshot de {ticker}. Rode dados/baixar.py primeiro."
        )
    if _sha256(csv) != registro["sha256"]:
        raise RuntimeError(
            f"{csv} não é o arquivo registrado no manifesto (sha256 diferente): "
            "foi alterado depois do download."
        )
    return pd.read_csv(csv, index_col="Date", parse_dates=True)


if __name__ == "__main__":
    hoje = date.today()

    if __doc__:
        p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
        p.add_argument("--ticker", default="PETR4.SA")
        p.add_argument(
            "--inicio",
            type=date.fromisoformat,
            default=hoje.replace(year=hoje.year - 5),
        )
        p.add_argument(
            "--fim",
            type=date.fromisoformat,
            default=hoje,
            help="exclusivo; o padrão (hoje) deixa de fora o pregão em andamento",
        )
        p.add_argument("--substituir", action="store_true")
        a = p.parse_args()
        print(
            json.dumps(
                baixar(a.ticker, a.inicio, a.fim, a.substituir),
                indent=2,
                ensure_ascii=False,
            )
        )

    else:
        raise RuntimeError(
            "Este módulo não é um script: use `uv run python dados/baixar.py`."
        )
