"""Gráfico de candles da PETR4 com janela deslizante, usado no slide 2.2 do seminário de Redes Neurais.

Baixa os pregões de 01/09/2026 a 28/10/2026 do Yahoo Finance (yfinance), destaca uma janela de
entrada de L = 15 pregões terminando em 23/10/2026 e o pregão-alvo (t+1), e anota a anatomia
OHLC do candle de 05/10/2026. Salva `petr4_candles.png` ao lado deste script.

Uso: python metodologia_2026_1/grafico_petr4_candles.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf
from matplotlib.patches import Rectangle

TICKER = "PETR4.SA"
INICIO, FIM = (
    "2026-09-01",
    "2026-10-29",
)  # FIM é exclusivo no yfinance: último pregão = 28/10
FIM_JANELA = "2026-10-23"  # último pregão da janela; o alvo é o pregão seguinte (26/10)
CANDLE_ANOTADO = "2026-10-05"
L = 15
SAIDA = Path(__file__).with_name("petr4_candles.png")

ALTA, BAIXA = "#2F5DA8", "#C8641E"
TINTA, SUAVE, FUNDO = "#1C2433", "#4A5262", "#F7F6F2"

d = yf.download(TICKER, start=INICIO, end=FIM, auto_adjust=False, progress=False)

if d is None:
    raise RuntimeError(
        f"Não foi possível baixar os dados de {TICKER} de {INICIO} a {FIM}"
    )

d.columns = [c[0] for c in d.columns]
d = d.reset_index()

fim_janela = int(d.index[d.Date == FIM_JANELA][0])
ini_janela, alvo = fim_janela - L + 1, fim_janela + 1

plt.rcParams.update(
    {
        "font.size": 15,
        "axes.edgecolor": SUAVE,
        "text.color": TINTA,
        "axes.labelcolor": SUAVE,
        "xtick.color": SUAVE,
        "ytick.color": SUAVE,
    }
)
fig, (ax, axv) = plt.subplots(
    2,
    1,
    figsize=(11, 7.2),
    dpi=200,
    sharex=True,
    gridspec_kw={"height_ratios": [4, 1], "hspace": 0.05},
)
fig.patch.set_facecolor(FUNDO)
for a in (ax, axv):
    a.set_facecolor(FUNDO)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    # janela de entrada e pregão-alvo
    a.axvspan(ini_janela - 0.5, fim_janela + 0.5, color=ALTA, alpha=0.10, lw=0)
    a.axvspan(alvo - 0.5, alvo + 0.5, color=BAIXA, alpha=0.18, lw=0)

# candle azul = fechou acima da abertura; laranja = abaixo
for i, r in d.iterrows():
    cor = ALTA if r.Close >= r.Open else BAIXA
    ax.vlines(i, r.Low, r.High, color=cor, lw=1.4)
    lo, hi = sorted((r.Open, r.Close))
    ax.add_patch(Rectangle((i - 0.32, lo), 0.64, max(hi - lo, 0.04), color=cor))
    axv.bar(i, r.Volume / 1e6, width=0.64, color=cor)

topo = d.High.max() + 1.2
ax.annotate(
    "",
    xy=(ini_janela - 0.5, topo),
    xytext=(fim_janela + 0.5, topo),
    arrowprops={"arrowstyle": "<->", "color": ALTA, "lw": 1.6},
)
ax.text(
    (ini_janela + fim_janela) / 2,
    topo + 0.35,
    f"janela de entrada: L = {L} pregões",
    ha="center",
    color=ALTA,
    fontsize=15,
    weight="bold",
)
ax.text(
    alvo, d.Low.min() - 0.6, "t+1", ha="center", color=BAIXA, fontsize=15, weight="bold"
)

# anatomia de um candle; rótulos espaçados à mão para não se sobreporem
k = int(d.index[d.Date == CANDLE_ANOTADO][0])
r = d.loc[k]
for y, y_rotulo, nome in (
    (r.High, 57.6, "máxima (High)"),
    (r.Close, 55.9, "fechamento (Close)"),
    (r.Open, 54.2, "abertura (Open)"),
    (r.Low, 52.5, "mínima (Low)"),
):
    ax.annotate(
        f"{nome}  R$ {y:.2f}".replace(".", ","),
        xy=(k - 0.4, y),
        xytext=(k - 13, y_rotulo),
        va="center",
        fontsize=13,
        color=TINTA,
        arrowprops={"arrowstyle": "-", "color": SUAVE, "lw": 0.8},
    )

ax.set_ylim(d.Low.min() - 1.2, topo + 1.6)
ax.set_ylabel("Preço (R$)")
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.1f}".replace(".", ","))
axv.set_ylabel("Volume\n(mi)")
marcas = sorted({t for t in range(0, len(d), 5) if abs(t - alvo) > 2} | {alvo})
axv.set_xticks(marcas)
axv.set_xticklabels([d.Date[t].strftime("%d/%m") for t in marcas], fontsize=12)
axv.set_xlim(-0.8, len(d) - 0.2)
fig.savefig(SAIDA, bbox_inches="tight", facecolor=FUNDO)
print(f"Gráfico salvo em {SAIDA}")
