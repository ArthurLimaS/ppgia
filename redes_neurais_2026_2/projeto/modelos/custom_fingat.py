"""FinGAT (Hsu et al., 2023) em duas variantes, ambas com o bloco GRU compartilhado.

- `fingat_repo` (Etapa 3a): o `CategoricalGraphAtt` do código dos autores
  (github.com/Roytsai27/Financial-GraphAttention, commit 0f7a2dd), com perda,
  inicialização, lr e passo do otimizador de lá. A cópia sem modificação e a lista do
  que mudou estão em `terceiros/fingat_original/` (ORIGEM.md).
- `fingat_artigo` (Etapa 3b): arquitetura e perda como o artigo descreve (§4, Eq. 1-16).

Entrada: [dias, ações, semanas, 5, features], com as ações em ordem de setor e o
mesmo número por setor (D-07). Saída: [dias, ações, 2] = (retorno previsto,
probabilidade de alta). Na comparação só entra a direção da PETR4 (D-06).
"""

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from comum.gru import BlocoGRU
from comum.janelas import FECHAMENTO, Conjunto, limites
from torch import nn

PETR4 = "PETR4.SA"
POR_SETOR = (
    6  # D-07: ações por setor (o pooling do repo exige setores do mesmo tamanho)
)
DIAS = 5  # pregões por semana (artigo, §5.1)
MEDIAS = (5, 10, 15, 20, 25, 30)  # Eq. 3
INICIO = max(MEDIAS)  # o repo descarta os 30 primeiros pregões (`iloc[30:]`)
DELTA, LAMBDA = 0.01, 1e-4  # Eq. 15 (artigo, §5.1)


# ---------------------------------------------------------------- dados


def selecionar(
    candidatas: pd.DataFrame, dfs: dict[str, pd.DataFrame]
) -> tuple[list[str], list[int], dict[str, str]]:
    """Universo (D-07): por setor, as `POR_SETOR` de maior volume financeiro médio no
    treino, entre as candidatas com todos os pregões da PETR4. Devolve os tickers em
    ordem de setor, o índice do setor de cada um e o motivo de cada exclusão.
    """
    ref = dfs[PETR4].index
    ini_val = limites(len(ref))[0]
    excluidas: dict[str, str] = {}
    linhas: list[dict[str, Any]] = []
    for t, setor in zip(candidatas["ticker"], candidatas["setor"], strict=True):
        if t not in dfs:
            excluidas[t] = "sem snapshot (o Yahoo Finance não devolveu dados)"
        elif not dfs[t].index.equals(ref):
            excluidas[t] = f"incompleta ({len(dfs[t])} pregões; a PETR4 tem {len(ref)})"
        else:
            d = dfs[t].iloc[:ini_val]
            linhas.append(
                {"ticker": t, "setor": setor, "vol": (d["Close"] * d["Volume"]).mean()}
            )
    ok = pd.DataFrame(linhas).sort_values(["setor", "vol"], ascending=[True, False])
    for t in ok.groupby("setor").tail(-POR_SETOR)["ticker"]:
        excluidas[t] = f"fora das {POR_SETOR} de maior volume financeiro do setor"
    ok = ok.groupby("setor").head(POR_SETOR)
    contagem = ok.groupby("setor")["ticker"].transform("size").to_numpy()
    for t in ok[contagem < POR_SETOR]["ticker"]:
        excluidas[t] = f"setor com menos de {POR_SETOR} ações completas"
    ok = ok[contagem == POR_SETOR]
    if PETR4 not in set(ok["ticker"]):
        raise ValueError("a PETR4 ficou fora do universo")
    codigos = {s: i for i, s in enumerate(dict.fromkeys(ok["setor"]))}
    return list(ok["ticker"]), [codigos[s] for s in ok["setor"]], excluidas


def _features(
    df: pd.DataFrame, repo: bool, ini_val: int, setor: int, n_setores: int
) -> np.ndarray:
    """[pregões, features]. Níveis de preço padronizados com média e desvio do treino
    (D-09c). Retorno e médias sobre o fechamento ajustado, que dá o rótulo comum (D-04).
    """
    adj, close = df[FECHAMENTO], df["Close"]

    def z(s: Any) -> Any:
        treino = s.iloc[:ini_val]
        return (s - treino.mean()) / treino.std(ddof=0)

    niveis = ("Open", "High", "Low", "Close", FECHAMENTO)
    colunas: list[Any] = (
        [z(adj)] if repo else [z(df[c]) for c in niveis]
    )  # repo: `nor_close`
    colunas.append(adj.pct_change().fillna(0.0))  # Eq. 1
    colunas += [df[c] / close - 1 for c in ("Open", "High", "Low")]  # Eq. 2
    colunas += [adj.rolling(f).mean() / adj - 1 for f in MEDIAS]  # Eq. 3, D-09(a)
    x = np.column_stack([c.to_numpy(np.float64) for c in colunas])
    if repo:  # o repo acrescenta o setor em one-hot (`label_<setor>`)
        x = np.hstack([x, np.tile(np.eye(n_setores)[setor], (len(x), 1))])
    return x


def montar(
    dfs: list[pd.DataFrame], setores: list[int], repo: bool, janela: int, petr4: int
) -> dict[str, Conjunto]:
    """Treino, validação e teste pela posição do pregão-alvo, como na GRU (D-02).

    A janela vira janela/5 semanas de 5 pregões. `y` = (retorno do alvo, alta), com
    alta = retorno ≥ 0; `anterior` e `real` são da PETR4.
    """
    if janela % DIAS:
        raise ValueError(f"a janela ({janela}) tem de ser múltipla de {DIAS}")
    n = len(dfs[0])
    ini_val, ini_teste = limites(n)
    if INICIO + janela >= ini_val:
        raise ValueError("a janela não deixa nenhum alvo no treino")
    n_setores = max(setores) + 1
    pares = zip(dfs, setores, strict=True)
    feats = np.stack(
        [_features(d, repo, ini_val, s, n_setores) for d, s in pares], axis=1
    ).astype(np.float32)  # [pregões, ações, features]
    ret = np.stack([d[FECHAMENTO].pct_change().to_numpy() for d in dfs], axis=1)
    adj = dfs[petr4][FECHAMENTO].to_numpy(np.float64)

    def conjunto(inicio: int, fim: int) -> Conjunto:
        alvos = np.arange(inicio, fim)
        x = feats[alvos[:, None] + np.arange(-janela, 0)]  # só pregões antes do alvo
        x = x.transpose(0, 2, 1, 3).reshape(len(alvos), len(dfs), -1, DIAS, x.shape[-1])
        y = np.stack([ret[alvos], ret[alvos] >= 0], axis=-1)
        return Conjunto(
            x=torch.from_numpy(np.ascontiguousarray(x)),
            y=torch.tensor(y, dtype=torch.float32),
            anterior=adj[alvos - 1],
            real=adj[alvos],
            datas=pd.DatetimeIndex(dfs[0].index[alvos]),
        )

    return {
        "treino": conjunto(INICIO + janela, ini_val),
        "validacao": conjunto(ini_val, ini_teste),
        "teste": conjunto(ini_teste, n),
    }


# ---------------------------------------------------------------- camadas


class GAT(nn.Module):
    """Uma camada, uma cabeça, densa (os grafos são completos e pequenos; D-09).

    `artigo=False` reproduz o `GATConv(d, d)` do PyTorch Geometric que o repo usa: a
    mesma matriz na mensagem e na atenção, viés, sem ativação. `artigo=True` segue as
    Eq. 7-8: W1 na mensagem, W2 na atenção, ReLU na saída. O laço próprio está na
    adjacência, não aqui: `adj[q, n]` verdadeiro quando `n` envia para `q`.
    """

    def __init__(self, d: int, artigo: bool):
        super().__init__()
        self.w1 = nn.Linear(d, d, bias=False)
        self.w2 = nn.Linear(d, d, bias=False) if artigo else self.w1
        self.r_destino = nn.Linear(d, 1, bias=False)  # r = [r_destino; r_origem]
        self.r_origem = nn.Linear(d, 1, bias=False)
        self.vies = None if artigo else nn.Parameter(torch.zeros(d))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.w2(x)  # [..., nós, d]
        e = self.r_destino(h) + self.r_origem(h).transpose(-1, -2)  # [..., nós, nós]
        beta = F.leaky_relu(e, 0.2).masked_fill(~adj, float("-inf")).softmax(-1)
        saida = beta @ self.w1(x)
        return torch.relu(saida) if self.vies is None else saida + self.vies


def grafos(setores: list[int], laco: bool) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Adjacência intra-setor [ações, ações], entre setores [setores, setores], e K."""
    s = torch.tensor(setores)
    n_setores, por_setor = int(s.max()) + 1, len(setores) // (int(s.max()) + 1)
    if not torch.equal(s, torch.arange(n_setores).repeat_interleave(por_setor)):
        raise ValueError(
            "as ações têm de vir em ordem de setor, o mesmo número em cada"
        )
    intra = s[:, None] == s[None, :]
    entre = torch.ones(n_setores, n_setores, dtype=torch.bool)
    if not laco:  # artigo: liga cada nó a todos os *outros* do grafo
        intra &= ~torch.eye(len(s), dtype=torch.bool)
        entre &= ~torch.eye(n_setores, dtype=torch.bool)
    return intra, entre, por_setor


def _saidas(f: torch.Tensor, retorno: nn.Linear, movimento: nn.Linear) -> torch.Tensor:
    return torch.stack(
        [retorno(f).squeeze(-1), torch.sigmoid(movimento(f)).squeeze(-1)], dim=-1
    )


# ---------------------------------------------------------------- 3a: repositório


class AttentionBlock(nn.Module):
    """Sem modificação (`model/graph_pool.py`)."""

    def __init__(self, time_step: int, dim: int):
        super().__init__()
        self.attention_matrix = nn.Linear(time_step, time_step)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs_t = torch.transpose(inputs, 2, 1)  # (batch_size, input_dim, time_step)
        attention_weight = self.attention_matrix(inputs_t)
        attention_probs = F.softmax(attention_weight, dim=-1)
        attention_probs = torch.transpose(attention_probs, 2, 1)
        attention_vec = torch.mul(attention_probs, inputs)
        attention_vec = torch.sum(attention_vec, dim=1)
        return attention_vec, attention_probs


class SequenceEncoder(nn.Module):
    """Modificado: `BlocoGRU` no lugar do `nn.GRU(input_dim, hidden_dim)`."""

    def __init__(self, input_dim: int, time_step: int, camadas: list[int]):
        super().__init__()
        self.encoder = BlocoGRU(input_dim, camadas)
        self.dim = self.encoder.saida
        self.attention_block = AttentionBlock(time_step, self.dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        seq_vector, _ = self.encoder(seq)
        seq_vector = self.dropout(seq_vector)
        attention_vec, _ = self.attention_block(seq_vector)
        return attention_vec.view(-1, 1, self.dim)  # prepare for concat


class FinGATRepo(nn.Module):
    """`CategoricalGraphAtt` com `use_gru=False` (o padrão do repo). Modificado: bloco
    GRU compartilhado, `GAT` denso no lugar do `GATConv`, setores e semanas
    parametrizados (o repo fixa 5 x 20 ações) e um lote de dias de uma vez.
    """

    def __init__(
        self, input_dim: int, camadas: list[int], semanas: int, setores: list[int]
    ):
        super().__init__()
        intra, entre, self.por_setor = grafos(setores, laco=True)  # o GATConv põe laço
        self.register_buffer("adj_intra", intra)
        self.register_buffer("adj_setor", entre)
        d = camadas[-1]
        self.pool_attention = AttentionBlock(self.por_setor, d)
        self.encoder_list = nn.ModuleList(
            SequenceEncoder(input_dim, DIAS, camadas) for _ in range(semanas)
        )
        self.cat_gat = GAT(d, artigo=False)
        self.inner_gat = GAT(d, artigo=False)
        self.weekly_attention = AttentionBlock(semanas, d)
        self.fusion = nn.Linear(d * 3, d)
        self.reg_layer = nn.Linear(d, 1)
        self.cls_layer = nn.Linear(d, 1)
        for p in self.parameters():  # train.py: "initialize parameters"
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n = x.shape[:2]
        weekly_embedding = torch.cat(
            [
                enc(x[:, :, w].reshape(b * n, DIAS, -1))
                for w, enc in enumerate(self.encoder_list)
            ],
            dim=1,
        )  # [b*n, semanas, d]
        weekly_att_vector, _ = self.weekly_attention(weekly_embedding)
        weekly_att_vector = weekly_att_vector.view(b, n, -1)
        inner_graph_embedding = self.inner_gat(weekly_att_vector, self.adj_intra)
        category_vectors, _ = self.pool_attention(
            weekly_att_vector.reshape(-1, self.por_setor, weekly_att_vector.shape[-1])
        )
        category_vectors = self.cat_gat(
            category_vectors.view(b, -1, category_vectors.shape[-1]), self.adj_setor
        )
        category_vectors = category_vectors.repeat_interleave(self.por_setor, dim=1)
        fusion_vec = torch.cat(
            (weekly_att_vector, category_vectors, inner_graph_embedding), dim=-1
        )
        fusion_vec = torch.relu(self.fusion(fusion_vec))
        return _saidas(fusion_vec, self.reg_layer, self.cls_layer)


def perda_repo(saida: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """train.py: MAE + BCE + ranking por produto, pesos 1 (`parse_basic_args`)."""
    r_hat, p = saida.unbind(-1)
    r, m = y.unbind(-1)
    rank = torch.relu(
        -(r_hat[..., :, None] * r_hat[..., None, :])
        * (r[..., :, None] * r[..., None, :])
    ).sum()
    return F.l1_loss(r_hat, r) + F.binary_cross_entropy(p, m) + rank


# ---------------------------------------------------------------- 3b: artigo


class Atencao(nn.Module):
    """Eq. 5-6: pesos = softmax no tempo de tanh(W0·h); saída = soma ponderada."""

    def __init__(self, d: int):
        super().__init__()
        self.w0 = nn.Linear(d, 1, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # [N, tempo, d] -> [N, d]
        return (torch.tanh(self.w0(h)).softmax(dim=1) * h).sum(dim=1)


class FinGATArtigo(nn.Module):
    def __init__(
        self, input_dim: int, camadas: list[int], semanas: int, setores: list[int]
    ):
        super().__init__()
        intra, entre, self.por_setor = grafos(setores, laco=False)
        self.register_buffer("adj_intra", intra)
        self.register_buffer("adj_setor", entre)
        d = camadas[-1]
        self.curto, self.att_curto = BlocoGRU(input_dim, camadas), Atencao(d)  # Eq. 4-6
        self.gat_intra = GAT(d, artigo=True)  # Eq. 7-8
        self.longo_a, self.att_a = BlocoGRU(d, camadas), Atencao(d)  # Eq. 9-10
        self.longo_g, self.att_g = BlocoGRU(d, camadas), Atencao(d)
        self.gat_setor = GAT(d, artigo=True)  # Eq. 12
        self.fusao = nn.Linear(3 * d, d, bias=False)  # Eq. 13
        self.retorno, self.movimento = nn.Linear(d, 1), nn.Linear(d, 1)  # Eq. 14

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, semanas = x.shape[:3]
        h, _ = self.curto(x.reshape(b * n * semanas, DIAS, -1))
        a = self.att_curto(h).view(b, n, semanas, -1)
        g = self.gat_intra(a.transpose(1, 2), self.adj_intra).transpose(
            1, 2
        )  # por semana
        ha, _ = self.longo_a(a.reshape(b * n, semanas, -1))
        hg, _ = self.longo_g(g.reshape(b * n, semanas, -1))
        tau_a, tau_g = self.att_a(ha).view(b, n, -1), self.att_g(hg).view(b, n, -1)
        z = tau_g.view(b, -1, self.por_setor, tau_g.shape[-1]).amax(dim=2)  # Eq. 11
        tau_p = self.gat_setor(z, self.adj_setor).repeat_interleave(
            self.por_setor, dim=1
        )
        f = torch.relu(self.fusao(torch.cat((tau_g, tau_a, tau_p), dim=-1)))
        return _saidas(f, self.retorno, self.movimento)


def perda_artigo(saida: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Eq. 15-16 sem o λ‖Θ‖², que entra no Adam (`decaimento` = 2λ: mesmo gradiente)."""
    r_hat, p = saida.unbind(-1)
    r, m = y.unbind(-1)
    rank = torch.relu(
        -(r_hat[..., :, None] - r_hat[..., None, :])
        * (r[..., :, None] - r[..., None, :])
    ).sum()
    return (1 - DELTA) * rank + DELTA * F.binary_cross_entropy(p, m, reduction="sum")


# ---------------------------------------------------------------- variantes

VARIANTES: dict[str, dict[str, Any]] = {
    # repo: train.py treina um dia por passo; lr e l2 são os padrões de parse_basic_args
    "fingat_repo": {
        "classe": FinGATRepo,
        "perda": perda_repo,
        "lote": 1,
        "lr": 0.05,
        "decaimento": 0.0,
        "sublote": None,
    },
    # artigo, §5.1: lote 128 (de dias, D-09e); lr da busca do artigo (D-09f)
    "fingat_artigo": {
        "classe": FinGATArtigo,
        "perda": perda_artigo,
        "lote": 128,
        "lr": 1e-3,
        "decaimento": 2 * LAMBDA,
        # 128 dias de uma vez estouram os 8 GB da GPU; a perda é soma, então acumular
        # o gradiente de 8 sublotes de 16 dá o mesmo passo
        "sublote": 16,
    },
}


def preparar(
    cfg: dict[str, Any],
    dfs: dict[str, pd.DataFrame] | None = None,
    candidatas: pd.DataFrame | None = None,
) -> tuple[dict[str, Conjunto], nn.Module, dict[str, Any], dict[str, Any]]:
    """Dados, modelo, argumentos extras do `treinar` e o registro do universo.

    `dfs` e `candidatas` substituem o snapshot e `universo.csv` (testes).
    """
    snapshot = None
    if candidatas is None:
        from dados.baixar import UNIVERSO

        candidatas = pd.read_csv(UNIVERSO)
    if dfs is None:
        import json

        from dados.baixar import MANIFESTO, carregar

        manifesto = json.loads(MANIFESTO.read_text(encoding="utf-8"))
        dfs = {t: carregar(t) for t in candidatas["ticker"] if t in manifesto}
        snapshot = {t: manifesto[t]["sha256"] for t in dfs}
    tickers, setores, excluidas = selecionar(candidatas, dfs)
    v = VARIANTES[cfg["modelo"]]
    dados = montar(
        [dfs[t] for t in tickers],
        setores,
        cfg["modelo"] == "fingat_repo",
        cfg["janela"],
        tickers.index(PETR4),
    )
    modelo = v["classe"](
        dados["treino"].x.shape[-1], cfg["camadas"], cfg["janela"] // DIAS, setores
    )
    universo = {
        "universo": tickers,
        "setores": setores,
        "excluidas": excluidas,
        "snapshot_sha256": snapshot and {t: snapshot[t] for t in tickers},
    }
    return (
        dados,
        modelo,
        {k: v[k] for k in ("decaimento", "sublote")} | {"perda_fn": v["perda"]},
        universo,
    )
