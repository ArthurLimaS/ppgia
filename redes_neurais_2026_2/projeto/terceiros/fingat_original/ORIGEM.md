# Origem: código dos autores do FinGAT

Os arquivos desta pasta são uma **cópia sem modificação** do repositório público dos autores do FinGAT:

| Campo        | Valor                                                                                                                 |
| ------------ | --------------------------------------------------------------------------------------------------------------------- |
| Repositório  | <https://github.com/Roytsai27/Financial-GraphAttention>                                                                |
| Commit       | `0f7a2dd86e0af0138a3e5976b559d18bd8807d35` (2021-07-06, "Update README.md"), o último em 2026-11-18                   |
| Copiado em   | 2026-11-18                                                                                                            |
| Artigo       | Hsu et al. (2023), _FinGAT: Financial Graph Attention Networks for Recommending Top-K Profitable Stocks_, IEEE TKDE 35(1) (`artigos/hsu_fingat_2023.pdf`, nota 2 e §5.1) |
| Licença      | **o repositório não tem arquivo de licença.** O uso aqui é acadêmico, com citação; não redistribuir fora do trabalho. |
| Arquivos     | `README.md`, `clean_data.py`, `parse_arg.py`, `train.py`, `model/graph_pool.py` (o `.pyc` do repo ficou de fora)      |

**O repositório não traz dados.** `train.py` e `clean_data.py` leem arquivos que não estão nele (`Taiwan_model_data_10_best.pickle`, `*_edge.npy`, `SP500_category.csv`, `SP500_dataset/`). Os dados deste projeto vêm do Yahoo Finance (`dados/baixar.py`, `dados/snapshot/`), e o universo de ações é o de `dados/universo.csv` (B3, D-07). O que vem do repositório é o **código**.

Esta pasta fica fora do ruff e do pyright (`pyproject.toml`) para continuar idêntica ao original. Ela não é importada por nada; o código usado no projeto é `modelos/fingat.py`.

## O que mudou em `modelos/fingat.py` (variante `fingat_repo`, Etapa 3a)

A variante `fingat_repo` é o `CategoricalGraphAtt` de `model/graph_pool.py`, treinado como em `train.py`. As classes `AttentionBlock` e `SequenceEncoder` mantêm o nome e o corpo do original, para o diff ficar legível. Mudou só o que a comparação exige:

| #   | Onde (original)                                | Original                                                                                     | No projeto                                                                                                   | Motivo                                                                             |
| --- | ---------------------------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| 1   | `SequenceEncoder` (`graph_pool.py`)            | `nn.GRU(input_dim, hidden_dim)`, `dim` = 16                                                  | `BlocoGRU(input_dim, camadas)` (`comum/gru.py`), configuração `f`: 1 camada de 128                           | mesma GRU nos quatro modelos (§3.2, D-03)                                          |
| 2   | `inner_gat`, `cat_gat`                         | `GATConv(hidden_dim, hidden_dim)` do PyTorch Geometric                                       | `GAT(d, artigo=False)`: densa, 1 cabeça, mesma matriz na mensagem e na atenção, LeakyReLU 0,2, viés, laço próprio | equivalente ao `GATConv` padrão sem instalar o PyTorch Geometric (D-09)            |
| 3   | `view(5, 20, -1)`, `AttentionBlock(20, …)`     | 5 setores × 20 ações fixos no código                                                          | S setores × K ações tirados do universo (8 × 6)                                                               | universo da B3 (D-07)                                                              |
| 4   | `forward(weekly_batch)`                        | lista de semanas de **um** dia                                                                | tensor `[dias, ações, semanas, 5, features]`, vários dias de uma vez                                         | mesmo cálculo por dia; serve à validação inteira numa chamada                      |
| 5   | `use_gru`                                      | ramo opcional (padrão `False`; com `True`, o `nn.GRU` sem `batch_first` percorre as ações)    | removido: só o caminho padrão                                                                                 | o padrão do repo é `False`                                                         |
| 6   | `train.py`, "initialize parameters"           | Xavier uniforme em todo peso com 2+ dimensões, depois de construir                            | o mesmo, dentro do construtor                                                                                 | um lugar só                                                                        |
| 7   | saída                                          | tupla `(reg_output, cls_output)` achatada                                                     | `[dias, ações, 2]` = (retorno, probabilidade de alta)                                                          | formato do laço de treino comum                                                    |
| 8   | `CategoricalGraph`, `CategoricalGraphPool`, `predict_toprank` | variantes e avaliação do repo                                                   | não copiados                                                                                                  | o `train.py` usa `CAT` por padrão; a avaliação é a do projeto                     |

O treino segue `train.py`: perda MAE + BCE + ranking por **produto** (`relu(−(r̂ᵢ·r̂ⱼ)(rᵢ·rⱼ))`), pesos 1 (`--alpha`, `--beta`, `--gamma`), sem L2, Adam com lr 0,05, um dia por passo do otimizador. Mudou:

| #   | Original (`train.py`, `clean_data.py`)                                                       | No projeto                                                                                  | Motivo                                                     |
| --- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| 9   | melhor época escolhida pelo MRR **no teste**                                                 | melhor época pela perda na validação (`comum/treino.py`)                                    | o teste não escolhe nada (§5)                              |
| 10  | MRR, Precision, ACC pelo sinal do retorno previsto                                           | acurácia de direção pela probabilidade ≥ 0,5 (PETR4 e todas as ações); a ACC do repo ao lado | só direção entra na comparação (D-06)                      |
| 11  | divisão 20 % treino / 80 % teste (`clean_data.py`)                                           | 80/10/10 pela data do alvo, os mesmos dias de validação e teste da GRU                       | D-02                                                       |
| 12  | `nor_close`, retorno e médias móveis sobre `Close`                                           | sobre `Adj Close`; `c_open`, `c_high`, `c_low` continuam `Open/Close − 1` (mesma razão)      | o rótulo de alta é o mesmo para os quatro modelos (D-04, §3.4) |
| 13  | `StandardScaler` ajustado na série inteira                                                    | média e desvio só do trecho de treino                                                       | sem vazamento (D-09c)                                      |
| 14  | "semanas" = janelas de 7 dias deslocadas de 1 dia (`before_day(week)`, 3 semanas)            | janela 120 = 24 semanas de 5 pregões, sem sobreposição                                      | janela comum aos quatro (D-01); 5 dias por semana como no artigo (§5.1) |
| 15  | ações e setores de Taiwan / S&P 500                                                          | `dados/universo.csv`: 6 ações por setor da B3, por volume financeiro no treino              | D-07                                                       |

A variante `fingat_artigo` (Etapa 3b) não usa este código: segue o texto do artigo (§4, Eq. 1–16).
