# Condições do estudo — comparação de 5 modelos na PETR4

> **RASCUNHO para revisão (2026-11-18).** Junta o que foi decidido nas Etapas 0 a 4 (decisões D-NN do [guia de implementação](guia-implementacao-projeto.md), §6.1) e propõe o que a Etapa 6 ainda tem de decidir. Cada ponto novo está marcado como **Proposta P-NN** e listado no fim (§9). Depois de aprovado, este documento **não muda durante a Etapa 7**; mudança feita depois de ver resultado do teste é registrada como tal, com data e motivo (§8).

## Índice

1. [Modelos](#1-modelos)
2. [Dados](#2-dados)
3. [Hiperparâmetros](#3-hiperparâmetros)
4. [Épocas e parada](#4-épocas-e-parada)
5. [Repetições](#5-repetições)
6. [Métricas](#6-métricas)
7. [Orçamento](#7-orçamento)
8. [Regras da execução](#8-regras-da-execução)
9. [Pontos para revisão](#9-pontos-para-revisão)
10. [Desvios dos artigos](#10-desvios-dos-artigos)

---

## 1. Modelos

Cinco modelos de dois artigos, mais a GRU. Todos usam o mesmo bloco GRU (`comum/gru.py`), na configuração escolhida na Etapa 2: o que muda entre eles é o que vem antes e depois desse bloco.

| #   | Modelo                               | `--modelo`       | Artigo                                    | O que acrescenta à GRU                              |
| --- | ------------------------------------ | ---------------- | ----------------------------------------- | --------------------------------------------------- |
| 1   | GRU                                  | `gru`            | Cho et al. (2014)                         | nada: é a base                                      |
| 2   | FinGAT, código dos autores           | `fingat_repo`    | Hsu et al. (2023), repositório deles      | grafo de ações por setor, hierarquia semana/setor   |
| 3   | Custom FinGAT, texto do artigo       | `fingat_artigo`  | Hsu et al. (2023)                         | idem, como o artigo descreve                        |
| 4   | HDFM, normalização A (min-max)       | `hdfm_minmax`    | Li et al. (2024)                          | decomposição do preço em 12 subséries, 12 GRUs      |
| 5   | HDFM, normalização B (diferença)     | `hdfm_diferenca` | Li et al. (2024)                          | idem, com a normalização da D-20 adaptada           |

- **Execuções do HDFM (P-01 e P-08, decidido em 2026-11-18):** dois eixos independentes. A **decomposição** (D-10): `causal`, em que cada amostra só vê a decomposição do próprio passado, e `inteira`, a série toda de uma vez, como o artigo aparenta (com vazamento). A **normalização** (D-14): A (min-max, a leitura literal da Eq. 20) e B (diferença por janela). Rodam **3** combinações:

  | Execução         | Papel                                                                                   |
  | ---------------- | --------------------------------------------------------------------------------------- |
  | A, causal        | modelo 4 da comparação (leitura literal do artigo; falha pela D-20)                     |
  | B, causal        | modelo 5 da comparação                                                                  |
  | B, inteira       | só a seção do vazamento: comparada com a B causal, mede quanto o vazamento infla o resultado (validação rápida: RMSE 0,44 contra 0,75) |

  A combinação A com série inteira não roda: a A já falha por outro motivo e não acrescenta nada à análise do vazamento. São **6 configurações** no total: GRU, 2 FinGAT e 3 HDFM.
- **O TEGRU saiu do estudo** (guia, Etapa 5): sem notícias para a PETR4, a contribuição principal dele não seria avaliada.

## 2. Dados

### 2.1. Snapshot

| Item                    | Valor                                                                                          |
| ----------------------- | ---------------------------------------------------------------------------------------------- |
| Série-alvo              | PETR4.SA, diária, fechamento **ajustado** (`Adj Close`) (D-04)                                  |
| Período                 | 2021-11-16 a 2026-11-13, **1.248 pregões**                                                      |
| Arquivo                 | `projeto/dados/snapshot/PETR4.SA.csv`, sha256 `fc93e695fec5e78dbba79421cc53be7808a73bb55223fe926b97b2516c91bc6f` |
| Fonte                   | Yahoo Finance via `yfinance` 1.5.1, `auto_adjust=False`, baixado em 2026-11-16                  |
| Universo do FinGAT      | 48 ações (abaixo), baixadas em 2026-11-18 nas mesmas datas; sha256 de cada uma no `manifesto.json` e no `config.json` de cada execução |
| Leitura                 | todo treino lê o snapshot (`dados/baixar.py: carregar`), que recusa o arquivo se o sha256 não bater; nenhum treino acessa a rede |

**Universo do FinGAT (D-07):** 8 setores da B3 × 6 ações, as de maior volume financeiro médio no trecho de treino, entre as candidatas de `dados/universo.csv` com os 1.248 pregões completos. Setores conferidos pelo autor em 2026-11-18.

| Setor (B3)                      | Ações                                         |
| ------------------------------- | --------------------------------------------- |
| Petróleo, Gás e Biocombustíveis | PETR4, PETR3, PRIO3, VBBR3, BRAV3, CSAN3      |
| Financeiro                      | ITUB4, BBDC4, BBAS3, B3SA3, BPAC11, ITSA4     |
| Materiais Básicos               | VALE3, SUZB3, GGBR4, CSNA3, USIM5, KLBN11     |
| Utilidade Pública               | SBSP3, EQTL3, CMIG4, ENGI11, CPFE3, TAEE11    |
| Consumo Cíclico                 | MGLU3, RENT3, LREN3, CYRE3, MRVE3, COGN3      |
| Consumo não Cíclico             | ABEV3, ASAI3, BEEF3, SLCE3, SMTO3, PCAR3      |
| Bens Industriais                | WEGE3, RAIL3, VAMO3, POMO4, ECOR3, RAPT4      |
| Saúde                           | HAPV3, RADL3, RDOR3, HYPE3, FLRY3, QUAL3      |

### 2.2. Divisão

Cronológica, 80/10/10 pela posição do **pregão-alvo** (D-02), sem embaralhar, sobre os 5 anos do snapshot. Os dias de validação e de teste são **os mesmos para todos os modelos**.

- **Validação cruzada temporal (`TimeSeriesSplit`) e 10 anos de dados: não** (P-10, decidido em 2026-11-18). Avaliar em vários períodos multiplicaria o treino pelo número de blocos (o FinGAT repo passaria de ~6 h a ~30 h com 5 blocos), e 10 anos exigiriam refazer o snapshot, o estudo da Etapa 2 e a decomposição causal, e encolheriam o universo do FinGAT. O teste num período só fica como limitação (guia, L-04) e trabalho futuro.

| Parte     | Posições dos alvos | Datas dos alvos            | Pregões | Dias de alta      |
| --------- | ------------------ | -------------------------- | ------- | ----------------- |
| Treino    | até 997            | até 2025-11-10             | 998     |                   |
| Validação | 998 a 1.122        | 2025-11-11 a 2026-05-18    | 125     | 66,4 %            |
| Teste     | 1.123 a 1.247      | 2026-05-19 a 2026-11-13    | 125     | 53,6 %            |

- **Tarefa:** com os 120 pregões até `t`, prever o pregão `t+1` (janela 120, D-01).
- **Primeiro alvo de treino** (D-17: na execução final, cada modelo usa tudo o que a janela alcança): GRU e HDFM na posição 120 (2022-05-11), **878 amostras**; FinGAT na posição 150 (2022-06-23; as médias de 30 dias consomem 30 pregões), **848 dias**. No estudo da Etapa 2 a GRU começava em 200 (798 amostras); a execução final não repete essa restrição.
- **O teste não escolhe nada:** janela, entrada, arquitetura e checkpoint saem da validação; o teste só é lido na Etapa 7.
- **O estudo da GRU (Etapa 2) usou este mesmo protocolo:** a mesma divisão, a escolha só na validação, no máximo 100 épocas com o checkpoint de menor perda de validação. Diferem só as sementes (3, não 5) e o primeiro alvo de treino (o mesmo para todas as janelas, D-17, como explicado acima).

### 2.3. Normalização (sem estatística do teste em nenhum modelo)

As quatro linhas abaixo são **a mesma etapa**: a transformação do que entra na rede (e do alvo) e a inversa na saída. O bloco GRU não tem normalização dentro dele. Há quatro porque cada modelo entrega dados diferentes ao bloco (§3). A regra comum: **nenhuma usa estatística da validação nem do teste**, só a do treino ou a da própria janela.

| Modelo            | Entrada                                                                                       | Alvo                                   | Decisão      |
| ----------------- | --------------------------------------------------------------------------------------------- | -------------------------------------- | ------------ |
| GRU               | janela OHLC ÷ último fechamento − 1 (abertura, máxima e mínima ajustadas por `Adj Close/Close`); volume ÷ média do volume da janela − 1 | variação relativa de `t+1`; o preço é reconstruído | D-20, D-21   |
| FinGAT (as duas)  | níveis de preço padronizados com média e desvio **do treino**, por ação; razões e médias móveis já são adimensionais | retorno de `t+1` e alta (retorno ≥ 0)  | D-09c        |
| HDFM A            | min-max por subsérie, ajustado no treino (Eq. 20)                                             | idem                                   | D-14         |
| HDFM B            | por janela e subsérie: (valor − último valor) ÷ desvio do passo de 1 dia da subsérie no treino | passo de `t+1`                         | D-14         |

## 3. Hiperparâmetros

Origem de cada valor: o artigo (com a seção), o código dos autores, ou a decisão D-NN que o fixou. **Valores em comum:** janela 120 (D-01), bloco GRU = 1 camada de 128 unidades (`f`, D-03), dispositivo GPU (RTX 3060 Ti, Etapa 2a), Adam.

**O que "a mesma GRU" quer dizer.** Todos os modelos usam a mesma classe (`comum/gru.py`) com a mesma configuração: 1 camada, 128 unidades, as mesmas portas e a mesma ativação. Nenhum modelo muda o número de camadas, o tamanho do estado oculto ou a forma da célula, e nenhum tem hiperparâmetro da GRU ajustado para ele. O que muda é o que cada artigo põe **em volta** do bloco, e isso é justamente o que o estudo mede:

| Modelo         | O que entra em cada GRU                         | Passos por sequência | Quantas GRUs            | Parâmetros do modelo |
| -------------- | ----------------------------------------------- | -------------------- | ----------------------- | -------------------- |
| GRU            | 5 features da PETR4 (OHLCV)                     | 120                  | 1                       | 51.969               |
| FinGAT repo    | 19 features, por ação e semana                  | 5 (uma semana)       | 24 (uma por semana)     | 1.457.620            |
| Custom FinGAT  | 15 features por ação e semana; depois, as semanas | 5; depois 24       | 3                       | 369.666              |
| HDFM           | 1 subsérie                                      | 120                  | 12 (uma por subsérie)   | 605.196              |

- **Número de features de entrada:** muda o tamanho da matriz que multiplica a entrada (3 × 128 × features), não a arquitetura. Não tem como ser igual: as features são parte do que cada artigo propõe.
- **Passos e número de GRUs:** a hierarquia semana → mês do FinGAT e a uma-GRU-por-subsérie do HDFM são a contribuição dos artigos, não uma mudança na GRU.
- **Pesos:** cada modelo treina a sua do zero; "mesma GRU" é mesma arquitetura e configuração, não os mesmos pesos.
- **O que isso não controla (L-03 do guia):** os modelos dos artigos têm de 7 a 28 vezes mais parâmetros que a GRU sozinha. Uma vantagem deles pode vir em parte desse tamanho, e não só da ideia do artigo. O relatório diz isso ao lado dos resultados.

### 3.1. GRU (`gru`)

| Hiperparâmetro      | Valor                         | Origem                                           |
| ------------------- | ----------------------------- | ------------------------------------------------ |
| Entrada             | `ohlcv` (5 features)          | D-05, estudo da Etapa 2                          |
| Janela              | 120                           | D-01, estudo da Etapa 2                          |
| Camadas             | 1 × 128 (51.969 parâmetros)   | D-03, estudo da Etapa 2                          |
| Saída               | linear → variação de `t+1`    | Etapa 1                                          |
| Perda               | MSE                           | HDFM, Tabela 2 (os únicos valores completos)     |
| Otimizador, lr      | Adam, 0,001                   | HDFM, Tabela 2                                   |
| Lote                | 64                            | HDFM, Tabela 2                                   |
| Dropout             | nenhum                        | P-02 (decidido em 2026-11-18: não usar)          |

### 3.2. FinGAT, código dos autores (`fingat_repo`)

| Hiperparâmetro         | Valor                                                                  | Origem                                       |
| ---------------------- | ---------------------------------------------------------------------- | -------------------------------------------- |
| Features por dia       | 19: fechamento padronizado, retorno, 3 razões de preço, 6 médias móveis, setor em one-hot (8) | `clean_data.py` do repo, com ajustes (D-04, D-09c) |
| Semanas                | 24 de 5 pregões                                                        | D-01 (o repo usa 3 janelas de 7 dias)        |
| Curto prazo            | uma GRU por semana (24 × `f`) + `AttentionBlock`, dropout 0,2          | repo (`SequenceEncoder`); GRU por D-03       |
| Grafos                 | GAT densa equivalente ao `GATConv` (1 cabeça, com laço) intra-setor e entre setores; pooling por atenção | repo (`CategoricalGraphAtt`), D-09b    |
| Longo prazo            | só atenção sobre as semanas (`use_gru=False`)                          | repo                                         |
| Perda                  | MAE + BCE + ranking por produto, pesos 1, sem L2                       | repo (`train.py`, `parse_basic_args`)         |
| Otimizador, lr         | Adam, 0,05                                                              | repo                                         |
| Passo                  | 1 dia por passo                                                         | repo (D-09e)                                 |
| Inicialização          | Xavier uniforme                                                         | repo                                         |
| Parâmetros             | 1.457.620                                                               |                                              |

### 3.3. Custom FinGAT, texto do artigo (`fingat_artigo`)

| Hiperparâmetro         | Valor                                                                  | Origem                                       |
| ---------------------- | ---------------------------------------------------------------------- | -------------------------------------------- |
| Features por dia       | 15: abertura, máxima, mínima, fechamento, ajustado (padronizados), retorno, 3 razões, 6 médias móveis | §4.1, Eq. 1–3; D-09a, D-09c |
| Semanas                | 24 de 5 pregões                                                        | §5.1 (5 dias por semana); D-01 (24 no lugar de 3) |
| Curto prazo            | uma GRU para todas as semanas + atenção (Eq. 5–6)                      | §4.1, Eq. 4–6; GRU por D-03                  |
| Intra-setor            | GAT por semana, 1 cabeça, sem laço (Eq. 7–8)                           | §4.1; D-09b                                  |
| Longo prazo            | GRU atentiva sobre `a` e sobre `g` (Eq. 9–10)                          | §4.1                                         |
| Setores                | max-pooling (Eq. 11) + GAT entre setores (Eq. 12)                      | §4.2                                         |
| Perda                  | (1 − δ)·ranking + δ·BCE + λ‖Θ‖², δ = 0,01, λ = 0,0001 (L2 via `weight_decay` = 2λ) | §4.3, Eq. 15–16; §5.1          |
| Otimizador, lr         | Adam, 0,001                                                             | §5.1 (busca em {0,0005; 0,001; 0,005}); D-09f |
| Lote                   | 128 dias (8 sublotes de 16, mesmo gradiente)                           | §5.1; D-09e                                  |
| Parâmetros             | 369.666                                                                 |                                              |

### 3.4. HDFM (`hdfm_minmax`, `hdfm_diferenca`)

| Hiperparâmetro            | Valor                                                                  | Origem                                   |
| ------------------------- | ---------------------------------------------------------------------- | ---------------------------------------- |
| CEEMDAN                   | 100 realizações, ε = 0,005                                              | D-11 (padrões da `EMD-signal`; o artigo não informa) |
| Entropia amostral         | m = 2, r = 0,2 × desvio                                                 | D-12                                     |
| K-means                   | K = 3 (cotovelo conferido na PETR4)                                     | D-12; artigo, §IV                        |
| VMD                       | K = 10 modos, α = 2000, τ = 0, na Co-IMF de maior entropia              | D-13 (padrões; o artigo não informa)     |
| Resíduo da VMD            | somado ao modo de maior frequência                                       | D-22                                     |
| Subséries                 | 12 = 10 modos + 2 Co-IMFs                                                | artigo, Fig. 2                           |
| Semente da decomposição   | 0, fixa para todas as sementes de treino                                 | Etapa 4 (custo da causal)                |
| GRU por subsérie          | `f` (1 × 128) + linear; 12 GRUs, 605.196 parâmetros                      | D-03 (o artigo usa 128-64-32)            |
| Fusão                     | soma simples das 12 previsões desnormalizadas                            | D-14                                     |
| Normalização              | A: min-max por subsérie; B: diferença por janela                         | D-14                                     |
| Perda, otimizador, lr     | MSE, Adam, 0,001                                                         | artigo, Tabela 2                         |
| Lote                      | 64                                                                       | artigo, Tabela 2                         |

## 4. Épocas e parada

**No máximo 100 épocas em todos os modelos** (P-03, decidido em 2026-11-18: a mesma configuração para todos). A época cujos pesos vão para o teste é a de menor perda de validação, e varia de uma execução para outra. É o valor do HDFM (Tabela 2) e o do estudo da Etapa 2 (a melhor época da GRU escolhida foi 35, 37 e 78). O código dos autores do FinGAT usa 20 (`--epochs 20`), e o texto do artigo não informa: os dois FinGAT ficam com 100, como os demais.

- **Checkpoint avaliado no teste:** o de **menor perda de validação** (`melhor.pt`), em todos os modelos. É o que o laço comum já grava.
- **Sem parada antecipada** (P-04, decidido em 2026-11-18): todas as execuções vão às 100 épocas, e o melhor checkpoint já faz o papel da parada antecipada. A validação só escolhe o checkpoint; não entra no treino. Cada execução grava a época escolhida (`melhor_epoca` no `metricas.json`), e o relatório a mostra.
- **Épocas iguais, não orçamento igual** (P-03): uma época custa coisas diferentes em cada modelo (FinGAT: um dia com 48 ações; HDFM: 12 GRUs), e isso aparece no §7.
- **Atenção no FinGAT:** a perda de validação dele é a perda do próprio modelo (dominada pelo ranking), não a acurácia de direção que entra na comparação. O checkpoint é escolhido por ela mesmo assim, porque escolher pela direção na validação seria otimizar a métrica comparada.

## 5. Repetições

- **5 sementes (0 a 4) em todas as configurações** (P-05, decidido em 2026-11-18). O FinGAT reporta a média de 10 execuções; 10 aqui levariam ~13 h, e não cabem no prazo (§7).
- A semente fixa `torch`, `numpy` e `random` e a ordem dos lotes; vai no `config.json` de cada execução.
- **O que se reporta por modelo:** **todas as métricas do §6** que se aplicam a ele (o FinGAT só tem as de direção, D-06), cada uma como média e desvio-padrão entre as sementes. Um apêndice traz a tabela por semente.

## 6. Métricas

Calculadas no **teste**, a partir das previsões gravadas (`previsoes_teste.csv`), com o checkpoint do §4. Recalculáveis sem treinar de novo.

| Grupo   | Métrica                     | Definição                                                                 | Modelos                    |
| ------- | --------------------------- | ------------------------------------------------------------------------- | -------------------------- |
| Preço   | RMSE, MAE (R$)              | sobre o fechamento ajustado previsto                                      | GRU, HDFM                  |
| Preço   | MAPE (%)                    | média de \|erro ÷ real\| × 100                                             | GRU, HDFM                  |
| Preço   | R²                          | 1 − SQ do erro ÷ SQ em torno da média do real                             | GRU, HDFM                  |
| Direção | Acurácia de direção         | fração de dias em que a alta prevista coincide com a real                 | todos                      |
| Direção | Acurácia em todas as ações  | a mesma, nas 48 ações (conferência com a ACC do artigo)                    | FinGAT                     |

- **Rótulo real de alta, igual para todos:** `real(t) − real(t−1) ≥ 0` no fechamento ajustado da PETR4.
- **Alta prevista:** quem prevê preço (GRU, HDFM): `previsto(t) − real(t−1) ≥ 0`; FinGAT: probabilidade de alta ≥ 0,5 (D-06). O `fingat_repo` também grava a ACC do jeito do `train.py` (sinal do retorno previsto), só como conferência.
- **Referência ingênua** ("amanhã = hoje"): linha própria nas tabelas. No teste, ela acerta a direção em 53,6 % dos dias (só prevê alta).
- **Incerteza** (P-06, decidido em 2026-11-18):
  - **Acurácia de direção:** intervalo de confiança de 95 % de Wilson para uma proporção binomial, sobre os 125 dias de teste. Com 53,6 % de acertos, o intervalo vai de ~44,9 % a ~62,1 % (± 8,6 pontos): duas acurácias cujos intervalos se sobrepõem tanto assim não são distinguíveis (R-06). Referências: Wilson (1927), que propõe o intervalo; Brown, Cai e DasGupta (2001), que comparam os intervalos para proporção binomial, mostram que o intervalo "de livro" (Wald) falha e recomendam o de Wilson (e o de Agresti–Coull para n acima de 40, que dá praticamente o mesmo intervalo com n = 125).
    - E. B. Wilson. Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association*, 22(158):209–212, 1927. doi:10.1080/01621459.1927.10502953.
    - L. D. Brown, T. T. Cai e A. DasGupta. Interval estimation for a binomial proportion. *Statistical Science*, 16(2):101–133, 2001. doi:10.1214/ss/1009213286.
  - **Todas as métricas:** média ± desvio-padrão entre as 5 sementes (§5).
- **O AcMAPE sai do relatório** (P-07, aprovado em 2026-11-18). Era a métrica mista do TEGRU, que saiu do estudo; continua no `metricas.json`, mas não entra nas tabelas.
- **FinGAT só em direção** (D-06): o retorno dele é treinado por ranking, sem escala; convertê-lo em preço daria um erro sem sentido. MRR e Precision do artigo não são calculados.

## 7. Orçamento

Tempos medidos na máquina do estudo (guia, §6.4): 100 épocas (P-03), 5 sementes (P-05), 6 configurações (§1).

| Configuração                | Pré-processamento                     | Tempo por época | Por semente | × 5 sementes  |
| --------------------------- | ------------------------------------- | --------------- | ----------- | ------------- |
| GRU                         | < 1 s                                 | 0,035 s         | ~4 s        | ~20 s         |
| FinGAT repo                 | ~1 s                                  | 44 s            | **~73 min** | **~6 h 05**   |
| Custom FinGAT               | ~1 s                                  | 2,0 s           | ~3,3 min    | ~17 min       |
| HDFM, 3 configurações       | causal: 13,7 min, uma vez (em cache)  | 0,25 s          | ~25 s cada  | ~6 min        |
| **Total**                   |                                       |                 |             | **~6 h 30**   |

- **O FinGAT repo é quase todo o custo:** um passo do otimizador por dia (848 por época) e 24 GRUs separadas, como no código dos autores.
- **Antes da execução longa, o ensaio geral (Etapa 7):** a fila inteira com 1 época por configuração, do começo à gravação das métricas.
- **Memória:** o maior pico é o do Custom FinGAT (6,8 GB de VRAM de 8, com o sublote de 16); um processo por vez, nunca em paralelo.

## 8. Regras da execução

- **Arquivo de configuração:** depois de aprovado, os valores deste documento vão para um arquivo lido pelo `rodar.py` e pela fila da Etapa 7 (um lugar só para os valores; este documento os descreve e aponta para ele).
- **Uma execução por (configuração, semente)**, cada uma num processo, com checkpoint por época e retomada automática; a fila pula o que terminou e registra o que falhou.
- **O teste só é lido depois de todas as execuções terminarem.** Nenhum valor deste documento muda depois disso; se mudar, a mudança entra numa seção "Alterações depois do teste", com data, motivo e os dois resultados.
- **Tabela final regenerável:** a tabela de resultados é calculada do zero a partir de `execucoes/`, por um script.

## 9. Pontos para revisão

| ID   | Ponto                                                     | Situação                                                                                           |
| ---- | --------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| P-01 | Quais execuções do HDFM existem e para que servem          | **Decidido (2026-11-18):** A causal e B causal na comparação; B inteira só para o vazamento (§1)    |
| P-02 | Dropout na GRU compartilhada                              | **Decidido (2026-11-18):** não usar. O dropout 0,2 do FinGAT repo fica, porque é do código deles    |
| P-03 | Épocas por modelo                                         | **Decidido (2026-11-18):** 100 em todos                                                             |
| P-04 | Parada antecipada                                         | **Decidido (2026-11-18):** não; 100 épocas no máximo, e o checkpoint de menor perda de validação (a época varia por execução) |
| P-05 | Sementes                                                  | **Decidido (2026-11-18):** 5 (0 a 4) em todas                                                       |
| P-06 | Incerteza nas métricas                                    | **Decidido (2026-11-18):** intervalo de Wilson de 95 % na direção (Wilson, 1927; Brown et al., 2001); média ± desvio entre sementes |
| P-07 | AcMAPE                                                    | **Decidido (2026-11-18):** sai do relatório                                                         |
| P-08 | Normalização A do HDFM na Etapa 7                         | **Decidido (2026-11-18):** roda, só com a decomposição causal (§1)                                  |
| P-09 | Hiperparâmetros dos modelos dos artigos                   | **Decidido:** coberto pelo §3; nenhuma busca                                                        |
| P-10 | Divisão 80/10/10 ou `TimeSeriesSplit` (e 10 anos de dados) | **Decidido (2026-11-18):** mantém 80/10/10 sobre 5 anos (§2.2)                                      |

**Resultados das validações rápidas que motivam as propostas** (3 épocas, validação; não escolhem nada, só mostram que o treino funciona):

| Modelo                     | RMSE  | Direção | Observação                                                                 |
| -------------------------- | ----- | ------- | -------------------------------------------------------------------------- |
| Ingênuo                    | 0,712 | 0,664   |                                                                            |
| GRU (100 épocas, Etapa 2)  | 0,688 | 0,672   | média de 3 sementes                                                        |
| FinGAT repo                |       | 0,664   | prevê alta em tudo depois de 3 épocas (prob. 0,53)                          |
| Custom FinGAT              |       | 0,664   | idem (prob. 0,50 a 0,51)                                                    |
| HDFM A, causal             | 5,88  | 0,368   | tendência fora da faixa do treino (D-20)                                   |
| HDFM B, causal             | 0,753 | 0,488   |                                                                            |
| HDFM B, inteira            | 0,436 | 0,768   | vazamento da decomposição (D-10)                                           |

## 10. Desvios dos artigos

Consolidação das decisões que afastam a implementação dos artigos; é a base da seção de limitações do relatório, junto com as limitações L-01 a L-09 do guia (§6.3).

| Modelo         | Desvio                                                                                       | Decisão |
| -------------- | -------------------------------------------------------------------------------------------- | ------- |
| Todos          | Janela 120, a mesma em todos (FinGAT usa 15; HDFM, 30)                                        | D-01    |
| Todos          | Divisão 80/10/10 (FinGAT 60/20/20; HDFM, os últimos 800 pontos como teste)                    | D-02    |
| Todos          | GRU de 1 × 128 em todos (HDFM usa 128-64-32; FinGAT, 16)                                       | D-03    |
| Todos          | Fechamento ajustado como alvo e rótulo de alta                                                | D-04    |
| Todos          | Uma ação-alvo (PETR4), 5 anos, ~125 dias de teste                                              | L-04    |
| GRU            | Normalização por janela, previsão da variação                                                  | D-20, D-21 |
| FinGAT (os dois) | Universo de 48 ações da B3 em 8 setores do mesmo tamanho (o artigo: Taiwan, S&P 500, NASDAQ) | D-07    |
| FinGAT (os dois) | Comparado só em direção da PETR4                                             | D-06    |
| FinGAT (os dois) | GAT densa própria no lugar do PyTorch Geometric                                              | D-09    |
| FinGAT (os dois) | Padronização dos níveis só com o treino                                                      | D-09c   |
| FinGAT repo    | As 15 modificações do código dos autores (`terceiros/fingat_original/ORIGEM.md`)              | D-08    |
| FinGAT repo    | O código dos autores difere do artigo (perda, longo prazo, pooling, lr)                        | L-08    |
| Custom FinGAT  | Médias móveis sobre o ajustado; lr 0,001; lote de dias                                         | D-09a, e, f |
| HDFM (os dois) | Decomposição causal como versão oficial (o artigo decompõe a série inteira)                   | D-10    |
| HDFM (os dois) | Parâmetros de CEEMDAN e VMD da biblioteca; m, r e K da entropia                                 | D-11, D-12, D-13 |
| HDFM (os dois) | Resíduo da VMD somado ao modo mais rápido                                                      | D-22    |
| HDFM (os dois) | Fusão por soma simples (o artigo fala em "soma ponderada", sem pesos)                           | D-14    |
| HDFM B         | Normalização por diferença no lugar do min-max da Eq. 20                                       | D-14    |
