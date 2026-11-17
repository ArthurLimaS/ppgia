# Plano da apresentação: seminário de Redes Neurais (2VA)

**Formato:** individual, presencial, 20 min.
**Base:** `selecao-artigos-seminario.md` (por que estes artigos) e `rastreabilidade-artigos-seminario.md` (de onde vem cada número). Todo dado citado nos slides vem deste último. A coluna **Fonte** de cada slide aponta para o item correspondente (ex.: "R-1.1 #16" = rastreabilidade, seção 1.1, item 16).

---

## 1. Estrutura e orçamento de tempo

| Seção                        | Slides | Tempo     |
| ---------------------------- | ------ | --------- |
| Título                       | 1      | 0:30      |
| 1. Introdução                | 2–3    | 2:00      |
| 2. Referencial teórico       | 4–5    | 2:00      |
| 3. Art. 1: HDFM (`li`)       | 6–10   | 4:40      |
| 4. Art. 2: FinGAT (`hsu`)    | 11–15  | 4:40      |
| 5. Art. 3: TEGRU (`haryono`) | 16–20  | 4:40      |
| 6. Síntese crítica           | 21     | 1:00      |
| 7. Referências               | 22     | 0:30      |
| **Total**                    | **22** | **20:00** |

**Duas mudanças em relação à estrutura proposta:**

1. **Subseção X.5 "Análise crítica" em cada artigo e seção 6 "Síntese crítica".** O enunciado exige "análise crítica sobre os métodos", e a estrutura original não tinha onde colocá-la. É também onde está o material mais forte que levantamos.
2. **22 slides em vez de 21.** Para caber em 20 min, os slides X.1 (ideia geral do artigo) ficam com **40 s**, e não 1 min. São só o "gancho" do artigo; o detalhe vem nos slides seguintes. Se preferir 21 slides com 1 min cada, a melhor fusão é juntar os slides 2 e 3 (linha do tempo e frentes) em um só.

**Ordem dos artigos:** segue as três frentes do plano de dissertação, na ordem (i) → (ii) → (iii). A narrativa é: mudar **o que entra** na rede → mudar **como ela modela** → ampliar **o que ela sabe**.

---

## 2. Slides

### Slide 1: Título · 0:30

- **Título sugerido:** "Redes recorrentes na previsão de séries financeiras: três frentes do estado da arte"
  > [Supervisor] Vamos usar "Previsão de Séries Temporais Financeiras" como título principal e "Recorte estado da arte sobre GRU" no subtítulo.
- Nome, disciplina, PPGIA/UFRPE, data.
- **Fala:** uma frase sobre o tema do mestrado: previsão de séries do mercado de ações com LSTM e GRU.

---

### Slide 2 · Seção 1.1: Introdução, linha do tempo · 1:00

**Conteúdo:**

- **Por que prever o mercado:** além do lucro, o mercado reflete o crescimento econômico e subsidia políticas (`li_accurate_2024`, `ramkumar_development_2023`, `sun_novel_2024`).
  > [Supervisor] Não tem necessidade de falar do "por que". Essa é uma apresentação técnica não de defesa do tema do mestrado.
- **Por que é difícil:** volatilidade, não estacionariedade e ruído.
  > [Supervisor] Motivo similar ao comentário anterior, se a gente não vai mostrar os dados agr, não é legal ficar falando características que são só palavras pro ouvinte.
- **Linha do tempo (visual principal):**
  1. **Estatísticos e ML clássico:** ARIMA, SVM, Random Forest, XGBoost (`mahjouby_predicting_2024`, `sharma_stock_2023`). Bons baselines, mas limitados em relações não lineares.
  2. **Deep Learning:** LSTM supera RNN simples e ARIMA (`k_leveraging_2023`, `miot_empirical_2020`).
  3. **Híbridos:** LSTM + ARIMA (`kiruba_risk_2025`, `ramkumar_development_2023`), ensembles de redes (`liu_application_2024`, `choi_hybrid_2023`).
  4. **Hoje:** três frentes de aprimoramento (próximo slide).

**Visual:** seta horizontal com 4 marcos e 1–2 nomes de modelo em cada.
**Fonte:** `metodologia_2026_1/main.tex:76–86`.

---

### Slide 3 · Seção 1.2: Introdução, as três frentes e os artigos escolhidos · 1:00

**Conteúdo:**

- **As três frentes atuais** (`main.tex:88`), cada uma com o artigo escolhido:

  | Frente                                              | Artigo                       | Ideia em uma linha                              |
  | --------------------------------------------------- | ---------------------------- | ----------------------------------------------- |
  | (i) Pré-processamento e extração de características | HDFM (Li et al., 2024)       | Decompor a série antes de prever                |
  | (ii) Atenção e relações temporais/estruturais       | FinGAT (Hsu et al., 2023)    | Atenção sobre o tempo e sobre o grafo de ativos |
  | (iii) Integração multimodal                         | TEGRU (Haryono et al., 2023) | Juntar preço e sentimento de notícias           |

- **Fio condutor:** nos três, o núcleo é uma **GRU**, encaixada em papéis diferentes: preditor de subséries, codificador temporal e decodificador de um Transformer.
- **Critério de escolha (uma linha):** ligação com a dissertação, rede neural no centro do método, avaliação rica o bastante para criticar.

**Visual:** tabela de 3 linhas com um ícone por frente.
**Fonte:** `selecao-artigos-seminario.md` §1 e §6.

---

### Slide 4 · Seção 2.1: Referencial, de RNN a GRU · 1:00

**Conteúdo (revisão rápida, a turma já viu):**

- **RNN:** $h_t = f(W[h_{t-1}, x_t])$. O estado oculto carrega a memória, mas o gradiente some em sequências longas.
- **GRU** (Cho et al., 2014): duas portas.
  - Atualização $z_t$: quanto do passado manter.
  - Reset $r_t$: quanto do passado usar para gerar o candidato $\tilde{h}_t$.
  - $h_t = (1 - z_t)\,h_{t-1} + z_t\,\tilde{h}_t$
- **GRU × LSTM:** a GRU é mais simples (2 portas contra 3, sem célula separada) e tem desempenho comparável.

  > [Supervisor] Trazer fórmula matemática não é legal. Acho melhor a gente mencionar que a GRU é um tipo de RNN, relembrar que a GRU possui uma lógica com duas portas (uma porta de reset e outra porta de atualização)
  > Não precisa trazer diagrama, se necessário for eu desenho na hora ou então abro no slide da disciplina.

**Visual:** diagrama de uma célula GRU (equivalente à Fig. 1 do `li`, p. 5), com as 4 equações ao lado.
**Fonte:** `li_accurate_2024`, III.D, Eq. (12)–(15), p. 5 (R-1.1).

---

### Slide 5 · Seção 2.2: Referencial, dados de mercado e formulação do problema · 1:00

**Conteúdo:**

- **OHLCV:** Open, High, Low, Close, Volume (um candle por dia). Inclua também o fechamento ajustado e o retorno diário.
- **Formulação comum aos três artigos:** janela deslizante de _L_ dias → previsão do dia seguinte.
- **Como avaliar:**
  - **Erro:** MAPE, RMSE, R².
  - **Direção:** acertou se subiu ou desceu?
  - Erro baixo **não garante** direção certa. Esta é a lente da análise crítica.

**Visual:** candlestick com OHLC anotado + janela deslizante destacando _L_ dias → dia _t_+1.

> [Supervisor] Monta o gráfico analisando dados reais da Petrobras prfv (ver "learning/pytrendseries/exploracao_pytrendseries.ipynb")
> **Fonte:** R-2.1 #4 (atributos do FinGAT); R-3.1 #10 e #17 (indicadores e definição de direção do TEGRU).

---

## Art. 1: HDFM, frente (i)

### Slide 6 · Seção 3.1: HDFM, ideia geral · 0:40

**Conteúdo:**

- _Accurate Stock Price Forecasting Based on Deep Learning and Hierarchical Frequency Decomposition_, IEEE Access, 2024.
- **Ideia:** "dividir para conquistar". A série de preço é ruidosa demais para uma rede só, então ela é **decomposta em subséries mais simples**, cada uma prevista separadamente.

  > [Supervisor] Aqui a ideia é "quebrar" pra conquistar. Uma série temporal financeira muitas vezes possui não-linearidade (mudanças no eixo X de 10% muda diferentemente o eixo Y) e não-estacionaridade (média vai movendo).

  > O que o modelo propoe é quebrar a série estudada para poder em segmentos menores e usar esses seguimentos mais simples como entrada para a rede.

- **Contribuições declaradas:**
  - decomposição hierárquica;
  - agrupamento das subséries para reduzir custo;
  - avaliação em três mercados.

**Fonte:** R-1.1 #1–2.

---

### Slide 7 · Seção 3.2: HDFM, arquitetura · 1:00

**Conteúdo (pipeline):**

1. **CEEMDAN** decompõe o fechamento em 10 IMFs, de alta a baixa frequência.
   > [Supervisor] Não consegui encontrar facilmente no artigo, o que é IMF? É uma sigla para o que? O que representa?
2. **Entropia amostral** mede a complexidade de cada IMF. O **K-means** (com _k_ pelo método do cotovelo) agrupa as IMFs em **3 Co-IMFs**, cada uma a soma das IMFs do grupo.
   > [Supervisor] Aqui acho que você alucinou, nada garante que precisam ser 3 imf's (acho que pode ser até n...)
3. A Co-IMF1 (a mais complexa, de alta frequência) é **redecomposta pela VMD**.
4. **Uma GRU por subsérie**: 3 camadas (128, 64, 32), janela de 30 dias.
5. **Fusão:** soma ponderada das previsões.

**Visual:** versão simplificada da Fig. 2 (p. 6): série → CEEMDAN → K-means → [VMD] → GRUs em paralelo → Σ.
**Fonte:** R-1.1 #3–8, #12–13.

---

### Slide 8 · Seção 3.3: HDFM, dados de teste · 1:00

**Conteúdo:**

- **3 índices:** SSEC (Xangai), SZI (Shenzhen) e **S&P 500**, do **Yahoo Finance**. Fechamento diário de 1990 a 05/2023, cerca de 8 mil observações por índice.
- **Teste:** as **últimas 800 observações**. Normalização min-max.
- **8 baselines:**
  - ingênuos e estatísticos: passeio aleatório, ARIMA;
  - ML clássico: SVR, RF;
  - redes: LSTM, GRU;
  - híbridos: CEEMDAN-GRU, McVCsB.
- **Métricas:** R², RMSE, MAE, MAPE e teste de Diebold-Mariano.

**Fonte:** R-1.1 #9–11, #14–15; período de teste: Fig. 7 (p. 11).

---

### Slide 9 · Seção 3.4: HDFM, resultados · 1:00

**Conteúdo:**

- **Tabela SSEC** (recorte da Tabela 3):

  | Modelo            | R²        | MAPE      |
  | ----------------- | --------- | --------- |
  | Passeio aleatório | 0,734     | 2,91%     |
  | ARIMA             | 0,779     | 2,64%     |
  | LSTM              | 0,946     | 1,21%     |
  | GRU               | 0,952     | 1,13%     |
  | CEEMDAN-GRU       | 0,975     | 0,84%     |
  | McVCsB            | 0,982     | 0,76%     |
  | **HDFM**          | **0,986** | **0,65%** |

- **Diebold-Mariano:** o HDFM é melhor que todos os baselines, nos 3 índices, a 1% de significância.
- **Ablações:** GRU > LSTM > RNN > DNN; CEEMDAN > EMD > EEMD; VMD é a melhor redecomposição.

> [Supervisor] a tabela é a 3 mesmo mas da pra trazer todas as métricas (RMSE e MAE) também. Agr a gente pode manter o recorte do SSEC mesmo

**Fonte:** R-1.1 #16–20 (Tabelas 3, 4, 5, 7).

---

### Slide 10 · Seção 3.5: HDFM, análise crítica · 1:00

**Conteúdo (4 pontos):**

1. **Vazamento não descartável.** O artigo não diz se a decomposição e a normalização usam só o treino; a Fig. 2 decompõe a série antes de qualquer divisão. CEEMDAN e VMD não são causais.
2. **Baseline ingênuo suspeito.** O passeio aleatório erra **mais que o ARIMA** (MAPE 2,91% × 2,64%), e sua implementação não é descrita. R² ≈ 0,98 em nível de preço impressiona pouco.
3. **Pouca evidência sobre LSTM × GRU.** A ablação foi feita na série bruta, não dentro do HDFM.
4. **Limites declarados:** a alta frequência continua mal prevista e só há previsão de 1 passo. Não há métrica de direção nem backtest.

**Fonte:** R-1.2 e R-1.3. Os itens 1 e 2 são inferências marcadas como 🔎 ou "não informado"; formular como pergunta, não como acusação.

> [Supervisor] Vamos remover os pontos 1 e 2. Não tenho tempo para validar essas inferências

---

## Art. 2: FinGAT, frente (ii)

### Slide 11 · Seção 4.1: FinGAT, ideia geral · 0:40

**Conteúdo:**

- _FinGAT: Financial Graph Attention Networks for Recommending Top-K Profitable Stocks_, IEEE TKDE, 2023 (submetido em 2020, publicado on-line em 2021).
- **Problema:** o investidor quer **quais ações comprar**, não o preço exato. Além disso, ações se influenciam, mas essas relações são difíceis de obter e mudam com o tempo.
- **Ideia:** aprender as relações latentes entre ações e entre setores **por atenção**, sem relações predefinidas, e **ordenar** as ações por retorno esperado.

**Fonte:** R-2.1 #1–3; datas no cabeçalho do PDF.

---

### Slide 12 · Seção 4.2: FinGAT, arquitetura · 1:00

**Conteúdo (fluxo na ordem certa):**

1. **GRU + atenção** sobre os 5 dias de cada semana → embedding semanal da ação.
2. **GAT intra-setor:** grafo completo entre ações do mesmo setor. A atenção aprende o peso de cada vizinho.
3. **GRU + atenção entre semanas** (longo prazo), em dois fluxos: com e sem o GAT.
4. **Max-pooling** das ações → embedding do setor → **GAT inter-setor**.
5. **Concatenação** (ação + intra-setor + setor) → duas cabeças:
   - retorno, com perda de **ordenação** par a par;
   - direção, com entropia cruzada.
   - $L = 0{,}99\,L_{rank} + 0{,}01\,L_{move}$

- **Atenção em uma linha:** $\beta_{qn} = \text{softmax}(\text{LeakyReLU}(r^\top[W a_q \,\|\, W a_n]))$, o peso que a ação _q_ dá à ação _n_.

**Visual:** versão simplificada da Fig. 2 do artigo (p. 3), com três blocos: tempo → ações → setores.
**Fonte:** R-2.1 #5–11 (Eq. 4–16).

---

### Slide 13 · Seção 4.3: FinGAT, dados de teste · 1:00

**Conteúdo:**

- **3 mercados:**

  | Mercado | Ações | Setores | Treino/val./teste (dias) |
  | ------- | ----- | ------- | ------------------------ |
  | Taiwan  | 100   | 5       | 579 / 193 / 193          |
  | S&P 500 | 424   | 9       | 579 / 193 / 193          |
  | NASDAQ  | 1.026 | 112     | 756 / 252 / 237          |

- **Entrada e saída:** 15 pregões (3 semanas) → retorno do 16º dia.
- **Atributos:** OHLC, fechamento ajustado, retorno, razões de preço e médias móveis de 5 a 30 dias.
- **Métricas:** de recomendação (MRR@K e Precision@K) e de direção (ACC). Os autores **descartam MAE e RMSE**.
- **Baselines:** MLP, GRU, GRU+Atenção, FineNet, RankLSTM. Média de 10 execuções.

**Fonte:** R-2.1 #4, #12–16.

---

### Slide 14 · Seção 4.4: FinGAT, resultados · 1:00

**Conteúdo:**

- **Recorte da Tabela 2 (K = 5, MRR):**

  | Modelo     | Taiwan    | S&P 500   |
  | ---------- | --------- | --------- |
  | GRU        | 0,312     | 0,116     |
  | RankLSTM   | 0,396     | 0,174     |
  | **FinGAT** | **0,439** | **0,197** |

- **Ganho médio sobre o RankLSTM (K = 5):** +12,23% em MRR e +7,93% em Precision.
- **Exceção:** no NASDAQ com K = 10, perde para o RankLSTM (−12% em MRR).
- **Ablação (Taiwan):**
  - tirar o GAT intra-setor causa a maior queda;
  - trocar a entropia cruzada por MSE derruba o desempenho.
- **Sem setor:** funciona bem, mas só foi testado no Taiwan, com subconjuntos de 10 ações.

> [Supervisor] Esse recorte não ta legal. A gente pode trazer todos os resutlados do S&P 500 mesmo

**Fonte:** R-2.1 #17–21 (Tabelas 2 e 3, Fig. 3).

---

### Slide 15 · Seção 4.5: FinGAT, análise crítica · 1:00

**Conteúdo (4 pontos):**

1. **"Sem relações predefinidas" é parcial.** Os setores são dados de entrada, e o grafo é completo dentro deles. Quem decide a estrutura é a atenção, e os autores propõem inferir o grafo como trabalho futuro.
2. **A direção quase não melhora.**
   - S&P 500: ACC de 0,5425 contra 0,535 do MLP. A perda de direção pesa só 1%.
   - NASDAQ: ACC de **0,26**, pior que o acaso, sem comentário dos autores.
3. **"Significativamente melhor" sem teste estatístico.** O artigo mostra a média de 10 execuções, sem desvio-padrão.
4. **Recomenda ações, mas não mede o retorno da carteira.** Não há backtest nem custos de transação, e o período das bases não é informado.

**Fonte:** R-2.2 e R-2.3.

---

## Art. 3: TEGRU, frente (iii)

### Slide 16 · Seção 5.1: TEGRU, ideia geral · 0:40

**Conteúdo:**

- _Transformer-Gated Recurrent Unit Method for Predicting Stock Price Based on News Sentiments and Technical Indicators_, IEEE Access, 2023.
- **Problema:** notícias afetam preços, mas o sentimento é qualitativo. É preciso transformá-lo em série numérica respeitando o tempo.
- **Três contribuições:**
  - indicador de sentimento por aprendizado ativo;
  - arquitetura **TEGRU**;
  - métrica **AcMAPE**, que pune erro de direção.

**Fonte:** R-3.1 #1–3.

---

### Slide 17 · Seção 5.2: TEGRU, arquitetura · 1:00

**Conteúdo (dois blocos):**

- **(a) Sentimento → números:**
  - manchetes de um portal indonésio → tradução para o inglês → RoBERTa gera os rótulos;
  - **IndoBERT** classifica aspecto e sentimento e é **retreinado mês a mês por aprendizado ativo**. A entropia escolhe o que entra no treino;
  - saem **35 indicadores diários** (7 estatísticas × 5 aspectos).
- **(b) TEGRU:**
  - entrada: indicadores técnicos e/ou de sentimento, filtrados por VIF < 5;
  - **encoder Transformer** (atenção multicabeça + feed-forward);
  - **Linear → GRU** no lugar do decoder;
  - saída: fechamento de _t_+1.

**Visual:** dois diagramas lado a lado. À esquerda, o pipeline de NLP (Fig. 3/4); à direita, a Fig. 5 (p. 8).
**Fonte:** R-3.1 #5, #8–11, #13–14.

---

### Slide 18 · Seção 5.3: TEGRU, dados de teste · 1:00

**Conteúdo:**

- **8 ações da bolsa indonésia**, via yfinance:
  - 5 de grande capitalização: BBCA, BBRI, TLKM, BMRI, ASII;
  - 3 de maior alta em mar./2022: INDX, SMMT, HITS.
- **Notícias:** jan./2017 a mar./2022.
- **5 cenários de atributos:**

  | Cenário | Entrada                           |
  | ------- | --------------------------------- |
  | UNI     | só o fechamento                   |
  | OTI     | OHLCV                             |
  | ETI     | indicadores técnicos expandidos   |
  | CAS     | fechamento + sentimento           |
  | ETS     | indicadores técnicos + sentimento |

- **Defasagens:** de 5 a 200 dias; os resultados são a média entre elas.
- **Baselines:** Transformer, Informer, FEDformer e o encoder sem a GRU.
- **Métricas:** R², RMSE, MSE, MAPE e **AcMAPE** = 2 × (1 − acurácia direcional) + MAPE/100.

**Fonte:** R-3.1 #4, #6–7, #12, #15–17.

---

### Slide 19 · Seção 5.4: TEGRU, resultados · 1:00

**Conteúdo:**

- **TEGRU contra outros Transformers:** melhor na maioria dos cenários. Vence em 5 de 8 ações no UNI e em 6 de 8 no ETI, e empata 4 a 4 com o FEDformer no ETS.
- **MAPE médio por cenário** (Fig. 8):

  | UNI   | OTI   | ETI   | CAS      | ETS  |
  | ----- | ----- | ----- | -------- | ---- |
  | 21,25 | 25,74 | 30,45 | **5,22** | 9,60 |

  O cenário com sentimento (CAS) é o melhor, e também tem o menor AcMAPE (1,00).

- **Para quem o sentimento importa:** mais para as ações de maior alta (INDX e SMMT) do que para as de grande capitalização.
- **Aprendizado ativo:** usa 63,4% menos dados de treino na tarefa de aspecto e 58,0% menos na de sentimento.

**Visual:** gráfico de barras do MAPE médio por cenário.
**Fonte:** R-3.1 #18–25 (Tabelas 7–11, Figs. 8–9).

---

### Slide 20 · Seção 5.5: TEGRU, análise crítica · 1:00

**Conteúdo (4 pontos):**

1. **Os próprios autores mostram o problema.** O melhor cenário acerta a direção em só **52%** dos casos. O BBCA com MAPE de 1,19% acerta **50%**, o mesmo que uma moeda.
2. **A média engana.**
   - O MAPE médio é dominado por INDX e SMMT (o INDX chega a 120% no UNI).
   - O cenário com sentimento cobre só 2017–2022, enquanto as séries de preço vêm desde o IPO.
3. **Falta o baseline óbvio.** Não há comparação com GRU ou LSTM isoladas. A frase "a GRU melhora o encoder em todas as ações" é contrariada nas Tabelas 7 e 9.
4. **Reprodutibilidade:** o artigo não informa a divisão treino/teste nem os hiperparâmetros. Na Tabela 12, a melhor configuração de cada ação é sempre **univariada**.

**Fonte:** R-3.2 e R-3.3.

---

### Slide 21 · Seção 6: Síntese crítica e o que levo para a dissertação · 1:00

**Conteúdo:**

- **Tabela-resumo:**

  |                     | HDFM                                          | FinGAT                              | TEGRU                              |
  | ------------------- | --------------------------------------------- | ----------------------------------- | ---------------------------------- |
  | Papel da GRU        | Preditor de subséries                         | Codificador temporal                | Decodificador do Transformer       |
  | Ganho declarado     | Erro (MAPE 0,65%)                             | Ordenação (+12% MRR)                | Erro (MAPE com sentimento)         |
  | Direção do preço    | **Não mede**                                  | **≈ 54%** (S&P 500)                 | **52%**                            |
  | Lacuna de protocolo | Vazamento possível, baseline ingênuo suspeito | Sem teste estatístico, sem backtest | Sem divisão treino/teste informada |

- **Lição comum:** nenhum dos três mostra ganho na direção do preço. Isso coincide com o diagnóstico do FinTSB (`hu_fintsb_2025`): erros baixos têm pouca relação com o retorno e falta padronização na avaliação.
- **Para a dissertação:**
  1. avaliar **erro e direção**;
  2. **dividir os dados antes** de qualquer pré-processamento;
  3. incluir **baselines ingênuos e recorrentes puros** (LSTM e GRU isoladas);
  4. testar em regimes distintos (FinTSB).

**Fonte:** R-1, R-2, R-3; `main.tex:90` e `:128`.

---

### Slide 22 · Seção 7: Referências · 0:30

- **Os 3 artigos analisados**, em destaque e em ABNT completa:
  - LI et al., 2024;
  - HSU; TSAI; LI, 2023;
  - HARYONO; SARNO; SUNGKONO, 2023.
- **Citados na introdução e na síntese**, em fonte menor: `mahjouby_predicting_2024`, `sharma_stock_2023`, `k_leveraging_2023`, `miot_empirical_2020`, `kiruba_risk_2025`, `ramkumar_development_2023`, `liu_application_2024`, `choi_hybrid_2023`, `sun_novel_2024`, `hu_fintsb_2025` + Cho et al. (2014) para a GRU.
- Entradas prontas em `metodologia_2026_1/referencias.bib`. Cho et al. (2014) **não está** no `.bib`; usar a referência [40] do `li` (p. 17).

---

## 3. Pontos em aberto

- **Correções pendentes em `selecao-artigos-seminario.md`** (§4 do relatório de rastreabilidade). Este plano já usa os fatos corrigidos (ordem do FinGAT, Linear → GRU no TEGRU, ablação na série bruta).
  > [Supervisor] Não cabe agr
- **Passeio aleatório do HDFM (slide 10, ponto 2):** se você quiser que o ponto vire evidência em vez de suspeita, dá para calcular o MAPE da previsão ingênua nos 3 índices com yfinance.
  > [Supervisor] Respondi lá já
- **Slides de arquitetura (7, 12, 17)** são os mais densos para 1 min. Se algum estourar, o candidato a ganhar tempo é o slide 5 (OHLCV), cujo conteúdo de métricas pode migrar para o slide 9.

> [Supervisor] Me faça o seguinte, os pontos que você cita como fonte, devem vir como comentários no slide
