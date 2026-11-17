# Seleção dos artigos para o seminário de Redes Neurais (2VA)

**Seminário:** individual, presencial, 20 min. São 3 artigos do estado da arte ligados ao tema do mestrado (redes neurais), com descrição dos métodos, do funcionamento e uma análise crítica.

**Fontes usadas:** `metodologia_2026_1/main.tex` (plano de dissertação), `artigos/refinamento_trabalhos_relacionados.txt` (fichamento dos 20 trabalhos) e `metodologia_2026_1/referencias.bib` (veículos de publicação).

---

## 1. Resultado

| Frente da introdução do plano                       | Artigo selecionado                       | Rede neural central                                       | Veículo           |
| --------------------------------------------------- | ---------------------------------------- | --------------------------------------------------------- | ----------------- |
| (i) Pré-processamento e extração de características | `li_accurate_2024` (HDFM)                | GRU sobre subséries decompostas (CEEMDAN + K-means + VMD) | IEEE Access, 2024 |
| (ii) Atenção e relações temporais/estruturais       | `hsu_fingat_2023` (FinGAT)               | GRU com atenção + Graph Attention Networks                | IEEE TKDE, 2023   |
| (iii) Integração multimodal                         | `haryono_transformer-gated_2023` (TEGRU) | Encoder Transformer + GRU, com sentimento de notícias     | IEEE Access, 2023 |

A escolha anterior se mantém, agora com justificativa explícita. A única troca que vale considerar é `haryono` → `anbaee_farimani_adaptive_2024` na frente (iii) (ver §5).

---

## 2. Critérios de seleção

Cada artigo precisa atender ao enunciado do seminário e, ao mesmo tempo, servir à dissertação:

| #   | Critério                                                                                                                         | Por que importa                                                                 |
| --- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| C1  | **Aderência à dissertação**: ações, séries diárias, redes recorrentes (LSTM/GRU), regressão de preço                             | O enunciado exige ligação com o tema do mestrado                                |
| C2  | **Rede neural no centro do método**                                                                                              | A disciplina é de redes neurais: o "funcionamento" a explicar tem de ser neural |
| C3  | **Estado da arte**: trabalho recente e veículo de qualidade                                                                      | Exigência do enunciado                                                          |
| C4  | **Material para análise crítica**: avaliação rica o bastante para discutir (ablação, testes estatísticos, limitações declaradas) | Parte explícita da nota                                                         |
| C5  | **Cabe em ~5–6 min** de apresentação                                                                                             | 20 min ÷ 3 artigos, mais introdução e síntese                                   |
| C6  | **Complementaridade**: cada artigo cobre uma frente diferente e o trio conta uma história coerente                               | Evita três apresentações redundantes                                            |

---

## 3. Triagem inicial (11 trabalhos eliminados de saída)

| Grupo                       | Trabalhos                                                               | Motivo da exclusão                                                                                                                                                      |
| --------------------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ML clássico                 | `mahjouby_predicting_2024`, `sharma_stock_2023`                         | A rede neural é secundária ou ausente (C2). Os protocolos são frágeis (sem divisão treino/teste e sem hiperparâmetros)                                                  |
| LSTM "puro"                 | `k_leveraging_2023`, `miot_empirical_2020`                              | Servem de base, não de estado da arte (C3). `miot` usa dados simulados e é de 2020                                                                                      |
| Híbridos com ARIMA/stacking | `kiruba_risk_2025`, `ramkumar_development_2023`, `liu_application_2024` | Publicados em conferências menores, com inconsistências graves no fichamento: dados fictícios, "acurácia" em regressão, métricas anunciadas e não apresentadas (C3, C4) |
| Benchmark                   | `hu_fintsb_2025`                                                        | É uma infraestrutura de avaliação, não um método. **Mas serve como lente da análise crítica** (§6)                                                                      |
| Fora do foco neural/recente | `anghinoni_time_2019`                                                   | O método central é detecção de comunidades em redes complexas. Os classificadores são genéricos e o trabalho é de 2019 (C2, C3)                                         |
| Fora do foco multimodal     | `wang_stock_2023`                                                       | ⚠️ **Não é multimodal**: usa só dados de negociação e indicadores técnicos, e o próprio fichamento aponta a integração de outras fontes como trabalho futuro. Ver §7    |
| Fora do domínio             | `wan_tcdformer_2024`                                                    | Benchmark genérico de previsão de longo horizonte, com uma única base financeira (câmbio) e sem redes recorrentes (C1)                                                  |

Os 9 restantes foram comparados por frente.

---

## 4. Comparação por frente

### Frente (i): pré-processamento e extração de características

| Candidato                     | A favor                                                                                                                                                                                                                                                                                                                                                                             | Contra                                                                                                                                                                                                                          |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`li_accurate_2024`** ✅     | A GRU é o preditor (C2). Usa Yahoo Finance com SSEC, SZI e **S&P 500**, a mesma fonte do plano (C1). Tem o baseline mais completo do grupo, incluindo **passeio aleatório**, ARIMA, SVR, RF, LSTM, GRU e híbridos. Aplica o **teste de Diebold-Mariano**. As ablações comparam **GRU × LSTM × RNN × densa** como preditor base, CEEMDAN × EMD × EEMD e VMD como redecomposição (C4) | Só faz previsão de um passo à frente. Custo alto (uma GRU por subsérie). Não traz métrica direcional                                                                                                                            |
| `sun_novel_2024`              | Decomposição (CEEMDAN) + LSTM, com acurácia direcional                                                                                                                                                                                                                                                                                                                              | O preditor final é **LightGBM**, não rede neural (C2). O fichamento aponta **vazamento** (decomposição aplicada antes da divisão treino/teste). Só detalha uma ação                                                             |
| `hosseini_pattern-based_2025` | Compara LSTM e GRU, tem ablação e teste de significância                                                                                                                                                                                                                                                                                                                            | A tarefa é classificação intradiária no mercado de carbono (C1). Os segmentos são **embaralhados** antes da divisão (vazamento). A parte neural é simples (LSTM + densa), e a contribuição está nos atributos feitos à mão (C2) |
| `wan_tcdformer_2024`          | Arquitetura neural rica                                                                                                                                                                                                                                                                                                                                                             | Fora do domínio (ver triagem)                                                                                                                                                                                                   |

**Por que `li_accurate_2024`:** de longe, é o artigo da frente (i) com a avaliação mais rigorosa. É também o único cuja ablação responde diretamente à pergunta central da dissertação, LSTM ou GRU (a GRU venceu nos três índices). A decomposição funciona como um módulo de entrada que se acopla a qualquer rede recorrente, exatamente como o plano descreve: "técnicas de decomposição de sinais (como CEEMDAN e VMD) ajudam a isolar ruídos".

### Frente (ii): atenção e arquiteturas temporais/estruturais

| Candidato                | A favor                                                                                                                                                                                                                                                                                                                                     | Contra                                                                                                                                                              |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`hsu_fingat_2023`** ✅ | Único artigo do conjunto em periódico **top-tier (IEEE TKDE)** (C3). É o mais denso em redes neurais: GRU hierárquica (intra e inter-semana), atenção temporal, **GAT** em dois níveis (ações e setores) e aprendizado multitarefa (C2). Avalia três mercados com 100, 424 e 1.026 ações (C4). Os autores declaram limitações honestas (C4) | A tarefa é **recomendação Top-K por ordenação**, não regressão de preço (C1 parcial). Exige mais tempo para explicar o grafo (C5)                                   |
| `luo_short-term_2024`    | CNN-BiLSTM-Atenção, a arquitetura mais próxima de "LSTM + atenção"                                                                                                                                                                                                                                                                          | Prevê **correlação** entre ações, não preço (C1). Só 8 ações de um mercado. Não informa se a divisão respeita a ordem cronológica                                   |
| `biswas_dual_2025`       | Regressão de preço com atenção; inclui backtesting                                                                                                                                                                                                                                                                                          | A TCN não é recorrente. Só 2 empresas do mesmo setor. O mesmo MAE aparece **ora como validação, ora como teste**. Baselines sem configuração descrita (C4 negativo) |

**Por que `hsu_fingat_2023`:** é o artigo que mais "ensina rede neural" (atenção sobre estados ocultos de GRU, atenção em grafos, multitarefa) e o de maior qualidade de publicação. O descompasso de tarefa (ordenação × regressão) é real, mas pode virar ponto da análise crítica em vez de defeito da seleção (§5). Os dois componentes também são transplantáveis para a dissertação: a atenção temporal sobre a GRU entra direto no baseline recorrente, e o GAT entre ativos é a frente exploratória citada no plano ("mapear a influência relacional entre diferentes ativos").

### Frente (iii): integração multimodal

| Candidato                               | A favor                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Contra                                                                                                                                                                                                                                                 |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **`haryono_transformer-gated_2023`** ✅ | Ações, dados diários, **yfinance** e regressão do fechamento (C1 pleno). A arquitetura **encoder Transformer + GRU** é exatamente uma das vias citadas no plano. O desenho experimental tem **5 cenários de atributos** (só preço → técnicos → técnicos expandidos → preço + sentimento → tudo) e janelas de 5 a 200 dias, ou seja, uma ablação de modalidades (C4). Propõe a métrica **AcMAPE** e os próprios autores mostram que **MAPE baixo coexiste com acerto direcional de ~50%** (C4) | Os baselines são só da família Transformer: **não compara com LSTM/GRU isoladas**. Grande parte da contribuição é o pipeline de NLP (tradução, RoBERTa, IndoBERT com aprendizado ativo), que não é o foco do seminário (C5). Usa só 8 ações indonésias |
| `anbaee_farimani_adaptive_2024`         | **Baseada em LSTM**, com regressão. Ablação com 1, 2 e 3 modalidades. **Teste de Wilcoxon**. Normalização adaptativa contra não estacionariedade. Simulação de negociação                                                                                                                                                                                                                                                                                                                     | Não trabalha com ações (câmbio, ouro e bitcoin **horários**) (C1). A fusão envolve três sub-redes, extração de conceitos por BERT + k-means e convolucional recorrente, o que é difícil de explicar em ~5 min (C5)                                     |
| `choi_hybrid_2023`                      | GRU + BERT + MLP-Mixer, com ablação GRU × LSTM                                                                                                                                                                                                                                                                                                                                                                                                                                                | Classificação de direção. **Só tem conjunto de validação, sem teste**. Revocação de 95% com especificidade de 41% indica viés para "alta"                                                                                                              |

**Por que `haryono_transformer-gated_2023`:** é o mais aderente ao cenário da dissertação (ações, diário, Yahoo Finance, regressão) e sua arquitetura é a que o plano propõe testar. O achado sobre MAPE × direção conecta com o plano, que prevê "métricas de acurácia direcional", e com o FinTSB ("erros baixos têm pouca correlação com o retorno").

---

## 5. Defesa de cada escolha: o que apresentar e o que criticar

### 5.1 `li_accurate_2024`: HDFM (frente i)

**Funcionamento (o que mostrar):**

1. CEEMDAN decompõe o fechamento em IMFs + resíduo.
2. Para cada IMF, calcula-se a **entropia amostral**. O K-means (com _k_ pelo método do cotovelo) agrupa as IMFs por complexidade, e as IMFs de cada grupo são somadas, o que reduz o número de modelos.
3. O grupo de **maior frequência** é redecomposto por **VMD**.
4. Uma **GRU por subsérie** prevê o dia seguinte com janela de 30 dias, e as previsões são somadas.

**Resultados-chave:** no SSEC, R² 0,9856, RMSE 27,25 e MAPE 0,65%, contra 1,13% da GRU pura e 1,21% da LSTM pura. O teste de Diebold-Mariano é significativo a 1% contra todos os baselines.

**Análise crítica:**

- **Possível vazamento na decomposição** _(verificar no PDF)_: CEEMDAN e VMD não são causais. Se forem aplicados à série inteira antes de separar as 800 observações de teste, as subséries de treino "enxergam" o futuro. Esse problema é recorrente na literatura de decomposição e já foi identificado no fichamento de `sun_novel_2024`.
- **R² ≈ 0,98 em nível de preço** é inflado pela persistência da série. A comparação que importa é contra o **passeio aleatório** (incluído, um ponto positivo) e o ganho de MAPE sobre a GRU pura.
- Os próprios autores admitem que as subséries de **alta frequência** (justamente o "ruído") continuam mal previstas.
- Não há métrica direcional nem simulação de carteira (_Real-World Mismatch_ do FinTSB). Só prevê um passo à frente.
- Custo: várias GRUs mais duas decomposições por série.

**Ligação com a dissertação:** a ablação GRU > LSTM > RNN alimenta a comparação LSTM × GRU, e a decomposição é candidata direta a "estratégia de otimização" sobre o baseline.

### 5.2 `hsu_fingat_2023`: FinGAT (frente ii)

**Funcionamento (o que mostrar):**

1. Atributos diários de cada ação (OHLC, retornos, médias móveis) em janelas de 15 pregões.
2. **GRU com atenção** dentro de cada semana (curto prazo), depois **GRU entre semanas** (longo prazo).
3. **GAT entre ações do mesmo setor** (grafo completamente conectado). A atenção aprende quais vizinhos importam.
4. **GAT entre setores**, cujas representações são propagadas de volta às ações.
5. Saída **multitarefa**: ordenação por retorno esperado e classificação de direção.

**Resultados-chave:** supera MLP, GRU, GRU+Atenção, FineNet e RankLSTM nas três bases, com +12,23% de MRR e +7,93% de precisão no topo sobre o RankLSTM. Mantém o desempenho mesmo sem a informação de setor.

**Análise crítica:**

- **"Sem relações predefinidas" é parcial**: o grafo é completamente conectado _dentro de setores predefinidos_, e quem infere a estrutura é a atenção. Os próprios autores propõem, como trabalho futuro, _inferir_ o grafo.
- **A ordenação melhora, mas a direção não**, nas palavras dos autores, sem "ganho significativo". Fica a pergunta: o modelo aprende sinal de retorno ou só ranking relativo?
- No NASDAQ com Top-10, perde para o RankLSTM: a vantagem não é uniforme.
- Grafo denso: custo O(n²) por setor, relevante no NASDAQ (1.026 ações, 112 setores).
- Pelo fichamento, não há custos de transação nem backtest. _(Verificar no PDF se existe simulação de retorno e quais baselines de grafo foram usados.)_
- A tarefa difere da dissertação. O ponto a defender é que **os módulos (atenção temporal sobre GRU e GAT entre ativos) são reutilizáveis em regressão**, não o modelo inteiro.

### 5.3 `haryono_transformer-gated_2023`: TEGRU (frente iii)

**Funcionamento (o que mostrar, com o NLP resumido em 1 slide):**

1. Manchetes de um portal indonésio passam por tradução e rotulação via RoBERTa. Um classificador IndoBERT de aspecto e sentimento é atualizado mensalmente por **aprendizado ativo** e gera **35 indicadores diários de sentimento**.
2. OHLCV expandido para **43 indicadores técnicos**, com seleção de atributos contra multicolinearidade.
3. **TEGRU**: codificação posicional → **encoder Transformer (atenção multicabeça)** → **GRU no lugar do decoder** → fechamento do dia seguinte.
4. Avaliação em 5 cenários de atributos × janelas de 5 a 200 dias × 8 ações, comparando com Transformer, encoder puro, Informer e FEDformer.

**Resultados-chave:** o aprendizado ativo reduz em até 63,4% os dados rotulados (acurácia de 0,87 e 0,88). A TEGRU vence na maioria dos cenários e empata com o FEDformer no cenário completo. O cenário "fechamento + sentimento" tem o menor MAPE médio (5,22) e o melhor AcMAPE. O sentimento ajuda mais as ações de alta valorização do que as de grande capitalização.

**Análise crítica:**

- **O achado mais forte é contra o próprio paradigma**: o melhor cenário acerta a direção em só **52%** dos casos, e configurações com MAPE de 1,19% têm acurácia direcional de ~50%. Métricas de erro baixas não bastam, como também afirma o FinTSB.
- **Falta o baseline recorrente puro** (GRU ou LSTM sozinhas). Por isso não dá para separar quanto do ganho vem do encoder Transformer e quanto da GRU.
- O sentimento vem de notícias **gerais**, não específicas de cada ação (os autores reconhecem), e passa por **tradução automática**, mais uma fonte de ruído.
- Amostra pequena (8 ações) de um único mercado. _(Verificar no PDF a divisão treino/teste e como os resultados são agregados entre janelas.)_

**Ligação com a dissertação:** é o modelo mais próximo do que o plano propõe na frente multimodal, e o desenho de 5 cenários é um molde pronto para a ablação de variáveis exógenas na dissertação.

---

## 6. Coerência do trio (a "história" da apresentação)

**Fio condutor: a GRU em três papéis.** Nos três artigos, uma rede recorrente com portas é o núcleo, mas encaixada de formas diferentes:

| Artigo | Onde entra a GRU                         | O que muda em volta dela                                         |
| ------ | ---------------------------------------- | ---------------------------------------------------------------- |
| HDFM   | Preditor de cada subsérie                | **A entrada**: a série é limpa e decomposta antes                |
| FinGAT | Codificador temporal de cada ação        | **A arquitetura**: atenção + grafo modelam relações entre ativos |
| TEGRU  | Decodificador após o encoder Transformer | **A informação**: entram dados exógenos (sentimento)             |

A progressão é: mudar o que entra → mudar como se modela → ampliar o que se sabe. São as três frentes da introdução do plano, na mesma ordem.

**Fio crítico transversal (slide de síntese):** nenhum dos três demonstra ganho na **direção** do preço. O HDFM não mede, o FinGAT não ganha e o TEGRU fica em 52%. Ligado ao FinTSB (_Standardization Deficit_ e _Real-World Mismatch_), isso justifica a escolha da dissertação de avaliar com MSE/RMSE/MAPE **e** acurácia direcional, sob regimes de mercado distintos.

**Ressalva assumida:** nenhum dos três é centrado em LSTM, embora a dissertação compare LSTM e GRU. Isso tem defesa: a ablação do HDFM mostra a GRU superando a LSTM, e o próprio plano trata as duas como intercambiáveis no núcleo. Se o professor ou o orientador fizer questão de uma LSTM, a troca está na §7.

**Orçamento de tempo sugerido (20 min):** ~2 min de contexto (problema e três frentes) + 3 × ~5,5 min por artigo + ~2 min de síntese crítica.

---

## 7. Riscos, alternativas e pendências

**Substituto principal:** `anbaee_farimani_adaptive_2024` no lugar do `haryono`, se a prioridade for (a) ter uma LSTM no trio ou (b) rigor estatístico (Wilcoxon, ablação de 1/2/3 modalidades). O custo é se afastar do cenário de ações diárias e ter uma arquitetura mais difícil de explicar no tempo.

**Reserva da frente (ii):** `luo_short-term_2024` (CNN-BiLSTM-Atenção), se o descompasso de tarefa do FinGAT pesar demais. Tem arquitetura mais próxima do baseline da dissertação, mas avaliação mais fraca.

**A verificar nos PDFs antes de montar os slides:**

- [ ] `li_accurate_2024`: a decomposição (CEEMDAN/VMD) é aplicada à série inteira ou só ao treino, de forma rolante? A normalização min-max é ajustada só no treino?
- [ ] `hsu_fingat_2023`: há simulação de retorno ou custos? Quais baselines de grafo foram usados?
- [ ] `haryono_transformer-gated_2023`: qual a divisão treino/teste? Como os resultados são agregados entre as 8 ações e as janelas?

**Inconsistência encontrada no plano de dissertação:** `wang_stock_2023` é citado como exemplo de integração multimodal em `metodologia_2026_1/main.tex:88` e `:126`, mas o fichamento registra que ele usa **só dados de negociação e indicadores técnicos**. Vale mover a citação para a frente (ii), já que é BiLSTM + Transformer + TCN, ou reescrever a frase.
