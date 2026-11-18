# Layout do relatório — comparação de 5 modelos na PETR4

> **RASCUNHO para revisão (2026-11-18).** Organização das seções do relatório (formato de artigo) e o que cada uma explica. Base: [`condicoes-do-estudo.md`](condicoes-do-estudo.md) e [`guia-implementacao-projeto.md`](guia-implementacao-projeto.md). Cada seção tem uma linha de **Insumos**: o que já existe e de onde vem, ou o que ainda falta. As lacunas estão juntas no fim (§ "O que falta para escrever"). A apresentação de 10 minutos não está aqui; tem guia próprio depois dos resultados.

**Título provisório:** _O que os extras acrescentam a uma GRU? Uma comparação controlada de FinGAT e HDFM na previsão diária da PETR4_

---

## Resumo

Um parágrafo com o problema, o método e os números principais. O problema: artigos recentes de previsão de ações propõem extras sobre redes recorrentes (grafos de ações, decomposição do sinal) e relatam ganhos, mas cada um com seus dados, sua divisão e sua configuração de rede, o que impede saber de onde vem o ganho. O método: os dois artigos são reimplementados sobre **a mesma GRU**, escolhida antes por um estudo próprio, e comparados com ela e com a referência ingênua na PETR4, no mesmo período de teste, com 5 sementes.

Os resultados entram em uma ou duas frases: o ganho (ou a falta de ganho) de cada modelo sobre a GRU e sobre o ingênuo, o tamanho do vazamento temporal do HDFM como publicado, e a constatação, se se confirmar, de que nenhum modelo distingue a direção do preço do acaso no período de teste. Fecha com a contribuição: o protocolo de comparação e a correção causal do HDFM.

- **Insumos:** números da Etapa 7. O resto está pronto.

## 1. Introdução

Contextualiza a previsão de preços de ações com redes neurais e a escolha da GRU (Cho et al., 2014) como base: arquitetura recorrente mais simples que a LSTM, e a que os dois artigos usam. Apresenta o problema que motiva o estudo: nos artigos, o ganho atribuído aos extras se mistura com diferenças de dados, divisão, normalização e tamanho da rede, e a avaliação costuma usar só métricas de preço, em que a referência "amanhã = hoje" já é difícil de bater.

Fecha com os objetivos e as contribuições: (i) uma GRU escolhida por estudo próprio (108 combinações de entrada, janela e arquitetura) e usada como bloco comum; (ii) a reimplementação do FinGAT em duas leituras, o código dos autores e o texto do artigo, que divergem; (iii) a reimplementação do HDFM com decomposição causal, e a medida do vazamento da versão publicada; (iv) a comparação dos cinco modelos com o mesmo protocolo. Termina com a organização do texto e um parágrafo curto sobre o escopo: o enunciado pede 4 arquiteturas de 3 artigos, e o estudo entrega 5 modelos de 2 artigos, porque o terceiro (TEGRU) depende de notícias que não existem para a PETR4.

- **Insumos:** prontos (guia §1, escopo revisto; Etapa 5 removida; `resultados-2b.md`).

## 2. Fundamentação teórica

Explica os componentes que os modelos combinam, na medida necessária para ler a metodologia: a célula GRU (portas de atualização e de reinício); atenção sobre passos de tempo; redes de atenção em grafos (GAT), incluindo o que é um grafo completo intra-setor e o papel do laço próprio; e a decomposição de sinais (EMD, CEEMDAN, VMD) junto com a entropia amostral, que o HDFM usa para agrupar as componentes.

Um segundo parágrafo trata de dois conceitos de avaliação que sustentam as decisões do estudo: o **vazamento temporal** (informação do futuro usada para treinar ou para montar a entrada, como numa decomposição feita sobre a série inteira) e a **normalização fora da faixa** (uma rede com saída limitada não extrapola um nível de preço que nunca viu, o que motiva a normalização por janela da D-20).

- **Insumos:** prontos em conteúdo; faltam as referências primárias de GAT, CEEMDAN, VMD e entropia amostral (lista no fim).

## 3. Trabalhos relacionados

Apresenta os dois artigos reproduzidos: o **FinGAT** (Hsu et al., 2023), que modela as relações entre ações do mesmo setor e entre setores com GAT sobre GRUs atentivas semanais e é treinado para ranquear ações; e o **HDFM** (Li et al., 2024), que decompõe o preço em subséries e treina uma GRU por subsérie. Para cada um: o que propõe, como avalia, e o ponto crítico já levantado no seminário (o FinGAT não ganha em direção; o HDFM não mede direção e, como publicado, decompõe a série inteira).

Um parágrafo final situa o que ficou de fora e por quê: o TEGRU (Haryono et al., 2023), cuja contribuição é o sentimento de notícias; e, se couber, trabalhos sobre a avaliação de modelos financeiros (por exemplo, o FinTSB) que apoiam a crítica de protocolo feita aqui.

- **Insumos:** prontos (`selecao-artigos-seminario.md`, `rastreabilidade-artigos-seminario.md`, `plano-apresentacao.md`).

## 4. Metodologia

### 4.1. Dados e tarefa

Descreve o snapshot: PETR4, fechamento ajustado, 1.248 pregões de 2021-11-16 a 2026-11-13, baixados uma vez do Yahoo Finance e lidos sempre do disco, com sha256 conferido. A tarefa: com os 120 pregões até `t`, prever `t+1`. A divisão cronológica 80/10/10 pelo dia-alvo, com as datas de validação e de teste, iguais para todos os modelos. Para o FinGAT, o universo de 48 ações da B3 (8 setores × 6) e o critério de seleção.

Um parágrafo sobre o achado da Etapa 0 que condiciona tudo: no fechamento ajustado, a validação e o teste estão acima de toda a faixa de preços do treino (7,72 a ~32 no treino; até ~60 no teste), o que leva à normalização por janela e explica a falha da normalização literal do HDFM.

- **Insumos:** prontos (`condicoes-do-estudo.md` §2; guia, Etapa 0 e D-20).

### 4.2. Protocolo de avaliação

Reúne as regras que tornam a comparação justa: nenhuma estatística de validação ou de teste em nenhuma normalização; o teste não escolhe nada (janela, arquitetura e checkpoint saem da validação); no máximo 100 épocas em todos, com o checkpoint de menor perda de validação; 5 sementes por configuração; mesmos dias de teste e mesmo rótulo de alta para todos.

O segundo parágrafo define as métricas: RMSE, MAE, MAPE e R² sobre o preço reconstruído (GRU e HDFM); acurácia de direção para todos, com a regra de cada tipo de modelo; a referência ingênua; o intervalo de Wilson de 95 % na acurácia de direção (Wilson, 1927; Brown, Cai e DasGupta, 2001) e média ± desvio entre sementes. Explica por que o FinGAT entra só em direção (o retorno dele é treinado por ranking, sem escala).

- **Insumos:** prontos (`condicoes-do-estudo.md` §4 a §6).

### 4.3. A GRU compartilhada

Descreve o estudo da Etapa 2: 2 entradas (fechamento ou OHLCV) × 6 janelas (5 a 200) × 9 arquiteturas (1 a 3 camadas, 16 a 128 unidades), 3 sementes, 100 épocas, mesmos dias-alvo para todas as janelas (D-17). Apresenta a escolha (OHLCV, janela 120, 1 camada de 128) e o critério (menor RMSE de validação).

Deixa explícito que o estudo **seguiu o mesmo protocolo da comparação final** (§4.2): a mesma divisão 80/10/10, a escolha feita só na validação (os mesmos 125 dias), o teste intocado, no máximo 100 épocas com o checkpoint de menor perda de validação. Só duas coisas diferem, e o texto diz por quê: (i) 3 sementes em vez de 5, porque eram 324 execuções; (ii) todas as janelas treinam a partir do mesmo dia-alvo (o 200º pregão, 798 amostras), para que a comparação entre janelas meça a janela e não a quantidade de dados; na execução final, a janela escolhida usa todos os dias que alcança (878 amostras). Refazer o estudo com 5 sementes não mudaria a conclusão: as primeiras colocadas diferem em 0,0002 a 0,0025 de RMSE, abaixo do desvio entre sementes (0,0015 a 0,0044).

Diz com franqueza o limite da escolha (L-07): ela ganha do ingênuo por só 3,4 % e fica dentro do ruído entre sementes, então é "a de menor erro", não "a claramente melhor". E explica o que "a mesma GRU" significa nos outros modelos: mesma classe e configuração, com entrada, comprimento de sequência e número de instâncias definidos por cada artigo.

- **Insumos:** prontos (`resultados-2b.md`; guia, Etapa 2 e L-07; `condicoes-do-estudo.md` §3).

### 4.4. Modelos

Um parágrafo por modelo, com um diagrama de blocos cada. **GRU:** bloco + camada linear, prevendo a variação relativa. **FinGAT repo:** o código dos autores com só os ajustes que a comparação exige (bloco GRU comum, GAT sem PyTorch Geometric, universo da B3). **Custom FinGAT:** o texto do artigo (GRU atentiva de curto prazo, GAT por semana, GRUs atentivas de longo prazo, max-pooling e GAT entre setores, perda de ranking por diferença).

Para o HDFM, a cadeia CEEMDAN → entropia amostral → K-means (K = 3) → VMD (10 modos) → 12 GRUs → soma, com as duas normalizações (A, a leitura literal da Eq. 20; B, por diferença) e as duas decomposições (causal, a oficial; série inteira, a do artigo). Inclui o que foi preciso decidir porque o artigo não diz (resíduo da VMD, parâmetros das bibliotecas).

- **Insumos:** texto pronto (`condicoes-do-estudo.md` §1 e §3; `terceiros/fingat_original/ORIGEM.md`). **Faltam os diagramas** dos quatro tipos de modelo.

### 4.5. Implementação e reprodutibilidade

Ambiente (Python, PyTorch 2.14 com CUDA, Windows 11, RTX 3060 Ti, 16 GB), dependências das decomposições e o que garante a reprodução: snapshot com hash, semente gravada, configuração completa em cada execução, checkpoint com retomada testada, previsões gravadas e métricas recalculáveis, testes de fumaça (vazamento temporal, máscara do grafo, retomada).

Um parágrafo sobre o código de terceiros: a origem do FinGAT (repositório e commit), a ausência de licença, as 15 modificações e onde estão listadas.

- **Insumos:** prontos (guia §2 e §6.4; `ORIGEM.md`; `testes/`).

## 5. Resultados

### 5.1. Comparação principal

A tabela central: uma linha por modelo (GRU, FinGAT repo, Custom FinGAT, HDFM A causal, HDFM B causal) e a referência ingênua; colunas RMSE, MAE, MAPE, R² e acurácia de direção com o intervalo de Wilson, todas como média ± desvio de 5 sementes no teste. Um gráfico de previsto contra real no período de teste para a GRU e os dois melhores modelos.

O texto lê a tabela contra a GRU, não só contra o ingênuo: o que cada extra acrescentou ao mesmo bloco, e se a diferença é maior que a variação entre sementes e que o intervalo da direção.

- **Insumos:** **Etapa 7** (execuções e o script da tabela final).

### 5.2. O vazamento da decomposição no HDFM

Compara a HDFM B com decomposição da série inteira (como o artigo aparenta) e com a causal, no mesmo protocolo. A diferença entre as duas é a medida de quanto o vazamento infla o resultado; na validação rápida, a primeira bateu o ingênuo com folga (RMSE 0,44 contra 0,71) e a causal não (0,75).

Inclui um exemplo visual de por que vaza: o valor de uma subsérie num dia muda quando a série ganha dias depois dele. É a seção que justifica a correção do estudo e a principal contribuição crítica do relatório.

- **Insumos:** **Etapa 7** (B inteira e B causal com 5 sementes). **Falta a figura** do vazamento (a mesma subsérie decomposta até `t` e até o fim).

### 5.3. Código dos autores contra texto do artigo (FinGAT)

Compara o FinGAT repo e o Custom FinGAT em direção, na PETR4 e nas 48 ações (a ACC do artigo), e lista as diferenças entre o código e o texto (perda, GRU de longo prazo, onde entra a GAT, pooling, taxa de aprendizado). O ponto: o "FinGAT" que se avalia depende de qual das duas fontes se toma.

- **Insumos:** **Etapa 7**; a lista de diferenças está pronta (`ORIGEM.md`; guia, Etapa 3).

### 5.4. Normalização e faixa de preços

Mostra a falha da normalização A do HDFM (min-max por subsérie, a leitura literal): a subsérie de tendência no teste passa de tudo o que o treino viu, e a rede não extrapola. Liga o resultado à decisão D-20 da GRU, que resolve o mesmo problema na entrada.

- **Insumos:** **Etapa 7**; a figura das subséries existe (`execucoes/hdfm_conferencia.png`), falta marcar nela a faixa do treino.

### 5.5. Custo computacional

Tabela com parâmetros, tempo por época, tempo total, pico de memória e a época escolhida de cada modelo, mais o custo único da decomposição causal (1.129 decomposições, 13,7 min). Comenta o custo relativo ao ganho: uma época do FinGAT repo (44 s) custa mais de 1.000 vezes uma da GRU (0,035 s).

- **Insumos:** parcialmente prontos (guia §6.4); a época escolhida e o tempo total vêm da Etapa 7.

## 6. Discussão

Interpreta os resultados à luz da pergunta do estudo: o que cada extra acrescenta a uma GRU fixa. Discute o que não se pode separar (os modelos dos artigos têm de 7 a 28 vezes mais parâmetros que a GRU; parte de um ganho pode ser tamanho), a direção perto do acaso (com 125 dias, o intervalo tem ± 8,6 pontos, e o ingênuo acerta 53,6 % só prevendo alta), e o peso do protocolo nos números publicados (vazamento, normalização, período de teste).

Retoma o fio crítico do seminário: nenhum dos artigos demonstra ganho em direção, e o estudo confirma ou contesta isso no mesmo protocolo para todos.

- **Insumos:** **Etapa 7**; o argumento está pronto (`selecao-artigos-seminario.md` §6; guia R-05, R-06, L-03).

## 7. Limitações

Lista consolidada: uma ação e um período de teste de 125 dias numa alta forte (L-04); a GRU escolhida sem vitória clara (L-07); a escolha de entrada só alcança a GRU (L-02); a GRU escolhida isolada, não dentro de cada modelo (L-03); FinGAT com código divergente do artigo (L-08) e universo com viés de sobrevivência (L-09); comparação desigual por construção (L-06); dois artigos em vez de três (L-05).

Um parágrafo com os desvios dos artigos que podem mudar o resultado (janela 120 em todos, GRU de 1 × 128 em todos, parâmetros de decomposição da biblioteca, normalizações), apontando para o apêndice com a lista completa.

- **Insumos:** prontos (guia §6.3; `condicoes-do-estudo.md` §10).

## 8. Conclusão e trabalhos futuros

Responde em poucas frases à pergunta do título com os números da Etapa 7 e resume as contribuições de método (protocolo comum, decomposição causal, leitura dupla do FinGAT).

Trabalhos futuros: avaliação em vários períodos (`TimeSeriesSplit`) e com mais anos de dados; outras ações como alvo; o TEGRU com uma fonte de notícias em português; e o que leva para a dissertação.

- **Insumos:** **Etapa 7**; trabalhos futuros prontos (`condicoes-do-estudo.md` §2.2; guia, Etapa 5).

## Referências

Os três artigos do seminário (Cho et al., 2014; Hsu et al., 2023; Li et al., 2024) e o TEGRU (Haryono et al., 2023), citado para justificar a exclusão; Wilson (1927) e Brown, Cai e DasGupta (2001); as referências da normalização por janela (D-20); e as fontes primárias dos componentes e das bibliotecas usadas.

- **Insumos:** parcialmente prontos; a lista do que falta está no fim.

## Apêndices

- **A. Hiperparâmetros completos** por modelo, com a origem de cada valor (`condicoes-do-estudo.md` §3).
- **B. Desvios dos artigos** e as decisões que os autorizaram (`condicoes-do-estudo.md` §10; guia §6.1).
- **C. Resultados por semente** de cada configuração (Etapa 7).
- **D. O estudo da GRU:** a grade completa e os efeitos marginais (`resultados-2b.md`).
- **E. Modificações no código do FinGAT** (`terceiros/fingat_original/ORIGEM.md`).

---

## O que falta para escrever

| Item                                                     | Para que seção     | Quando                          |
| -------------------------------------------------------- | ------------------ | ------------------------------- |
| Execuções da Etapa 7 (6 configurações × 5 sementes)      | §5, §6, §8, Resumo | Etapa 7                          |
| Script da tabela final: média ± desvio, intervalo de Wilson, ingênuo, tabela por semente | §5, Apêndice C | Etapa 7 (já previsto no guia) |
| Figuras: previsto × real no teste; vazamento da decomposição; subséries com a faixa do treino; efeitos marginais da Etapa 2 | §4.3, §5 | Etapa 7 (script de figuras) |
| Diagramas de blocos dos 4 tipos de modelo                | §4.4               | pode ser feito já               |
| **Formato do relatório pedido no enunciado** (template, limite de páginas, idioma) | todas | **a confirmar com o autor**: não está nos documentos do projeto |
| Referências da normalização por janela (D-20): escolher entre Jiang et al. (2017), Zeng et al. (2023), Passalis et al. (2020), Salinas et al. (2020) | §2, §4.1 | escrita |
| Referências primárias, a conferir: GAT (Veličković et al., 2018); CEEMDAN (Torres et al., 2011); VMD (Dragomiretskiy e Zosso, 2014); entropia amostral (Richman e Moorman, 2000); Adam (Kingma e Ba, 2015); `EMD-signal` e `vmdpy`; FinTSB (`artigos/hu_fintsb_2025.pdf`) | §2, §3, §4 | escrita |
