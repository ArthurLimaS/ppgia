# Guia de implementação — projeto de Redes Neurais (comparação de 5 modelos)

Este guia diz **como** implementar, validar e executar a comparação entre cinco modelos de previsão para a PETR4 (a GRU e quatro variantes de dois artigos). **O que** a disciplina pede está no enunciado do projeto (código + relatório em formato de artigo + apresentação de 10 min; 4 arquiteturas comparadas, com as limitações identificadas). **O que** cada artigo descreve está nos PDFs de `artigos/`; os números já levantados deles estão em [`rastreabilidade-artigos-seminario.md`](rastreabilidade-artigos-seminario.md). Se este guia e um artigo discordarem sem que a diferença esteja registrada como decisão (§6.1), vale o artigo, e o guia é corrigido.

| #   | Modelo                                            | Fonte                                                               | Papel                        |
| --- | ------------------------------------------------- | ------------------------------------------------------------------- | ---------------------------- |
| 1   | GRU (`gru`)                                       | Cho et al. (2014), `artigos/cho2014learning.pdf`                    | Baseline                     |
| 2   | FinGAT, código dos autores (`fingat_repo`)        | Hsu et al. (2023), `artigos/hsu_fingat_2023.pdf`; repositório deles | Muda **como a rede modela**  |
| 3   | Custom FinGAT, texto do artigo (`fingat_artigo`)  | Hsu et al. (2023), `artigos/hsu_fingat_2023.pdf`                    | Muda **como a rede modela**  |
| 4   | HDFM, normalização A: min-max por subsérie (D-14) | Li et al. (2024), `artigos/li_accurate_2024.pdf`                    | Muda **o que entra na rede** |
| 5   | HDFM, normalização B: diferença por janela (D-14) | Li et al. (2024), `artigos/li_accurate_2024.pdf`                    | Muda **o que entra na rede** |

**Escopo revisto em 2026-11-18.** O enunciado pede 4 arquiteturas de 3 artigos. O estudo compara **5 modelos de 2 artigos**: onde o artigo e o código dos autores divergem (FinGAT, D-08) ou o artigo é ambíguo num ponto que muda o resultado (HDFM, D-14), as duas leituras viram modelos. O **TEGRU (Haryono et al., 2023) saiu do estudo**: ver a Etapa 5.

## Índice

1. [Escopo e ponto de partida](#1-escopo-e-ponto-de-partida)
2. [Organização do código](#2-organização-do-código)
3. [Desenho comum aos modelos](#3-desenho-comum-aos-modelos)
4. [Ciclo de uma etapa](#4-ciclo-de-uma-etapa)
5. [Plano por etapas](#5-plano-por-etapas)
6. [Pontos em aberto e riscos](#6-pontos-em-aberto-e-riscos)

---

## 1. Escopo e ponto de partida

| Entra                                                                                         | Fica de fora                                                                                |
| --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Os 5 modelos da tabela acima, treinados e avaliados sobre a PETR4                             | O TEGRU (Haryono et al., 2023): saiu do estudo em 2026-11-18 (Etapa 5)                      |
| Previsão de um passo: janela de N pregões → pregão seguinte (N sai da Etapa 2)                | Previsão de vários passos à frente                                                          |
| Validação rápida de cada modelo (1 a 3 épocas) e medição do tempo por época                   | Busca de hiperparâmetros dos modelos dos artigos (FinGAT, HDFM): só o que a Etapa 6 decidir |
| Estudo da GRU na Etapa 2: entrada × janela × arquitetura, para escolher a GRU compartilhada   | Séries de outras ações como alvo                                                            |
| Documento das condições do estudo (métricas, épocas, hiperparâmetros) antes da execução longa | Os baselines dos próprios artigos (ARIMA, SVR, Informer, FEDformer, RankLSTM…)              |
| Execução longa com retomada, e os resultados em arquivo                                       | O relatório em formato de artigo e os slides (têm guia próprio depois dos resultados)       |

Situação em 2026-11-16:

- **Código:** nenhum. `redes_neurais_2026_2/` só tem os documentos do seminário. O único script do repositório que baixa a PETR4 é `metodologia_2026_1/grafico_petr4_candles.py` (yfinance).
- **Dependências já no `pyproject.toml`:** `torch`, `yfinance`, `matplotlib`. Faltam as de decomposição do HDFM e `scikit-learn` (Etapa 0).
- **Máquina de desenvolvimento (medido):** WSL2, 12 núcleos lógicos, **4 GB de RAM visíveis no WSL**, **sem GPU** (`nvidia-smi` não existe). Serve para escrever o código e para os testes de fumaça; **não é a máquina do estudo**.
- **Máquina do estudo (informado em 2026-11-16, não medido):** Ryzen 5 5500 (6 núcleos, 12 threads), RTX 5060 com 6 GB de VRAM, 16 GB de RAM. É nela que rodam a Etapa 2 (estudo da GRU), as medições de tempo e a Etapa 7. Ver R-01 e R-02 no §6.2.
- **Prazo de entrega: 2026-11-19** (R-08).
- **Código dos artigos:** o FinGAT **cita um repositório** (`github.com/Roytsai27/Financial-GraphAttention`, nota 2 e §5.1 do artigo), ao contrário do que se supunha. O HDFM não cita código. O enunciado permite usar código-base público; como usar o do FinGAT é a decisão D-08 (decidida em 2026-11-18: cópia em `projeto/terceiros/fingat_original/`, ver Etapa 3).

**Regra central deste guia:** onde o artigo é ambíguo, incompleto ou inaplicável à PETR4, o agente **para e pergunta**. Pode propor solução, mas não implementa a parte afetada antes da resposta. As ambiguidades já encontradas na leitura estão no §6.1, cada uma com a etapa que ela bloqueia; as que aparecerem durante a implementação entram na mesma tabela.

---

## 2. Organização do código

```text
redes_neurais_2026_2/
├─ guia-implementacao-projeto.md      este guia
├─ condicoes-do-estudo.md             criado na Etapa 6 (métricas, épocas, hiperparâmetros)
└─ projeto/
   ├─ dados/
   │  ├─ baixar.py                    baixa do yfinance e grava o snapshot
   │  ├─ universo.csv                 candidatas do FinGAT: ticker, setor da B3 (D-07)
   │  └─ snapshot/                    CSVs baixados + manifesto (datas, versão do yfinance); commitado
   ├─ comum/
   │  ├─ janelas.py                   série → (janela de N, alvo), divisão temporal, normalização
   │  ├─ gru.py                       o bloco GRU compartilhado (§3.2)
   │  ├─ treino.py                    laço de treino, checkpoint, retomada, log, tempo por época
   │  └─ metricas.py                  métricas de preço e de direção (§3.4)
   ├─ modelos/
   │  ├─ gru_baseline.py
   │  ├─ custom_fingat.py             duas variantes: fingat_repo (3a) e fingat_artigo (3b)
   │  └─ hdfm.py                      decomposição + um GRU por subsérie + fusão
   ├─ terceiros/fingat_original/      cópia sem modificação do código dos autores do FinGAT;
   │                                  ORIGEM.md: commit, licença e o que mudou (D-08)
   ├─ rodar.py                        ponto de entrada: --modelo, --epocas, --semente, --dispositivo,
   │                                  --entrada, --janela, --camadas (retoma sozinho)
   ├─ grade_gru.py                    fila do estudo da Etapa 2 (chama rodar.py para cada combinação)
   ├─ testes/                         testes de fumaça (formas, vazamento temporal, retomada)
   └─ execucoes/                      uma pasta por execução (config, log, checkpoints, previsões); fora do git
```

- **Um ponto de entrada só** (`rodar.py`). A validação rápida e a execução longa são o mesmo comando com outro `--epocas`; não há um caminho "de teste" que possa divergir do caminho real.
- **Retomada automática:** rodar de novo o mesmo comando continua do último checkpoint; se a execução já terminou, não faz nada; se a pasta tem outra configuração, recusa. Não há flag `--retomar`.
- **Testes:** `uv run python -m unittest discover -s redes_neurais_2026_2/projeto/testes` (biblioteca padrão; nenhuma dependência de teste).
- **Comando:** `uv run python redes_neurais_2026_2/projeto/rodar.py --modelo gru --epocas 3`.
- **O que é comum fica em `comum/`** e é escrito na Etapa 1 com a GRU; os outros modelos reutilizam. Cada modelo de artigo só acrescenta o que é dele (features, grafo, decomposição, encoder).
- **Previsões gravadas, métricas recalculáveis:** cada execução grava as previsões do conjunto de teste (data, real, previsto). As métricas saem desse arquivo, então mudar ou acrescentar métrica na Etapa 6 não exige treinar de novo.

---

## 3. Desenho comum aos modelos

### 3.1. Dados e tarefa

- **Série:** PETR4.SA, diária, últimos 5 anos a contar do dia do download (~1.250 pregões). O download acontece **uma vez**; o snapshot é gravado em `dados/snapshot/` com manifesto, e todo treino lê o snapshot. Motivo: o yfinance reajusta preços passados a cada provento e limita requisições; sem snapshot, duas execuções não são comparáveis.
- **Tarefa comum:** com os N pregões até `t`, prever o pregão `t+1`. A janela N é um parâmetro (`--janela`), escolhido na Etapa 2 entre 5, 15, 30, 60, 120 e 200 (D-01).
- **Divisão temporal 80/10/10** (D-02), em ordem cronológica, sem embaralhar entre treino, validação e teste: ~1.000 / 125 / 125 pregões.
- **Normalização por janela (D-20, opção c), na GRU:** os preços de cada janela são divididos pelo **último fechamento da própria janela**, e a rede prevê a **variação relativa** do pregão seguinte, `fechamento(t+1)/fechamento(t) − 1`. O preço previsto é reconstruído na saída (`fechamento(t)·(1 + variação prevista)`), e as métricas de preço são calculadas sobre ele. Não há estatística ajustada no treino, então não há o que vazar, e a entrada fica na mesma faixa em qualquer nível de preço. Motivo: no fechamento ajustado, validação e teste estão acima de tudo o que o treino viu (Etapa 0).
- **Volume na entrada `ohlcv`:** não é preço, então não pode ser dividido pelo fechamento. Como normalizá-lo é a decisão D-21.
- **HDFM:** decompõe o nível do preço; a normalização das subséries é D-14, a decidir à luz da D-20.
- **O que cada modelo recebe e devolve:**

| Modelo | Entrada (janela de N pregões)                                                | Saída                                                       | Comparável em               |
| ------ | ---------------------------------------------------------------------------- | ----------------------------------------------------------- | --------------------------- |
| GRU    | fechamento ou OHLCV da PETR4, o que a Etapa 2 escolher (D-05)                | fechamento de `t+1`                                         | preço e direção             |
| HDFM   | cada subsérie da decomposição do fechamento, uma GRU por subsérie            | soma das previsões das subséries = fechamento de `t+1`      | preço e direção             |
| FinGAT | features do artigo para **todas as ações do universo**, N/5 semanas × 5 dias | por ação: razão de retorno prevista e probabilidade de alta | **só direção** (ver abaixo) |

- **FinGAT e a comparação (confirmando a OBS 2):** o FinGAT prevê duas coisas por ação, a razão de retorno do dia seguinte e a probabilidade de alta (Eq. 14). O retorno, porém, é treinado com perda de **ranking por pares** (Eq. 16), que só cobra a **ordem** entre as ações, não o valor: a escala da saída não é ancorada em nada. Converter esse retorno em preço (`p̂ = p·(1+r̂)`) daria um erro sem sentido. Por isso o FinGAT entra na comparação **só em direção**, pela saída de movimento da PETR4. O artigo faz o mesmo: descarta MAE e RMSE de propósito (§5.1) e reporta MRR, Precision e ACC.
- **A escolha de entrada da Etapa 2 só alcança a GRU.** O HDFM é univariado por construção (decompõe o fechamento), e o FinGAT tem as features do próprio artigo (L-02).
- **A janela escolhida vale para todos.** Cada artigo usa a sua (FinGAT 15; HDFM 30), então a janela comum é um desvio nos dois (D-01). No FinGAT a janela vira N/5 semanas de 5 pregões; todas as candidatas são múltiplas de 5, mas o artigo só testou de 1 a 4 semanas (Fig. 4a–b).

### 3.2. Bloco GRU compartilhado (OBS 3)

- `comum/gru.py` define **um** bloco (entrada `[lote, tempo, features]` → sequência de estados ocultos e último estado), sobre o `torch.nn.GRU`. Todos os modelos o usam; nenhum declara a própria GRU.
- O bloco recebe a **lista de tamanhos das camadas** (`--camadas 128,64,32`). Camadas de tamanhos diferentes são `nn.GRU` de uma camada empilhadas, porque o `num_layers` do PyTorch exige o mesmo tamanho em todas.
- O que muda entre os modelos é só o que vem **antes** (decomposição, features) e **depois** (atenção, GAT, fusão) do bloco. Assim a diferença de resultado em relação ao baseline mede o que cada artigo acrescentou.
- **A configuração do bloco sai do estudo da Etapa 2** (D-03), porque os artigos discordam: o HDFM usa 3 camadas (128, 64, 32); o FinGAT usa 16 unidades e relata piora com 32 e 64 (Fig. 4c–d). A configuração escolhida é usada em todos os modelos: a ideia do estudo é medir quanto os extras de cada artigo melhoram **a mesma** GRU.
- **Limite da ideia, a registrar no relatório:** o FinGAT usa o bloco três vezes por ação (dias da semana; semanas de `a`; semanas de `g`), sempre seguido de atenção, e o HDFM treina uma instância por subsérie. "Mesma arquitetura de GRU" significa mesma classe e mesma configuração, não mesmo número de parâmetros no modelo todo.

### 3.3. Treino

- Um laço só (`comum/treino.py`) para todos: Adam, perda do modelo (MSE para GRU e HDFM; a perda multitarefa do FinGAT), validação ao fim de cada época.
- **Mede e grava o tempo de cada época** no log da execução (objetivo 2 da etapa b). A primeira época é reportada à parte, porque inclui aquecimento.
- Semente fixa (`torch`, `numpy`, `random`) recebida por argumento e gravada na configuração da execução.
- Checkpoint e retomada já nascem aqui (§5, Etapa 7), para serem validados nas execuções curtas e não descobertos na longa.

### 3.4. Métricas (lista de partida; a lista final é da Etapa 6)

| Grupo   | Métrica             | Origem                           | Modelos   |
| ------- | ------------------- | -------------------------------- | --------- |
| Preço   | RMSE, MAE, MAPE, R² | HDFM (Eq. 16–19); TEGRU (§III-G) | GRU, HDFM |
| Direção | Acurácia de direção | FinGAT (ACC); TEGRU (Eq. 9–11)   | todos     |
| Mista   | AcMAPE              | TEGRU (Eq. 13)                   | GRU, HDFM |

- **Origem das métricas:** RMSE a R² e a definição de direção vêm dos artigos do estudo; a direção para quem prevê preço e o AcMAPE vêm da formulação do TEGRU (Haryono et al., 2023), que saiu do estudo mas continua citado como fonte dessas duas fórmulas.
- **Direção para quem prevê preço**: alta se `previsto(t+1) − real(t) ≥ 0`. **Para o FinGAT:** alta se a probabilidade de movimento ≥ 0,5. O rótulo real é o mesmo para todos.
- **Referência ingênua:** "amanhã = hoje" (Random Walk, que o HDFM usa como baseline). Não é um quinto modelo; é a linha que diz se um RMSE é bom. Custa uma linha em `metricas.py`.

---

## 4. Ciclo de uma etapa

As etapas dos modelos dos artigos (3 a 5) e a parte 2a da Etapa 2 seguem o mesmo ciclo, com uma parada no meio. O estudo da GRU (parte 2b) tem o roteiro próprio, na Etapa 2.

```text
(i) decisões em aberto  ──►  (ii) implementar  ──►  (iii) validação rápida  ──►  (iv) relatório
    da etapa respondidas        o modelo              (1 a 3 épocas)              e aprovação
```

**(i) Decisões.** Antes de escrever código, o agente apresenta as decisões do §6.1 que bloqueiam a etapa (o que o artigo diz, as opções, a proposta) e espera a resposta. A resposta é registrada na coluna "Decisão" da tabela.

**(ii) Implementar.** Só o modelo da etapa, reutilizando `comum/`. Ambiguidade nova encontrada no meio: o agente para, registra no §6.1 e pergunta.

**(iii) Validação rápida.** No máximo 3 épocas. **O objetivo é saber se o treino funciona, não se o resultado é bom.** Nenhum hiperparâmetro é ajustado para "melhorar" o número nesta fase.

**(iv) Relatório da etapa**, curto, apresentado para aprovação:

| Item              | Conteúdo                                                                                 |
| ----------------- | ---------------------------------------------------------------------------------------- |
| Funciona?         | perda de treino por época (cai?), sem NaN, formas dos tensores conferidas                |
| Tempo             | segundos por época (1ª à parte), tempo de pré-processamento, pico de memória             |
| Projeção          | horas para 50, 100 e 200 épocas **na máquina do estudo** (o tempo é medido nela)         |
| Desvios do artigo | cada ponto em que a implementação difere do artigo, com a decisão (D-NN) que o autorizou |
| Pendências        | o que ficou para decidir na Etapa 6                                                      |

**Pronto quando** (toda etapa de modelo):

- `rodar.py --modelo <m> --epocas 3` termina sem erro e grava configuração, log, checkpoint e previsões do teste;
- a perda de treino da última época é menor que a da primeira;
- os testes de fumaça passam: formas; **nenhuma janela usa dado posterior ao alvo** (teste de vazamento temporal); normalização ajustada só no treino; interromper e retomar dá o mesmo resultado de rodar direto;
- o tempo por época e o pico de memória estão no relatório;
- todo desvio do artigo tem uma decisão registrada no §6.1.

---

## 5. Plano por etapas

**Regras para todas as etapas**

- **Nunca assumir.** Lacuna do artigo vira linha no §6.1 e pergunta; a parte afetada espera.
- **Git só de leitura para o agente** (`status`, `diff`, `log`); commit e branch são do autor do projeto.
- **Execução curta antes da longa:** nenhuma execução acima de 3 épocas antes da Etapa 7. Exceção única: o estudo da GRU (Etapa 2, parte 2b), que precisa de treino de verdade para comparar configurações.
- **Tempo só vale se medido na máquina do estudo.** O que roda na máquina de desenvolvimento prova que o código funciona; não entra no §6.4.
- **O conjunto de teste não escolhe nada.** Toda escolha (janela, entrada, arquitetura, época do checkpoint) usa a validação; o teste só é lido na Etapa 7.
- Dependência nova só com o motivo registrado na etapa que a pede (`uv add`).

### Etapa 0 — Ambiente e dados

Entregáveis:

- Dependências: `pandas` agora; as do HDFM (`scikit-learn` para o K-means, `EMD-signal`, `vmdpy`) só na Etapa 4, depois de D-10.
- `dados/baixar.py` e o snapshot da PETR4 (OHLCV e fechamento ajustado, 5 anos), com manifesto.
- D-02 e D-04 já decididas; D-01, D-03 e D-05 são resolvidas pelo estudo da Etapa 2.
- **Máquina do estudo pronta:** o repositório, `uv sync`, e o PyTorch com CUDA enxergando a RTX 5060 (`torch.cuda.is_available()`). A série 50 da NVIDIA exige um build recente do PyTorch/CUDA; conferir antes de tudo, porque sem isso o estudo roda em CPU (R-02). A máquina é **Windows 11 nativo** (D-18): o `uv` e o projeto rodam direto no Windows, sem WSL, e o PyTorch vem do índice de CUDA do PyTorch (o pacote padrão do PyPI para Windows é só CPU), configurado no `pyproject.toml` de modo que a máquina de desenvolvimento (Linux, CPU) continue instalando normalmente. O código usa `pathlib` e não depende de comandos de shell, para rodar igual nos dois sistemas.

Pronto quando: o snapshot existe e é relido sem rede; um gráfico do fechamento confere com o de `grafico_petr4_candles.py`; a máquina do estudo (Windows 11) roda um tensor na GPU e passa nos testes de fumaça.

**Parte desta máquina concluída em 2026-11-16; falta a conferência na máquina do estudo.**

- **Snapshot:** `dados/snapshot/PETR4.SA.csv`, 1.248 pregões, de 2021-11-16 a 2026-11-13 (o pregão de 16/11, em andamento, ficou de fora), sem valor ausente nem volume zero; `manifesto.json` com datas, sha256 e versão do yfinance (1.5.1). `carregar()` relê do disco sem importar o yfinance e recusa o arquivo se o sha256 não bater. `baixar.py` recusa sobrescrever sem `--substituir`. Um `.gitattributes` na pasta impede o git de trocar o fim de linha no Windows, o que mudaria o hash.
- **Conferência:** no período do gráfico de candles (01/09 a 28/10/2026, 40 pregões), o snapshot e um download novo coincidem (diferença de 7e-15).
- **Divisão 80/10/10:** 998 / 125 / 125 pregões; validação de 2025-11 a 2026-05, teste de 2026-05 a 2026-11.
- **Dependências:** `pandas` no `pyproject.toml`. PyTorch pelo índice `cu130` só no Windows (`[tool.uv.sources]`); o `uv.lock` ficou com `torch 2.13.0` (PyPI) no Linux e `2.14.1+cu130` no Windows — versões diferentes, porque o índice de CUDA não tem a 2.13.0 para Windows. O `scikit-learn` foi adiado para a Etapa 4, que é quem o usa.
- **Achado que virou decisão (D-20):** a validação e o teste estão **fora da faixa de preços do treino**. No fechamento ajustado, o treino vai de 7,72 a ~32; a validação chega a ~47 e o teste a 59,68. O fechamento ajustado de 2021 é 3,4 vezes menor que o bruto (27,27 contra 8,06), por causa dos proventos.
- **A fazer na máquina do estudo:** `uv sync` e `uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"`. Exige driver NVIDIA recente (CUDA 13).

### Etapa 1 — GRU: implementação (item a)

Entregáveis: `comum/janelas.py`, `comum/gru.py`, `comum/treino.py`, `comum/metricas.py`, `modelos/gru_baseline.py`, `rodar.py`, `grade_gru.py` e os testes de fumaça.

- É a etapa que mais escreve código comum; as seguintes só acrescentam.
- A GRU baseline é o bloco compartilhado + uma camada linear que dá o fechamento de `t+1`.
- **A GRU nasce configurável nos três eixos do estudo da Etapa 2**, por argumento, sem código separado por variante:
  - `--entrada univariado|ohlcv` (1 ou 5 features por pregão; o alvo é sempre o fechamento ajustado);
  - `--janela N`;
  - `--camadas`, a lista de tamanhos (§3.2).
- `grade_gru.py` é a fila do estudo: percorre as combinações, chama `rodar.py` para cada uma, pula as que já terminaram e junta os resultados numa tabela. É a mesma mecânica de fila da Etapa 7, escrita já aqui.

Pronto quando: os testes de fumaça passam com dados sintéticos (ex.: uma senoide, que a GRU tem de aprender em poucas épocas — prova de que o laço de treino funciona, separada da dificuldade da PETR4), nas duas entradas, em pelo menos duas janelas e nas arquiteturas `a` e `i` da Etapa 2.

**Concluída em 2026-11-16 para a entrada `univariado`; a entrada `ohlcv` foi acrescentada em 2026-11-17 (D-21).**

- **Código:** `comum/janelas.py` (janelas, divisão cronológica 80/10/10 pela posição do alvo, normalização por janela da D-20, primeiro alvo de treino configurável para a D-17), `comum/gru.py` (GRUs empilhadas a partir da lista de tamanhos), `comum/treino.py` (Adam, MSE, tempo por época, `ultimo.pt` e `melhor.pt` com gravação atômica, parada em perda não finita), `comum/metricas.py`, `modelos/gru_baseline.py`, `rodar.py`, `grade_gru.py`.
- **Entrada e alvo na mesma unidade:** a janela entra como `fechamento/último fechamento − 1` (o último ponto é sempre 0) e o alvo é `fechamento(t+1)/fechamento(t) − 1`. Subtrair 1 só centra a entrada em zero; é a opção (c) da D-20.
- **Cada execução grava** em `execucoes/<nome>/`: `config.json` (com o sha256 do snapshot), `log.csv`, `ultimo.pt`, `melhor.pt`, `previsoes_validacao.csv`, `previsoes_teste.csv` e `metricas.json`. O `metricas.json` **só tem métricas de validação** (e a referência ingênua); as previsões de teste ficam gravadas, mas nada as lê antes da Etapa 7.
- **Testes de fumaça (10, todos passam):** valores e formas das janelas; divisão cronológica; **vazamento temporal** (multiplicar por 3 os pregões de teste não muda nenhuma amostra de treino nem de validação, nas janelas 5 e 200); mesmos dias-alvo entre janelas (D-17); reconstrução exata do preço; métricas com valores calculados à mão; formas do bloco nas arquiteturas `a` e `i`; a GRU aprende uma senoide e fica abaixo de metade do RMSE da referência ingênua (arquiteturas `i` com janela 15 e `a` com janela 30); retomar dá previsões idênticas a rodar direto; configuração diferente na mesma pasta é recusada.
- **Conferência com a PETR4 nesta máquina** (só para provar o caminho; tempo não vale): 3 épocas nos dois extremos, a perda cai nos dois. A fila `grade_gru.py` rodou a grade reduzida com 1 época.
- **Desvios do planejado:** a retomada é automática (sem `--retomar`); os testes usam `unittest`; `--pasta` permite gravar a mesma configuração em pastas diferentes (necessário na 2a, para medir CPU e GPU).
- **`ohlcv` (D-21, 2026-11-17):** `comum/janelas.py` monta 5 colunas (abertura, máxima, mínima, fechamento, volume) e `janelas.ENTRADAS` passou a ser o dicionário nome → nº de features. Testes novos: valores calculados à mão, volume com média zero em toda janela, vazamento temporal nas duas entradas e a senoide aprendida com `ohlcv`. A grade passa a ter 36 combinações na reduzida e 108 na completa.

### Etapa 2 — GRU: validação com a PETR4 e escolha da GRU compartilhada (item b)

A etapa tem duas partes. A primeira é a validação rápida do pedido original; a segunda é o estudo que resolve D-01, D-03 e D-05.

**Por que o estudo existe.** O objetivo do projeto é medir quanto os extras de cada artigo (grafo e hierarquia no FinGAT, decomposição no HDFM) melhoram uma GRU. Para isso todos os modelos usam **a mesma GRU**, e ela precisa ser escolhida com critério, não copiada de um dos artigos. A janela, a entrada e a arquitetura definem essa GRU; por isso as três decisões são validadas aqui, com dados, e não respondidas de antemão.

#### 2a — Validação rápida (o treino funciona? quanto custa uma época?)

- Na máquina do estudo, PETR4, 5 anos, **3 épocas no máximo**, pelo ciclo do §4.
- Rodar os **extremos de custo** da grade, para ter o piso e o teto do tempo por época:
  - mais barata: univariado, janela 5, arquitetura `i` (1 camada de 16);
  - mais cara: OHLCV, janela 200, arquitetura `a` (128 → 64 → 32).
- Medir cada extremo **em CPU e em GPU**. Com ~1.000 amostras e redes deste tamanho, a GPU pode ser mais lenta que a CPU (o custo de mandar lotes pequenos para a placa supera o ganho); o dispositivo do estudo é o que for mais rápido, por medição. A memória não é o limite aqui: a maior configuração tem perto de 100 mil parâmetros.
- Resultado: o treino funciona, e o **tempo por época** dos dois extremos vai para o §6.4. Com eles se calcula o orçamento da parte 2b.

**Situação (2026-11-17): código pronto; falta rodar na máquina do estudo.**

- `medir_2a.py` roda os dois extremos (`barata`: univariado, janela 5, `i`; `cara`: OHLCV, janela 200, `a`) com 3 épocas, em CPU e, se `torch.cuda.is_available()`, em GPU, nos mesmos alvos de treino da grade (D-17, 798 amostras). Grava em `execucoes/2a_<extremo>_<dispositivo>/` e imprime as linhas da tabela do §6.4. `rodar.py` passou a gravar `preprocessamento_s` no `metricas.json`.
- Nesta máquina (Linux, só CPU) o script termina sem erro e a perda de treino cai nos dois extremos (`barata`: 2,0e-2 → 5,3e-4, 929 parâmetros; `cara`: 1,0e-2 → 6,2e-4, 98.529 parâmetros). Esses tempos não valem (§5, regras) e as pastas dessa execução foram apagadas.
- Na máquina do estudo: `uv run python redes_neurais_2026_2/projeto/medir_2a.py`; copiar a tabela impressa para o §6.4.

#### 2b — Estudo da GRU (entrada × janela × arquitetura)

Os três eixos, como decididos em 2026-11-16:

**1. Entrada da rede** (resolve D-05)

| Opção        | Features por pregão                          |
| ------------ | -------------------------------------------- |
| `univariado` | fechamento ajustado                          |
| `ohlcv`      | abertura, máxima, mínima, fechamento, volume |

**2. Janela** (resolve D-01)

| Janela | Justificativa                           |
| ------ | --------------------------------------- |
| 5      | mínimo testado no TEGRU                 |
| 15     | valor do FinGAT (3 semanas × 5 pregões) |
| 30     | valor do HDFM                           |
| 60     | dobro de 30                             |
| 120    | dobro de 60                             |
| 200    | máximo testado no TEGRU                 |

**3. Arquitetura da GRU** (resolve D-03)

| ID  | Camadas          | Origem | Grade reduzida |
| --- | ---------------- | ------ | -------------- |
| `a` | 3: 128 → 64 → 32 | HDFM   | sim            |
| `b` | 3: 64 → 32 → 16  |        |                |
| `c` | 2: 128 → 64      |        |                |
| `d` | 2: 64 → 32       |        | sim            |
| `e` | 2: 32 → 16       |        |                |
| `f` | 1: 128           |        |                |
| `g` | 1: 64            |        |                |
| `h` | 1: 32            |        |                |
| `i` | 1: 16            | FinGAT | sim            |

**Tamanho da grade**

| Grade    | Combinações         | Quando                                                            |
| -------- | ------------------- | ----------------------------------------------------------------- |
| Completa | 2 × 6 × 9 = **108** | só se der para treinar as 108 **com qualidade** (ver o orçamento) |
| Reduzida | 2 × 6 × 3 = **36**  | o padrão, se a completa não couber: arquiteturas `a`, `d` e `i`   |

**Orçamento e ordem de corte.** Com os tempos da 2a: `tempo total = Σ (tempo por época da combinação) × épocas × sementes`. O total precisa caber no que sobra até 19/11 **depois** de reservar o tempo das Etapas 3 a 7 e do relatório (R-08). Se não couber, corta-se nesta ordem, e cada corte vira uma linha em L-01 (§6.3):

1. grade completa → reduzida (`a`, `d`, `i`);
2. sementes por combinação → 1;
3. épocas → o que couber (a "forma simplificada").

O agente apresenta a conta e a proposta de grade, épocas e sementes **antes** de soltar a fila, e espera a aprovação.

**Regras do estudo**

- **A escolha é pela validação, nunca pelo teste.** Critério proposto: menor RMSE do fechamento na validação, no melhor checkpoint de cada execução; a acurácia de direção é reportada ao lado. O teste fica intocado até a Etapa 7; escolher a GRU olhando o teste inflaria o resultado dos quatro modelos.
- **Mesmos dias-alvo para todas as janelas** (D-17, opção a: todas treinam só nos alvos que a janela de 200 alcança, ~800). Uma janela de 200 só consegue prever a partir do 201º pregão; uma de 5, a partir do 6º. Sem cuidado, cada janela seria avaliada em dias diferentes e treinada com quantidades diferentes de amostras, e a comparação mediria isso, não a janela.
- **Tudo o mais fixo** entre as combinações: perda MSE, Adam 0,001, lote 64 (os valores do HDFM, Tabela 2, os únicos completos entre os artigos), normalização por janela (D-20), mesma semente (ou o mesmo conjunto de sementes).
- **A referência ingênua** ("amanhã = hoje") entra na tabela. Se nenhuma combinação ganhar dela no RMSE da validação, isso é resultado a reportar, e a escolha passa a ser a menos pior, com o registro em L-01.
- Cada combinação é uma execução normal de `rodar.py`, com checkpoint e retomada; a fila pula o que já terminou.

**Relatório da 2b** (apresentado para aprovação; é a análise desta etapa):

| Item                | Conteúdo                                                                                                                                |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Tabela completa     | uma linha por combinação: entrada, janela, arquitetura, parâmetros, tempo por época, melhor época, RMSE/MAE/MAPE e direção na validação |
| Efeito de cada eixo | três gráficos ou tabelas marginais: por entrada, por janela, por arquitetura (média e dispersão das demais)                             |
| Ruído               | com mais de uma semente: a variação entre sementes da mesma combinação. Diferenças menores que ela **não** contam como diferença        |
| Custo               | tempo total gasto; o que foi cortado em relação à grade completa                                                                        |
| Proposta de escolha | **uma** combinação (entrada, janela, arquitetura) para a GRU compartilhada, com o motivo; e a segunda colocada                          |
| Consequências       | o que a escolha implica para as Etapas 3 a 5 (ver abaixo)                                                                               |

**O que a escolha implica para os outros modelos** (a conferir no relatório, antes de aprovar):

- **Janela:** vale para todos. No FinGAT vira N/5 semanas; no HDFM, o tamanho da janela de cada subsérie.
- **Entrada:** vale só para a GRU. HDFM e FinGAT não a recebem (§3.1; L-02).
- **Arquitetura:** vale para todos, e multiplica o custo onde o bloco se repete: 12 vezes no HDFM (uma GRU por subsérie) e 3 vezes por ação no FinGAT. Uma arquitetura grande que ganhe por pouco na GRU isolada pode ser inviável ali. Se as primeiras colocadas estiverem dentro do ruído, a proposta é a **menor** delas.
- **Limite da escolha (L-03):** a melhor GRU isolada não é necessariamente a melhor dentro de cada modelo — o FinGAT relata piora acima de 16 unidades (Fig. 4c–d). O estudo aceita isso de propósito: o que se compara é o efeito dos extras sobre uma GRU fixa.

**Pronto quando:**

- 2a: os critérios do §4 nos dois extremos; tempos por época (CPU e GPU) no §6.4;
- 2b: todas as combinações da grade aprovada terminaram ou têm a falha registrada; o relatório foi apresentado; a combinação escolhida está registrada em D-01, D-03 e D-05; os cortes estão em L-01.

### Etapa 3 — FinGAT (item c)

Decisões que bloqueiam: **D-06, D-07, D-08, D-09**, e a GRU escolhida na Etapa 2 (janela e arquitetura). Todas decididas em 2026-11-18.

**Duas variantes (D-08).** O código dos autores não implementa o que o artigo descreve. Por isso o FinGAT entra em duas versões, as duas com a GRU da Etapa 2 (`f`, 1 camada de 128; janela 120 = 24 semanas de 5 pregões):

| Ponto       | 3a — `fingat_repo` (código dos autores)                                  | 3b — `fingat_artigo` (texto do artigo)                                           |
| ----------- | ------------------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| Features    | `nor_close`, retorno, 3 razões, 6 médias, setor em one-hot (19)          | abertura, máxima, mínima, fechamento, ajustado, retorno, 3 razões, 6 médias (15) |
| Curto prazo | uma GRU **por semana** + atenção do repo (`AttentionBlock`), dropout 0,2 | uma GRU para todas as semanas + atenção da Eq. 5–6                               |
| Intra-setor | GAT **depois** de juntar as semanas, com laço próprio                    | GAT **em cada semana** (Eq. 7–8), sem laço                                       |
| Longo prazo | só atenção sobre as semanas (`use_gru=False`)                            | GRU atentiva sobre `a` e sobre `g` (Eq. 9–10)                                    |
| Setor       | pooling por atenção (exige setores do mesmo tamanho)                     | max-pooling (Eq. 11)                                                             |
| Perda       | MAE + BCE + ranking por **produto**, pesos 1, sem L2                     | `0,99·ranking por diferença + 0,01·BCE + 1e-4·‖Θ‖²` (Eq. 15–16)                  |
| Otimização  | Adam, lr 0,05, um dia por passo, Xavier                                  | Adam, lr 0,001, lote de 128 dias, inicialização padrão                           |
| Parâmetros  | 1.457.620 (24 GRUs de 128)                                               | 369.666 (3 GRUs de 128)                                                          |

- **Onde está:** `modelos/custom_fingat.py`. A cópia sem modificação do repositório está em `terceiros/fingat_original/`, e o `ORIGEM.md` dela registra origem (URL, commit `0f7a2dd`, data, falta de licença) e as 15 modificações da 3a.
- **Comando:** `uv run python redes_neurais_2026_2/projeto/rodar.py --modelo fingat_repo --janela 120 --camadas 128 --epocas 3` (ou `--modelo fingat_artigo`). Lote e lr saem da variante quando a linha de comando não os dá.
- **Universo (D-07):** `dados/universo.csv` tem 74 candidatas em 8 setores da B3; `dados/baixar.py --universo` baixou-as nas datas da PETR4 (2021-11-16 a 2026-11-13). 66 têm os 1.248 pregões; 7 não existem mais no Yahoo Finance e RAIZ4 tem 83 pregões (L-09). Ficam as 6 de maior volume financeiro de cada setor no treino: **48 ações**. A lista e o motivo de cada exclusão vão no `config.json` de cada execução (`universo`, `excluidas`). Setores conferidos e aprovados pelo autor em 2026-11-18 (planilha da B3, link em D-07).
- **Mesmos dias de validação e teste da GRU:** a divisão é pela posição do alvo na série da PETR4 (D-02). O treino começa no pregão 150 (30 para as médias + 120 de janela): 848 dias.
- **Sublote na 3b:** 128 dias × 48 ações × 24 semanas num passo só estouram os 8 GB da GPU. A perda do artigo é uma soma sobre o lote, então `comum/treino.py` acumula o gradiente de 8 sublotes de 16 dias, o que dá o mesmo passo do lote de 128 (testado). Não é desvio do artigo.
- **Retomada com dropout:** o checkpoint passou a guardar o estado do gerador aleatório global (`comum/treino.py`); sem isso, retomar a 3a (que tem dropout) daria máscaras diferentes.
- **Saídas:** `previsoes_<parte>.csv` (PETR4: data, anterior, real, `prob_alta`) e `previsoes_<parte>_todas.csv` (todas as ações: retorno real, retorno previsto, `prob_alta`). O `metricas.json` traz a acurácia de direção da PETR4, a de todas as ações e a do jeito do `train.py` (sinal do retorno).

**Relatório da validação rápida (2026-11-18, máquina do estudo, GPU RTX 3060 Ti, 3 épocas):**

| Item                 | 3a — `fingat_repo`                                                      | 3b — `fingat_artigo`                                         |
| -------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------ |
| Funciona?            | sim: perda de treino 2,36 → 0,72, sem NaN; melhor época na validação: 3 | sim: perda de treino 82,9 → 46,6, sem NaN; melhor época: 3   |
| Tempo                | 45,1 s na 1ª época, 43,8 s nas demais; pré-processamento 0,7 s          | 3,1 s na 1ª época, 2,0 s nas demais; pré-processamento 0,7 s |
| Memória (pico)       | VRAM 794 MB, RAM 1,9 GB                                                 | **VRAM 6,8 GB** (de 8), RAM 3,7 GB                           |
| Projeção             | 50 épocas 37 min; 100 épocas 73 min; 200 épocas 2,4 h                   | 50 épocas 1,7 min; 100 épocas 3,3 min; 200 épocas 6,5 min    |
| Direção na validação | PETR4 0,664; todas 0,517                                                | PETR4 0,664; todas 0,517                                     |

- **As duas ainda não aprenderam direção:** depois de 3 épocas, preveem "alta" para todas as ações em todos os dias (probabilidade entre 0,50 e 0,53). Por isso a acurácia da PETR4 é a do ingênuo (0,664), e a de todas as ações é a fração de dias de alta (0,517). Pela regra do §4, nada foi ajustado; é pendência da Etapa 6 (épocas, lr; o lr 0,05 da 3a é alto para o Adam).
- **A 3a é 22 vezes mais lenta por época que a 3b:** um passo por dia (848 por época) e 24 GRUs separadas. Pesa no orçamento da Etapa 6.
- **O pico de VRAM da 3b (6,8 GB)** deixa pouca folga; se outra coisa usar a GPU durante a Etapa 7, reduzir o sublote (não muda o resultado).
- **Testes de fumaça (`testes/test_fingat.py`, 8 testes; a suíte toda tem 20, todos passam):** seleção do universo (incompleta e menor volume saem, setores em ordem); formas da entrada e da saída nas duas variantes; formas de `a`, `g`, `τA`, `τG`, `τ(π)` na 3b; **máscara intra-setor** (mudar as ações de um setor não altera a saída da GAT nas do outro, com e sem laço); laço próprio na adjacência; **vazamento temporal** (multiplicar por 3 os preços do teste não muda treino nem validação, nas duas variantes); sublote dá o mesmo passo que o lote inteiro; 3 épocas com a perda caindo e retomar (2 + 1 épocas) dá previsões idênticas a rodar direto, nas duas variantes.
- **Desvios do artigo:** na 3b, só os registrados em D-09 (média sobre o ajustado; normalização com o treino; lr 0,001; lote de dias) e a janela comum (D-01). Na 3a, os do `ORIGEM.md` e a tabela acima (L-08).
- **Pendências para a Etapa 6:** épocas e lr das duas variantes; se as duas variantes vão para a Etapa 7 ou só uma.

**Texto original do plano** (o que o artigo define; vale para a 3b):

- **Features por ação e dia** (§4.1): abertura, máxima, mínima, fechamento, fechamento ajustado, razão de retorno; razões de preço (Eq. 2); médias móveis de 5, 10, 15, 20, 25 e 30 dias (Eq. 3; ver D-09). As médias de 30 dias consomem os primeiros 30 pregões de cada série.
- **Curto prazo:** bloco GRU sobre os 5 dias de cada semana + atenção (Eq. 4–6) → `a`.
- **Intra-setor:** grafo completo entre as ações do mesmo setor, GAT (Eq. 7–8) → `g`.
- **Longo prazo:** GRU atentiva sobre as semanas de `a` e de `g` (Eq. 9–10; 3 semanas no artigo, N/5 aqui) → `τA`, `τG`.
- **Setores:** max-pooling de `τG` por setor (Eq. 11), grafo completo entre setores, GAT (Eq. 12) → `τ(π)`.
- **Fusão e saídas** (Eq. 13–14): retorno e movimento.
- **Perda** (Eq. 15–16): `(1−δ)·ranking + δ·movimento + λ‖Θ‖²`, com δ = 0,01 e λ = 0,0001; lote 128; Adam. A dimensão das GRUs é a da Etapa 2 (16 no artigo); a dos GATs acompanha a saída da GRU.

Particularidades da etapa:

- **É o único modelo com várias ações.** Precisa do universo e dos setores (D-07) e de um snapshot maior (um CSV por ticker). Ações sem os 5 anos completos saem do universo, e a saída fica registrada no `config.json` de cada execução (`excluidas`).
- **Uma época do FinGAT não é comparável à da GRU:** cada instância é um dia com todas as ações. O relatório reporta o tempo por época e o número de instâncias.
- Avaliação na validação rápida: acurácia de direção da PETR4 e de todas as ações (a segunda é a ACC do artigo e serve de conferência).

Pronto quando: os critérios do §4; as formas de `a`, `g`, `τA`, `τG`, `τ(π)` conferidas num teste; a máscara do grafo intra-setor testada (ação de um setor não recebe atenção de outro).

### Etapa 4 — HDFM (item d)

Decisões que bloqueiam: **D-10, D-11, D-12, D-13, D-14**, e a GRU escolhida na Etapa 2 (janela e arquitetura; a entrada é sempre a subsérie, univariada).

O que o artigo define e será seguido:

1. CEEMDAN no fechamento → IMFs + resíduo (§III-A).
2. Entropia amostral de cada IMF; K-means sobre as entropias → Co-IMFs, somando as IMFs do mesmo grupo (§III-B, III-E). Nos três índices do artigo, K = 3.
3. VMD na Co-IMF1, a de maior entropia (§III-C, IV-F; a Fig. 11 mostra 10 modos).
4. Uma GRU por subsérie: cada modo do VMD e cada Co-IMF restante (Fig. 2).
5. Fusão: soma das previsões → fechamento previsto.

**Dois modelos (D-14), cada um com as duas decomposições (D-10 c):** a normalização A (min-max por subsérie, Eq. 20) e a B (diferença por janela, D-20 adaptada) são os modelos 4 e 5. Cada um roda com a decomposição da série inteira (a) e com a causal (b); a diferença entre (a) e (b) é a medida do vazamento, que vai para as limitações.

Particularidades da etapa:

- **Windows:** a CEEMDAN da `EMD-signal` 1.10 abre um processo por núcleo; o script que a chama precisa de `if __name__ == "__main__"`, senão trava (visto em 2026-11-18).
- **Ordem das subséries entre decomposições (só na causal):** Co-IMFs ordenadas pela entropia do grupo; modos da VMD ordenados pela frequência central (a `vmdpy` não os ordena).
- **Custo multiplicado:** com 10 modos + 2 Co-IMFs são **12 GRUs**. O tempo "por época" do HDFM é o das 12 somadas; a projeção do relatório usa esse total.
- **A decomposição é cacheada em disco** (por snapshot e parâmetros): CEEMDAN é lento e tem ruído aleatório, então também recebe a semente.
- **Vazamento temporal (D-10) é a decisão mais importante do projeto**; ver o §6.1. O teste de vazamento da etapa cobre a decomposição, não só as janelas.
- Conferência da decomposição antes de treinar: a soma das subséries reconstrói o fechamento (erro numérico pequeno); o gráfico das IMFs e das entropias lembra a Fig. 8 do artigo (entropia decrescente da IMF1 para a última).

Pronto quando: os critérios do §4; a reconstrução conferida; o relatório traz o tempo da decomposição à parte do tempo de treino.

**Concluída em 2026-11-18 (código e validação rápida).**

- **Código:** `modelos/hdfm.py`. `decompor` faz CEEMDAN → entropia amostral (NumPy) → K-means (K = 3) → VMD na Co-IMF de maior entropia → 12 subséries: os 10 modos, do mais lento ao mais rápido, com o resíduo da VMD somado ao mais rápido (D-22), e as Co-IMFs 2 e 3. `montar` gera as amostras, normaliza (A ou B, estatísticas só do treino) e devolve, por parte, a fusão (desfaz a normalização e soma). `HDFM` = 12 GRUs da baseline (`f`), sem parâmetro em comum, treinadas com a MSE média das 12 pelo laço comum.
- **Dependências** (`uv add`): `EMD-signal` 1.10.0 (CEEMDAN), `vmdpy` 0.2 (VMD), `scikit-learn` (K-means). Instalam no Python do projeto (R-04 resolvido).
- **Comando:** `uv run python redes_neurais_2026_2/projeto/rodar.py --modelo hdfm_diferenca --decomposicao causal --janela 120 --camadas 128 --epocas 3` (`--modelo hdfm_minmax` para a A; `--decomposicao inteira` é o padrão).
- **Cache:** `execucoes/cache_hdfm/`, um `.npy` por decomposição, com chave = sha256 da série e dos parâmetros. A causal calcula 1.129 decomposições (as séries `[0..t]`, t de 119 a 1.247) em 12 processos: **13,7 min** na primeira vez, instantâneo depois. A semente da decomposição é fixa (0) e não acompanha a de treino, para não refazer a causal a cada semente.
- **Conferência** (`uv run python redes_neurais_2026_2/projeto/modelos/hdfm.py`): entropias decrescentes da IMF1 ao resíduo (como na Fig. 8 do artigo), cotovelo em K = 3 (D-12), reconstrução com erro 1e-14, figura em `execucoes/hdfm_conferencia.png`.
- **Windows:** cada CEEMDAN roda com `parallel=False`; o paralelismo é entre decomposições (`ProcessPoolExecutor`), o que rende mais (5,5 s por decomposição de ~1.000 pontos num núcleo).
- **Histórico curto na causal:** a CEEMDAN dos 120 primeiros pontos da PETR4 dá 5 componentes, o bastante para K = 3. Com menos de 3, `decompor` para com erro.

**Relatório da validação rápida** (máquina do estudo, GPU RTX 3060 Ti, 3 épocas, 878 amostras de treino, 605.196 parâmetros):

| Modelo                    | Decomposição | Perda de treino (1ª → 3ª) | RMSE val. | MAE   | MAPE   | Direção   |
| ------------------------- | ------------ | ------------------------- | --------- | ----- | ------ | --------- |
| HDFM A (`hdfm_minmax`)    | inteira      | 0,109 → 0,023             | 5,74      | 4,71  | 11,6 % | 0,352     |
| HDFM A (`hdfm_minmax`)    | causal       | 0,100 → 0,011             | 5,88      | 4,77  | 11,7 % | 0,368     |
| HDFM B (`hdfm_diferenca`) | inteira      | 0,808 → 0,334             | **0,436** | 0,339 | 0,89 % | **0,768** |
| HDFM B (`hdfm_diferenca`) | causal       | 0,883 → 0,647             | 0,753     | 0,569 | 1,48 % | 0,488     |
| Referência ingênua        |              |                           | 0,712     |       |        | 0,664     |

- **Funciona:** a perda cai nas quatro, sem NaN. Tempo por época: 0,5 s na 1ª, 0,25 s nas demais; VRAM 428 MB, RAM 1,2 GB. Projeção: 50 épocas ~13 s, 100 ~25 s, 200 ~50 s, mais a decomposição, feita uma vez.
- **A normalização A falha como a D-20 previa:** a Co-IMF3 (tendência) passa, na validação e no teste, de tudo o que o treino viu; a rede não extrapola, e o erro é 8 vezes o do ingênuo.
- **O vazamento da D-10 aparece já em 3 épocas:** com a B, a decomposição da série inteira bate o ingênuo com folga (RMSE 0,44 contra 0,71; direção 0,77), e a causal não (0,75; 0,49). A diferença entre as duas é o tamanho do vazamento: é a limitação mais forte para o relatório (L-06).
- Nada foi ajustado (§4): os números mostram que o treino funciona, não escolhem nada.
- **Testes de fumaça (`testes/test_hdfm.py`, 7; a suíte toda tem 27, todos passam):** entropia igual à de uma implementação por força bruta, e menor num seno que em ruído; 12 subséries que somam a série (comprimentos par e ímpar); decomposição determinística; a fusão desfaz a normalização (A e B, inteira e causal); **vazamento:** multiplicar por 3 o trecho de teste não muda treino nem validação na causal, e muda na inteira; 3 épocas com a perda caindo, e retomar dá previsões idênticas (A inteira, B inteira, B causal).
- **Desvios do artigo:** D-10 (decomposição causal além da do artigo), D-11 e D-13 (parâmetros das bibliotecas), D-12 (m, r, K), D-14 (normalizações A e B; fusão por soma simples), D-22 (resíduo da VMD), D-03 (GRU `f` no lugar de 128-64-32) e D-01 (janela 120 no lugar de 30).
- **Pendências para a Etapa 6:** épocas; se a normalização A, que já se mostrou inviável pela D-20, vai para a Etapa 7 ou fica só como registro.

### Etapa 5 — TEGRU: removida em 2026-11-18

**Decisão do autor (2026-11-18): o TEGRU (Haryono et al., 2023) não entra no estudo.** A contribuição principal do artigo é o indicador de sentimento de notícias, e não há fonte de notícias para a PETR4 (o dataset do artigo é de notícias indonésias). Sem ela, sobraria só a arquitetura, com quase nenhum hiperparâmetro informado (D-15 e D-16 tinham tudo em aberto). No lugar, o estudo ganhou variantes dos dois artigos restantes (FinGAT em 3a/3b, HDFM com as duas normalizações da D-14): 5 modelos de 2 artigos, em vez de 4 de 3.

**Onde está o que saiu:** todo o detalhamento do TEGRU (o plano desta etapa, as decisões D-15 e D-16, a limitação das notícias, a linha da tabela de tempos) está na versão do guia do commit `10c3b08`: `git show 10c3b08:redes_neurais_2026_2/guia-implementacao-projeto.md`. Nenhum código do TEGRU chegou a ser escrito. Ficaram menções ao TEGRU só nos registros de decisões já tomadas (D-01 a D-05, D-21, a grade da Etapa 2), porque explicam escolhas feitas antes da remoção, e como fonte de duas métricas (§3.4). A numeração das etapas foi mantida.

### Etapa 6 — Condições do estudo (item f)

Sem código novo de modelo. Com os modelos validados e os tempos medidos (§6.4), decide-se e grava-se em [`condicoes-do-estudo.md`](condicoes-do-estudo.md):

| Seção do documento  | O que fixa                                                                                                        |
| ------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Dados               | snapshot usado (hash e datas do manifesto), divisão treino/validação/teste, normalização                          |
| Hiperparâmetros     | uma tabela por modelo: valor, e a origem de cada valor (artigo, com a seção; ou decisão D-NN)                     |
| Épocas e parada     | épocas por modelo; critério de escolha do checkpoint (melhor validação ou última época); parada antecipada ou não |
| Repetições          | quantas sementes por modelo (o FinGAT reporta a média de 10 execuções); cabe no orçamento de tempo?               |
| Métricas            | a lista final do §3.4, com as fórmulas e as definições de direção                                                 |
| Orçamento           | horas previstas por modelo = tempo por época × épocas × sementes; total e folga                                   |
| Desvios dos artigos | a consolidação das decisões D-NN: é a base da seção de limitações do relatório                                    |

- **Regularização e variantes (anotado em 2026-11-16):** conferir aqui se vale ligar dropout ou parâmetros parecidos na GRU compartilhada. O dropout treina, a cada passo, uma sub-rede diferente da mesma arquitetura, o que aproxima "testar várias arquiteturas de uma vez" sem multiplicar execuções. Nenhum dos três artigos informa dropout na GRU; se entrar, entra igual nos quatro modelos e é registrado como decisão.
- **Limitação a levar ao relatório (L-07):** a GRU compartilhada (`ohlcv`, janela 120, `f`) foi escolhida por menor RMSE de validação, sem vitória clara sobre a referência ingênua (3,4% melhor; direção empatada) nem sobre as outras combinações (diferenças dentro do ruído entre sementes). Dizer isso na seção de limitações, junto com os desvios dos artigos.
- **A janela e a GRU já chegam decididas** da Etapa 2; aqui só se confirma que continuam viáveis com os tempos medidos em todos os modelos.
- **Épocas iguais para todos ou orçamento igual?** Os modelos têm custos por época muito diferentes (HDFM = 12 GRUs; FinGAT = todas as ações). A decisão é tomada aqui, com os números na mão.
- Se algum modelo for inviável na máquina do estudo, as saídas do enunciado valem nesta ordem: menos épocas → entrada menor → base menor → menos camadas/neurônios; trocar o modelo por um clássico exige avisar o professor.
- Depois de aprovado, o documento **não muda durante a Etapa 7**. Mudança depois de ver resultado do teste é registrada como tal, com data e motivo.

Pronto quando: o documento está aprovado, e `rodar.py` lê as condições de um arquivo de configuração que corresponde a ele (um só lugar para os valores; o `.md` os descreve e aponta para o arquivo).

**Concluída em 2026-11-18.**

- **[`condicoes-do-estudo.md`](condicoes-do-estudo.md) aprovado**, com os pontos P-01 a P-10 decididos (§9 dele). Os principais: 6 configurações (GRU, FinGAT repo, Custom FinGAT, HDFM A causal, HDFM B causal na comparação; HDFM B com a série inteira só para medir o vazamento); no máximo 100 épocas em todos, com o checkpoint de menor perda de validação, sem parada antecipada; 5 sementes (0 a 4); sem dropout na GRU compartilhada; AcMAPE fora; intervalo de Wilson de 95 % na acurácia de direção; divisão 80/10/10 sobre 5 anos mantida, sem `TimeSeriesSplit`.
- Os dois pontos abertos acima ficaram resolvidos assim: **dropout** não entra (P-02); **épocas iguais** para todos, não orçamento igual (P-03).
- **Arquivo de configuração:** `projeto/condicoes.json` (configurações, sementes, épocas, janela, GRU, lote, lr, papel de cada configuração e o tempo por época medido, usado na previsão de término). Quem o lê é a fila (`projeto/fila.py`), que passa os valores ao `rodar.py`. Os hiperparâmetros internos de cada modelo continuam no código (`VARIANTES`, `DELTA`, `LAMBDA` em `modelos/custom_fingat.py`; `PARAMS` em `modelos/hdfm.py`), e o §3 do documento aponta para eles.
- **Orçamento previsto:** ~6 h 30, quase todo do FinGAT repo (~73 min por semente).
- **Layout do relatório:** [`layout-relatorio.md`](layout-relatorio.md), com o que cada seção explica e de onde sai cada insumo.

### Etapa 7 — Execução (item g)

**Antes de soltar a execução longa** (o que pode dar errado e a medida já tomada):

| Risco                                                   | Medida                                                                                                                                    |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| O Windows suspende, hiberna ou reinicia por atualização | Desligar suspensão e pausar atualizações durante a execução; mesmo assim, checkpoint a cada época e `--retomar`                           |
| O terminal é fechado ou a sessão do Windows cai         | A fila grava o log em arquivo e guarda o estado em disco; abrir o terminal e rodar a fila de novo retoma do último checkpoint             |
| Falta de memória (16 GB de RAM, 6 GB de VRAM)           | Pico de RAM e de VRAM medido nas etapas 2 a 5; um processo por modelo, em sequência, nunca em paralelo                                    |
| Uma falha num modelo derruba os outros                  | Um processo por (modelo, semente); a fila continua com o próximo e registra a falha                                                       |
| Perda vira NaN na época 80                              | O laço detecta NaN/inf, para aquela execução, mantém o último checkpoint bom e registra                                                   |
| Checkpoint corrompido por queda no meio da gravação     | Gravar em arquivo temporário e renomear (troca atômica); manter o melhor e o último                                                       |
| yfinance fora do ar, limitado ou com preços reajustados | Nenhum treino acessa a rede: tudo lê o snapshot (§3.1)                                                                                    |
| Resultado não reproduzível                              | Semente, configuração completa e hash do snapshot gravados em cada pasta de `execucoes/`                                                  |
| Retomada dá resultado diferente de rodar direto         | Checkpoint guarda modelo, otimizador, época e estado do gerador aleatório; testado no ciclo de cada etapa (§4)                            |
| Refazer a decomposição do HDFM a cada retomada          | Cache em disco (Etapa 4)                                                                                                                  |
| Erro só aparece no fim (ex.: ao gravar as previsões)    | **Ensaio geral:** a fila inteira, com a configuração real, 1 época por modelo, do começo à gravação das métricas, antes da execução longa |
| Disco cheio de checkpoints                              | Só "melhor" e "último" por execução; tamanho previsto conferido no ensaio geral                                                           |
| Não saber se travou ou se está só lento                 | Log com data e hora por época e a previsão de término; o tempo por época esperado é o do §6.4                                             |

Entregáveis: a fila de execuções (um script que roda cada modelo e semente em sequência, pulando o que já terminou), o ensaio geral aprovado, a execução longa, e uma tabela de resultados gerada a partir das previsões gravadas.

**Verificação de saúde (2026-11-18), na máquina do estudo:**

| Risco da tabela acima                  | Situação                                                                                                                         |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Suspensão e hibernação                 | ok na tomada (plano "Equilibrado": nunca suspende nem hiberna); na bateria suspende em 10 min. **Falta:** pausar o Windows Update antes de soltar |
| Terminal fechado / retomada pela fila  | ok: `projeto/fila.py` lê `projeto/condicoes.json`, roda cada (configuração, semente) num processo, pula o que terminou e retoma o resto; trava contra duas filas ao mesmo tempo |
| Memória                                | ok: maior pico 6,8 GB de VRAM (Custom FinGAT); a fila roda um processo por vez                                                   |
| Falha isolada                          | ok: `rodar.py` sai com código ≠ 0; a fila segue com a próxima e registra a falha em `fila.log` e `fila_situacao.csv`            |
| NaN                                    | ok: o laço para, mantém os checkpoints e sai com erro. A execução fica sem previsões e conta como falha (risco maior: FinGAT repo, lr 0,05) |
| Checkpoint corrompido                  | ok: `ultimo.pt`, `melhor.pt`, `config.json` e `metricas.json` gravados com troca atômica                                        |
| Rede / yfinance                        | ok: tudo lê o snapshot, com sha256                                                                                              |
| Reprodutibilidade                      | ok: semente, configuração, sha256, versão do PyTorch e commit do código (com aviso de alteração não commitada) em cada `config.json`; `config.json` e `metricas.json` gravados com troca atômica |
| Retomada igual a rodar direto          | **ok também na GPU:** GRU, HDFM B causal, Custom FinGAT (3 épocas) e FinGAT repo (2 épocas), dados reais: duas execuções diretas e direto × retomado dão previsões de validação idênticas (diferença 0) |
| Decomposição do HDFM a cada retomada   | ok: cache em `execucoes/cache_hdfm/` (1.129 decomposições, 74 MB)                                                               |
| Erro só no fim                         | ok: **ensaio geral** (`fila.py --ensaio`, 1 época, em `execucoes/ensaio/`): 30/30 execuções ok em 8,5 min, do começo às previsões; rodar de novo pulou as 30. A primeira tentativa achou um erro de codificação na fila (a saída do `rodar.py` em cp1252 derrubava a fila no Windows), corrigido forçando UTF-8 |
| Disco                                  | ok: 287 GB livres; uma execução ocupa de 1 MB (GRU) a 24 MB (FinGAT repo); as 30 do ensaio ocuparam 295 MB                       |
| Travou ou está lento                   | ok: cada época imprime data e hora (`<execução>/saida.log`); a fila registra o início de cada execução com a previsão de término (`execucoes/fila.log`); no ensaio, previsto 18:16, terminou 18:13 |

Pronto quando: todas as execuções previstas em `condicoes-do-estudo.md` terminaram ou têm a falha registrada; a tabela de métricas é regenerada do zero a partir de `execucoes/`; nenhuma condição do estudo mudou sem registro.

**Andamento (2026-11-18): pronta para soltar a execução longa.**

| Entregável                                   | Situação                                                                                                     |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Verificação de saúde                         | feita (tabela acima)                                                                                         |
| Fila (`projeto/fila.py`)                     | pronta: lê `projeto/condicoes.json`; 6 configurações × 5 sementes = 30 execuções, das baratas para a cara (GRU, HDFM, Custom FinGAT em ~25 min; depois o FinGAT repo, ~6 h) |
| Ensaio geral                                 | **aprovado:** 30/30 ok com 1 época                                                                            |
| Execução longa                               | **a soltar**                                                                                                 |
| Script da tabela de resultados               | a fazer (pode ser escrito e testado com os dados do ensaio enquanto a longa roda)                            |
| Script de figuras                            | a fazer (idem)                                                                                               |

**Antes de soltar:**

1. Commitar o código: cada execução grava o commit no `config.json`; com mudança não commitada, ele marca `alteracoes_nao_commitadas: true`, e o commit gravado não reproduz o código que rodou.
2. Pausar o Windows Update e manter a máquina na tomada.
3. Num terminal próprio, a partir da raiz do repositório: `uv run python redes_neurais_2026_2/projeto/fila.py`. Se cair, o mesmo comando retoma. Acompanhar em `projeto/execucoes/fila.log`; a saída de cada execução fica em `<execução>/saida.log`.

**Depois da execução longa:**

- o script de resultados gera, do zero e a partir de `execucoes/`, as tabelas do relatório (`layout-relatorio.md` §5 e Apêndice C): métricas no teste por modelo (média ± desvio de 5 sementes), intervalo de Wilson na direção, referência ingênua, a acurácia nas 48 ações do FinGAT, a tabela do vazamento (HDFM B inteira × causal) e a de custo;
- o script de figuras gera previsto × real no teste, o vazamento da decomposição, as subséries do HDFM com a faixa do treino e os efeitos marginais da Etapa 2;
- conferir `fila_situacao.csv`: execução que falhou (ex.: NaN no FinGAT repo) entra no relatório como falha, com a causa, sem trocar nenhuma condição.

---

## 6. Pontos em aberto e riscos

### 6.1. Decisões a tomar (ambiguidades e lacunas)

Cada linha: o que falta, o que o artigo diz, e uma **proposta** — que só vale depois de aprovada. A coluna "Decisão" vazia quer dizer pergunta em aberto.

**Comuns**

| ID   | Questão                           | O que se sabe                                                                                                                                                      | Decisão (2026-11-16)                                                                                                                                                                                                                                                                                                                                |
| ---- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D-01 | Janela                            | FinGAT usa 15 (3×5). HDFM usa 30 (§IV-B). TEGRU testa 5 a 200, e o melhor varia por ação (Tabela 12).                                                              | **Validada na Etapa 2:** testar 5, 15, 30, 60, 120 e 200; a escolhida vale para os quatro modelos. **Valor escolhido (2026-11-18): janela 120** (menor RMSE de validação da melhor combinação, `resultados-2b.md`; escolha dentro do ruído, ver L-07).                                                                                              |
| D-02 | Divisão treino/validação/teste    | FinGAT usa 60/20/20; HDFM, os últimos 800 pontos como teste; TEGRU não informa.                                                                                    | **80/10/10**, em ordem cronológica, sobre os 5 anos.                                                                                                                                                                                                                                                                                                |
| D-03 | Configuração da GRU compartilhada | HDFM: 3 camadas (128, 64, 32), tanh, lote 64, Adam 0,001, 100 épocas (Tabela 2). FinGAT: 16 unidades, e 32/64 pioram (Fig. 4). TEGRU: não informa.                 | **Validada na Etapa 2:** 9 arquiteturas (`a` a `i`), ou só `a`, `d` e `i` se o tempo não permitir as 9 com qualidade. A escolhida vale para os quatro modelos. Tem de caber na máquina do estudo (Ryzen 5 5500, RTX 5060 6 GB, 16 GB). **Valor escolhido (2026-11-18): `f`, 1 camada de 128** (51.969 parâmetros; grade completa rodada, ver L-07). |
| D-04 | Preço ajustado ou bruto           | A PETR4 paga proventos altos; o fechamento bruto tem degraus nas datas ex. HDFM e TEGRU usam "close" do Yahoo, sem dizer qual. O FinGAT usa os dois como features. | **Aprovado:** baixar os dois; alvo e entrada de GRU, HDFM e TEGRU = fechamento **ajustado**; FinGAT recebe os dois, como no artigo.                                                                                                                                                                                                                 |
| D-05 | Entrada da GRU e do TEGRU         | A GRU do HDFM é univariada (fechamento). O TEGRU tem cenário univariado (UNI) e OHLCV (OTI).                                                                       | **Validada na Etapa 2:** `univariado` contra `ohlcv`. A escolhida vale para a GRU e o TEGRU (§3.1). **Valor escolhido (2026-11-18): `ohlcv`** (ver L-07).                                                                                                                                                                                           |

**Estudo da GRU e máquina (bloqueiam a Etapa 0 e a parte 2b)**

| ID   | Questão                                            | O que se sabe                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Proposta                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Decisão                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ---- | -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D-17 | Dias-alvo iguais entre as janelas                  | Com 80/10/10 por data, a validação e o teste são os mesmos dias para qualquer janela (a janela de um alvo de validação pode buscar entradas no trecho de treino; isso não é vazamento). No treino, a janela de 200 perde os primeiros 200 alvos (~800 amostras contra ~995 da janela de 5).                                                                                                                                                                    | **(a)** todas as janelas treinam só nos alvos que a de 200 alcança (~800): comparação limpa, e 20% do treino jogado fora; **(b)** cada janela usa tudo o que consegue: mais dados para as curtas, e a diferença de quantidade se mistura ao efeito da janela. Sugestão: **(a)** no estudo da 2b; o treino final da Etapa 7 usa tudo o que a janela escolhida alcança.                                                                                                                                                                                                                                                                                             | **(a), aprovado em 2026-11-16:** no estudo da 2b, todas as janelas treinam só nos alvos que a de 200 alcança; o treino final da Etapa 7 usa tudo o que a janela escolhida alcança.                                                                                                                                                                                                                                                             |
| D-18 | Sistema da máquina do estudo                       | Não informado: Windows nativo, WSL ou Linux. Muda a instalação do PyTorch com CUDA e as medidas contra interrupção da Etapa 7.                                                                                                                                                                                                                                                                                                                                 | Informar; o guia é ajustado.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | **Windows 11 nativo** (2026-11-16).                                                                                                                                                                                                                                                                                                                                                                                                            |
| D-19 | Épocas e sementes do estudo da GRU                 | Dependem dos tempos da parte 2a e do prazo (19/11).                                                                                                                                                                                                                                                                                                                                                                                                            | Decidir depois da 2a, com a conta do orçamento (Etapa 2, "Orçamento e ordem de corte"). Ponto de partida: até 100 épocas com parada antecipada pela validação, 3 sementes; cortar pela ordem da Etapa 2.                                                                                                                                                                                                                                                                                                                                                                                                                                                          | **Adiada para depois da 2a** (aprovado em 2026-11-16).                                                                                                                                                                                                                                                                                                                                                                                         |
| D-20 | Normalização quando o teste sai da faixa do treino | Achado da Etapa 0: no fechamento ajustado o treino vai de 7,72 a ~32, a validação chega a ~47 e o teste a ~60. Com min-max ajustado só no treino (§3.1), validação e teste ficam entre 1 e 2,1, valores que a rede nunca viu; GRU com tanh não extrapola nível, e o erro mediria isso, não o modelo. O HDFM normaliza "os dados" (Eq. 20) sem dizer se inclui o teste; seus índices de teste ficam dentro da faixa do treino, então o problema não aparece lá. | **(a)** min-max de nível só no treino, como está: honesto, e provavelmente todos os modelos falham no teste pelo mesmo motivo; **(b)** min-max sobre a série inteira: resolve a faixa, mas usa o máximo do teste (vazamento); **(c)** normalizar cada janela pelo seu último fechamento e prever a variação relativa de `t+1`; o preço previsto é reconstruído exatamente, as métricas continuam em preço, e não há vazamento. Sugestão: **(c)** na GRU e no TEGRU. No HDFM a decomposição é feita sobre o nível, e a subsérie de tendência tem o mesmo problema: a normalização dele fica para D-14, à luz desta decisão. Bloqueia `comum/janelas.py` (Etapa 1). | **(c), aprovada em 2026-11-16**, com a divisão cronológica mantida (embaralhar a divisão foi descartado: é vazamento). Vale para a GRU e o TEGRU; a do HDFM fica em D-14. As referências que sustentam a escolha são definidas na escrita do relatório; candidatas levantadas: Jiang et al. (2017, arXiv:1706.10059), Zeng et al. (2023, arXiv:2205.13504), Passalis et al. (2020, arXiv:1902.07892), Salinas et al. (2020, arXiv:1704.04110). |
| D-21 | Normalização do volume na entrada `ohlcv`          | A D-20 normaliza preços pelo último fechamento da janela; o volume tem outra unidade e outra escala (9 a 320 milhões no snapshot). O TEGRU, que usa OHLCV, não informa a normalização. Segundo ponto: o snapshot só tem o fechamento ajustado; abertura, máxima e mínima vêm brutas, e misturá-las com o fechamento ajustado (D-04) daria candles incoerentes.                                                                                                 | Ajustar abertura, máxima e mínima pelo mesmo fator do dia (`Adj Close/Close`) e normalizá-las como o fechamento (D-20). Dividir o volume de cada pregão pela **média do volume da própria janela** (mesma lógica da D-20: sem estatística do treino). Alternativa: logaritmo do volume com média e desvio do treino. Bloqueia só a entrada `ohlcv` da Etapa 1.                                                                                                                                                                                                                                                                                                    | **Proposta aprovada em 2026-11-17:** abertura, máxima e mínima ajustadas por `Adj Close/Close` e normalizadas como o fechamento; volume dividido pela média do volume da própria janela. Em todas as colunas subtrai-se 1 (só centra em zero, como na D-20). Janela com volume médio zero é recusada.                                                                                                                                          |

**FinGAT (bloqueiam a Etapa 3)**

| ID   | Questão                           | O que se sabe                                                                                                                                                                                                                                                                                                                                                                           | Proposta                                                                                                                                                                                                                                                                                                                                                                    | Decisão                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ---- | --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D-06 | Como o FinGAT entra na comparação | Prevê retorno (treinado por ranking, sem escala) e movimento (§3.1).                                                                                                                                                                                                                                                                                                                    | Só direção da PETR4, pela saída de movimento. Reportar também a ACC sobre todas as ações, como conferência com o artigo.                                                                                                                                                                                                                                                    | **Aprovada (2026-11-18).** O `metricas.json` traz também a ACC que o `train.py` do repo calcula (sinal do retorno previsto), como conferência.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| D-07 | Universo de ações e setores       | O modelo não existe com uma ação só: precisa de ações do mesmo setor e de vários setores. O artigo usa 100 ações/5 setores (Taiwan) e 424/9 (S&P 500).                                                                                                                                                                                                                                  | Ações do Ibovespa com 5 anos completos, setor pela classificação setorial da B3, em `universo.csv` (lista fixa, conferida à mão; o campo de setor do yfinance é instável). Tamanho a decidir: ~50 a 80 ações, limitado pela memória. O setor da PETR4 precisa ter outras ações (PETR3, PRIO3, VBBR3…).                                                                      | **Aprovada (2026-11-18), com setores do mesmo tamanho:** 6 ações por setor (o pooling por atenção do repo exige o mesmo número em todos), as de maior volume financeiro médio (`Close × Volume`) no trecho de treino, entre as candidatas de `dados/universo.csv` com os 1.248 pregões da PETR4. Resultado: **8 setores × 6 = 48 ações** (Etapa 3). **Classificação setorial da B3:** <https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/acoes/consultas/classificacao-setorial/> (planilha "Classificação Setorial", coluna "Setor Econômico"). A coluna `setor` de `universo.csv` foi preenchida de memória e **conferida e aprovada pelo autor em 2026-11-18**.                 |
| D-08 | Usar o código público do FinGAT   | O artigo cita `github.com/Roytsai27/Financial-GraphAttention`. Ainda não foi conferido se está no ar, se roda, e se bate com o artigo.                                                                                                                                                                                                                                                  | Implementar a partir do artigo, com o bloco GRU compartilhado, e usar o repositório **só para tirar dúvidas** do que o texto não define (D-09), citando-o. Usar o código deles direto quebraria a OBS 3.                                                                                                                                                                    | **Decidido (2026-11-18): fork do repositório, em duas variantes.** O código dos autores difere do artigo (perda, GRU de longo prazo, GAT por semana, pooling, lr; lista na Etapa 3). Por isso: **3a** (`fingat_repo`) = o código do repo; **3b** (`fingat_artigo`) = o texto do artigo. Nas duas, a GRU é a da Etapa 2 (D-03). A cópia sem modificação está em `projeto/terceiros/fingat_original/`, e o `ORIGEM.md` dela registra a origem (URL, commit `0f7a2dd`, data) e cada modificação. **O repositório não traz dados:** dele vem só o código; os dados são do Yahoo Finance. **Não tem licença:** uso acadêmico com citação.                                                                       |
| D-09 | Lacunas do texto                  | (a) A Eq. 3 como impressa divide por `(adjclose − 1)`; parece erro de impressão de `média/adjclose − 1`. (b) Número de cabeças e de camadas do GAT, e se o nó atende a si mesmo. (c) Normalização das features de preço bruto. (d) Épocas. (e) O "lote de 128" é de dias ou de pares dia-ação. (f) Taxa de aprendizado: o artigo busca em {0,0005; 0,001; 0,005} sem dizer a escolhida. | Conferir cada item no repositório (D-08) e trazer a resposta; onde o repositório não resolver: (a) `média/adjclose − 1`; (b) 1 camada, 1 cabeça, sem laço próprio, como nas Eq. 7–8; (c) normalizar por ação, com estatísticas do treino; (e) lote de dias; (f) 0,001. GAT denso próprio em PyTorch (os grafos são completos e pequenos), sem instalar o PyTorch Geometric. | **Aprovada (2026-11-18).** O que o repo responde, e o que cada variante usa: (a) o repo calcula `média/Close − 1`: confirma a leitura `média/adjclose − 1`; as duas usam o ajustado. (b) `GATConv` padrão: 1 camada, 1 cabeça, **com** laço próprio → 3a com laço; 3b sem laço (o artigo liga cada ação "a todas as outras", §4.1). (c) o repo padroniza o fechamento na série inteira (vazamento) → as duas usam só o treino. (d) o repo usa 20 épocas → Etapa 6. (e) o `train.py` dá **um passo por dia** (o `--batch_size 128` não é usado) → 3a um dia; 3b lote de 128 dias. (f) o repo usa lr 0,05, fora da busca do artigo → 3a 0,05; 3b 0,001. GAT denso nas duas; na 3a, equivalente ao `GATConv`. |

**HDFM (bloqueiam a Etapa 4)**

| ID   | Questão                                    | O que se sabe                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Proposta                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Decisão                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ---- | ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| D-10 | **Decomposição e vazamento temporal**      | O artigo não diz se CEEMDAN e VMD são aplicados à série inteira ou só ao treino. A Fig. 8 mostra a decomposição dos ~8.000 pontos (treino + teste). Decompor a série inteira põe informação do futuro em cada ponto das subséries: o valor de uma IMF em `t` depende de pontos depois de `t`. **Medido em 2026-11-18 (máquina do estudo, EMD-signal 1.10.0, 100 realizações, 12 processos):** uma CEEMDAN leva 13 s na série inteira (1.248 pontos) e 12 s no treino (998); as duas dão 7 componentes, e a soma reconstrói a série (erro 7e-15). Uma VMD com K = 10 leva 0,2 s. Logo, (b) só em validação e teste (~250 decomposições) custa ~50 min; (b) também no treino, cada janela decomposta só com o próprio passado (~1.100), ~3,7 h. As duas cabem em cache. | Três caminhos: **(a)** série inteira, como o artigo aparenta — simples, mas com vazamento, e o resultado sai inflado; **(b)** causal: para cada dia de teste, decompor só o passado até ele — honesto, porém ~190 decomposições e um problema novo (o número de IMFs muda entre janelas); **(c)** rodar (a) e (b) e reportar a diferença. Sugestão: **(c)** se o custo de (b) couber; é a limitação mais forte que o projeto pode demonstrar, e o enunciado pede limitações. Decidir depois de medir o tempo de uma decomposição. | **(c), aprovada em 2026-11-18:** rodar (a) série inteira e (b) causal, e reportar a diferença. (b) é causal **também no treino** (aprovado em 2026-11-18): cada amostra usa a decomposição só do próprio passado; ~1.100 decomposições, ~3,7 h, em cache.                                                                                                                                                                                                                                                                                  |
| D-11 | Parâmetros do CEEMDAN                      | Não informados (número de realizações, amplitude do ruído).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Padrões da biblioteca `EMD-signal` (PyEMD): 100 realizações, ε = 0,005; semente fixa. Conferir antes se a biblioteca instala no Python do projeto (R-04).                                                                                                                                                                                                                                                                                                                                                                         | **Aprovada (2026-11-18):** padrões da `EMD-signal` (100 realizações, ε = 0,005), semente fixa. No relatório: o artigo não informa; usamos os padrões da biblioteca.                                                                                                                                                                                                                                                                                                                                                                        |
| D-12 | Entropia amostral e K-means                | A Fig. 8 testa m ∈ {1, 2} e r ∈ {0,1; 0,2}, mas o texto não diz qual gerou a Tabela 6. K sai do "método do cotovelo", sem o critério automático; deu 3 nos três índices.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | m = 2, r = 0,2·desvio-padrão (o usual), implementação própria em NumPy (poucas linhas). K = 3 fixo, como o resultado do artigo, conferindo o gráfico do cotovelo para a PETR4; se o cotovelo for outro, parar e perguntar.                                                                                                                                                                                                                                                                                                        | **Aprovada (2026-11-18):** m = 2, r = 0,2·desvio-padrão, implementação própria em NumPy; K = 3, conferindo o cotovelo na PETR4 (se der outro, parar e perguntar). **Cotovelo conferido (2026-11-18):** na série inteira da PETR4, entropias 1,41 / 0,76 / 0,45 / 0,42 / 0,07 / 0,03 / 0,00 (decrescentes, como na Fig. 8); inércia do K-means 1,54 / 0,40 / 0,075 / 0,003 para K = 1 a 4, plana a partir de K = 3. Grupos: {IMF1}, {IMF2–4}, {IMF5–resíduo}. |
| D-13 | Parâmetros do VMD e alvo da redecomposição | Número de modos, α e τ não informados (a Fig. 11 tem 10 modos). O texto fala em "limiar predeterminado" de entropia, sem o valor; na prática só a Co-IMF1 é redecomposta.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | `vmdpy`, K = 10, α = 2000, τ = 0 (padrões usuais); redecompor só a Co-IMF de maior entropia.                                                                                                                                                                                                                                                                                                                                                                                                                                      | **Aprovada (2026-11-18):** padrões usuais da `vmdpy` (K = 10, α = 2000, τ = 0), redecompondo só a Co-IMF de maior entropia. No relatório: o artigo não informa; usamos os padrões.                                                                                                                                                                                                                                                                                                                                                         |
| D-14 | Normalização e fusão das subséries         | A Eq. 20 normaliza "os dados", sem dizer se cada subsérie tem o próprio min-max. O texto fala em "soma ponderada" das previsões, sem pesos; a Fig. 2 mostra só "Fusion".                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Min-max por subsérie, ajustado no treino; fusão = soma simples (pesos 1), que é a inversa exata da decomposição.                                                                                                                                                                                                                                                                                                                                                                                                                  | **Decidido (2026-11-18): implementar as duas leituras, como no FinGAT.** Fusão = soma simples (pesos 1), a inversa exata da decomposição, nas duas. Normalização **A** (Eq. 20 ao pé da letra): min-max por subsérie, ajustado no treino; a tendência no teste sai da faixa do treino (D-20). Normalização **B** (D-20 adaptada): por janela e subsérie, subtrai-se o último valor e divide-se pelo desvio-padrão das variações de um dia da subsérie no treino; a rede prevê o passo de amanhã. A e B são os modelos 4 e 5 da comparação. |
| D-22 | Resíduo da VMD | Achado da Etapa 4: a VMD não reconstrói exatamente a Co-IMF que decompõe. Na PETR4, o resíduo (Co-IMF1 − soma dos 10 modos) tem desvio 0,047 e máximo 0,23, ~12% do desvio da variação diária do fechamento (0,39). O artigo fala em 12 subséries somadas e não menciona o resíduo. | Somar o resíduo ao modo de maior frequência: continuam 12 subséries (Fig. 2) e a soma volta a reconstruir o fechamento. Alternativas: ignorar (erro fixo na previsão), 13ª subsérie, ou as duas leituras. | **Aprovada (2026-11-18):** somar ao modo de maior frequência. |

### 6.2. Riscos

| ID   | Risco                              | Situação                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ---- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| R-01 | Memória                            | A máquina de desenvolvimento tem 4 GB visíveis no WSL: serve para testes de fumaça, não para o FinGAT com o universo inteiro. A máquina do estudo tem 16 GB de RAM e 6 GB de VRAM, folgados para estes modelos; o pico é medido em cada etapa mesmo assim.                                                                                                                                                                |
| R-02 | GPU da máquina do estudo           | A RTX 5060 precisa de um build recente do PyTorch com CUDA; se não funcionar, o estudo roda em CPU (Etapa 0). Mesmo funcionando, redes pequenas com ~1.000 amostras podem treinar mais rápido em CPU: a Etapa 2a mede os dois. CEEMDAN e VMD rodam só em CPU.                                                                                                                                                             |
| R-03 | Poucos dados                       | ~1.000 janelas de treino são pouco para 3 camadas de GRU e para as 12 GRUs do HDFM; o HDFM usou ~7.000 pontos de treino. Sobreajuste é provável e é resultado a reportar, não a esconder. A alternativa (série mais longa) muda o pedido e fica para decisão na Etapa 6.                                                                                                                                                  |
| R-04 | Bibliotecas de decomposição        | `EMD-signal` e `vmdpy` são pouco mantidas; o Python do sistema é 3.13. Conferir a instalação no começo da Etapa 4; se falhar, opções: outra versão de Python no `uv`, ou outra implementação — perguntar antes.                                                                                                                                                                                                           |
| R-05 | Comparação desigual por construção | O FinGAT treina com dezenas de ações, os outros só com a PETR4; o HDFM pode ter vazamento (D-10). Cada um vira item da seção de limitações do relatório.                                                                                                                                                                                                                                                                  |
| R-06 | Direção perto de 50%               | Os próprios artigos reportam acurácia de direção entre 50% e 57% (FinGAT, Tabela 2). Com ~125 dias de teste, diferenças de poucos pontos não são significativas; decidir na Etapa 6 como reportar (ex.: intervalo de confiança, várias sementes).                                                                                                                                                                         |
| R-07 | Tempo do HDFM                      | Uma "época" = 12 treinos. Se a projeção estourar o orçamento, a saída do enunciado é reduzir épocas ou modos do VMD — decisão da Etapa 6, registrada como desvio.                                                                                                                                                                                                                                                         |
| R-08 | Prazo: entrega em 2026-11-19       | Três dias a partir de 2026-11-16 para as Etapas 0 a 7, o relatório e a apresentação. O estudo da GRU (2b) disputa as mesmas horas de máquina que os três modelos dos artigos e a execução final. Regra: o orçamento da 2b é o que sobra depois de reservar as Etapas 3 a 7; os cortes seguem a ordem da Etapa 2 e vão para o §6.3. Se o HDFM não couber em 2 dias de treino, a ordem de saída é a do enunciado (Etapa 6). |
| R-09 | Escolha da GRU dentro do ruído     | Com ~125 dias de validação e uma semente, a diferença entre combinações pode ser acaso. O relatório da 2b mostra a variação entre sementes quando houver; se não houver (corte 2), a escolha é registrada como frágil em L-01.                                                                                                                                                                                            |

### 6.3. Limitações do estudo

Lista viva: cada etapa acrescenta o que simplificou ou deixou de fazer. É a base da seção de limitações do relatório, junto com os desvios dos artigos (§6.1).

| ID   | Limitação                                    | Situação                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ---- | -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| L-01 | Estudo da GRU simplificado pelo prazo        | **Sem cortes** (2026-11-17): grade completa (108 combinações), 100 épocas, 3 sementes, na GPU; 324 execuções em ~22 min de treino. Números em `resultados-2b.md`. Cuidado na conclusão: as diferenças entre combinações são pequenas frente ao ruído entre sementes.                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| L-02 | A escolha de entrada só alcança a GRU        | `univariado` contra `ohlcv` só vale para a GRU; o HDFM é sempre univariado e o FinGAT usa as features do artigo.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| L-03 | GRU escolhida isolada                        | A melhor GRU sozinha pode não ser a melhor dentro de cada modelo (o FinGAT relata piora acima de 16 unidades). Os modelos dos artigos não são avaliados na configuração de GRU que os autores usaram.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| L-04 | Uma ação, um período                         | Só a PETR4, 5 anos, ~125 dias de teste. O teste (2026-05 a 2026-11) é um trecho de alta forte, com preços acima de tudo o que o treino viu (D-20).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| L-05 | Dois artigos em vez de três                  | O TEGRU saiu do estudo (Etapa 5): sem notícias da PETR4, a contribuição principal dele não seria avaliada. A comparação tem 5 modelos de 2 artigos (variantes do FinGAT e do HDFM), contra 4 de 3 pedidos no enunciado.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| L-06 | Comparação desigual por construção           | Ver R-05: o FinGAT treina com várias ações e só é comparado em direção; o HDFM depende de D-10: na validação rápida, a decomposição da série inteira dá RMSE 0,44 e a causal 0,75 (Etapa 4).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| L-07 | GRU escolhida sem vitória clara              | **A registrar na seção de limitações do relatório.** Decisão do usuário em 2026-11-18: a base é a combinação de menor RMSE médio de validação (`ohlcv`, janela 120, arquitetura `f`; RMSE 0,6878). Ela bate o ingênuo (0,7118) por só 3,4%, e a grade toda varia pouco (0,688 a 0,710, desvio 0,0048) frente ao ruído entre sementes (0,0015 em média; 0,0023 nesta combinação). A acurácia de direção (~0,66) empata com a do ingênuo (0,664). Logo a escolha não vem de vitória clara sobre a base ingênua nem sobre as outras combinações: é a de menor erro numérico. Isso contraria a regra do §2b ("dentro do ruído, a menor"); a menor equivalente seria `ohlcv`, janela 30, `i`. Números em `resultados-2b.md`. |
| L-08 | Código do FinGAT diverge do artigo           | O repositório dos autores não implementa o que o artigo descreve (perda com MAE e ranking por produto, sem GRU de longo prazo, GAT depois de juntar as semanas, pooling por atenção, lr 0,05). Por isso há duas variantes (D-08), e o resultado do "FinGAT" depende de qual se toma. A lista está na Etapa 3 e em `terceiros/fingat_original/ORIGEM.md`.                                                                                                                                                                                                                                                                                                                                                                |
| L-09 | Universo do FinGAT com viés de sobrevivência | As candidatas são ações negociadas hoje; as que foram deslistadas ou trocaram de código no período (ELET3, ELET6, CCRO3, EMBR3, CRFB3, STBP3, ODPV3) não têm histórico no Yahoo Finance e ficaram de fora, e RAIZ4 tem só 83 pregões. A escolha das 6 por setor usa só o trecho de treino.                                                                                                                                                                                                                                                                                                                                                                                                                              |

### 6.4. Tempos medidos (máquina do estudo)

Preenchida no relatório de cada etapa; é a entrada da Etapa 6.

| Modelo                                            | Etapa | Pré-processamento | 1ª época | Demais épocas (média) | Instâncias de treino | Pico de memória             | Projeção 100 épocas |
| ------------------------------------------------- | ----- | ----------------- | -------- | --------------------- | -------------------- | --------------------------- | ------------------- |
| GRU, mais barata (univariado, janela 5, `i`), CPU | 2a    | 0,01 s            | 0,19 s   | 0,045 s               | 798                  | RAM 736 MB                  | 4,6 s               |
| GRU, mais barata, GPU (RTX 3060 Ti)               | 2a    | 0,00 s            | 0,41 s   | 0,045 s               | 798                  | VRAM 25 MB, RAM 1.143 MB    | 4,9 s               |
| GRU, mais cara (OHLCV, janela 200, `a`), CPU      | 2a    | 0,01 s            | 3,94 s   | 3,758 s               | 798                  | RAM 891 MB                  | 376 s (6,3 min)     |
| GRU, mais cara, GPU (RTX 3060 Ti)                 | 2a    | 0,01 s            | 0,35 s   | 0,094 s               | 798                  | VRAM 238 MB, RAM 1.170 MB   | 9,6 s               |
| FinGAT 3a (`fingat_repo`), GPU (RTX 3060 Ti)      | 3     | 0,72 s            | 45,1 s   | 43,8 s                | 848 dias × 48 ações  | VRAM 794 MB, RAM 1.868 MB   | 4.378 s (73 min)    |
| FinGAT 3b (`fingat_artigo`), GPU (RTX 3060 Ti)    | 3     | 0,67 s            | 3,13 s   | 1,96 s                | 848 dias × 48 ações  | VRAM 6.834 MB, RAM 3.737 MB | 197 s (3,3 min)     |
| HDFM (12 GRUs), GPU (RTX 3060 Ti)                 | 4     | causal: 821 s (1.129 decomp., uma vez); inteira: 13 s | 0,51 s   | 0,25 s                | 878                  | VRAM 428 MB, RAM 1.172 MB | 25 s                |
