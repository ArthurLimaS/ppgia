# Análise Exploratória Completa: pytrendseries

Este documento apresenta uma análise aprofundada da biblioteca `pytrendseries` (v0.1.12), baseada estritamente na inspeção do código-fonte e documentação do repositório. O objetivo é avaliar sua adequação para pesquisas de detecção de tendências em séries temporais financeiras.

---

## 1. Objetivo da biblioteca

A biblioteca `pytrendseries` tem como objetivo principal a **detecção de tendências** (tanto de alta quanto de baixa) em séries temporais, com foco em dados financeiros como preços de ações.

- **Problema que resolve:** Identificar automaticamente os intervalos de tempo onde um ativo está em uma tendência definida, calculando a magnitude (drawdown/drawup) e a duração desses movimentos. Isso é crucial para gestão de risco e para a criação de variáveis-alvo em modelos de Machine Learning.
- **Entrada esperada:** Um `pandas.DataFrame` contendo uma única coluna numérica (preços ou observações) e um `DatetimeIndex`.
- **Saída produzida:** 
  - DataFrames com intervalos das tendências detectadas (datas de início/fim, preços extremos, duração e percentual de movimento).
  - Métricas de risco como Máximo Drawdown, Drawdown Atual e Tempo Submerso (Time Under Water).
  - Rótulos (labels) para classificação supervisionada (1 para alta, -1 para baixa, 0 para neutro).
- **Principais casos de uso:** Análise de risco de portfólio, geração de dados para treinamento de redes neurais ou modelos de classificação, e visualização de ciclos de mercado.

---

## 2. Arquitetura do projeto

O repositório é organizado de forma modular dentro de `src/pytrendseries/`:

| Arquivo | Responsabilidade Principal |
| :--- | :--- |
| `__init__.py` | Exporta as funções públicas e define a versão. |
| `detecttrend.py` | Módulo central. Contém `detecttrend`, `get_trends_labels` e o scanner otimizado `_scan` (Numba). |
| `maximum_drawdown.py` | Implementação do cálculo de Máximo Drawdown (global e por janelas). |
| `current_drawdown.py` | Calcula o status de drawdown no momento atual (relativo ao pico mais recente). |
| `time_under_water.py` | Calcula o tempo necessário para recuperação de perdas (TUW). |
| `vizplot.py` | Funções de visualização gráfica usando `matplotlib`. |

**Dependências Relevantes:**
- **`numba`**: Usada para compilação JIT do algoritmo de varredura (`_scan`), garantindo alta performance em loops complexos.
- **`pandas` e `numpy`**: Base para manipulação e cálculo vetorial de dados.
- **`matplotlib`**: Para geração dos gráficos de tendências e drawdowns.

---

## 3. Fluxo de execução

Ao chamar a função principal `detecttrend()`, o seguinte fluxo ocorre:

1.  **Validação e Preparação (`_treat_parameters`):** Verifica se o DataFrame tem uma coluna, se os parâmetros são inteiros e se o índice é datetime.
2.  **Conversão para NumPy:** Os preços são extraídos como arrays `float64` contíguos para otimizar o acesso内存.
3.  **Varredura Sequencial (`_scan` via Numba):**
    - O array é percorrido ponto a ponto.
    - Para cada início de movimento (ex: preço caiu em relação ao anterior), o algoritmo olha à frente até o limite do `window`.
    - Identifica o Pico (máximo local) e o Vale (mínimo local) dentro desse segmento.
    - Valida se a duração é `>= limit`.
4.  **Cálculo de Métricas:** Calcula o percentual de drawdown/drawup e a distância temporal entre os extremos.
5.  **Montagem do DataFrame:** Organiza os resultados em um formato tabular ordenado cronologicamente.

---

## 4. Métricas produzidas

1.  **Peak / Valley:** O preço mais alto e mais baixo encontrados no segmento da tendência.
2.  **Peak Date / Valley Date:** As datas correspondentes a esses preços.
3.  **drawdown / drawup:** A variação percentual entre o pico e o vale. 
    - *Interpretação:* Mede a intensidade da perda (drawdown) ou ganho (drawup).
4.  **time_span:** O número de períodos (dias, horas, etc.) entre o pico e o vale.
    - *Interpretação:* Mede a velocidade do movimento.
5.  **MaxDD:** O maior drawdown registrado dentro de uma janela específica ou na série inteira.
6.  **Time Underwater (TUW):** O número de períodos que o preço permaneceu abaixo do pico mais recente.
7.  **Status:** Indica se um drawdown foi "Recovered" (recuperado) ou está "Ongoing" (em andamento).
8.  **label:** Rótulo numérico ou string para uso em ML (ex: `1` para uptrend, `-1` para downtrend).

---

## 5. Algoritmo de detecção de tendências (Detalhado)

O cerne da biblioteca é a função `_scan` em `detecttrend.py`. Trata-se de uma varredura sequencial com look-ahead.

### Pseudocódigo do `_scan`:
```text
INÍCIO
  i = 0
  ENQUANTO i < n - 1:
    price_curr = prices[i]
    price_next = prices[i+1]
    
    // Verifica se inicia uma tendência
    SE (is_down E price_next < price_curr) OU (NÃO is_down E price_next > price_curr):
      ENCONTRAR fim_da_janela = min(i + window, n)
      brk = -1
      
      // Procura por um "rompimento" (reversão) dentro da janela
      PARA j DE i ATÉ fim_da_janela:
        SE (is_down E prices[j] > price_curr) OU (NÃO is_down E prices[j] < price_curr):
          brk = j
          QUEBRA
      
      // Define o segmento a ser analisado
      SEG_INICIO = i
      SEG_FIM = (brk SE brk != -1 SENÃO fim_da_janela)
      
      // Encontra o extremo (Pico para baixa, Vale para alta)
      EXTREMO = prices[SEG_INICIO]
      PARA j DE SEG_INICIO ATÉ SEG_FIM:
        ATUALIZAR EXTREMO (máximo se is_down, mínimo se NÃO is_down)
      
      // Encontra a posição do extremo (primeira ocorrência)
      POS_EXTREMO = posição de EXTREMO em prices[SEG_INICIO:SEG_FIM]
      
      // Encontra o extremo oposto (Vale para baixa, Pico para alta)
      SE brk != -1:
        OPOSTO_INICIO = POS_EXTREMO + 1
        OPOSTO_FIM = SEG_FIM
        OPOSTO = preço em OPOSTO_INICIO
        // Encontra OPOSTO (mínimo se is_down, máximo se NÃO is_down)
      
      // Validação de duração
      SE (POS_OPOSTO - POS_EXTREMO) >= limit:
        REGISTRA tendência (Peak, Valley, Datas, Preços)
        i = POS_OPOSTO - 1 // Pula para o fim da tendência encontrada
    
    i = i + 1
FIM
```

**Detalhes de Implementação (Numba `_scan`):**
- **Tratamento de Empates:** No código, `date_max` e `date_min` são calculados selecionando a **última** ocorrência de preço extremo no segmento, enquanto os índices `loc_max` e `loc_min` usam a **primeira** ocorrência. Isso afeta como as datas são registradas no DataFrame final.
- **Lógica de Inversão:** O algoritmo define `pmax` como o extremo de início (mínimo para alta, máximo para baixa) e `pmin` como o extremo oposto. Os nomes das variáveis (`pk`, `vl`) no retorno são atribuídos de acordo com a direção (`is_down`).
- **Janela Parcial:** Se o preço não reverte antes do limite do `window` mas atinge o `limit` de duração, a tendência é registrada usando os extremos encontrados dentro do intervalo disponível.
- **Otimização:** O índice `i` avança para `loc_min - 1` ao final de uma tendência detectada, garantindo que o próximo passo comece logo após o fim da tendência atual, evitando sobreposição.

---

## 6. Fórmulas matemáticas

1.  **Drawdown (Downtrend):**
    $$ Drawdown = \frac{|Peak - Valley|}{Peak} $$
    *(Implementado em `detecttrend.py` linha 155)*

2.  **Drawup (Uptrend):**
    $$ Drawup = \frac{|Peak - Valley|}{Valley} $$
    *(Implementado em `detecttrend.py` linha 163)*

3.  **Drawdown Atual (Current Drawdown):**
    $$ Current\_DD = \left( \frac{Current\_Price}{Last\_Peak} - 1 \right) \times 100 $$
    *(Implementado em `current_drawdown.py` linha 49)*

4.  **Máximo Drawdown (Vetorial):**
    Calculado usando o pico acumulado (`running_peak`):
    $$ DD_t = \frac{Price_t}{RunningPeak_t} - 1 $$
    O MaxDD é o valor mínimo de $DD_t$ (maior magnitude negativa).

---

## 7. Parâmetros configuráveis

- **`trend`** (string, padrão 'downtrend'):
  - *Impacto:* Define se o algoritmo procurará por movimentos de queda ('downtrend') ou subida ('uptrend').
- **`window`** (int, padrão 21):
  - *Impacto:* Define o horizonte de busca. Um valor maior captura tendências de longo prazo, mas pode ignorar reversões rápidas. Um valor menor é mais sensível a ruídos.
- **`limit`** (int, padrão 5):
  - *Impacto:* O filtro de ruído. Define o número mínimo de barras que a tendência deve durar para ser considerada válida. Aumentar este valor filtra flutuações microscópicas.

---

## 8. Dependências entre funções

```text
pytrendseries (público)
├── detecttrend()
│   ├── _treat_parameters() [Validação]
│   ├── _scan() [Numba JIT - Lógica principal]
│   └── pd.DataFrame() [Formatando saída]
├── get_trends_labels()
│   ├── detecttrend() [Uptrend]
│   └── detecttrend() [Downtrend]
├── maxdrawdown()
│   ├── _calculate_max_drawdown_vectorized()
│   │   └── np.maximum.accumulate()
│   └── pd.concat()
├── calculate_time_under_water()
│   ├── np.maximum.accumulate()
│   └── np.diff() [Identificando períodos]
└── plot_trend()
    └── plt.axvspan() [Destacando intervalos]
```

---

## 9. Complexidade computacional

- **`_scan` (Detecttrend):** A complexidade é aproximadamente $O(N \cdot W)$ no pior caso, onde $N$ é o número de observações e $W$ o tamanho da janela. No entanto, como o índice `i` avança significativamente a cada tendência encontrada, na prática é muito eficiente, especialmente com a JIT-compilation do Numba.
- **`maxdrawdown` / `tuw`:** São $O(N)$ pois utilizam operações vetORIZADAS do NumPy que percorrem o array uma única vez.
- **Complexidade Espacial:** $O(N)$ para armazenar os arrays de preços e resultados intermediários.

---

## 10. Limitações

1.  **Univariado:** A biblioteca espera uma única coluna de preços. Não suporta séries multivariadas (ex: OHLC) diretamente sem pré-processamento.
2.  **Janela Fixa:** Tendências que duram mais do que o `window` definido podem ser quebradas ou ignoradas, dependendo da lógica de reversão.
3.  **Ausência de Sazonalidade:** O algoritmo não leva em conta padrões sazonais ou cíclicos intrínsecos; ele reage puramente à variação de preço.
4.  **Dependência de Numba:** Para rodar, requer a instalação correta do Numba, que pode ter problemas de compatibilidade em alguns ambientes (ex: Apple Silicon sem configuração específica).

---

## 11. Relação com literatura

Embora o código não mencione artigos específicos (como visto na busca), as métricas implementadas são padrão na indústria financeira:
- **Drawdown:** Métrica clássica de risco de investimento (usada por hedge funds e gestoras).
- **Time Under Water (TUW):** Conceito presente na análise de risco de performance de portfólios (frequente em estudos de "Risk of Ruin").
- **Labeling (1, -1, 0):** Abordagem comum em finanças quantitativas para transformar preços em sinais de classificação supervisionada.

---

## 12. Resumo Executivo

A biblioteca `pytrendseries` é uma ferramenta **eficiente e bem estruturada** para a detecção de tendências em séries temporais financeiras. Ela se destaca pelo uso de **Numba** para garantir performance em algoritmos de varredura que seriam lentos em Python puro.

**O que ela faz:**
- Identifica automaticamente movimentos de alta e baixa em dados de preços.
- Calcula métricas de risco fundamentais como Drawdown e Time Underwater.

**Como detecta tendências:**
- Utiliza uma abordagem de **sliding window** (janela deslizante) com validação de duração mínima (`limit`). O algoritmo busca por picos e vales locais dentro de um horizonte temporal definido (`window`).

**O que o pesquisador deve saber:**
- A detecção é **sequencial e não sobreposta**.
- É ideal para criar **variáveis-alvo (labels)** para modelos de machine learning.
- Para uso científico, o pesquisador deve estar ciente de que a sensibilidade do algoritmo depende fortemente da escolha dos parâmetros `window` e `limit`, que devem ser calibrados de acordo com a frequência dos dados (diário, intraday, etc.).
