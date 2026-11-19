# Resultados da Etapa 7 (teste)

Gerado em 2026-11-19 00:56 por `projeto/resultados.py`, a partir de `execucoes/`. Métricas no **teste** (125 dias), média ± desvio-padrão entre sementes; na direção, entre colchetes, o intervalo de Wilson de 95 % para a acurácia média (Wilson, 1927; Brown, Cai e DasGupta, 2001). Condições: `condicoes-do-estudo.md`.

## 5.1. Comparação principal

| Configuração | Sementes | RMSE (R$) | MAE (R$) | MAPE (%) | R² | Direção [IC 95 % Wilson] |
|---|---|---|---|---|---|---|
| GRU | 5 | 1.0453 ± 0.0157 | 0.7247 ± 0.0162 | 1.589 ± 0.035 | 0.9715 ± 0.0009 | 0.475 ± 0.020 [0.390; 0.562] |
| HDFM A (min-max), causal | 5 | 2.7472 ± 0.2207 | 2.2099 ± 0.2269 | 4.671 ± 0.504 | 0.8021 ± 0.0313 | 0.469 ± 0.007 [0.384; 0.556] |
| HDFM B (diferença), causal | 5 | 1.2980 ± 0.0318 | 1.0252 ± 0.0297 | 2.277 ± 0.067 | 0.9560 ± 0.0021 | 0.454 ± 0.021 [0.370; 0.542] |
| Custom FinGAT (texto do artigo) | 5 | — | — | — | — | 0.536 ± 0.000 [0.449; 0.621] |
| FinGAT (código dos autores) | 5 | — | — | — | — | 0.536 ± 0.000 [0.449; 0.621] |
| Referência ingênua | — | 1.0308 | 0.7150 | 1.559 | 0.9723 | 0.536 [0.449; 0.621] |

## 5.2. Vazamento da decomposição (HDFM B)

| Configuração | Sementes | RMSE (R$) | MAE (R$) | MAPE (%) | R² | Direção [IC 95 % Wilson] |
|---|---|---|---|---|---|---|
| HDFM B (diferença), causal | 5 | 1.2980 ± 0.0318 | 1.0252 ± 0.0297 | 2.277 ± 0.067 | 0.9560 ± 0.0021 | 0.454 ± 0.021 [0.370; 0.542] |
| HDFM B (diferença), inteira | 5 | 0.3852 ± 0.0070 | 0.2498 ± 0.0038 | 0.524 ± 0.007 | 0.9961 ± 0.0001 | 0.930 ± 0.015 [0.871; 0.963] |

## 5.3. FinGAT: código dos autores × texto do artigo

| Configuração | Sementes | PETR4 [IC 95 % Wilson] | 48 ações (prob. ≥ 0,5) | 48 ações (sinal do retorno) |
|---|---|---|---|---|
| Custom FinGAT (texto do artigo) | 5 | 0.536 ± 0.000 [0.449; 0.621] | 0.511 ± 0.000 | 0.502 ± 0.012 |
| FinGAT (código dos autores) | 5 | 0.536 ± 0.000 [0.449; 0.621] | 0.511 ± 0.000 | 0.502 ± 0.012 |

## 5.5. Custo

| Configuração | Sementes | Parâmetros | s por época | Treino total (s) | Época escolhida | VRAM (MB) |
|---|---|---|---|---|---|---|
| GRU | 5 | 51969 ± 0 | 0.035 ± 0.001 | 3.7 ± 0.1 | 48.2 ± 32.0 | 153 ± 0 |
| HDFM A (min-max), causal | 5 | 605196 ± 0 | 0.255 ± 0.024 | 25.7 ± 2.4 | 93.8 ± 7.5 | 429 ± 1 |
| HDFM B (diferença), causal | 5 | 605196 ± 0 | 0.254 ± 0.027 | 25.6 ± 2.6 | 41.4 ± 3.9 | 429 ± 1 |
| HDFM B (diferença), inteira | 5 | 605196 ± 0 | 0.252 ± 0.028 | 25.4 ± 2.8 | 45.8 ± 11.2 | 429 ± 1 |
| Custom FinGAT (texto do artigo) | 5 | 369666 ± 0 | 2.117 ± 0.011 | 212.6 ± 1.1 | 80.2 ± 21.6 | 6835 ± 1 |
| FinGAT (código dos autores) | 5 | 1457620 ± 0 | 44.291 ± 3.746 | 4428.3 ± 371.7 | 77.4 ± 17.4 | 795 ± 3 |

## Execuções que faltam

nenhuma
