# Como a validação está implementada

Este documento explica o que o conjunto de validação faz no estudo, onde está o código de cada parte, como se garante que nem a validação nem o teste vazam para o treino, e o resultado da verificação feita em 2026-11-19 sobre as 30 execuções da Etapa 7. As condições do estudo estão em [`condicoes-do-estudo.md`](condicoes-do-estudo.md); os números, em [`resultados-7.md`](resultados-7.md).

**Resumo:** a lógica está correta. Verificado nas 30 execuções reais e nos dados reais: os mesmos 125 dias de validação e de teste em todos os modelos; a época avaliada no teste é sempre a de menor perda de validação; nenhuma estatística, seleção ou amostra de treino muda quando a validação e o teste mudam, exceto no HDFM com decomposição da série inteira, que vaza por construção (D-10) e está no estudo justamente para medir isso. Há pontos de atenção na interpretação (§7), não defeitos.

## 1. Para que serve a validação

| Uso                                                     | Onde                                          |
| ------------------------------------------------------- | --------------------------------------------- |
| Escolher, em cada execução, a época cujos pesos vão para o teste | `comum/treino.py`, laço de treino (§4)  |
| Escolher a GRU compartilhada (entrada, janela, arquitetura) no estudo da Etapa 2 | `grade_gru.py`, `resultados-2b.md` |
| Escolher qual semente aparece nas figuras (a de RMSE de validação mediano) | `figuras.py: semente_mediana`    |

Nada mais é escolhido. O **teste não escolhe nada**: as previsões de teste são gravadas por cada execução, mas só o `resultados.py` as lê, depois de todas as execuções terminarem.

## 2. A divisão

- **Onde:** `comum/janelas.py:23` (`FRACOES = (0.8, 0.1)`) e `comum/janelas.py:40` (`limites`). Com os 1.248 pregões do snapshot, a validação começa na posição 998 e o teste na 1.123.
- **Pela posição do pregão-alvo, não da janela.** Uma amostra pertence à validação se o dia que ela prevê está entre as posições 998 e 1.122. A janela de entrada dessa amostra pode buscar dias do trecho de treino; isso não é vazamento, porque são dias anteriores ao alvo, que um previsor real teria.
- **Os mesmos dias para todos os modelos.** Os três `montar` chamam a mesma função `limites` sobre a série da PETR4:

  | Modelo  | Onde as partes são montadas                 | Treino (alvos)  | Validação | Teste       |
  | ------- | ------------------------------------------- | --------------- | --------- | ----------- |
  | GRU     | `comum/janelas.py:103-105`                  | 120 a 997       | 998 a 1.122 | 1.123 a 1.247 |
  | FinGAT  | `modelos/custom_fingat.py:133-135`          | 150 a 997       | idem      | idem        |
  | HDFM    | `modelos/hdfm.py:184-186`                   | 120 a 997       | idem      | idem        |

  O FinGAT começa o treino em 150 porque as médias de 30 dias consomem os 30 primeiros pregões.
- **Datas:** validação de 2025-11-11 a 2026-05-18, teste de 2026-05-19 a 2026-11-13, 125 dias cada.
- **As janelas só olham para trás:** `alvos[:, None] + np.arange(-janela, 0)` (`comum/janelas.py:84`; `modelos/custom_fingat.py:121`). No HDFM causal, a entrada do alvo `t` vem da decomposição de `[0..t-1]` e o rótulo, da de `[0..t]` (`modelos/hdfm.py:179-180`).

## 3. O que cada modelo aprende só com o treino

Tudo o que é ajustado aos dados usa só o trecho anterior à validação. Assim, a validação mede o modelo como ele seria usado: com o passado apenas.

| Modelo | O que é ajustado                                   | Com quais dados                                  | Onde                                  |
| ------ | -------------------------------------------------- | ------------------------------------------------ | ------------------------------------- |
| GRU    | nada (normalização pela própria janela, D-20/D-21) | —                                                | `comum/janelas.py`                    |
| FinGAT | média e desvio dos níveis de preço, por ação       | pregões 0 a 997                                  | `modelos/custom_fingat.py:81` (`z`)  |
| FinGAT | quais 6 ações de cada setor entram (volume financeiro) | pregões 0 a 997                              | `modelos/custom_fingat.py:54`         |
| HDFM A | mínimo e máximo de cada subsérie                   | entradas e alvos das amostras de treino          | `modelos/hdfm.py:191-192`             |
| HDFM B | desvio do passo de 1 dia de cada subsérie           | amostras de treino                               | `modelos/hdfm.py:195`                 |
| HDFM causal | a decomposição de cada amostra                | só o passado do alvo                             | `modelos/hdfm.py:145, 179-180`        |
| HDFM inteira | a decomposição                               | **a série inteira** (vazamento de propósito, D-10) | `modelos/hdfm.py:140`               |

## 4. Como a validação escolhe a época

Em `comum/treino.py`, a cada época:

1. **Treino** (`:100-111`): o modelo entra em modo de treino (`modelo.train()`, que liga o dropout do FinGAT repo), os lotes vêm só do conjunto de treino, em ordem sorteada por um gerador com a semente (`:102`), e o otimizador dá um passo por lote.
2. **Validação** (`:115`): `prever` (`:43-46`) põe o modelo em modo de avaliação (`modelo.eval()`, sem dropout) e roda sem gradiente (`@torch.no_grad()`). A validação nunca atualiza peso.
3. **Escolha** (`:136-138`): se a perda de validação é a menor até ali, os pesos vão para `melhor.pt`. Empate fica com a época anterior (comparação estrita).
4. **Retomada** (`:85-95`): o `ultimo.pt` guarda a menor perda de validação já vista; uma execução retomada continua a mesma escolha.
5. **Perda não finita** (`:120-125`): a execução para com erro, sem tocar no `melhor.pt` já gravado.

Todas as execuções vão até as 100 épocas (não há parada antecipada, P-04); a época escolhida é gravada como `melhor_epoca` no `metricas.json`.

**A perda que escolhe é a de cada modelo**, a mesma do treino:

| Modelo        | Perda de validação                                                  |
| ------------- | ------------------------------------------------------------------- |
| GRU           | MSE da variação relativa do fechamento                              |
| FinGAT repo   | MAE + BCE + ranking por produto (código dos autores)                |
| Custom FinGAT | (1 − δ)·ranking + δ·BCE, somada sobre os dias (Eq. 15–16, sem o L2) |
| HDFM          | MSE média das 12 subséries normalizadas                             |

## 5. O que vai para o teste, e quando

Em `rodar.py`, depois do treino:

1. Carrega `melhor.pt` (`:197-200`), os pesos da época escolhida pela validação.
2. Prevê validação e teste (`:202`) e grava `previsoes_validacao.csv` e `previsoes_teste.csv`.
3. Grava no `metricas.json` **só as métricas de validação** e as do ingênuo na validação (`:242-263`).

O teste só é lido pelo `resultados.py`, que roda depois da fila e calcula as métricas a partir das previsões gravadas. Como nenhuma escolha dependeu dele, as métricas de teste estimam o desempenho sem viés de seleção.

## 6. Verificação (2026-11-19)

Script: [`projeto/verificar_validacao.py`](projeto/verificar_validacao.py), rodado depois da execução longa. Para repetir: `uv run python redes_neurais_2026_2/projeto/verificar_validacao.py` (sai com código 1 se algo falhar).

| Checagem                                                                                  | Resultado |
| ----------------------------------------------------------------------------------------- | --------- |
| As 30 execuções têm os mesmos dias de validação e de teste                                 | ok        |
| Validação: 2025-11-11 a 2026-05-18; teste: 2026-05-19 a 2026-11-13; 125 dias cada          | ok        |
| Em todas, `melhor_epoca` = época de menor perda de validação no `log.csv` = época do `melhor.pt` | ok   |
| Nenhum `metricas.json` guarda métrica de teste                                             | ok        |
| GRU (univariado e OHLCV): multiplicar por 3 os preços da validação e do teste não muda nenhuma amostra de treino; os alvos de treino vêm antes da validação | ok |
| FinGAT (as duas variantes): o mesmo; nem o universo de 48 ações muda                        | ok        |
| HDFM A e B causais: o mesmo; nem as estatísticas de normalização mudam                      | ok        |
| HDFM A e B com a série inteira: o treino **muda** (o vazamento que a D-10 quer medir)       | ok (esperado) |

Os testes de fumaça cobrem o mesmo princípio com dados sintéticos, mas perturbando só o teste (`testes/test_fumaca.py`, `test_fingat.py`, `test_hdfm.py`). A verificação acima é mais forte: perturba também a validação e usa os dados reais.

## 7. Pontos de atenção (não são defeitos)

- **A validação escolhe por uma perda que não é a métrica do relatório.** Na GRU, a MSE da variação relativa é quase proporcional à MSE do preço. No HDFM, a escolha é pela MSE das 12 subséries normalizadas, que dá peso igual a subséries de escalas diferentes e não é o RMSE do fechamento. Nos FinGAT, é uma perda dominada pelo ranking, não a acurácia de direção. Escolher pela métrica do relatório na validação seria possível, mas não foi o combinado (`condicoes-do-estudo.md` §4).
- **HDFM A escolheu épocas no fim do treino** (93,8 ± 7,5 de 100): a perda de validação ainda caía. Com mais épocas, o checkpoint poderia ser outro; o limite de 100 vale para todos (P-03).
- **A validação e o teste são períodos diferentes.** Na validação, 66,4 % dos dias são de alta; no teste, 53,6 %. A GRU escolhida batia o ingênuo na validação do estudo da Etapa 2 (RMSE 0,688 contra 0,712, −3,4 %) e perde dele no teste da Etapa 7 (1,045 contra 1,031, +1,4 %). Parte disso é essa mudança de regime; parte é o efeito de escolher, entre 108 combinações, a que foi melhor na validação: o número da validação da vencedora tende a ser otimista. O teste não sofre desse viés.
- **No HDFM com a série inteira, a validação também vaza**, então até a escolha da época dele usa o futuro. É coerente com o papel dele no estudo: reproduzir o protocolo do artigo para medir o vazamento.
- **Uma amostra de validação usa dias do treino na janela de entrada.** É o esperado num previsor que só olha o passado; o que importa é que nenhuma amostra de treino use dias da validação, e isso foi verificado.
