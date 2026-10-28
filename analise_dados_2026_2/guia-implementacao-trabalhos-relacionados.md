# Guia de implementação — Seção "Trabalhos Relacionados"

Guia específico desta entrega (artigo da disciplina de Análise de Dados, 2026.2).
Criado em 2026-10-27. Pode ser substituído no futuro. Todos os caminhos são relativos à raiz do repositório (`ppgia/`).

**Como chamar em uma nova conversa:**

> Leia `analise_dados_2026_2/guia-implementacao-trabalhos-relacionados.md` e comece a implementação da etapa E01.

Esse comando autoriza **ler e analisar** a etapa. **Não** autoriza redigir nem editar o manuscrito. Ver §5.1.

**Roteiro de leitura para o agente:** §1 (visão rápida) → §4 (decisões) → §5 (regras) → §6 (procedimento padrão) → ficha da etapa em §7. Antes de agir, confira o estado real dos arquivos (§6, passo A1). Não confie apenas no quadro de etapas.

---

## 1. Visão rápida

### 1.1 Objetivo

Construir a seção 3, "Trabalhos Relacionados", de `analise_dados_2026_2/latex/main.tex`. Cada um dos 20 artigos aprovados recebe exatamente um parágrafo, distribuído em subseções que seguem a introdução do Plano de Dissertação (`metodologia_2026_1/main.tex`).

### 1.2 Estado atual verificado (2026-10-27)

- A seção 3 do `main.tex` ainda contém apenas os três parágrafos de instrução do template. Nenhuma das 20 chaves está citada no `main.tex`.
- O esqueleto de subseções ainda não foi criado. Isso é feito na E00.
- `artigos/` contém apenas `.gitignore`, rastreado no commit `990cbee`. **Nenhum dos 20 arquivos `.md` existe ainda.**
- As 20 chaves existem em `analise_dados_2026_2/latex/referencias.bib`. O arquivo é idêntico, byte a byte, a `metodologia_2026_1/referencias.bib`.
- Os arquivos de `metodologia_2026_1/` estão rastreados (commit `85107ee`).
- O preâmbulo usa `biblatex` com `biber`, no estilo `authoryear`, com sobrenomes em maiúsculas e vírgula antes do ano.
- A compilação do `main.tex` atual **não** foi verificada nesta data. Compilar é tarefa do usuário: o agente nunca compila (D15).

### 1.3 Quadro das etapas

Os estados possíveis estão definidos em §5.7. O campo "Pré-requisitos pendentes" lista o que falta para a etapa poder ser iniciada.

| ID  | Destino | Chave                             | Estado       | Pré-requisitos pendentes |
| --- | ------- | --------------------------------- | ------------ | ------------------------ |
| E00 | —       | Preparação, inventário e proteção | Não iniciada | —                        |
| E01 | 3.1     | `mahjouby_predicting_2024`        | Não iniciada | E00; `.md`               |
| E02 | 3.1     | `sharma_stock_2023`               | Não iniciada | E00; `.md`               |
| E03 | 3.2     | `k_leveraging_2023`               | Não iniciada | E00; `.md`; PB-1         |
| E04 | 3.2     | `miot_empirical_2020`             | Não iniciada | E00; `.md`               |
| E05 | 3.3     | `kiruba_risk_2025`                | Não iniciada | E00; `.md`               |
| E06 | 3.3     | `ramkumar_development_2023`       | Não iniciada | E00; `.md`               |
| E07 | 3.3     | `liu_application_2024`            | Não iniciada | E00; `.md`               |
| E08 | 3.4.1   | `sun_novel_2024`                  | Não iniciada | E00; `.md`               |
| E09 | 3.4.1   | `li_accurate_2024`                | Não iniciada | E00; `.md`               |
| E10 | 3.4.1   | `hosseini_pattern-based_2025`     | Não iniciada | E00; `.md`               |
| E11 | 3.4.1   | `wan_tcdformer_2024`              | Não iniciada | E00; `.md`               |
| E12 | 3.4.2   | `biswas_dual_2025`                | Não iniciada | E00; `.md`               |
| E13 | 3.4.2   | `luo_short-term_2024`             | Não iniciada | E00; `.md`               |
| E14 | 3.4.2   | `hsu_fingat_2023`                 | Não iniciada | E00; `.md`; PB-2         |
| E15 | 3.4.2   | `anghinoni_time_2019`             | Não iniciada | E00; `.md`               |
| E16 | 3.4.3   | `choi_hybrid_2023`                | Não iniciada | E00; `.md`               |
| E17 | 3.4.3   | `wang_stock_2023`                 | Não iniciada | E00; `.md`               |
| E18 | 3.4.3   | `haryono_transformer-gated_2023`  | Não iniciada | E00; `.md`               |
| E19 | 3.4.3   | `anbaee_farimani_adaptive_2024`   | Não iniciada | E00; `.md`               |
| E20 | 3.5     | `hu_fintsb_2025`                  | Não iniciada | E00; `.md`               |
| E21 | Seção 3 | Revisão integrada                 | Não iniciada | E01–E20 concluídas       |

Na coluna de pré-requisitos, "`.md`" significa que o arquivo `artigos/<chave>.md` ainda não existe. Os artigos são adicionados aos poucos: cada etapa depende apenas do **próprio** `.md`, não da presença dos 20 (D16).

### 1.4 Pendências e bloqueios

| ID   | Descrição                                                                                                                                                                                                                                                               | Afeta                                  | Quem resolve                    |
| ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- | ------------------------------- |
| PF-1 | Nenhum dos 20 arquivos `artigos/<chave>.md` existe ainda. Eles serão adicionados aos poucos, e cada etapa só exige o próprio arquivo (D16).                                                                                                                             | Cada etapa de E01–E20, individualmente | Usuário                         |
| PB-1 | Em `k_leveraging_2023`, o primeiro autor está registrado como `K, Thinakaran`, e a citação sai `(K et al., 2023)`. "K" provavelmente é uma inicial, então a identidade do sobrenome é duvidosa. O coautor `V, Dilli Ganesh` tem o mesmo padrão, sem bloquear a citação. | E03 (bloqueia)                         | Usuário, no `.bib`              |
| PB-2 | O título de `hsu_fingat_2023` contém `Top-{KK}`, que sai "Top-KK" na lista de referências. O CSV registra "Top-K".                                                                                                                                                      | E14 (bloqueia)                         | Usuário, no `.bib`              |
| PB-3 | O coautor `S, Raghavi`, de `ramkumar_development_2023`, tem o mesmo padrão de PB-1. Afeta só a lista de referências.                                                                                                                                                    | E06 (informar, não bloqueia)           | Usuário, se quiser              |
| PE-1 | O escopo de comparação é provisório (D07). Se a Introdução ou os Objetivos do `main.tex` forem preenchidos, confira se há divergência.                                                                                                                                  | E01–E21                                | Agente verifica; usuário decide |

### 1.5 Próximo passo

Executar a **E00**: inventário e verificação de proteção, sem edição, e depois o esqueleto da seção, que exige autorização específica.

---

## 2. Mapa das fontes

| Caminho                                                     | Papel                                                                                                 | Permissão                                                              |
| ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `analise_dados_2026_2/latex/main.tex`                       | Manuscrito. A seção 3 tem `\label{sec:trabalhos-relacionados}` e termina antes de `\section{Método}`. | Editar **somente** a seção 3, com autorização (§5.2).                  |
| `analise_dados_2026_2/latex/referencias.bib`                | Bibliografia efetivamente usada pelo `main.tex` (`\addbibresource`).                                  | Somente leitura.                                                       |
| `metodologia_2026_1/main.tex`                               | Plano de Dissertação. A introdução é a base de organização.                                           | Somente leitura.                                                       |
| `metodologia_2026_1/referencias.bib`                        | Bibliografia do plano. Não vale como prova de disponibilidade no artigo.                              | Somente leitura.                                                       |
| `metodologia_2026_1/refinamento_trabalhos_relacionados.csv` | Planilha de apoio à análise.                                                                          | Somente leitura.                                                       |
| `artigos/<chave>.md`                                        | Artigos-fonte. São eles que sustentam as afirmações sobre cada publicação.                            | Somente leitura. Nunca reproduzir integralmente em arquivo versionado. |
| `.claude/skills/redacao-trabalhos-relacionados/SKILL.md`    | Skill de redação. É o fluxo base de cada etapa de artigo.                                             | Somente leitura (não alterar).                                         |
| `analise_dados_2026_2/template.md`                          | Template original da disciplina, rastreado. Guarda o texto removido na E00.                           | Somente leitura.                                                       |

### 2.1 Plano de Dissertação: parágrafos usados

Referência em `metodologia_2026_1/main.tex`. As linhas foram verificadas em 2026-10-27. Se mudarem, localize o parágrafo pela frase de abertura.

| Linha      | Abertura                                                                | Papel                                                                                                   |
| ---------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| 76, 78, 80 | "A previsão do comportamento…", "Nesse contexto…", "Além de seu papel…" | Contexto de motivação. Citam `li`, `ramkumar`, `sun` e `hosseini`. Não geram subseção nem resumo extra. |
| 82         | "A complexidade inerente…"                                              | Base da 3.1                                                                                             |
| 84         | "Para superar essas barreiras…"                                         | Base da 3.2                                                                                             |
| 86         | "Trabalhos recentes \cite{kiruba…"                                      | Base da 3.3 (cita também `choi`, ver D04)                                                               |
| 88         | "Atualmente, o aprimoramento…"                                          | Base da 3.4 e das subsubseções 3.4.1 a 3.4.3                                                            |
| 90         | "Por fim, é fundamental…"                                               | Base da 3.5                                                                                             |
| 92         | "A partir do mapeamento…"                                               | Só contexto. Não gera subseção (D05).                                                                   |
| 120–128    | Seção "Metodologia"                                                     | Contexto opcional. Usar em comparações só se confirmado (D07).                                          |

### 2.2 Planilha CSV

- Leitura: delimitador `,`, UTF-8 sem BOM, quebras de linha CRLF. Há campos com quebras de linha internas, então use um leitor de CSV real (em Python, `csv.reader(open(p, encoding='utf-8', newline=''), delimiter=',')`). Não conte linhas físicas.
- Colunas: `Título`, `Problema de Pesquisa`, `Objetivos`, `Contribuições`, `Trabalhos Futuros`. **Não** há colunas de método, resultados ou limitações: essas informações vêm só do artigo.
- Não há coluna de chave. A associação abaixo foi feita pelo título. O número é o registro de dados (1 = primeiro registro após o cabeçalho).
- Os registros 3 (`palaniappan_review_2024`), 11 (`singh_harnessing_2023`) e 14 (`cheekati_machine_2025`) estão **fora do escopo**.

| Chave                            | Registro | Observação verificada                                            |
| -------------------------------- | -------- | ---------------------------------------------------------------- |
| `mahjouby_predicting_2024`       | 16       | —                                                                |
| `sharma_stock_2023`              | 19       | Trabalhos Futuros: "Não fala sobre" (confirmar no artigo)        |
| `k_leveraging_2023`              | 13       | —                                                                |
| `miot_empirical_2020`            | 21       | Problema registrado como citação literal em inglês (parafrasear) |
| `kiruba_risk_2025`               | 17       | Trabalhos Futuros: "Não fala sobre" (confirmar no artigo)        |
| `ramkumar_development_2023`      | 8        | —                                                                |
| `liu_application_2024`           | 7        | —                                                                |
| `sun_novel_2024`                 | 2        | —                                                                |
| `li_accurate_2024`               | 5        | —                                                                |
| `hosseini_pattern-based_2025`    | 15       | —                                                                |
| `wan_tcdformer_2024`             | 22       | Objetivos vazio                                                  |
| `biswas_dual_2025`               | 1        | —                                                                |
| `luo_short-term_2024`            | 18       | —                                                                |
| `hsu_fingat_2023`                | 9        | —                                                                |
| `anghinoni_time_2019`            | 23       | Problema de Pesquisa vazio                                       |
| `choi_hybrid_2023`               | 12       | —                                                                |
| `wang_stock_2023`                | 4        | —                                                                |
| `haryono_transformer-gated_2023` | 20       | —                                                                |
| `anbaee_farimani_adaptive_2024`  | 6        | —                                                                |
| `hu_fintsb_2025`                 | 10       | —                                                                |

### 2.3 Artigos-fonte

O arquivo de cada artigo é `artigos/<chave>.md`, com o nome exatamente igual à chave do `.bib`. Exemplo: `artigos/hosseini_pattern-based_2025.md`. Se o arquivo não existir, a etapa correspondente não pode ser iniciada. Não procure nem invente outros nomes.

---

## 3. Estrutura final aprovada

O parágrafo de cada artigo vai na subseção de destino. Dentro de cada subseção, os parágrafos seguem a ordem dos IDs.

| Nº    | Comando e título                                                               | Label                     | Etapas                        |
| ----- | ------------------------------------------------------------------------------ | ------------------------- | ----------------------------- |
| 3.1   | `\subsection{Abordagens Estatísticas e de Aprendizado de Máquina Clássico}`    | `sec:tr-classicos`        | E01, E02                      |
| 3.2   | `\subsection{Redes Neurais Recorrentes e LSTM}`                                | `sec:tr-recorrentes`      | E03, E04                      |
| 3.3   | `\subsection{Modelos Híbridos}`                                                | `sec:tr-hibridos`         | E05, E06, E07                 |
| 3.4   | `\subsection{Estratégias Recentes de Aprimoramento}`                           | `sec:tr-aprimoramento`    | — (só contém as subsubseções) |
| 3.4.1 | `\subsubsection{Pré-processamento e Extração de Características}`              | `sec:tr-preprocessamento` | E08–E11                       |
| 3.4.2 | `\subsubsection{Mecanismos de Atenção e Arquiteturas Temporais e Estruturais}` | `sec:tr-atencao`          | E12–E15                       |
| 3.4.3 | `\subsubsection{Integração de Dados Multimodais}`                              | `sec:tr-multimodal`       | E16–E19                       |
| 3.5   | `\subsection{Padronização e Avaliação Metodológica}`                           | `sec:tr-avaliacao`        | E20                           |

A seção não terá parágrafo de abertura, transições nem síntese final, salvo aprovação futura (D06). A 3.4 fica sem texto próprio antes da 3.4.1.

---

## 4. Decisões aprovadas

Todas foram aprovadas pelo usuário em 2026-10-27. Não reabra nenhuma delas sem motivo concreto (§5.6).

| ID  | Decisão                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| D01 | O escopo são as **20 referências distintas** citadas na introdução do plano. Não buscar, acrescentar nem substituir artigos. Remover um artigo só com aprovação explícita, registrada aqui e refletida no quadro.                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| D02 | Não se aplica a exigência do template de "dois trabalhos por integrante", porque o autor é mestrando.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| D03 | A estrutura, os títulos, os labels e a ordem das etapas são os de §3 e §1.3.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| D04 | `choi_hybrid_2023` tem **um único** resumo, na 3.4.3 (E16). A dimensão híbrida pode ser descrita nesse resumo. A 3.3 não recebe parágrafo sobre `choi`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| D05 | O parágrafo "A partir do mapeamento…" do plano é só contexto e não gera subseção.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| D06 | Não acrescentar parágrafos de abertura, transições ou síntese final sem aprovação.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| D07 | **Escopo de comparação (provisório).** O artigo da disciplina é uma análise de dados sobre o projeto de dissertação, com pergunta de pesquisa provavelmente de **regressão com LSTM/GRU** sobre séries temporais financeiras ("95% de certeza", segundo o usuário). As comparações usam só esse núcleo. Outros elementos do plano (Yahoo Finance, FinTSB, métricas, decomposição, atenção, multimodalidade como proposta própria) só entram se estiverem confirmados no `main.tex` ou se o usuário os confirmar na etapa. Se o `main.tex` vier a divergir, pare e pergunte. Não use automaticamente o problema da dissertação como problema do artigo. |
| D08 | As exigências do template de graduação (clusterização ou classificação como perguntas de pesquisa, app Flutter, KDD) não orientam esta seção. O template serve só como guia (§5.3).                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| D09 | **O preâmbulo não muda.** Citações: `\parencite{chave}` gera `(HSU, TSAI e LI, 2023)` e `(ANGHINONI et al., 2019)`. `\textcite{chave}` gera `HSU, TSAI e LI (2023)`. Isso substitui o exemplo com ponto e vírgula do pedido original. Ajustes de formato ficam para depois, se o usuário pedir.                                                                                                                                                                                                                                                                                                                                                        |
| D10 | `\subsubsection` fica com a formatação padrão do `article`, sem ajuste no preâmbulo.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| D11 | `artigos/` é protegida pelo `artigos/.gitignore` existente (`*` e `!.gitignore`). **Não** criar `.gitkeep`, porque não há diferença de efeito. Nenhuma etapa escreve em `artigos/`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| D12 | A E00 remove os três parágrafos de instrução do template na seção 3 e cria o esqueleto de §3. O texto original continua preservado em `analise_dados_2026_2/template.md`. A E00 exige autorização específica no momento da execução.                                                                                                                                                                                                                                                                                                                                                                                                                   |
| D13 | As pendências bibliográficas PB-1 e PB-2 são resolvidas na primeira etapa que usa a referência (E03 e E14). A etapa para até o usuário corrigir o `.bib`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| D14 | Termos técnicos em inglês vão em `\textit{}` no parágrafo, como no plano (ex.: `\textit{Machine Learning}`). Resolve a antiga PE-2.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| D15 | **O agente nunca compila.** Não executar `latexmk`, `pdflatex`, `biber` nem qualquer outra ferramenta de build do LaTeX, nem mesmo em cópia temporária. O usuário compila manualmente e reporta o resultado.                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| D16 | Os artigos-fonte são adicionados aos poucos. Uma etapa de artigo pode começar assim que o **próprio** `artigos/<chave>.md` estiver na pasta, sem esperar os demais. A E00 não depende de nenhum artigo.                                                                                                                                                                                                                                                                                                                                                                                                                                                |

---

## 5. Regras permanentes

### 5.1 Autonomia e autorização

1. Separe sempre o que é **fato verificado**, **decisão aprovada** (§4), **sugestão** e **pendência**.
2. Sugestões, silêncio, respostas a perguntas e inferências **não** são aprovação. A ausência de dúvidas também não autoriza escrever.
3. "Comece a implementação da etapa X" autoriza **apenas** a leitura, a análise e o relatório prévio (§6, passos A e B).
4. Redigir e editar o manuscrito exige confirmação expressa, como "Pode inserir no .tex" ou "Autorizado", dada **depois** do relatório da etapa.
5. Se surgir uma decisão essencial durante a execução, pare e consulte o usuário.
6. Se perceber que uma ação extrapolou o autorizado, informe exatamente o que aconteceu e interrompa novas alterações. Comunicar a ação não a torna permitida.

### 5.2 Proteção de arquivos e Git

- **Somente leitura:** todo `metodologia_2026_1/`, todo `artigos/` (inclusive `.gitignore`), qualquer `.bib`, a skill e `template.md`. É proibido editar, renomear, mover, excluir ou sobrescrever esses arquivos, inclusive para "facilitar a leitura".
- **Editável com autorização:** apenas a seção 3 de `analise_dados_2026_2/latex/main.tex`, no ponto autorizado, e este guia (§5.6). Não altere o preâmbulo, outras seções, o `README.md` nem configurações.
- **Git:** não fazer commit, não usar `git add -f`, não reescrever histórico, não alterar o rastreamento e não apagar arquivos locais.
- **Compilação:** proibida para o agente (D15). Se o usuário reportar erros ou avisos de compilação, trate na etapa apenas os ligados à seção 3.
- Não reproduzir integralmente o conteúdo dos artigos-fonte neste guia nem em outro arquivo versionado. Registre só a localização das evidências.
- Não buscar publicações nem usar serviços externos para obter ou processar fontes.

### 5.3 Bibliografia, citações e template

- Cite **somente** com `\parencite{chave}` ou `\textcite{chave}` (`\cite` também funciona). `\citep` e `\citet` não existem, porque o `natbib` não está carregado. Nunca escreva citação em texto puro.
- A chave precisa existir em `analise_dados_2026_2/latex/referencias.bib`. Estar no `.bib` da metodologia não basta.
- Se faltar uma entrada, se a identidade bibliográfica for duvidosa ou se for preciso corrigir dados, **interrompa a etapa** e peça ao usuário que cadastre ou corrija. Nunca edite o `.bib`.
- Não altere preâmbulo, pacotes ou estilo para resolver incompatibilidades. Reporte e aguarde.
- O template da disciplina é só um guia. Contradições ou instruções vagas relevantes devem ser reportadas, não seguidas cegamente. Por exemplo: a introdução do template pede citação com iniciais, `(SOBRENOME, A., 2000)`, enquanto a seção 3 pede só sobrenome e ano. Vale D09.

### 5.4 Requisitos de cada resumo

Cada artigo recebe **exatamente um parágrafo**, salvo pedido explícito posterior de expansão. O parágrafo é contínuo e integra, na medida das evidências:

1. **Problema de pesquisa e objetivos.**
2. **Método ou abordagem.**
3. **Contribuições:** o que o artigo entregou (modelo, método, base de dados, análise).
4. **Resultados:** o desempenho do que foi entregue (métricas, comparações). Não misture com contribuições.
5. **Limitações:** problemas reconhecidos pelos autores no próprio trabalho ("sabemos que há este problema").
6. **Trabalhos futuros:** próximos passos que os autores sugerem para a área. Superar uma limitação pode aparecer como trabalho futuro, mas a limitação em si é o problema reconhecido.
7. **Semelhanças e diferenças** em relação ao trabalho atual, dentro do escopo de D07.
8. **Citação** com a chave correspondente (§5.3).

Regras de evidência:

- Afirmações sobre a publicação vêm do `artigos/<chave>.md`. O CSV e o plano servem de apoio. Se divergirem do artigo, reporte a divergência e pergunte.
- Registre no relatório onde está cada evidência (seção, título ou página do `.md`).
- Quando faltar uma informação, diga qual dos dois casos se aplica:
  - **"não encontrado no arquivo fornecido"**: a extração pode estar incompleta (seções faltando, texto truncado, tabelas ou figuras perdidas);
  - **"ausente no artigo"**: o texto está completo e não trata do item.

  - Na dúvida, use o primeiro e pergunte. Nunca conclua que o artigo é fraco só porque a extração não traz uma informação.

- Não invente conteúdo. Use paráfrase acadêmica em português. Não copie trechos do plano, do CSV ou dos artigos como se fossem redação nova. O CSV contém citações literais, como no registro 21.
- Termos técnicos em inglês vão em `\textit{}` (D14).

### 5.5 Inserção no manuscrito

- Coloque o parágrafo na subseção de destino da etapa, em ordem de ID. Ele fica depois dos parágrafos de IDs menores do mesmo bloco, antes dos de IDs maiores e antes do próximo título.
- Cada parágrafo é identificado pela chave citada nele (`\parencite{chave}` ou `\textcite{chave}`). Não use marcadores adicionais.
- Não reescreva parágrafos de outras etapas. Se notar um problema neles, reporte.

### 5.6 Atualização deste guia

- **Permitido sem consulta:** registrar progresso factual, verificações executadas (com data e resultado real), pendências descobertas e decisões **expressamente aprovadas** pelo usuário. Esses registros vão em §1.3, §1.4, §1.5, §4, na ficha da etapa e em §8.
- **Exige proposta e aprovação antes de mudar:** requisitos, escopo, estrutura, títulos, labels, ordem, agrupamento ou procedimento. Preserve os IDs. Se uma etapa for removida ou criada, registre o fato em §8 em vez de renumerar.
- Se uma nova fonte entrar em conflito com uma decisão, explique qual decisão foi afetada e por quê, e aguarde.
- Nunca declare que uma compilação ou verificação foi executada sem tê-la executado.

### 5.7 Estados das etapas

| Estado                 | Significado                                                                                                         |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Não iniciada           | A análise ainda não começou. Os pré-requisitos pendentes ficam listados no quadro.                                  |
| Em análise             | Leitura e relatório prévio em andamento (passos A e B).                                                             |
| Bloqueada              | A etapa foi iniciada, mas está parada por um bloqueio registrado, como fonte ausente, `.bib` ou decisão do usuário. |
| Aguardando autorização | O relatório foi entregue e as dúvidas resolvidas, mas falta a autorização expressa.                                 |
| Em execução autorizada | Redação ou edição autorizadas, em andamento.                                                                        |
| Concluída              | **Todos** os critérios de conclusão foram verificados. Existir texto no manuscrito não basta.                       |

---

## 6. Procedimento padrão das etapas de artigo (E01–E20)

Este procedimento segue o fluxo da skill `.claude/skills/redacao-trabalhos-relacionados/SKILL.md` e acrescenta as regras deste guia. Se a ficha da etapa (§7) trouxer algo específico, aplique-o junto com este procedimento.

### A. Verificação e leitura (sem editar nada)

1. **Estado real.** Confira:
   - se a E00 está concluída, ou seja, se o esqueleto existe no `main.tex`;
   - se `artigos/<chave>.md` existe;
   - se a chave existe no `.bib` do projeto;
   - se a chave já aparece no `main.tex` (busca pela chave, e também pelo sobrenome do primeiro autor e pelo título, para achar menções sem `\cite`).

   > Se o quadro (§1.3) divergir do estado real, reporte a divergência antes de seguir.

2. **Pare** se faltar o `.md` desta etapa (a falta de outros artigos não bloqueia, D16), se faltar a chave no `.bib`, se houver pendência PB da etapa ou se a E00 não estiver concluída. Registre a etapa como Bloqueada e informe o motivo.
3. Leia **integralmente** o `artigos/<chave>.md`. Avalie se a extração parece completa.
4. Leia o registro do CSV (§2.2) e os parágrafos do plano em que o artigo aparece (ficha da etapa).
5. Confira se o `main.tex` foi preenchido com escopo próprio (Introdução e Objetivos). Se divergir de D07, pare e pergunte.
6. Confira se o que o plano diz do artigo bate com o artigo. Exemplo: o plano afirma que `kiruba` e `ramkumar` integram LSTM e ARIMA. Divergências que afetem o agrupamento devem ser reportadas e perguntadas. Não mude o agrupamento por conta própria.

### B. Relatório prévio (no chat)

Apresente de forma objetiva:

- Pontos extraídos: problema e objetivos, método, contribuições, resultados, limitações e trabalhos futuros, cada um com a localização da evidência ou com "não encontrado no arquivo fornecido" ou "ausente no artigo" (§5.4).
- Semelhanças e diferenças propostas em relação ao trabalho atual, marcadas como **sugestão**.
- Status bibliográfico: chave confirmada, pendências e a forma da citação.
- Local exato de inserção (§5.5): subseção e posição relativa aos parágrafos já existentes.
- Divergências entre o artigo, o CSV e o plano, e dúvidas objetivas.

Atualize a etapa no guia para "Em análise" e, depois do relatório, para "Aguardando autorização" ou "Bloqueada".

### C. Pausa

Aguarde as respostas e depois uma **autorização expressa** para redigir e inserir. Responder às perguntas não é autorização.

### D. Redação e inserção (somente após autorização)

- Redija o parágrafo seguindo §5.4 e insira-o conforme §5.5. Altere apenas esse ponto da seção 3.
- Apresente no chat o trecho inserido ou o diff.
- Ajustes nesse mesmo parágrafo, quando pedidos pelo usuário, fazem parte da etapa.

### E. Verificação e conclusão

Critérios comuns de conclusão, todos verificáveis:

- [ ] Existe exatamente um parágrafo com a chave da etapa na subseção de destino, e nenhum outro parágrafo de resumo dessa chave na seção 3.
- [ ] O parágrafo cobre os itens de §5.4, ou registra a ausência deles conforme a regra de evidência.
- [ ] A citação usa `\parencite` ou `\textcite` com uma chave existente no `.bib` do projeto. Não há citação em texto puro.
- [ ] A posição respeita a ordem dos IDs. Fora o parágrafo, nada mudou no `main.tex`, o que pode ser conferido com `git diff`.
- [ ] O usuário validou o trecho inserido.

**Compilação:** o agente não compila (D15). A citação é conferida por leitura: a chave existe no `.bib` do projeto e o comando está bem formado. Em §8, registre "compilação a cargo do usuário". Se o usuário reportar um problema de compilação ligado ao parágrafo, trate-o dentro da etapa.

Ao concluir, atualize a ficha, o quadro (§1.3), as pendências (§1.4), o próximo passo (§1.5) e o registro (§8).

---

## 7. Fichas das etapas

### E00 — Preparação, inventário e proteção

- **Objetivo:** confirmar as fontes e a proteção de `artigos/`, e preparar a seção 3 com o esqueleto aprovado.
- **Destino:** seção 3 do `main.tex` (só a parte 3).
- **Pré-requisitos:** nenhum. A E00 não depende da presença de artigos em `artigos/`, e concluí-la não exige que eles existam (D16).

**Parte 1 — Inventário (sem edição):**

1. Liste quais dos 20 arquivos `artigos/<chave>.md` existem. Registre os faltantes em PF-1.
2. Confirme que as 20 chaves estão em `analise_dados_2026_2/latex/referencias.bib` e compare esse arquivo com `metodologia_2026_1/referencias.bib` (informativo).
3. Leia o CSV com o leitor real e confirme a associação de §2.2.
4. Confira a seção 3 atual. Em 2026-10-27 ela tinha só os três parágrafos do template e nenhuma citação das 20 chaves.

**Parte 2 — Proteção e Git (sem edição):**

1. `git ls-files artigos`: o esperado é apenas `artigos/.gitignore`. Se houver qualquer fonte rastreada, **reporte e peça autorização específica** antes de qualquer mudança de rastreamento. Não use `git rm`, não apague nada e não faça commit.
2. `git check-ignore -v` com caminhos de teste que não precisam existir:
   - `artigos/teste.md` e `artigos/sub/teste.md` devem aparecer como **ignorados**;
   - `artigos/.gitignore` **não** pode ser ignorado.

   Registre o resultado. Não crie `.gitkeep` (D11).

3. Confirme que não há regra no `.gitignore` da raiz que anule a proteção, como uma negação que reinclua `artigos/`.

**Parte 3 — Esqueleto da seção (exige autorização específica):**

Apresente o plano a seguir e aguarde autorização expressa. Depois dela:

1. Remova da seção 3 os três parágrafos de instrução do template, que começam com:
   - "Escreva aqui dois trabalhos relacionados por pessoa do grupo";
   - "Cada parágrafo deve dar um resumo sobre o trabalho";
   - "Segundo \sout{(Smith e Jones, 1999)}".

   O texto continua preservado em `analise_dados_2026_2/template.md`.

2. Mantenha a linha `\section{Trabalhos Relacionados}\label{sec:trabalhos-relacionados}` e o bloco de comentário anterior a ela. Logo depois, insira:

```latex
\subsection{Abordagens Estatísticas e de Aprendizado de Máquina Clássico}\label{sec:tr-classicos}

\subsection{Redes Neurais Recorrentes e LSTM}\label{sec:tr-recorrentes}

\subsection{Modelos Híbridos}\label{sec:tr-hibridos}

\subsection{Estratégias Recentes de Aprimoramento}\label{sec:tr-aprimoramento}

\subsubsection{Pré-processamento e Extração de Características}\label{sec:tr-preprocessamento}

\subsubsection{Mecanismos de Atenção e Arquiteturas Temporais e Estruturais}\label{sec:tr-atencao}

\subsubsection{Integração de Dados Multimodais}\label{sec:tr-multimodal}

\subsection{Padronização e Avaliação Metodológica}\label{sec:tr-avaliacao}
```

3. Não altere o comentário de abertura da seção 4 (`% 4. Método`) nem nada fora da seção 3.

**Alterações permitidas:** só a Parte 3, depois da autorização.

**Critérios de conclusão:**

- [ ] O inventário e a verificação de Git estão registrados em §8, com os resultados reais.
- [ ] Nenhuma fonte de `artigos/` está rastreada, ou a situação foi reportada e decidida pelo usuário.
- [ ] Os três parágrafos do template saíram da seção 3, e os oito títulos e labels de §3 estão presentes, na ordem.
- [ ] O `git diff` do `main.tex` mostra mudanças só na seção 3.

Não compile (D15). Ao compilar, o usuário verá o aviso `Empty bibliography`, que é esperado enquanto nada for citado.

**Estado:** Não iniciada. **Próximo passo:** executar as Partes 1 e 2 e apresentar a Parte 3 para autorização.

### Fichas E01–E20

Os campos comuns a todas as etapas de artigo valem para cada ficha abaixo:

- **Objetivo:** um parágrafo de resumo do artigo, conforme §5.4.
- **Fonte:** `artigos/<chave>.md`, não localizado em 2026-10-27.
- **Pré-requisitos:** E00 concluída e o `.md` **desta etapa** presente. Os demais artigos não precisam estar na pasta (D16).
- **Procedimento e condições de parada:** §6.
- **Alterações permitidas após autorização:** inserir o parágrafo na subseção de destino, conforme §5.5.
- **Critérios de conclusão:** §6.E.
- **Estado conhecido do texto, em 2026-10-27:** a chave não está citada no `main.tex`.

Cada ficha traz só o que é específico da etapa. "Plano" indica as linhas de `metodologia_2026_1/main.tex` (§2.1) e "CSV" indica o registro (§2.2).

#### E01 — `mahjouby_predicting_2024` → 3.1

- Plano: 82. CSV: 16.
- Estado: Não iniciada. Pendências: PF-1. Próximo passo: aguardar a E00 e o `.md`.

#### E02 — `sharma_stock_2023` → 3.1

- Plano: 82. CSV: 19 (Trabalhos Futuros "Não fala sobre"; confirmar no artigo).
- Estado: Não iniciada. Pendências: PF-1. Próximo passo: aguardar a E00 e o `.md`.

#### E03 — `k_leveraging_2023` → 3.2

- Plano: 84 e 120 (metodologia). CSV: 13.
- **Bloqueio PB-1:** antes de redigir, peça ao usuário que confirme ou corrija no `.bib` o nome do primeiro autor (hoje `K, Thinakaran`) e o coautor `V, Dilli Ganesh`. Use o próprio artigo como evidência da grafia.
- Estado: Não iniciada. Pendências: PF-1, PB-1. Próximo passo: aguardar a E00, o `.md` e a correção do `.bib`.

#### E04 — `miot_empirical_2020` → 3.2

- Plano: 84. CSV: 21 (problema registrado como citação literal; parafrasear). Entrada `@misc` (arXiv).
- Estado: Não iniciada. Pendências: PF-1. Próximo passo: aguardar a E00 e o `.md`.

#### E05 — `kiruba_risk_2025` → 3.3

- Plano: 86. CSV: 17 (Trabalhos Futuros "Não fala sobre"; confirmar no artigo).
- Confirmar a integração LSTM/ARIMA afirmada pelo plano (§6.A6).
- Estado: Não iniciada. Pendências: PF-1. Próximo passo: aguardar a E00 e o `.md`.

#### E06 — `ramkumar_development_2023` → 3.3

- Plano: 78 (contexto) e 86. CSV: 8.
- Confirmar a integração LSTM/ARIMA afirmada pelo plano (§6.A6). Informar PB-3, que não bloqueia.
- Estado: Não iniciada. Pendências: PF-1. Próximo passo: aguardar a E00 e o `.md`.

#### E07 — `liu_application_2024` → 3.3

- Plano: 86. CSV: 7.
- Estado: Não iniciada. Pendências: PF-1. Próximo passo: aguardar a E00 e o `.md`.

#### E08 — `sun_novel_2024` → 3.4.1

- Plano: 78 (contexto), 88 e 124 (metodologia). CSV: 2.
- Estado: Não iniciada. Pendências: PF-1. Próximo passo: aguardar a E00 e o `.md`.

#### E09 — `li_accurate_2024` → 3.4.1

- Plano: 76 (contexto), 88 e 124 (metodologia). CSV: 5.
- Estado: Não iniciada. Pendências: PF-1. Próximo passo: aguardar a E00 e o `.md`.

#### E10 — `hosseini_pattern-based_2025` → 3.4.1

- Plano: 80 (contexto), 88 e 124 (metodologia). CSV: 15.
- Estado: Não iniciada. Pendências: PF-1. Próximo passo: aguardar a E00 e o `.md`.

#### E11 — `wan_tcdformer_2024` → 3.4.1

- Plano: 88. CSV: 22 (Objetivos vazio).
- Estado: Não iniciada. Pendências: PF-1. Próximo passo: aguardar a E00 e o `.md`.

#### E12 — `biswas_dual_2025` → 3.4.2

- Plano: 88 e 124 (metodologia). CSV: 1.
- Estado: Não iniciada. Pendências: PF-1. Próximo passo: aguardar a E00 e o `.md`.

#### E13 — `luo_short-term_2024` → 3.4.2

- Plano: 88. CSV: 18.
- Estado: Não iniciada. Pendências: PF-1. Próximo passo: aguardar a E00 e o `.md`.

#### E14 — `hsu_fingat_2023` → 3.4.2

- Plano: 88 e 124 (metodologia). CSV: 9.
- **Bloqueio PB-2:** antes de redigir, peça ao usuário que corrija no `.bib` o título `Top-{KK}`. Use o artigo como evidência do título correto. O ano no `.bib` é 2023 e o DOI é `10.1109/TKDE.2021.3079496`. Se o artigo indicar outra data de publicação, reporte.
- Estado: Não iniciada. Pendências: PF-1, PB-2. Próximo passo: aguardar a E00, o `.md` e a correção do `.bib`.

#### E15 — `anghinoni_time_2019` → 3.4.2

- Plano: 88. CSV: 23 (Problema de Pesquisa vazio).
- Estado: Não iniciada. Pendências: PF-1. Próximo passo: aguardar a E00 e o `.md`.

#### E16 — `choi_hybrid_2023` → 3.4.3

- Plano: 86 (bloco híbrido), 88 e 126 (metodologia). CSV: 12.
- Resumo único (D04). Pode descrever a dimensão híbrida. Nenhum parágrafo sobre ele na 3.3.
- Estado: Não iniciada. Pendências: PF-1. Próximo passo: aguardar a E00 e o `.md`.

#### E17 — `wang_stock_2023` → 3.4.3

- Plano: 88, 120 e 126 (metodologia). CSV: 4.
- Estado: Não iniciada. Pendências: PF-1. Próximo passo: aguardar a E00 e o `.md`.

#### E18 — `haryono_transformer-gated_2023` → 3.4.3

- Plano: 88, 120 e 126 (metodologia). CSV: 20.
- Estado: Não iniciada. Pendências: PF-1. Próximo passo: aguardar a E00 e o `.md`.

#### E19 — `anbaee_farimani_adaptive_2024` → 3.4.3

- Plano: 88. CSV: 6.
- Estado: Não iniciada. Pendências: PF-1. Próximo passo: aguardar a E00 e o `.md`.

#### E20 — `hu_fintsb_2025` → 3.5

- Plano: 90 e 122 (metodologia, FinTSB como _baseline_). CSV: 10. Entrada `@misc` (arXiv).
- O uso do FinTSB no artigo da disciplina **não** está confirmado (D07). Pergunte antes de usá-lo como semelhança.
- Estado: Não iniciada. Pendências: PF-1. Próximo passo: aguardar a E00 e o `.md`.

### E21 — Revisão integrada da seção

- **Objetivo:** conferir a seção 3 completa. Esta etapa **não** autoriza reescrever a seção nem tocar em outras seções.
- **Pré-requisitos:** E01–E20 concluídas, ou removidas por decisão registrada em §4.
- **Procedimento:**
  1. **Cobertura:** cada artigo do escopo vigente tem exatamente um parágrafo, na subseção correta e em ordem de ID. Não há resumo de artigo fora do escopo.
  2. **Duplicação:** nenhuma chave tem dois resumos. `choi` aparece só na 3.4.3.
  3. **Estrutura:** os títulos e labels de §3 estão intactos. Não há parágrafos de abertura, transição ou síntese sem aprovação.
  4. **Consistência:** terminologia, tempo verbal e termos em inglês em `\textit{}` (D14) uniformes, e a distinção entre contribuições, resultados, limitações e trabalhos futuros mantida.
  5. **Comparações:** todas estão dentro do escopo de D07 e não se contradizem.
  6. **Citações:** só `\parencite` ou `\textcite`, todas as chaves resolvidas e a lista de referências sem pendências PB abertas.
  7. **Compilação:** não compile (D15). Peça ao usuário que compile e reporte erros e avisos, e trate só os ligados à seção 3.
- **Alterações permitidas após autorização:** só correções pontuais apontadas no relatório e aprovadas uma a uma.
- **Critérios de conclusão:** relatório da revisão entregue, correções aprovadas aplicadas e todos os itens 1 a 7 conferidos e registrados em §8.
- **Estado:** Não iniciada.

---

## 8. Registro de mudanças e verificações

| Data       | Etapa | Registro                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ---------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 2026-10-27 | —     | Guia criado (v1), com as decisões D01–D13 aprovadas pelo usuário.                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| 2026-10-27 | —     | Verificado durante a criação do guia: as 20 chaves foram extraídas da introdução do plano e conferidas com o agrupamento aprovado. O `.bib` do projeto contém as 20 e é idêntico (`cmp`) ao da metodologia. O CSV foi lido com `csv.reader` (delimitador `,`, UTF-8, CRLF, 23 registros, 5 colunas). `artigos/` tem só `.gitignore`, rastreado. O `git check-ignore` confirmou que `artigos/x.md` e `artigos/sub/y.md` são ignorados e que `artigos/.gitignore` não é. A seção 3 do `main.tex` não cita nenhuma das 20 chaves. |
| 2026-10-27 | —     | Formato de citação testado numa cópia temporária, fora do projeto, com o preâmbulo atual: `(HSU, TSAI e LI, 2023)`, `(MIOT e DRIGOUT, 2020)`, `(K et al., 2023)` e `HSU, TSAI e LI (2023)`. O título "Top-KK" aparece na lista de referências. A compilação do `main.tex` do projeto não foi verificada.                                                                                                                                                                                                                       |
| 2026-10-27 | —     | Guia v2: incorporados os comentários do usuário. Novas decisões D14 (termos em inglês em `\textit{}`, que resolve a PE-2), D15 (o agente nunca compila; critérios de compilação substituídos por conferência de citações por leitura) e D16 (artigos adicionados aos poucos; cada etapa só exige o próprio `.md`). PF-1 reescrita. Comentários removidos.                                                                                                                                                                      |
