# Template acadêmico da disciplina — versão LaTeX

Projeto LaTeX local (sem Overleaf) que reproduz o template da disciplina. O conteúdo vem de `../template.md` e a apresentação (fontes, medidas, margens, alinhamentos) de `../template.html`, um export do Google Docs. Os dois arquivos originais foram mantidos intactos.

O texto do template é composto por **instruções e exemplos para o autor**: nada foi reescrito, preenchido ou inventado na conversão.

## Estrutura

```
analise_dados_2026_2/latex/
├── main.tex         documento completo (preâmbulo + conteúdo)
├── referencias.bib  base bibliográfica (biblatex/biber)
├── main.pdf         PDF gerado pela compilação
├── README.md        este arquivo
└── figuras/      imagens usadas no documento (ver mapa abaixo)
```

Arquivos gerados pela compilação (`main.aux`, `main.log`, `main.out`, `main.fls`, `main.fdb_latexmk`, `main.bbl`, `main.bcf`, `main.blg`, `main.run.xml`) podem ser apagados a qualquer momento com `latexmk -C`.

As referências ficam em **`referencias.bib`** (biblatex + biber, estilo `authoryear`). A seção "Referências" é gerada por `\printbibliography` e lista apenas as obras citadas no texto com `\cite`/`\parencite`/`\textcite`. As citações de exemplo do template (ex.: `(Knuth, 1984)`) continuam como texto simples, e a lista manual de exemplos foi removida.

### Mapa das figuras

Os arquivos em `../images/` **não** estão na ordem em que aparecem no documento. A correspondência usada (confirmada por comparação de conteúdo e proporção das imagens) é:

| `figuras/` (usada no `main.tex`) | arquivo de origem   | aparece em                                 |
| -------------------------------- | ------------------- | ------------------------------------------ |
| `eda.png`                        | `images/image2.png` | Figura 1 (Análise Exploratória)            |
| `cotovelo.png`                   | `images/image5.png` | Figura 2, 1º painel (método do cotovelo)   |
| `silhueta-6-clusters.png`        | `images/image3.png` | Figura 2, 2º painel                        |
| `silhueta-10-clusters.png`       | `images/image4.png` | Figura 2, 3º painel                        |
| `silhueta-3-a-8-clusters.png`    | `images/image6.png` | Figura 2, 4º painel                        |
| `dados-clusterizados.png`        | `images/image1.png` | Figura 3 (dados originais x clusterizados) |

As quatro imagens de silhueta/cotovelo eram, no documento original, quatro imagens seguidas no mesmo parágrafo; aqui formam uma única figura (`figura 2`) com os mesmos painéis e as mesmas larguras relativas.

## Compilação

Arquivo principal: **`main.tex`** — é o arquivo a abrir no VS Code.

```bash
latexmk -pdf main.tex        # compila (executa o pdflatex quantas vezes precisar)
latexmk -pvc -pdf main.tex   # recompila automaticamente a cada alteração
latexmk -C                   # apaga os arquivos intermediários
```

**Compilador: `pdflatex`** (via `latexmk`), com **`biber`** para a bibliografia; o `latexmk` executa o `biber` sozinho. O documento usa apenas fontes padrão (Times, via pacote `mathptmx`), sem `fontspec`, e o `pdflatex` é o compilador padrão da extensão LaTeX Workshop, o que dispensa configuração no VS Code. Neste ambiente o executável `biber` precisa estar no `PATH` (há um symlink em `~/.local/bin/biber`, como os demais executáveis do TinyTeX).

### VS Code (LaTeX Workshop)

1. Instale a extensão **LaTeX Workshop** (`james-yu.latex-workshop`).
2. Abra `analise_dados_2026_2/latex/main.tex`.
3. `Ctrl+Alt+B` compila; `Ctrl+Alt+V` abre o PDF ao lado.

Nenhum arquivo em `.vscode/` foi criado ou alterado: os padrões da extensão (`latexmk` + `pdflatex`) já compilam este projeto.

### Estado da compilação neste ambiente

Compilado e verificado com:

- TeX Live 2026 (`pdfTeX 3.141592653-2.6-1.40.29`), `latexmk` 4.88, padrões de hifenização do português (`babel-portuges`);
- instalação em espaço de usuário (`~/.TinyTeX`), sem `sudo` e sem alterar configurações do sistema.

Resultado: **8 páginas, 0 erros**. Avisos esperados: `Empty bibliography` (enquanto nada for citado) e `csquotes missing` (recomendação do biblatex; ver abaixo). Sem referências indefinidas e sem `Overfull`/`Underfull` boxes.

Em uma instalação mínima de TeX Live, os pacotes usados podem ser instalados com:

```bash
tlmgr install psnfss titlesec caption ulem changepage tools \
              babel-portuges geometry hyperref xcolor
```

Todos já vêm em uma instalação completa do TeX Live.

## Como editar o conteúdo

- Todo o conteúdo está em `main.tex`, na ordem do documento. Troque os textos de instrução pelas seções do seu trabalho e apague os avisos entre colchetes (por exemplo `[Legenda para os resultados do treino]`) e a lista "ATENÇÃO PARA DISCUTIR E APRESENTAR" quando não precisar mais dela.
- Os comandos `\section`, `\subsection`, `\begin{itemize}`, `figure` e `table` são semânticos: os números são calculados pelo LaTeX, nunca digitados.
- Foi preservada literalmente a redação do original, inclusive marcas de digitação que existem nele: "tradulção", "inclídos", "J2ohn Wiley & Sons", "segundo (Smith e Jones, 1999)" riscado com `\sout` e o "à esquerda" em minúscula na legenda da Figura 3.
- Se preferir que figuras e tabelas fiquem exatamente no ponto de inserção (e não flutuando para o topo da página), acrescente `\usepackage{float}` ao preâmbulo e troque `[htbp]` por `[H]`.

## Citações e referências cruzadas

### Referências cruzadas (prontas para uso)

Basta escrever `\ref{chave}` (ou `\autoref{chave}`) e a numeração sai correta. Chaves já existentes:

- Seções: `sec:introducao`, `sec:objetivos`, `sec:referencial`, `sec:assunto-problema`, `sec:aprendizado-maquina`, `sec:trabalhos-relacionados`, `sec:metodo`, `sec:etapa-1`, `sec:etapa-2`, `sec:analise-exploratoria`, `sec:aplicativo`, `sec:pergunta-3`, `sec:resultados`, `sec:primeira-pergunta`, `sec:segunda-pergunta`, `sec:consideracoes`
- Figuras: `fig:eda`, `fig:cotovelo-silhueta`, `fig:dados-clusterizados`
- Tabelas: `tab:resultados-treino`, `tab:resultados-teste`

Quando acrescentar figuras/tabelas/equações, use os mesmos prefixos (`fig:`, `tab:`, `eq:`) e coloque o `\label` sempre **depois** do `\caption` (nas figuras e tabelas) para que a referência capture o número correto.

### Citações bibliográficas

A bibliografia está ativa: `main.tex` carrega `biblatex` (backend `biber`, estilo `authoryear`) e `\addbibresource{referencias.bib}`.

- cite com `\parencite{chave}` → `(SOBRENOME, 2000)` ou `\textcite{chave}` → `SOBRENOME (2000)`; `\cite{chave}` também funciona. `\citep`/`\citet` são do `natbib` e **não** existem neste projeto;
- as chaves válidas são as de `referencias.bib`; só as obras citadas entram na seção "Referências";
- compile com `latexmk -pdf main.tex`. Pacotes: `tlmgr install biblatex biber`;
- o aviso `csquotes missing` é uma recomendação do biblatex para citações com aspas em documentos `babel`; para eliminá-lo, `tlmgr install csquotes` e `\usepackage{csquotes}` no preâmbulo (não aplicado).

O preâmbulo ajusta o `authoryear` para o formato do template: sobrenome em maiúsculas e vírgula antes do ano (`\mkbibnamefamily` e `nameyeardelim`). Com 4 ou mais autores sai `SOBRENOME et al.`; com 2 ou 3, `SOBRENOME1 e SOBRENOME2`. Como `\mkbibnamefamily` é global, o sobrenome também sai em maiúsculas na lista de referências. As iniciais dos autores não aparecem na citação.

## Diferenças em relação ao original (e por quê)

Nada de conteúdo foi descartado. As diferenças abaixo são de numeração, tipografia ou de recurso, e todas foram decididas ou estão justificadas:

1. **Numeração automática.** O documento original tem numeração inconsistente, que o LaTeX passou a calcular:
   - os títulos de subseção `7.1` e `7.2` são digitados no meio do texto no original (parágrafos em negrito, e não títulos); aqui são `\subsection`, com a mesma aparência (Times 12pt negrito) e numeração automática — o que também permite que "Seção 7.1" no texto vire `\ref{sec:primeira-pergunta}`;
   - a seção final é "9. Considerações Finais", sem que exista a seção 8; no PDF ela aparece como **8**;
   - no original as duas últimas legendas de figura estão numeradas como "Figura 2" (a segunda deveria ser 3); no PDF são **Figura 2** e **Figura 3**;
   - o original escreve às vezes "1. Introdução", às vezes "7 Resultados e Discussão", às vezes "9 . Considerações Finais"; o padrão adotado é `n. Título` em todas as seções e subseções.
2. **Fonte das legendas.** No HTML as legendas das Figuras 2 e 3 estão em "Helvetica Neue" negrito 10pt (artefato do export do Google Docs, que misturou a fonte); aqui todas as legendas ficam na mesma fonte do resto do documento, Times 10pt negrito. A legenda da Figura 1 continua centralizada e as demais justificadas, como no original.
3. **Linhas das tabelas.** No original as linhas internas são cinza-claro (`#cccccc`) e as linhas que separam os blocos de modelos são pretas; aqui todas são `\hline` preto.
4. **Rodapé.** O link `https://www.todamateria.com.br/normas-abnt-trabalhos/` é uma nota de rodapé; no export do Google Docs o texto dela aparece no fim do arquivo, depois de uma régua horizontal. Aqui é um `\footnote` de verdade, que sai ao pé da página em que é citado (página 1).
5. **Tamanho de página e margens.** O HTML não tem regra `@page`; as dimensões foram deduzidas da área de texto (`max-width: 451,4pt` + `padding: 72pt` nos quatro lados, o que soma exatamente a largura de uma folha A4). Adotado **A4 com margens de 2,54cm** (72pt).
6. **Larguras.** As colunas da tabela foram redistribuídas para a mesma largura total do original (~15cm, com o mesmo preenchimento interno de 2pt), de modo que nenhuma palavra transborde a célula. As larguras das figuras seguem as do HTML.
7. **Texto com duas versões.** Onde o Markdown e o HTML divergem em detalhes de digitação (por exemplo `disciplina . à esquerda` no HTML, sem espaço no Markdown), foi seguido o Markdown, que é a fonte de conteúdo.
8. **Ferramenta.** O HTML declara a fonte "Times"; aqui é usado o `mathptmx`, o clone livre do Times distribuído com o TeX Live. A métrica é muito próxima, mas não idêntica à do Times New Roman usado pelo Google Docs.

## Pendências e revisão visual

- **Compilação:** nada pendente; o projeto compila sem erros nem avisos.
- **Fidelidade visual:** o PDF foi renderizado e comparado página a página com o layout do HTML; as diferenças conhecidas são as listadas acima. Vale uma conferência sua, no PDF, de dois pontos que dependem de fonte instalada no seu sistema: as métricas do Times e a posição final das figuras/tabelas (elas são flutuantes: o LaTeX pode colocá-las no topo ou no pé de uma página próxima — por exemplo, a Tabela 2 foi para a página 7).
- **Conteúdo:** nada a completar aqui; falta o preenchimento do artigo pelo autor, como previsto pelo próprio template.
- Opcional: para não versionar os intermediários do LaTeX, o `.gitignore` de `analise_dados_2026_2/` já cobre os intermediários do LaTeX e do biber (`*.aux`, `*.log`, `*.out`, `*.fls`, `*.fdb_latexmk`, `*.synctex.gz`, `*.bbl`, `*.bcf`, `*.blg`, `*.run.xml`) e os PDFs.
