---
name: redacao-trabalhos-relacionados
description: Analisa um artigo científico em Markdown e, mediante autorização explícita e validação prévia de referências, redige exatamente um parágrafo estruturado na seção "Trabalhos Relacionados" de um documento LaTeX (.tex).
---

# Redação de Trabalhos Relacionados

Use esta skill para analisar um artigo científico individual e integrá-lo de forma controlada à seção **"Trabalhos Relacionados"** de um manuscrito em LaTeX.

O agente operando esta skill **não toma decisões autônomas diante de ambiguidades**, **nunca edita arquivos bibliográficos (.bib)** e **jamais altera o documento .tex sem autorização prévia e explícita do usuário**.

---

## 1. Entradas Esperadas

### Obrigatórias
1. **Caminho do arquivo LaTeX:** caminho do `.tex` do trabalho (ex.: `analise_dados_2026_2/latex/main.tex`).
2. **Caminho do artigo a analisar:** caminho do `.md` contendo o conteúdo do artigo a ser sintetizado.

### Opcionais (suporte e contextualização)
3. **Plano de Dissertação / Proposta:** arquivo com a visão geral do projeto do usuário.
4. **Planilha ou notas de estudo:** síntese prévia de problemas, métodos, contribuições e trabalhos futuros.
5. **Template ou guia de diretrizes da disciplina:** instruções e orientações fornecidas pelos docentes.

---

## 2. Limites Estritos de Autonomia e Comportamento

1. **Sem decisões unilaterais:** diante de termos ambíguos, conflitos conceituais ou dados faltantes, pare e pergunte de forma objetiva.
2. **Sugestões não são decisões aprovadas:** você pode sugerir formulações ou abordagens com justificativa, mas nunca prossiga como se a sugestão tivesse sido aceita tacitamente.
3. **Consulta x Edição de Bibliografia (.bib):**
   - É permitida a **leitura** de arquivos `.bib` vinculados ao projeto para validar chaves de citação.
   - É **estritamente proibido editar** arquivos `.bib`.
   - Se o artigo analisado não possuir entrada bibliográfica no `.bib`, pare imediatamente e solicite ao usuário o cadastro da referência antes de redigir.
4. **Verificação de duplicidade:** verifique se o artigo já está citado ou descrito na seção. Se houver menção prévia, apresente o trecho encontrado e pergunte se deve substituir, expandir ou preservar.
5. **Transparência ativa:** se qualquer ação fora do escopo estritamente autorizado for executada acidentalmente, reporte o ocorrido com clareza e interrompa imediatamente o fluxo.
6. **Crítica aos templates:** templates de disciplinas podem conter orientações contraditórias ou incompletas. Não assuma que as instruções externas são perfeitas; aponte incoerências e confirme a conduta com o usuário.

---

## 3. Estrutura Obrigatória do Parágrafo

Salvo instrução em contrário dada explicitamente pelo usuário, cada artigo deve ser sintetizado em **exatamente um parágrafo contínuo**, contendo os quatro blocos fundamentais:

1. **Problema e Objetivos:** o problema de pesquisa abordado e os objetivos centrais do estudo.
2. **Método e Abordagem:** as técnicas, algoritmos, modelos ou metodologias empregadas.
3. **Resultados, Contribuições, Limitações e Trabalhos Futuros:**
   - Evidências e contribuições explícitas reportadas pelos autores.
   - Limitações e trabalhos futuros explicitamente declarados no texto.
   - *Atenção:* se limitações ou trabalhos futuros não constarem no artigo, registre essa ausência na análise prévia. Nunca invente informações não sustentadas pela fonte.
4. **Relação com o Trabalho Atual (Semelhanças e Diferenças):**
   - Em que ponto a técnica ou problemática se assemelha ao seu projeto.
   - Qual é a distinção clara e diferencial da sua abordagem em relação ao trabalho analisado.

### Padrão de Citação
- Use comandos semânticos do LaTeX (ex.: `\cite{chave}` ou `\citep{chave}`, conforme o pacote configurado no projeto).
- O formato textual deve seguir o estilo autor-data (ex.: `ANGHINONI et al., 2019`).
- Nunca escreva manualmente citações estáticas em texto puro se o projeto estiver configurado com BibTeX/BibLaTeX.

---

## 4. Fluxo Operacional em Quatro Etapas

### Etapa 1: Leitura e Mapeamento de Fontes
- Leia integralmente o artigo-fonte (`.md`).
- Localize a seção **"Trabalhos Relacionados"** (ou equivalente) no arquivo `.tex` informado.
- Caso existam arquivos opcionais (Plano de Dissertação, planilha, notas), cruze as informações para enriquecer o contexto.
- Verifique no `.bib` se existe a chave de citação correspondente ao artigo.

### Etapa 2: Análise Prévia e Formulação de Dúvidas (Sem editar o .tex)
Apresente ao usuário um relatório prévio e objetivo contendo:
- **Resumo dos pontos extraídos:** problema, método, resultados/contribuições, limitações e trabalhos futuros identificados no artigo.
- **Diferenciais identificados:** proposta de relação de semelhança e diferença com o trabalho atual.
- **Status bibliográfico:** confirmação da chave existente no `.bib` ou solicitação de cadastro se estiver ausente.
- **Local exato de inserção:** trecho ou posição do `.tex` onde o parágrafo será inserido.
- **Dúvidas e ambiguidades:** perguntas pontuais caso haja pontos abertos ou divergências entre o artigo e as notas/planilha.

### Etapa 3: Pausa para Autorização Explícita
- **Pare e aguarde a resposta do usuário.**
- O fato de responder a todas as perguntas pendentes **não** constitui autorização automática para edição.
- Aguarde uma confirmação expressa (ex.: *"Pode inserir no .tex"* ou *"Autorizado"*).

### Etapa 4: Inserção Controlada no LaTeX
- Somente após a autorização explícita, insira o parágrafo acordado no arquivo `.tex`.
- Limite a alteração estritamente ao ponto de inserção autorizado na seção de Trabalhos Relacionados.
- Não altere preâmbulos, outras seções ou configurações sem solicitação prévia.
- Apresente o diff ou trecho inserido no chat para validação final.
