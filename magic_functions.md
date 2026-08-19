`%matplotlib inline` é um **comando mágico** (magic command) do IPython/Jupyter que configura o backend de renderização do Matplotlib.

### O que ele faz especificamente:

1. **Renderiza gráficos dentro do notebook:** Os plots gerados são exibidos diretamente abaixo da célula que os criou, como imagens estáticas embutidas no output.
2. **Dispensa o `plt.show()`:** Com esse backend, você não precisa chamar `plt.show()` explicitamente para visualizar o gráfico (embora ainda funcione se você chamar).
3. **Salva no arquivo do notebook:** Como as imagens são estáticas (geralmente PNG), elas ficam armazenadas no `.ipynb`, permitindo que você veja os gráficos mesmo sem reexecutar o código.

### Comparação com outros backends

| Comando                | Comportamento                                                             |
| ---------------------- | ------------------------------------------------------------------------- |
| `%matplotlib inline`   | Imagens estáticas embutidas. Simples e portátil.                          |
| `%matplotlib notebook` | Gráficos **interativos** (zoom, pan) dentro do Jupyter Notebook clássico. |
| `%matplotlib widget`   | Interatividade no **JupyterLab** (requer `ipympl` instalado).             |
