# Cinza

**Cinza** é uma biblioteca estatística desenvolvida em Python para facilitar a realização de análises estatísticas, testes de hipóteses, inferência e visualizações diretamente sobre dados estruturados.

A proposta da biblioteca é tornar o processo de análise estatística mais **organizado, reproduzível e acessível**, reduzindo a necessidade de implementar manualmente cálculos estatísticos recorrentes.

O projeto foi desenvolvido com foco em aplicações práticas de **Data Analysis, Business Intelligence, experimentação e tomada de decisão baseada em dados**.

## 📊 O que é possível fazer com a Cinza?

A biblioteca reúne diferentes etapas de um processo de análise estatística em uma única estrutura, permitindo trabalhar desde a exploração inicial dos dados até análises inferenciais mais avançadas.

### Estatística Descritiva

Ferramentas destinadas à exploração e compreensão inicial dos dados, permitindo analisar medidas de tendência central, dispersão e características das distribuições.

### 📐 Inferência Estatística

A Cinza possui recursos para construção de **intervalos de confiança**, incluindo estimativas para médias e proporções. Os resultados retornam informações como estimativa, tamanho da amostra, erro padrão, margem de erro, nível de confiança e intervalo estimado.

### 🧪 Testes de Hipóteses

A biblioteca implementa diferentes testes estatísticos para comparação, associação e investigação de hipóteses.

Entre eles:

* Teste t para uma amostra
* Teste t para duas amostras independentes
* Teste t de Welch
* Teste t para amostras pareadas
* Testes Z para uma e duas proporções
* ANOVA de um fator
* ANOVA de dois fatores
* ANOVA com interação
* Teste post-hoc de Tukey
* Teste Qui-Quadrado de independência
* Teste Qui-Quadrado de ajustamento
* Correlação de Pearson
* Correlação de Spearman
* Teste do Sinal
* Teste de McNemar

Os testes retornam resultados estruturados contendo estatísticas, graus de liberdade, p-valores, níveis de significância e decisões sobre as hipóteses, conforme o procedimento realizado.

## 🧪 Experimentação e Testes A/B

A Cinza também possui recursos voltados para **experimentação estatística**, incluindo funções para estimar o tamanho de amostra necessário para testes A/B.

É possível calcular o tamanho amostral necessário para:

* Comparação de proporções
* Comparação de médias
* Comparação de médias em amostras pareadas

Os cálculos consideram parâmetros como **nível de significância, poder estatístico, desvio padrão e diferença mínima detectável**.
Isso permite utilizar a biblioteca em cenários como:

* Experimentação de produtos
* Testes de conversão
* Marketing
* Vendas
* Comparação de grupos
* Avaliação de intervenções
* Análise de experimentos A/B

## 📈 Visualização Estatística

A Cinza também possui funções específicas para auxiliar na exploração visual dos dados.

Entre as visualizações disponíveis estão:

* Boxplots
* Histogramas
* QQ-Plots
* Análise visual de variância
* Visualização de interação entre fatores
* Gráficos de resíduos

As funções de visualização podem trabalhar diretamente com `pandas.DataFrame`, permitindo integrar a exploração gráfica ao restante do fluxo estatístico.
A análise visual de variância também permite apresentar médias, medianas, desvio padrão e intervalos de confiança dos grupos.

## 🧠 Filosofia do projeto

A Cinza não pretende ser apenas uma coleção de fórmulas estatísticas.

A ideia é construir uma **camada de análise estatística em Python**, na qual os procedimentos matemáticos, estatísticos e suas interpretações possam ser utilizados de maneira estruturada dentro de projetos de análise de dados.

O objetivo é aproximar:

**Dados → Estatística → Evidência → Decisão**

em um único fluxo de trabalho.

## 🛠️ Tecnologias

A biblioteca utiliza o ecossistema científico do Python, incluindo:

* Python
* NumPy
* Pandas
* SciPy
* Statsmodels
* Scikit-learn
* Matplotlib
* Seaborn

Essas bibliotecas são utilizadas como base para cálculos estatísticos, manipulação de dados, modelagem e visualização.

## 🚧 Projeto em desenvolvimento

A **Cinza** está em desenvolvimento e novas funcionalidades estatísticas podem ser incorporadas ao longo do projeto.

A intenção é expandir progressivamente a biblioteca para abranger diferentes áreas da estatística aplicada, mantendo uma estrutura organizada e orientada à análise de dados.

---

> **Cinza**
> Uma biblioteca para transformar dados em evidências estatísticas.
