# Planejamento Ágil - Projeto de Análise Preditiva

## 📋 Product Backlog Priorizado (Sprint 2)
As tarefas do projeto foram desdobradas em Histórias de Usuário (User Stories), priorizadas tecnicamente e agora contam com estimativas ágeis (Story Points) para guiar o desenvolvimento do modelo preditivo e do MVP.

| Prioridade | ID | História de Usuário (User Story) | Critérios de Aceite | Story Points (Esforço) |
| :--- | :--- | :--- | :--- | :--- |
| **1 (Altíssima)** | US01 | **Como** pesquisador,  realizar o levantamento de datasets públicos (Dados.gov.br, IPEA, Atlas da Violência), para compor a base de dados do projeto. | Os datasets devem ser baixados ou acessíveis via API; Devem conter histórico de ocorrências e metadados geográficos. | **3 pts** |
| **2 (Alta)** | US03 | **Como** engenheiro de dados,  higienizar e integrar as bases distintas, criando atributos como sazonalidade e médias móveis. | Colunas padronizadas; Dados nulos tratados; Features de engenharia criadas. | **5 pts** |
| **3 (Alta)** | US02 | **Como** cientista de dados,  conduzir uma análise exploratória (EDA) inicial, validando as distribuições para o problema. | Valores nulos identificados; Distribuição de classes avaliada; Notebook entregue. | **5 pts** |
| **4 (Média)** | US04 | **Como** cientista de dados,  treinar modelos preditivos conduzindo experimentos no Google Colab, para prever áreas com tendência à escalada da violência. | Random Forest e Regressão Logística testados; Arquivo `.joblib` exportado e métricas comparadas. | **8 pts** |
| **5 (Média)** | US05 | **Como** gestor de segurança pública,  visualizar as zonas de alto risco em um dashboard interativo, para alocar recursos antecipadamente. | Aplicação rodando em Streamlit; Mapas interativos funcionais. | **13 pts** |
| **6 (Baixa)** | US06 | **Como** desenvolvedor,  documentar a metodologia DSRM no artigo da SBC e atualizar o repositório para reprodutibilidade. | Artigo atualizado com métricas; `README.md` atualizado com uso do MVP. | **5 pts** |

## 🎯 Casos de Uso Principais (Sprints 1 e 2)
* **UC01 - Ingerir e Limpar Dados:** O cientista de dados faz o upload dos datasets no Google Colab, e o sistema processa a higienização (tratamento de nulos e padronização) e cria features (média móvel) para gerar um dataframe único.
* **UC02 - Treinar Modelo Preditivo:** O cientista de dados seleciona as features (idade, região, histórico) e o algoritmo Random Forest processa o treinamento, retornando métricas de precisão (Acurácia, F1-Score) e exportando o modelo baseline (`.joblib`).
* **UC03 - Visualizar Zonas de Risco:** O gestor público acessa o Dashboard, filtra por município ou estado, e o sistema exibe um mapa de calor apontando as áreas com maior probabilidade de escalada da violência (previsto para Sprint 3).

---

## 📊 Quadro Kanban (Status: Fim da Sprint 2)
O quadro abaixo reflete a transição das entregas de estruturação de dados (Sprint 1) e pipeline de Machine Learning (Sprint 2) para a fase de construção do Dashboard MVP (Sprint 3).

| A Fazer (To Do - Sprint 3) | Em Validação (Review - Sprint 2) | Concluído (Done - Sprints 0 e 1) |
| :--- | :--- | :--- |
| **(US05)** Desenvolver dashboard interativo do MVP utilizando Streamlit [13 pts]. | **(US04)** Exportar modelo Random Forest treinado e versionar na pasta `models` do repositório [8 pts]. | **(US01)** Realizar levantamento de datasets públicos [3 pts]. |
| **(US06)** Atualizar `README.md` com instruções de execução da aplicação final [2 pts]. | **(US06)** Redigir a seção de Metodologia, detalhando o ciclo de vida do ML e o uso da DSRM no artigo formato SBC [3 pts]. | **(US03)** Higienizar, agrupar e realizar o pré-processamento das bases de dados [5 pts]. |
| | | **(US02)** Conduzir análise exploratória (EDA) e validar variáveis [5 pts]. |
| | | Canvas, Backlog original e repositório inicial no GitHub. |
