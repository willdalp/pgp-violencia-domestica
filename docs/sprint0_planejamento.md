# Planejamento Ágil - Projeto de Análise Preditiva

## 📋 Product Backlog Priorizado (Sprint 1)
As tarefas do projeto foram desdobradas em Histórias de Usuário (User Stories) e priorizadas tecnicamente para guiar o desenvolvimento do modelo preditivo e do MVP.

| Prioridade | ID | História de Usuário (User Story) | Critérios de Aceite |
| :--- | :--- | :--- | :--- |
| **1 (Altíssima)** | US01 | **Como** pesquisador, **quero** realizar o levantamento de datasets públicos (Dados.gov.br, IPEA, Atlas da Violência)[cite: 1], **para** compor a base de dados do projeto. | Os datasets devem ser baixados ou acessíveis via API; Devem conter histórico de ocorrências e metadados geográficos. |
| **2 (Alta)** | US02 | **Como** cientista de dados, **quero** conduzir uma análise exploratória (EDA) inicial, **para** validar se os dados são suficientes e relevantes para o problema da violência doméstica[cite: 1]. | Valores nulos identificados; Distribuição de classes avaliada; Notebook executável entregue. |
| **3 (Alta)** | US03 | **Como** engenheiro de dados, **quero** higienizar e integrar as bases distintas[cite: 1], **para** criar um dataset único e estruturado para o treinamento. | Colunas padronizadas; Dados inconsistentes tratados ou removidos. |
| **4 (Média)** | US04 | **Como** cientista de dados, **quero** treinar modelos preditivos conduzindo experimentos no Google Colab[cite: 1], **para** prever áreas com tendência à escalada da violência. | Pelo menos dois modelos testados; Métricas de avaliação (Acurácia, F1-Score, Recall) computadas. |
| **5 (Média)** | US05 | **Como** gestor de segurança pública, **quero** visualizar as zonas de alto risco em um dashboard interativo[cite: 1], **para** alocar viaturas e recursos antecipadamente. | Aplicação rodando em Streamlit[cite: 1]; Mapas interativos funcionais. |
| **6 (Baixa)** | US06 | **Como** desenvolvedor, **quero** documentar o repositório completo e os resultados[cite: 1], **para** garantir a reprodutibilidade científica do projeto. | `README.md` atualizado com o passo a passo de instalação e uso do MVP. |

---

## 📊 Quadro Kanban (Status: Sprint 1)
O quadro abaixo reflete a transição das entregas de planejamento (Sprint 0) para a fase de exploração de dados (Sprint 1).

| A Fazer (To Do) | Em Andamento (Doing) | Concluído (Done) |
| :--- | :--- | :--- |
| Higienizar e integrar as bases distintas em um dataset único (US03)[cite: 1]. | Realizar levantamento de datasets públicos (Dados.gov.br, IPEA) (US01)[cite: 1]. | Elaboração do Business Model Canvas. |
| Treinar modelos preditivos no Google Colab (US04)[cite: 1]. | Conduzir análise exploratória (EDA) inicial (US02). | Criação do repositório no GitHub (setup inicial das pastas e arquivos). |
| Desenvolver dashboard MVP em Streamlit (US05)[cite: 1]. | Atualização do Product Backlog com Histórias de Usuário. | Redação inicial do Artigo Científico (Introdução, Problema, Objetivos e Justificativa)[cite: 1]. |
| Documentar o código e atualizar o repositório (US06)[cite: 1]. | | |
