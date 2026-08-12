# Planejamento Ágil - Projeto de Análise Preditiva

## 📋 Product Backlog Priorizado (Sprint 1)
As tarefas do projeto foram desdobradas em Histórias de Usuário (User Stories) e priorizadas tecnicamente para guiar o desenvolvimento do modelo preditivo e do MVP.

| Prioridade | ID | História de Usuário (User Story) | Critérios de Aceite |
| :--- | :--- | :--- | :--- |
| **1 (Altíssima)** | US01 | **Como** pesquisador,  realizar o levantamento de datasets públicos (Dados.gov.br, IPEA, Atlas da Violência), **para** compor a base de dados do projeto. | Os datasets devem ser baixados ou acessíveis via API; Devem conter histórico de ocorrências e metadados geográficos. |
| **2 (Alta)** | US02 | **Como** cientista de dados,  conduzir uma análise exploratória (EDA) inicial, **para** validar se os dados são suficientes e relevantes para o problema da violência doméstica. | Valores nulos identificados; Distribuição de classes avaliada; Notebook executável entregue. |
| **3 (Alta)** | US03 | **Como** engenheiro de dados,  higienizar e integrar as bases distintas, **para** criar um dataset único e estruturado para o treinamento. | Colunas padronizadas; Dados inconsistentes tratados ou removidos. |
| **4 (Média)** | US04 | **Como** cientista de dados,  treinar modelos preditivos conduzindo experimentos no Google Colab, **para** prever áreas com tendência à escalada da violência. | Pelo menos dois modelos testados; Métricas de avaliação (Acurácia, F1-Score, Recall) computadas. |
| **5 (Média)** | US05 | **Como** gestor de segurança pública,  visualizar as zonas de alto risco em um dashboard interativo, **para** alocar viaturas e recursos antecipadamente. | Aplicação rodando em Streamlit; Mapas interativos funcionais. |
| **6 (Baixa)** | US06 | **Como** desenvolvedor,  documentar o repositório completo e os resultados, **para** garantir a reprodutibilidade científica do projeto. | `README.md` atualizado com o passo a passo de instalação e uso do MVP. |

---

## 📊 Quadro Kanban (Status: Sprint 1)
O quadro abaixo reflete a transição das entregas de planejamento (Sprint 0) para a fase de exploração de dados (Sprint 1).

| A Fazer (To Do) | Em Andamento (Doing) | Concluído (Done) |
| :--- | :--- | :--- |
| Higienizar e integrar as bases distintas em um dataset único (US03). | Realizar levantamento de datasets públicos (Dados.gov.br, IPEA) (US01). | Elaboração do Business Model Canvas. |
| Treinar modelos preditivos no Google Colab (US04). | Conduzir análise exploratória (EDA) inicial (US02). | Criação do repositório no GitHub (setup inicial das pastas e arquivos). |
| Desenvolver dashboard MVP em Streamlit (US05). | Atualização do Product Backlog com Histórias de Usuário. | Redação inicial do Artigo Científico (Introdução, Problema, Objetivos e Justificativa). |
| Documentar o código e atualizar o repositório (US06). | | |
