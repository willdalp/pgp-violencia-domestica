# Escalada da Violência Doméstica: Previsão e Apoio à Decisão com Machine Learning

**Equipe:** José de Bortoli e Willian Z. Dal Pont  
**Disciplina:** Planejamento e Gestão de Projetos - UFFS Campus Chapecó  

## Sobre o Projeto
Este projeto tem como objetivo desenvolver uma solução tecnológica completa utilizando Machine Learning aplicado aos dados de Segurança Pública Brasileira. Focaremos em analisar e prever a escalada da violência doméstica, fornecendo uma aplicação interativa (dashboard/mapas) para auxiliar gestores de segurança na tomada de decisão e alocação de recursos.

## ⚙️ Status Atual: MVP e Sprint 3
Nesta etapa, o painel interativo foi consolidado com as seguintes funcionalidades:
* **Leitura Dinâmica de Dados Reais:** A aplicação consome diretamente o arquivo `base_modelo_violencia_domestica_sc.csv`, extraindo automaticamente mesorregiões, meses disponíveis e totais de ocorrências.
* **Padronização de Texto:** Tratamento automatizado de strings com a biblioteca `unicodedata` para alinhar as entradas do usuário com os codificadores (`LabelEncoder`) do modelo.
* **Interpretabilidade e Histórico:** Exibição do peso das variáveis (Feature Importance) na decisão do Random Forest e plotagem gráfica de toda a série histórica do município selecionado.

## Estrutura do Repositório
* `/app`: Código-fonte da aplicação final publicada no Streamlit. *(Nota: se o seu app.py estiver na raiz, você pode remover ou ajustar este diretório)*.
* `/article`: Artigo científico construído de forma incremental.
* `/data`: Dados brutos e processados utilizados no projeto (inclui o dataset histórico CSV).
* `/docs`: Documentações complementares (Canvas, diagramas).
* `/models`: Modelos de ML treinados e exportados (arquivos `.joblib`).
* `/notebooks`: Notebooks do Google Colab para experimentação e pré-processamento.
* `/src`: Scripts auxiliares e de processamento.
* `/tests`: Testes unitários e de integração.

## Ferramentas Utilizadas
* **Experimentação e Modelagem:** Python, Google Colab, Pandas, NumPy, Scikit-Learn.
* **Interface e Deploy:** Streamlit Community Cloud.
* **Versionamento e Gestão:** GitHub, GitHub Projects (Kanban).

## 🚀 Como Executar Localmente

1. **Clone o repositório:**
   ```bash
   git clone [https://github.com/willdalp/pgp-violencia-domestica.git](https://github.com/willdalp/pgp-violencia-domestica.git)
   cd pgp-violencia-domestica
