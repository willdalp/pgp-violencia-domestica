# Escalada da Violência Doméstica: Previsão e Apoio à Decisão com Machine Learning

**Equipe:** José de Bortoli e Willian Z. Dal Pont  
**Disciplina:** Planejamento e Gestão de Projetos - UFFS Campus Chapecó  
**Aplicação Pública (Streamlit):** [Acessar MVP Online](https://pgp-violencia-domestica-9g82dfx3xjfjhjgnqdog67.streamlit.app)

## Sobre o Projeto
Este projeto tem como objetivo desenvolver uma solução tecnológica completa utilizando Machine Learning aplicado aos dados de Segurança Pública Brasileira. Focaremos em analisar e prever a escalada da violência doméstica, fornecendo uma aplicação interativa (dashboard/mapas) para auxiliar gestores de segurança na tomada de decisão e alocação de recursos.

## 🚀 Status Atual: Evolução na Sprint 4
A partir da coleta de feedback de uso do MVP e da revisão do modelo de negócio, foram implementadas as seguintes melhorias:
* **Simplificação de UX e Ingestão Automática:** Substituição do preenchimento manual complexo de variáveis pela seleção direta do município alvo, recuperando automaticamente o período mais recente registrado na base de dados (`app.py` vs. `app-old.py`).
* **Diagnóstico Textual Automatizado:** Inclusão de uma seção de diagnóstico executivo que compara o volume recente de ocorrências com a média histórica local do município e destaca a influência da respectiva mesorregião.
* **Explicabilidade Avançada (Waterfall SHAP):** Adição do gráfico *Waterfall SHAP*, detalhando passo a passo como cada variável preditiva eleva ou reduz a probabilidade de escalada em relação à média histórica base ($E[f(X)]$), acompanhado da série temporal completa de ocorrências locais.
* **Revisão do Modelo de Negócio e Expansão de Dados:** Atualização do Business Model Canvas (`docs/Business Model Canvas 2.png`), evolução do artigo científico com os capítulos de Resultados, Discussão e Limitações (`/article`) e estruturação de novas bases de suporte em `/data` (`base_populacao.xlsx`, `base_calendario_feriados.xlsx`, `municipios_sc.json`, `feminicidio.parquet` e `geral.parquet`).

## ⚙️ Entrega Anterior: Consolidação do MVP (Sprint 3)
Nesta etapa anterior, o painel interativo foi estruturado com as seguintes funcionalidades base:
* **Leitura Dinâmica de Dados Reais:** A aplicação consome diretamente o arquivo `base_modelo_violencia_domestica_sc.csv`, extraindo automaticamente mesorregiões, meses disponíveis e totais de ocorrências.
* **Padronização de Texto:** Tratamento automatizado de strings com a biblioteca `unicodedata` para alinhar as entradas do usuário com os codificadores (`LabelEncoder`) do modelo.
* **Interpretabilidade e Histórico:** Exibição do peso das variáveis (Feature Importance) na decisão do Random Forest e plotagem gráfica de toda a série histórica do município selecionado.

## Estrutura do Repositório
* `app.py` / `app-old.py`: Código-fonte atualizado da aplicação publicada no Streamlit e registro da versão anterior para comparação de evolução.
* `/article`: Artigo científico construído de forma incremental (formato SBC).
* `/data`: Dados brutos e processados utilizados no projeto (inclui o dataset histórico CSV, bases Parquet, população, feriados e malha municipal).
* `/docs`: Documentações complementares (Business Model Canvas atualizado).
* `/models`: Modelos de ML treinados e exportados (arquivos `.joblib`).
* `/notebooks`: Notebooks do Google Colab para experimentação e pré-processamento.
* `/src`: Scripts auxiliares e de processamento.
* `/tests`: Testes unitários e de integração.

## Ferramentas Utilizadas
* **Experimentação, Modelagem e XAI:** Python, Google Colab, Pandas, NumPy, Scikit-Learn, SHAP, Matplotlib.
* **Interface e Deploy:** Streamlit Community Cloud.
* **Versionamento, Documentação e Gestão:** GitHub, GitHub Projects (Kanban), Overleaf (LaTeX).

## 💻 Como Executar Localmente

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/willdalp/pgp-violencia-domestica.git
   cd pgp-violencia-domestica
   ```

2. **Instale as dependências e execute a aplicação:**
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
