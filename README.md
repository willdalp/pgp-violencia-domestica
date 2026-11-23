# Escalada da Violência Doméstica em SC: Previsão e Apoio à Decisão com Machine Learning

**Equipe:** José Vítor Tessaro de Bortoli e Willian Z. Dal Pont  
**Disciplina:** Planejamento e Gestão de Projetos (GCS817) — UFFS Campus Chapecó  
**Aplicação Publicada (Streamlit Cloud):** [Acessar SC Segura ML](https://pgp-violencia-domestica-9g82dfx3xjfhjgnqdog67.streamlit.app)

---

## Sobre o Projeto
Solução tecnológica ponta a ponta aplicando **Machine Learning** e **Inteligência Artificial Explicável (XAI/SHAP)** sobre bases públicas da Secretaria de Estado da Segurança Pública de Santa Catarina (SSP/SC) integradas a indicadores demográficos e socioeconômicos do IBGE. A ferramenta estima mensalmente a tendência de escalada da violência doméstica nos **295 municípios catarinenses**, apoiando gestores públicos na alocação preventiva de recursos.

---

## Evolução da Sprint 4 (`v0.2.0-sprint4`)
Com base na validação Lean Startup do MVP da Sprint 3 e nas diretrizes do framework **CRISP-ML(Q)**, a solução evoluiu em três frentes:

1. **Modelo Preditivo v2 (16 atributos integrados + validação temporal estrita):** incorporação de taxas por 100 mil mulheres, médias móveis (3 e 6 meses), desvios locais, sazonalidade cíclica (seno/cosseno), feminicídios, população feminina e alfabetização, eliminando *data leakage* com corte temporal em `2024-09-01` (14.750 amostras de treino e 6.490 de teste).
2. **Painel de Decisão em 3 Abas:**
   * **Panorama Estadual (Ranking):** priorização dos 295 municípios com filtro por mesorregião, limiar de alerta ajustável, gráfico regional e exportação em CSV.
   * **Análise Municipal & SHAP:** ingestão automática do último registro municipal, simulador de cenários (*What-if*), gráfico Waterfall SHAP (Top 10 fatores) e série histórica interativa (Plotly).
   * **Auditoria & Qualidade (CRISP-ML(Q)):** comparativo quantitativo v1 → v2, matriz de confusão e importância global das variáveis.
3. **Garantia de Qualidade e Gestão:** suíte de testes automatizados (`/tests`), revisão do Business Model Canvas (`v2`) e documentação da validação Lean Startup em `/docs`.

### Evidência de Impacto: Benchmark v1 vs. v2 (Split Temporal)

| Modelo | Acurácia | Precisão | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `v1_Regressao_Logistica_Baseline` (3 features) | 55,10% | 0,00% | 0,00% | 0,0000 | 0,5101 |
| `v1_Random_Forest_Baseline` (3 features) | 66,39% | 64,50% | 55,94% | 0,5992 | 0,7090 |
| `v2_Regressao_Logistica_Integrada` (16 features) | 70,22% | 67,55% | 64,79% | 0,6614 | 0,7641 |
| **`v2_Random_Forest_Tunado` (Produção)** | **71,06%** | **67,10%** | **69,77%** | **0,6841** | **0,7788** |
| `v2_Gradient_Boosting_Integrado` (16 features) | 71,57% | 69,19% | 66,13% | 0,6763 | 0,7863 |

*Ganho do modelo em produção (RF v2 vs. RF v1):* **+4,67 p.p.** em Acurácia, **+13,83 p.p.** em Recall, **+0,0849** em F1-Score e **+0,0698** em ROC-AUC.

---

## Estrutura do Repositório
```text
pgp-violencia-domestica/
├── .github/workflows/   # CI automatizado (pytest)
├── article/             # Artigo científico (SBC / LaTeX)
├── data/                # Base consolidada (21.240 registros, 295 municípios)
├── docs/                # Business Model Canvas (v1 e v2) e Validação Lean Startup
├── models/              # Artefatos v2 (.joblib) e auditoria (metrics_sprint4.json)
├── notebooks/           # Notebook do Google Colab (pré-processamento e modelagem)
├── src/                 # Scripts auxiliares
├── tests/               # Testes unitários, de integração e de aceite (pytest)
├── app.py               # Aplicação principal publicada no Streamlit Cloud
├── requirements.txt     # Dependências de execução
└── LICENSE
