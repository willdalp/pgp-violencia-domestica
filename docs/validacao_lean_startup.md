# Validação Lean Startup — Ciclo Construir–Medir–Aprender (Sprint 4)

**Projeto:** Análise Preditiva e Mapeamento da Escalada da Violência Doméstica em Santa Catarina  
**Versão Validada:** `v0.2.0-sprint4` (comparada ao MVP `v0.1.0-sprint3`)  
**Aplicação Publicada:** [SC Segura ML (Streamlit Cloud)](https://pgp-violencia-domestica-9g82dfx3xjfhjgnqdog67.streamlit.app)

---

## 1. Estratégia de Validação (Lean Startup + CRISP-ML(Q))

Seguindo o ciclo **Construir–Medir–Aprender** (*Lean Startup*) alinhado à fase de avaliação e garantia de qualidade do **CRISP-ML(Q)**, a validação da Sprint 4 foi conduzida de forma rigorosa e reproduzível em duas camadas complementares:

1. **Validação Empírica de Hipóteses (*Backtesting* Temporal Estrito e Testes de Aceite):** verificação quantitativa do modelo `v2` frente ao baseline `v1` em dados futuros inéditos (*out-of-time*), sem vazamento temporal (*data leakage*), somada à suíte automatizada de testes em Integração Contínua (CI).
2. **Avaliação Heurística e *Cognitive Walkthrough* (Usabilidade v1 → v2):** inspeção técnica estruturada sob as Heurísticas de Nielsen e simulação de tarefas operacionais reais de um gestor de segurança pública, comparando objetivamente o MVP da Sprint 3 (`v1`) com a evolução da Sprint 4 (`v2`).

---

## 2. Validação Empírica de Hipóteses (Dados e Modelo)

| ID | Hipótese Testada | Critério de Sucesso (Definido *a priori*) | Evidência Medida (Split Temporal: 09/2024–06/2026, $n = 6.490$) | Resultado |
| :--- | :--- | :--- | :--- | :---: |
| **H1 (Valor Preditivo)** | Integrar atributos temporais (médias móveis, desvios, defasagens, sazonalidade) e demográficos (IBGE) melhora substancialmente a previsão frente ao baseline de 3 variáveis. | Ganho simultâneo em Acurácia, Recall, F1-Score e ROC-AUC, atingindo Acurácia $\ge 70\%$ e F1 $\ge 0{,}65$. | **RF v1 → RF v2:** Acurácia `66,39% → 71,06%` (+4,67 p.p.); Recall `55,94% → 69,77%` (+13,83 p.p.); F1 `0,5992 → 0,6841` (+0,0849); ROC-AUC `0,7090 → 0,7788` (+0,0698). | **Validada** |
| **H2 (Redução de Falsos Negativos)** | Regularizar o Random Forest com `class_weight="balanced"` eleva a sensibilidade (Recall) para segurança pública sem degradar a precisão. | Recall $\ge 65\%$ mantendo Precisão acima do baseline (`64,50%`). | Recall atingiu `69,77%` (`2.033` escaladas reais detectadas vs. `881` falsos negativos) com Precisão de `67,10%` (+2,60 p.p. sobre v1), alcançando o maior F1-Score (`0,6841`) entre os 5 modelos avaliados. | **Validada** |
| **H3 (Sinal de Feminicídios Mensais)** | A contagem mensal municipal de feminicídios contribui diretamente para o poder discriminativo do classificador. | Importância global (Feature Importance) $\ge 0{,}01$. | Importância global medida: `0,000` (devido à alta esparsidade mensal: `362` ocorrências distribuídas em `21.240` registros município × mês). | **Refutada** |
| **H4 (Confiabilidade em Produção)** | A suíte automatizada de testes (unitários, integração e aceite) assegura integridade dos 295 municípios, das 6 mesorregiões e das explicações SHAP. | 100% dos testes aprovados automaticamente em CI (GitHub Actions). | 5/5 testes aprovados em CI, validando cobertura dos 295 municípios, 6 mesorregiões oficiais, ausência de valores nulos, inferência ponta a ponta e *quality gates* da v2. | **Validada** |

---

## 3. Avaliação Heurística e *Cognitive Walkthrough* (MVP v1 → v2)

### 3.1. Inspeção pelas Heurísticas de Nielsen

| Heurística | Gargalo Detectado no MVP v1 (Sprint 3) | Melhoria Implementada na v2 (Sprint 4) | Impacto Verificado |
| :--- | :--- | :--- | :--- |
| **Prevenção de Erros** | Mapeamento manual (`MAPA_MESORREGIOES`) sujeito a falhas de grafia e fallback genérico para "Oeste Catarinense"; dependência de entradas manuais no fluxo inicial. | Ingestão direta da mesorregião oficial da base SSP/SC (`6` mesorregiões, `0` registros sem classificação) e carregamento automático do mês mais recente (`06/2026`). | Eliminação de erros de classificação regional e de preenchimento manual pelo operador. |
| **Flexibilidade e Eficiência de Uso** | Consulta restrita a um município por vez; impossibilidade de identificar rapidamente áreas prioritárias no estado. | Criação da aba **Panorama Estadual (Ranking)** cobrindo os 295 municípios com filtro por mesorregião, limiar de alerta configurável (30%–80%) e exportação CSV. | Redução da triagem regional de dezenas de consultas isoladas para uma visão consolidada em poucos cliques. |
| **Visibilidade do Status do Sistema** | Bloco de auditoria exibia apenas mensagem estática (`st.success`), sem métricas reais; indicador delta exibia seta ascendente em "Risco Controlado". | Criação da aba **Auditoria & Qualidade (CRISP-ML(Q))** alimentada por `metrics_sprint4.json` (benchmark, matriz de confusão e importância global) e calibração do delta frente à média estadual (`56,5%`). | Transparência quantitativa contínua sobre desempenho, erros e cobertura temporal da base (`2020-07` a `2026-06`). |
| **Correspondência com o Mundo Real** | Gráfico Waterfall SHAP restrito a 3 variáveis e com artefato visual `f(x) = -0`; série histórica estática sem linha de referência. | Waterfall SHAP com Top 10 das 16 variáveis integradas, prevenção de `-0`, série interativa Plotly (ocorrências absolutas e taxa/100k) com linha de média histórica e simulador *What-if*. | Explicabilidade alinhada à análise real de risco (ex.: Chapecó em `06/2026` com `229` casos, taxa `178,7/100k` e alerta de `61,0%`, detalhado por população, desvio da média móvel e sazonalidade). |

### 3.2. *Cognitive Walkthrough* — Simulação de Tarefas Operacionais

| Cenário Operacional (Gestor SSP/SC) | Fluxo na v1 (Sprint 3) | Fluxo na v2 (Sprint 4) | Ganho Mensurado |
| :--- | :--- | :--- | :--- |
| **T1:** Identificar os municípios em alerta no Norte Catarinense e exportar lista para planejamento. | Inviável (exigia consultar manualmente cada município da região individualmente). | Aba 1 → Filtrar *Norte Catarinense* → Filtrar *Alerta Preventivo* → *Baixar CSV* (3 ações). | Fluxo antes inviável executado em < 15 segundos, revelando concentração regional (>88% dos municípios em alerta em 06/2026). |
| **T2:** Investigar os fatores de risco em Chapecó e simular o efeito de uma redução de ocorrências. | Exibia apenas 3 variáveis no SHAP e não suportava simulação de cenários. | Aba 2 → Selecionar *CHAPECO* → Expandir *Simulador What-if* → Recálculo instantâneo de taxa, probabilidade e SHAP. | Suporte direto a análises de sensibilidade para ações preventivas locais. |
| **T3:** Verificar se a nova versão do modelo reduziu falsos negativos frente ao baseline. | Indisponível na aplicação. | Aba 3 → Consulta direta ao comparativo v1→v2 (`+13,83 p.p.` em Recall) e à matriz de confusão (`881` FN vs. `2.033` TP). | Governança de qualidade acessível no próprio painel de decisão. |

---

## 4. Decisões Estratégicas: Perseverar vs. Pivotar/Ajustar

| Eixo do Produto | Decisão | Evidência que Embasou a Decisão |
| :--- | :---: | :--- |
| **Arquitetura Random Forest + XAI (SHAP)** | **PERSEVERAR** | Atingiu o maior F1-Score (`0,6841`) e maior Recall (`69,77%`) no split temporal, mantendo explicabilidade nativa em escala de probabilidade via `TreeExplainer`. |
| **Engenharia de Atributos Temporais e Taxa/100k** | **PERSEVERAR** | Os 3 atributos de maior peso no modelo v2 são *Desvio da Média Móvel (3m)*, *Taxa/100k (Atual)* e *Variação da Taxa (Último Mês)*, impulsionando o salto de `+13,83 p.p.` em Recall. |
| **Modelo de Interação (Consulta Unitária → Panorama Estadual + *What-if*)** | **PIVOTAR / EVOLUIR** | A inspeção heurística e o *Cognitive Walkthrough* (T1 e T2) mostraram que a consulta isolada não atendia à priorização estadual; a arquitetura em 3 abas com ranking e exportação CSV corrigiu o gargalo. |
| **Variável de Feminicídios Mensais** | **DESCARTAR / REFORMULAR (Sprint 5)** | Hipótese **H3 refutada** (`importância = 0,000`). Reformular na Sprint 5 para indicadores acumulados em janelas longas (12 meses) ou agregados por mesorregião. |
| **Validação de Campo com Gestores Públicos** | **ENCAMINHAR (Limitação / Trabalho Futuro)** | Como esta sprint concentrou-se em validação empírica e inspeção heurística interna, a avaliação de usabilidade em campo com operadores da SSP/SC fica registrada como limitação formal do estudo e próxima etapa de extensão. |
