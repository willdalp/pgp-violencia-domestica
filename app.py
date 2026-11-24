import json
import os
import unicodedata

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st

# ============================================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ============================================================
st.set_page_config(
    page_title="SC Segura ML — Previsão de Escalada (Violência Doméstica)",
    page_icon="🚨",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_FEATURES = [
    "municipio_cod",
    "mesoregiao_cod",
    "populacao_feminina",
    "alfabetizacao_total",
    "ocorrencias",
    "taxa_violencia_100k",
    "taxa_violencia_mes_anterior",
    "taxa_violencia_3_meses_atras",
    "taxa_violencia_6_meses_atras",
    "media_3_meses",
    "media_6_meses",
    "desvio_media_3m",
    "delta_ultimo_mes",
    "feminicidios",
    "mes_seno",
    "mes_cosseno",
]

DEFAULT_FEATURE_LABELS = [
    "Município",
    "Mesorregião",
    "População Feminina",
    "Taxa de Alfabetização",
    "Ocorrências (Mês Atual)",
    "Taxa Violência / 100k (Atual)",
    "Taxa Violência / 100k (Mês -1)",
    "Taxa Violência / 100k (Mês -3)",
    "Taxa Violência / 100k (Mês -6)",
    "Média Móvel Taxa (3 meses)",
    "Média Móvel Taxa (6 meses)",
    "Desvio da Média Móvel (3m)",
    "Variação Taxa (Último Mês)",
    "Feminicídios (Mês Atual)",
    "Sazonalidade (Mês Seno)",
    "Sazonalidade (Mês Cosseno)",
]


def normalize_text(text):
    if not isinstance(text, str):
        return "" if pd.isna(text) else str(text)
    return (
        unicodedata.normalize("NFKD", text)
        .encode("ASCII", "ignore")
        .decode("ASCII")
        .strip()
        .upper()
    )


def resolve_path(filename, subfolder):
    candidates = [
        os.path.join(BASE_DIR, subfolder, filename),
        os.path.join(BASE_DIR, filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


# ============================================================
# 2. CARREGAMENTO DOS ARTEFATOS E DADOS (COM CACHE)
# ============================================================
@st.cache_resource
def load_models():
    rf_model = joblib.load(resolve_path("modelo_v2_rf.joblib", "models"))
    le_mun = joblib.load(resolve_path("le_mun.joblib", "models"))
    le_meso = joblib.load(resolve_path("le_meso.joblib", "models"))
    scaler = joblib.load(resolve_path("scaler_v2.joblib", "models"))
    explainer = shap.TreeExplainer(rf_model)

    metrics_path = resolve_path("metrics_sprint4.json", "models")
    metadata = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return rf_model, le_mun, le_meso, scaler, explainer, metadata


@st.cache_data
def load_data():
    csv_path = resolve_path("base_modelo_violencia_domestica_sc.csv", "data")
    df = pd.read_csv(csv_path)
    df["periodo"] = pd.to_datetime(df["periodo"])
    df["municipio_norm"] = df["municipio"].apply(normalize_text)
    if "mesoregiao" not in df.columns:
        df["mesoregiao"] = "Não Informada"
    return df.sort_values(["municipio", "periodo"]).reset_index(drop=True)


@st.cache_data
def build_state_ranking(_df, _rf_model, _le_mun, _le_meso, _scaler, feature_cols):
    ultimo_periodo = _df["periodo"].max()
    df_latest = _df[_df["periodo"] == ultimo_periodo].copy()

    valid_muns = set(_le_mun.classes_)
    valid_mesos = set(_le_meso.classes_)
    df_latest = df_latest[
        df_latest["municipio"].isin(valid_muns)
        & df_latest["mesoregiao"].isin(valid_mesos)
    ].copy()

    df_latest["municipio_cod"] = _le_mun.transform(df_latest["municipio"])
    df_latest["mesoregiao_cod"] = _le_meso.transform(df_latest["mesoregiao"])

    X_raw = df_latest[feature_cols].values
    X_scaled = _scaler.transform(X_raw)

    probs = _rf_model.predict_proba(X_scaled)[:, 1]
    preds = (probs >= 0.5).astype(int)

    df_latest["prob_escalada"] = probs
    df_latest["prob_escalada_pct"] = np.round(probs * 100, 1)
    df_latest["predicao"] = preds
    df_latest["status"] = np.where(
        preds == 1, "Alerta Preventivo", "Cenário Estável"
    )

    return df_latest.sort_values("prob_escalada", ascending=False).reset_index(drop=True), ultimo_periodo


try:
    rf_model, le_mun, le_meso, scaler, explainer, metadata = load_models()
    df_historico = load_data()
except Exception as e:
    st.error(f"Erro ao carregar artefatos do modelo v2 ou base de dados: {e}")
    st.info(
        "Verifique se os arquivos da Sprint 4 (`modelo_v2_rf.joblib`, `le_mun.joblib`, "
        "`le_meso.joblib`, `scaler_v2.joblib`, `metrics_sprint4.json` em `/models` e "
        "`base_modelo_violencia_domestica_sc.csv` em `/data`) foram enviados ao repositório."
    )
    st.stop()

FEATURES = metadata.get("features", DEFAULT_FEATURES)
FEATURE_LABELS = metadata.get("feature_labels", DEFAULT_FEATURE_LABELS)

df_ranking, ultimo_periodo = build_state_ranking(
    df_historico, rf_model, le_mun, le_meso, scaler, FEATURES
)
prob_media_estadual = float(df_ranking["prob_escalada"].mean() * 100)


# ============================================================
# 3. CABEÇALHO E NAVEGAÇÃO EM ABAS
# ============================================================
st.title("🚨 Previsão de Escalada: Violência Doméstica em Santa Catarina")
st.markdown(
    f"Plataforma de apoio à decisão baseada em **Machine Learning (Random Forest v2 + SHAP)** "
    f"sobre dados integrados da SSP/SC e IBGE (**{df_historico['municipio'].nunique()} municípios**, "
    f"base atualizada até **{ultimo_periodo.strftime('%m/%Y')}**)."
)

tab_panorama, tab_municipio, tab_auditoria = st.tabs(
    [
        "🗺️ Panorama Estadual (Ranking)",
        "🔍 Análise Municipal & SHAP",
        "⚙️ Auditoria & Qualidade (CRISP-ML(Q))",
    ]
)


# ============================================================
# ABA 1: PANORAMA ESTADUAL (RANKING DE PRIORIZAÇÃO)
# ============================================================
with tab_panorama:
    st.subheader("Priorização Estadual de Risco para o Mês Subsequente")

    col_f1, col_f2, col_f3 = st.columns([2, 1.5, 1.5])
    with col_f1:
        mesorregioes_disp = sorted(df_ranking["mesoregiao"].unique().tolist())
        meso_filtro = st.multiselect(
            "Filtrar por Mesorregião:",
            options=mesorregioes_disp,
            default=mesorregioes_disp,
        )
    with col_f2:
        limiar_alerta = st.slider(
            "Limiar de Alerta (%):",
            min_value=30,
            max_value=80,
            value=50,
            step=5,
            help="Ajusta o ponto de corte de probabilidade para classificar um município em alerta preventivo.",
        )
    with col_f3:
        status_filtro = st.selectbox(
            "Exibir Status:",
            options=["Todos", "Apenas Alerta Preventivo", "Apenas Cenário Estável"],
        )

    df_view = df_ranking[df_ranking["mesoregiao"].isin(meso_filtro)].copy()
    df_view["status_dinamico"] = np.where(
        df_view["prob_escalada_pct"] >= limiar_alerta,
        "Alerta Preventivo",
        "Cenário Estável",
    )

    if status_filtro == "Apenas Alerta Preventivo":
        df_view = df_view[df_view["status_dinamico"] == "Alerta Preventivo"]
    elif status_filtro == "Apenas Cenário Estável":
        df_view = df_view[df_view["status_dinamico"] == "Cenário Estável"]

    total_muns = len(df_view)
    qtd_alerta = int((df_view["status_dinamico"] == "Alerta Preventivo").sum())
    pct_alerta = (qtd_alerta / total_muns * 100) if total_muns > 0 else 0.0
    media_prob_view = df_view["prob_escalada_pct"].mean() if total_muns > 0 else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Municípios Filtrados", f"{total_muns}")
    k2.metric("Em Alerta Preventivo", f"{qtd_alerta}", f"{pct_alerta:.1f}% do recorte", delta_color="off")
    k3.metric("Probabilidade Média de Escalada", f"{media_prob_view:.1f}%")
    k4.metric("Mês de Referência", ultimo_periodo.strftime("%Y-%m"))

    if df_view.empty:
        st.warning("Nenhum município corresponde aos filtros selecionados.")
    else:
        g1, g2 = st.columns([3, 2])
        with g1:
            top15 = df_view.head(15).sort_values("prob_escalada_pct", ascending=True)
            fig_top = px.bar(
                top15,
                x="prob_escalada_pct",
                y="municipio",
                color="prob_escalada_pct",
                color_continuous_scale="Reds",
                orientation="h",
                hover_data=["mesoregiao", "ocorrencias", "taxa_violencia_100k"],
                labels={
                    "prob_escalada_pct": "Probabilidade de Escalada (%)",
                    "municipio": "Município",
                    "mesoregiao": "Mesorregião",
                    "ocorrencias": "Ocorrências no Mês",
                    "taxa_violencia_100k": "Taxa / 100k Mulheres",
                },
                title="Top 15 Municípios com Maior Probabilidade de Escalada",
            )
            fig_top.update_layout(coloraxis_showscale=False, height=430, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig_top, use_container_width=True)

        with g2:
            resumo_meso = (
                df_view.groupby("mesoregiao", as_index=False)
                .agg(
                    municipios=("municipio", "count"),
                    em_alerta=("status_dinamico", lambda s: (s == "Alerta Preventivo").sum()),
                    prob_media=("prob_escalada_pct", "mean"),
                )
            )
            resumo_meso["pct_alerta"] = np.round(
                resumo_meso["em_alerta"] / resumo_meso["municipios"] * 100, 1
            )
            resumo_meso = resumo_meso.sort_values("pct_alerta", ascending=True)

            fig_meso = px.bar(
                resumo_meso,
                x="pct_alerta",
                y="mesoregiao",
                orientation="h",
                text="pct_alerta",
                color="prob_media",
                color_continuous_scale="YlOrRd",
                labels={
                    "pct_alerta": "% de Municípios em Alerta",
                    "mesoregiao": "Mesorregião",
                    "prob_media": "Prob. Média (%)",
                },
                title="Concentração de Alertas por Mesorregião",
            )
            fig_meso.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_meso.update_layout(height=430, margin=dict(l=10, r=25, t=45, b=10))
            st.plotly_chart(fig_meso, use_container_width=True)

        st.markdown("### 📋 Tabela Completa de Priorização Municipal")
        tabela_export = df_view[
            [
                "municipio",
                "mesoregiao",
                "status_dinamico",
                "prob_escalada_pct",
                "ocorrencias",
                "taxa_violencia_100k",
                "media_3_meses",
                "populacao_feminina",
            ]
        ].rename(
            columns={
                "municipio": "Município",
                "mesoregiao": "Mesorregião",
                "status_dinamico": "Classificação",
                "prob_escalada_pct": "Prob. Escalada (%)",
                "ocorrencias": "Ocorrências (Atual)",
                "taxa_violencia_100k": "Taxa / 100k (Atual)",
                "media_3_meses": "Média Móvel 3m (/100k)",
                "populacao_feminina": "Pop. Feminina",
            }
        )
        tabela_export["Taxa / 100k (Atual)"] = tabela_export["Taxa / 100k (Atual)"].round(1)
        tabela_export["Média Móvel 3m (/100k)"] = tabela_export["Média Móvel 3m (/100k)"].round(1)

        st.dataframe(tabela_export, use_container_width=True, hide_index=True)

        csv_bytes = tabela_export.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 Baixar Ranking Filtrado (CSV)",
            data=csv_bytes,
            file_name=f"ranking_risco_sc_{ultimo_periodo.strftime('%Y_%m')}.csv",
            mime="text/csv",
        )


# ============================================================
# ABA 2: ANÁLISE MUNICIPAL, SIMULADOR WHAT-IF & SHAP
# ============================================================
with tab_municipio:
    st.subheader("Consulta Individual, Simulação de Cenários e Explicabilidade (XAI)")

    municipios_disponiveis = sorted(df_ranking["municipio"].unique().tolist())
    municipio_selecionado = st.selectbox(
        "Selecione o Município Alvo:",
        options=municipios_disponiveis,
        index=municipios_disponiveis.index("CHAPECO") if "CHAPECO" in municipios_disponiveis else 0,
    )

    df_mun_hist = df_historico[df_historico["municipio"] == municipio_selecionado].sort_values("periodo")
    registro_base = df_mun_hist.iloc[-1].copy()
    mesorregiao_real = registro_base["mesoregiao"]
    periodo_ref = registro_base["periodo"]

    st.info(
        f"**{municipio_selecionado}** ({mesorregiao_real}) — Referência mais recente: "
        f"**{periodo_ref.strftime('%m/%Y')}** | Ocorrências registradas: **{int(registro_base['ocorrencias'])}** | "
        f"Taxa: **{registro_base['taxa_violencia_100k']:.1f} casos / 100 mil mulheres** | "
        f"População feminina: **{int(registro_base['populacao_feminina']):,}**."
    )

    # Simulador What-if (Melhoria centrada no gestor)
    usar_simulacao = False
    ocorrencias_sim = int(registro_base["ocorrencias"])
    feminicidios_sim = int(registro_base["feminicidios"])

    with st.expander("🧪 Simulador de Cenário (What-if) — Testar impacto de variações no mês atual"):
        usar_simulacao = st.checkbox(
            "Ativar simulação manual (altera os valores do mês atual e recalcula taxas e SHAP em tempo real)"
        )
        sc1, sc2 = st.columns(2)
        with sc1:
            ocorrencias_sim = st.number_input(
                "Ocorrências simuladas no mês atual:",
                min_value=0,
                max_value=5000,
                value=int(registro_base["ocorrencias"]),
                step=1,
                disabled=not usar_simulacao,
            )
        with sc2:
            feminicidios_sim = st.number_input(
                "Feminicídios simulados no mês atual:",
                min_value=0,
                max_value=50,
                value=int(registro_base["feminicidios"]),
                step=1,
                disabled=not usar_simulacao,
            )

    # Monta vetor de entrada (real ou simulado)
    entrada = registro_base.copy()
    entrada["municipio_cod"] = le_mun.transform([municipio_selecionado])[0]
    entrada["mesoregiao_cod"] = le_meso.transform([mesorregiao_real])[0]

    if usar_simulacao:
        pop_fem = float(entrada["populacao_feminina"])
        taxa_sim = (ocorrencias_sim / pop_fem) * 100000.0 if pop_fem > 0 else 0.0
        entrada["ocorrencias"] = float(ocorrencias_sim)
        entrada["taxa_violencia_100k"] = taxa_sim
        entrada["desvio_media_3m"] = taxa_sim - float(entrada["media_3_meses"])
        entrada["delta_ultimo_mes"] = taxa_sim - float(entrada["taxa_violencia_mes_anterior"])
        entrada["feminicidios"] = float(feminicidios_sim)

    X_input = np.array([[float(entrada[col]) for col in FEATURES]])
    X_scaled = scaler.transform(X_input)

    prob_alta = float(rf_model.predict_proba(X_scaled)[0, 1])
    predicao = 1 if prob_alta >= 0.5 else 0
    prob_pct = prob_alta * 100.0
    diff_estadual = prob_pct - prob_media_estadual

    media_hist_casos = float(df_mun_hist["ocorrencias"].mean())
    casos_atual = int(entrada["ocorrencias"])
    pct_media = ((casos_atual - media_hist_casos) / media_hist_casos * 100) if media_hist_casos > 0 else 0.0

    st.markdown("---")
    res1, res2, res3 = st.columns([2.5, 1.5, 1.5])
    with res1:
        tag_sim = " (CENÁRIO SIMULADO)" if usar_simulacao else ""
        if predicao == 1:
            st.error(f"### 📈 ALERTA PREVENTIVO — {municipio_selecionado}{tag_sim}")
        else:
            st.success(f"### 📉 CENÁRIO ESTÁVEL — {municipio_selecionado}{tag_sim}")
    with res2:
        st.metric(
            label="Probabilidade de Escalada",
            value=f"{prob_pct:.1f}%",
            delta=f"{diff_estadual:+.1f} p.p. vs média SC",
            delta_color="inverse",
        )
    with res3:
        st.metric(
            label="Taxa Atual (/100k mulheres)",
            value=f"{entrada['taxa_violencia_100k']:.1f}",
            delta=f"{entrada['desvio_media_3m']:+.1f} vs média 3m",
            delta_color="inverse",
        )

    # Diagnóstico Textual Automatizado
    if abs(pct_media) < 5:
        rel_media = f"praticamente alinhado à média histórica local ({media_hist_casos:.1f} casos/mês)."
    elif pct_media > 0:
        rel_media = f"**{abs(pct_media):.1f}% acima** da média histórica local ({media_hist_casos:.1f} casos/mês)."
    else:
        rel_media = f"**{abs(pct_media):.1f}% abaixo** da média histórica local ({media_hist_casos:.1f} casos/mês)."

    st.markdown("#### 📄 Diagnóstico Automatizado")
    if predicao == 1:
        st.markdown(
            f"O modelo projeta **tendência de aumento** na taxa de violência doméstica de **{municipio_selecionado}** para o próximo mês.\n"
            f"* **Volume e Taxa Local:** Com **{casos_atual} ocorrências** ({rel_media}), a taxa atual está em "
            f"**{entrada['taxa_violencia_100k']:.1f}/100k** frente à média móvel trimestral de **{entrada['media_3_meses']:.1f}/100k**.\n"
            f"* **Contexto Regional e Sazonal:** A mesorregião **{mesorregiao_real}** combinada ao padrão sazonal do período "
            f"e às defasagens históricas elevou o risco acima do limiar de alerta (50%)."
        )
    else:
        st.markdown(
            f"O modelo projeta **estabilidade ou retração** na taxa de violência doméstica de **{municipio_selecionado}** para o próximo mês.\n"
            f"* **Volume e Taxa Local:** Com **{casos_atual} ocorrências** ({rel_media}), a taxa atual está em "
            f"**{entrada['taxa_violencia_100k']:.1f}/100k** frente à média móvel trimestral de **{entrada['media_3_meses']:.1f}/100k**.\n"
            f"* **Contexto Regional e Sazonal:** Os indicadores históricos recentes e a dinâmica da mesorregião **{mesorregiao_real}** "
            f"não sinalizam escalada adicional para o período subsequente."
        )

    # Explicabilidade SHAP (Waterfall)
    st.markdown("---")
    st.markdown("#### 🧠 Explicabilidade da Decisão (Waterfall SHAP — Top 10 Fatores)")
    st.caption(
        "Barras vermelhas aumentam a probabilidade de escalada; barras azuis reduzem. "
        "E[f(X)] é a probabilidade base do modelo e f(x) é a probabilidade final calculada para este município."
    )

    shap_values = explainer.shap_values(X_scaled)
    if isinstance(shap_values, list):
        shap_val_target = np.array(shap_values[1][0], dtype=float)
        expected_val = float(explainer.expected_value[1])
    elif len(np.shape(shap_values)) == 3:
        shap_val_target = np.array(shap_values[0, :, 1], dtype=float)
        expected_val = float(explainer.expected_value[1])
    else:
        shap_val_target = np.array(shap_values[0], dtype=float)
        expected_val = float(
            explainer.expected_value[0]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        )

    # Evita zero negativo (-0.0) por precisão de ponto flutuante
    shap_val_target = np.where(np.abs(shap_val_target) < 1e-6, 0.0, shap_val_target)

    fig_shap, _ = plt.subplots(figsize=(9, 4.8))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_val_target,
            base_values=expected_val,
            data=np.round(X_input[0], 2),
            feature_names=FEATURE_LABELS,
        ),
        max_display=10,
        show=False,
    )
    plt.tight_layout()
    st.pyplot(fig_shap)
    plt.close(fig_shap)

    # Série Histórica Interativa (Plotly)
    st.markdown("---")
    st.markdown("#### 📈 Contexto Histórico Local (2020–2026)")
    metrica_graf = st.radio(
        "Métrica da Série Temporal:",
        options=["Ocorrências Absolutas", "Taxa por 100 mil Mulheres"],
        horizontal=True,
    )
    col_y = "ocorrencias" if metrica_graf == "Ocorrências Absolutas" else "taxa_violencia_100k"
    media_y = float(df_mun_hist[col_y].mean())

    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Scatter(
            x=df_mun_hist["periodo"],
            y=df_mun_hist[col_y],
            mode="lines+markers",
            name=metrica_graf,
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=5),
        )
    )
    fig_hist.add_hline(
        y=media_y,
        line_dash="dash",
        line_color="#d62728",
        annotation_text=f"Média Histórica: {media_y:.1f}",
        annotation_position="top right",
    )
    fig_hist.update_layout(
        title=f"Série Temporal — {municipio_selecionado} ({metrica_graf})",
        xaxis_title="Período",
        yaxis_title=metrica_graf,
        hovermode="x unified",
        height=380,
        margin=dict(l=10, r=10, t=45, b=10),
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# ============================================================
# ABA 3: AUDITORIA E QUALIDADE DO MODELO (CRISP-ML(Q))
# ============================================================
with tab_auditoria:
    st.subheader("Auditoria Técnica, Evidência de Evolução (v1 → v2) e Governança CRISP-ML(Q)")

    if not metadata or "benchmark" not in metadata:
        st.warning("Arquivo `metrics_sprint4.json` não encontrado em `/models`.")
    else:
        bench = pd.DataFrame(metadata["benchmark"])
        v1_rf = bench[bench["modelo"] == "v1_Random_Forest_Baseline"].iloc[0]
        v2_rf = bench[bench["modelo"] == "v2_Random_Forest_Tunado"].iloc[0]

        st.markdown("#### 🚀 Impacto Quantitativo da Evolução")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Acurácia (Split Temporal)",
            f"{v2_rf['acuracia']*100:.2f}%",
            f"{(v2_rf['acuracia'] - v1_rf['acuracia'])*100:+.2f} p.p.",
        )
        m2.metric(
            "Recall (Sensibilidade)",
            f"{v2_rf['recall']*100:.2f}%",
            f"{(v2_rf['recall'] - v1_rf['recall'])*100:+.2f} p.p.",
        )
        m3.metric(
            "F1-Score (Classe Positiva)",
            f"{v2_rf['f1_score']:.4f}",
            f"{(v2_rf['f1_score'] - v1_rf['f1_score']):+.4f}",
        )
        m4.metric(
            "ROC-AUC",
            f"{v2_rf['roc_auc']:.4f}",
            f"{(v2_rf['roc_auc'] - v1_rf['roc_auc']):+.4f}",
        )

        st.markdown("---")
        c_bench, c_cm = st.columns([3, 2])

        with c_bench:
            st.markdown("#### 📊 Benchmark Completo (Validação Temporal Estrita)")
            bench_plot = bench[
                bench["modelo"].isin(["v1_Random_Forest_Baseline", "v2_Random_Forest_Tunado"])
            ].melt(
                id_vars="modelo",
                value_vars=["acuracia", "precisao", "recall", "f1_score", "roc_auc"],
                var_name="Métrica",
                value_name="Valor",
            )
            bench_plot["modelo"] = bench_plot["modelo"].replace(
                {
                    "v1_Random_Forest_Baseline": "v1 Baseline (Sprint 2 — 3 features)",
                    "v2_Random_Forest_Tunado": "v2 Evolução (Sprint 4 — 16 features)",
                }
            )
            fig_comp = px.bar(
                bench_plot,
                x="Métrica",
                y="Valor",
                color="modelo",
                barmode="group",
                text_auto=".3f",
                color_discrete_sequence=["#95a5a6", "#1f77b4"],
                title="Evolução do Random Forest: Baseline v1 vs. Modelo v2",
            )
            fig_comp.update_layout(yaxis_range=[0, 1], height=370, legend_title="")
            st.plotly_chart(fig_comp, use_container_width=True)

        with c_cm:
            st.markdown("#### 🎯 Matriz de Confusão (Produção v2)")
            cm = np.array(v2_rf["matriz_confusao"])
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                color_continuous_scale="Blues",
                x=["Previsto: Estável (0)", "Previsto: Escalada (1)"],
                y=["Real: Estável (0)", "Real: Escalada (1)"],
                title=f"Partição de Teste ({metadata.get('n_registros_teste', 0):,} amostras)",
            )
            fig_cm.update_layout(height=370, coloraxis_showscale=False)
            st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("#### 📋 Tabela Comparativa de Todos os Modelos Avaliados")
        tabela_bench = bench[["modelo", "acuracia", "precisao", "recall", "f1_score", "roc_auc"]].rename(
            columns={
                "modelo": "Modelo",
                "acuracia": "Acurácia",
                "precisao": "Precisão",
                "recall": "Recall",
                "f1_score": "F1-Score",
                "roc_auc": "ROC-AUC",
            }
        )
        st.dataframe(tabela_bench, use_container_width=True, hide_index=True)

        # Importância Global das Variáveis
        if "feature_importances" in metadata:
            st.markdown("#### 🔎 Importância Global das Variáveis (Feature Importance")
            df_imp = pd.DataFrame(
                list(metadata["feature_importances"].items()),
                columns=["Atributo", "Importância"],
            ).sort_values("Importância", ascending=True)
            fig_imp = px.bar(
                df_imp,
                x="Importância",
                y="Atributo",
                orientation="h",
                text_auto=".3f",
                color="Importância",
                color_continuous_scale="Teal",
            )
            fig_imp.update_layout(height=460, coloraxis_showscale=False, margin=dict(l=10, r=20, t=20, b=10))
            st.plotly_chart(fig_imp, use_container_width=True)

        # Bloco de Governança CRISP-ML(Q)
        st.markdown("#### 🛡️ Governança e Monitoramento (CRISP-ML(Q))")
        p_tr = metadata.get("periodo_treino", ["N/A", "N/A"])
        p_te = metadata.get("periodo_teste", ["N/A", "N/A"])
        st.success(
            f"**Versão em Produção:** `{metadata.get('versao', 'v2.0-sprint4')}` (`{metadata.get('modelo_producao')}`) | "
            f"**Treino:** {p_tr[0]} a {p_tr[1]} ({metadata.get('n_registros_treino', 0):,} registros) | "
            f"**Teste Temporal:** {p_te[0]} a {p_te[1]} ({metadata.get('n_registros_teste', 0):,} registros) | "
            f"**Cobertura:** {metadata.get('n_municipios', 295)} municípios de SC."
        )
