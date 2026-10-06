import streamlit as st
import joblib
import numpy as np
import pandas as pd
import unicodedata
import shap
import matplotlib.pyplot as plt

# 1. Configuração da Página
st.set_page_config(page_title="MVP - Violência SC", page_icon="🚨", layout="centered")

st.title("🚨 Previsão de Escalada: Violência Doméstica (SC)")
st.markdown("""
Este MVP utiliza **Machine Learning (Random Forest)** para prever a tendência de aumento das 
ocorrências de violência doméstica nos municípios de Santa Catarina para o mês subsequente.
""")

# Função auxiliar para remover acentos e cedilha, padronizando a busca
def normalize_text(text):
    if not isinstance(text, str):
        return text
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')

# 2. Carregamento dos Modelos e da Base de Dados (Cache)
@st.cache_resource
def load_models():
    rf_model = joblib.load('models/modelo_baseline_rf.joblib')
    le_mun = joblib.load('models/le_mun.joblib')
    le_meso = joblib.load('models/le_meso.joblib')
    scaler = joblib.load('models/scaler.joblib')
    
    # Prepara o explicador SHAP
    explainer = shap.TreeExplainer(rf_model)
    return rf_model, le_mun, le_meso, scaler, explainer

@st.cache_data
def load_data():
    return pd.read_csv('data/base_modelo_violencia_domestica_sc.csv')

try:
    rf_model, le_mun, le_meso, scaler, explainer = load_models()
    df_historico = load_data()
except Exception as e:
    st.error(f"Erro de infraestrutura ou carregamento de arquivos: Detalhes: {e}")
    st.stop()

st.markdown("---")
st.subheader("📊 Entrada de Variáveis Preditivas")

# 3. Interface de Usuário (Apenas Município)
mapa_municipios = {normalize_text(m): m for m in le_mun.classes_}
municipios_disponiveis = sorted(list(mapa_municipios.keys()))

municipio_selecionado_norm = st.selectbox("Município Alvo:", municipios_disponiveis)
municipio_selecionado = mapa_municipios[municipio_selecionado_norm]

df_historico['municipio_norm'] = df_historico['municipio'].apply(normalize_text)
df_mun_atual = df_historico[df_historico['municipio_norm'] == municipio_selecionado_norm]

# --- Lógica Automática para Mesorregião ---
mapa_meso = {normalize_text(m): m for m in le_meso.classes_}

if 'mesoregiao' in df_historico.columns and not df_mun_atual.empty:
    meso_csv_norm = normalize_text(df_mun_atual['mesoregiao'].iloc[0])
    mesoregiao = mapa_meso.get(meso_csv_norm, le_meso.classes_[0])
else:
    mesoregiao = le_meso.classes_[0]

# --- Lógica Automática para Período e Ocorrências ---
if not df_mun_atual.empty:
    periodo_selecionado = df_mun_atual['periodo'].max()
    dado_filtrado = df_mun_atual[df_mun_atual['periodo'] == periodo_selecionado]
    col_ocorrencias = 'ocorrencias' if 'ocorrencias' in df_mun_atual.columns else df_mun_atual.columns[2]
    casos_mes_anterior = int(dado_filtrado[col_ocorrencias].values[0]) if not dado_filtrado.empty else 10
    
    st.info(f"Base para predição: Mês mais recente registrado (**{periodo_selecionado}**). "
            f"Ocorrências registradas em **{municipio_selecionado}**: **{casos_mes_anterior}** casos.")
else:
    periodo_selecionado = "N/A"
    casos_mes_anterior = 10
    st.warning(f"Não foram encontrados dados históricos suficientes para **{municipio_selecionado}**. Utilizando valor padrão de 10 ocorrências para a predição.")

# 4. Execução do Algoritmo e Resultados
if st.button("Executar Algoritmo de Predição", type="primary"):
    with st.spinner("Processando dados e consultando o modelo Random Forest..."):
        
        mun_cod = le_mun.transform([municipio_selecionado])[0]
        meso_cod = le_meso.transform([mesoregiao])[0]
        
        X_input = np.array([[mun_cod, meso_cod, casos_mes_anterior]])
        X_scaled = scaler.transform(X_input)
        
        predicao = rf_model.predict(X_scaled)[0]
        probabilidades = rf_model.predict_proba(X_scaled)[0]
        confianca = np.max(probabilidades) * 100
        
        # --- CÁLCULOS ESTATÍSTICOS ---
        if not df_mun_atual.empty and len(df_mun_atual) >= 2:
            df_ord = df_mun_atual.sort_values('periodo')
            media_historica = df_ord[col_ocorrencias].mean()
            casos_retrasado = df_ord[col_ocorrencias].iloc[-2] if len(df_ord) > 1 else casos_mes_anterior
            
            var_mensal = ((casos_mes_anterior - casos_retrasado) / casos_retrasado * 100) if casos_retrasado > 0 else 0.0
            var_media = ((casos_mes_anterior - media_historica) / media_historica * 100) if media_historica > 0 else 0.0
        else:
            media_historica = casos_mes_anterior
            var_mensal = 0.0
            var_media = 0.0

        st.markdown("---")
        st.subheader("🔍 Resultado da Avaliação Preditiva")
        
        res_col1, res_col2 = st.columns([3, 2])
        with res_col1:
            if predicao == 1:
                st.error(f"**ALERTA PREVENTIVO PARA {municipio_selecionado.upper()}** 📈")
            else:
                st.success(f"**CENÁRIO ESTÁVEL PARA {municipio_selecionado.upper()}** 📉")
                
        with res_col2:
            cor_delta = "inverse" if predicao == 1 else "normal"
            indicador_delta = "Risco de Escalada" if predicao == 1 else "Risco Controlado"
            st.metric(label="Grau de Confiança (Modelo)", value=f"{confianca:.1f}%", delta=indicador_delta, delta_color=cor_delta)

        # --- DIAGNÓSTICO SIMPLIFICADO ---
        st.markdown("### 📄 Diagnóstico Simplificado")
        
        if predicao == 1:
            st.markdown(f"""
            O modelo projeta uma **tendência de alta** no volume de ocorrências para **{municipio_selecionado}** no próximo período.

            * **Último registro:** {casos_mes_anterior} casos.
            * **Média histórica local:** {media_historica:.1f} casos/mês.
            * **Contexto regional:** A mesorregião **{mesoregiao}** apresenta padrões que elevam a probabilidade de risco predita pelo modelo.
            """)
        else:
            st.markdown(f"""
            O modelo projeta **estabilidade ou redução** no volume de ocorrências para **{municipio_selecionado}** no próximo período.

            * **Último registro:** {casos_mes_anterior} casos.
            * **Média histórica local:** {media_historica:.1f} casos/mês.
            * **Contexto regional:** O histórico recente de **{mesoregiao}** mantém a estimativa em patamares dentro do esperado.
            """)
