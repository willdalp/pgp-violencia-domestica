import streamlit as st
import joblib
import numpy as np

# Configuração da Página
st.set_page_config(page_title="MVP - Violência SC", page_icon="🚨", layout="centered")

st.title("🚨 Previsão de Escalada: Violência Doméstica (SC)")
st.markdown("""
Este MVP utiliza **Machine Learning (Random Forest)** para prever a tendência de aumento das 
ocorrências de violência doméstica nos municípios de Santa Catarina para o mês subsequente.
""")

# Carregamento do Modelo e Transformadores 
@st.cache_resource
def load_models():
    # Agora puxando o nome exato que está no seu GitHub!
    rf_model = joblib.load('models/modelo_baseline_rf.joblib')
    le_mun = joblib.load('models/le_mun.joblib')
    le_meso = joblib.load('models/le_meso.joblib')
    scaler = joblib.load('models/scaler.joblib')
    return rf_model, le_mun, le_meso, scaler

try:
    rf_model, le_mun, le_meso, scaler = load_models()
except Exception as e:
    st.error(f"Erro de infraestrutura: Não foi possível carregar os modelos. Detalhes: {e}")
    st.stop()

st.markdown("---")
st.subheader("📊 Entrada de Variáveis Preditivas")

# Interface de Usuário 
col1, col2 = st.columns(2)
with col1:
    municipio = st.selectbox("Município Alvo:", le_mun.classes_)
with col2:
    mesoregiao = st.selectbox("Mesorregião de Pertencimento:", le_meso.classes_)

casos_mes_anterior = st.number_input("Total de Ocorrências no Mês Anterior:", min_value=0, value=10, step=1)

# Botão de Execução
if st.button("Executar Algoritmo de Predição", type="primary"):
    with st.spinner("Processando dados e consultando o modelo Random Forest..."):
        # Transformando entradas textuais em variáveis numéricas
        mun_cod = le_mun.transform([municipio])[0]
        meso_cod = le_meso.transform([mesoregiao])[0]
        
        # Montando o array
        X_input = np.array([[mun_cod, meso_cod, casos_mes_anterior]])
        
        # Padronizando os dados
        X_scaled = scaler.transform(X_input)
        
        # Predição
        predicao = rf_model.predict(X_scaled)[0]
        
        st.markdown("---")
        st.subheader("🔍 Resultado da Predição Preditiva")
        
        # Visualização clara dos resultados para o Gestor Público
        if predicao == 1:
            st.error(f"**ALERTA PREVENTIVO PARA {municipio.upper()}** 📈")
            st.write("O algoritmo estima uma **TENDÊNCIA DE ALTA** nas ocorrências de violência doméstica para o próximo mês. Recomenda-se alertar as redes de proteção locais e patrulhas preventivas da PM-SC.")
        else:
            st.success(f"**CENÁRIO ESTÁVEL PARA {municipio.upper()}** 📉")
            st.write("O algoritmo estima que as ocorrências se manterão **estáveis ou sofrerão redução** no próximo mês. Manter os protocolos normais de atendimento.")