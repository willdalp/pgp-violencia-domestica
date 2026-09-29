import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. Configuração da Página

st.set_page_config(page_title="MVP - Violência SC", page_icon="🚨", layout="centered")

st.title("🚨 Previsão de Escalada: Violência Doméstica (SC)")
st.markdown("""
Este MVP utiliza **Machine Learning (Random Forest)** para prever a tendência de aumento das 
ocorrências de violência doméstica nos municípios de Santa Catarina para o mês subsequente.
""")

# 2. Carregamento dos Modelos (Cache)

@st.cache_resource
def load_models():
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

# 3. Interface de Usuário (Inputs)

col1, col2 = st.columns(2)
with col1:
    municipio = st.selectbox("Município Alvo:", le_mun.classes_)
with col2:
    mesoregiao = st.selectbox("Mesorregião de Pertencimento:", le_meso.classes_)

casos_mes_anterior = st.number_input("Total de Ocorrências no Mês Anterior:", min_value=0, value=10, step=1)

# 4. Execução do Algoritmo e Resultados

if st.button("Executar Algoritmo de Predição", type="primary"):
    with st.spinner("Processando dados e consultando o modelo Random Forest..."):
        
        # Transformando entradas textuais em numéricas
        mun_cod = le_mun.transform([municipio])[0]
        meso_cod = le_meso.transform([mesoregiao])[0]
        
        # Montando array e padronizando
        X_input = np.array([[mun_cod, meso_cod, casos_mes_anterior]])
        X_scaled = scaler.transform(X_input)
        
        # Predição e Probabilidade
        predicao = rf_model.predict(X_scaled)[0]
        probabilidades = rf_model.predict_proba(X_scaled)[0]
        confianca = np.max(probabilidades) * 100
        
        st.markdown("---")
        st.subheader("🔍 Resultado da Avaliação Preditiva")
        
        res_col1, res_col2 = st.columns([3, 2])
        with res_col1:
            if predicao == 1:
                st.error(f"**ALERTA PREVENTIVO PARA {municipio.upper()}** 📈")
                st.write("O algoritmo estima uma **TENDÊNCIA DE ALTA** nas ocorrências de violência doméstica para o próximo mês. Recomenda-se alertar as redes de proteção locais e patrulhas preventivas da PM-SC.")
            else:
                st.success(f"**CENÁRIO ESTÁVEL PARA {municipio.upper()}** 📉")
                st.write("O algoritmo estima que as ocorrências se manterão **estáveis ou sofrerão redução** no próximo mês. Manter os protocolos normais de atendimento.")
                
        with res_col2:
            cor_delta = "inverse" if predicao == 1 else "normal"
            indicador_delta = "Risco de Escalada" if predicao == 1 else "Risco Controlado"
            st.metric(label="Grau de Confiança (Modelo)", value=f"{confianca:.1f}%", delta=indicador_delta, delta_color=cor_delta)

        # 5. Interpretabilidade (Feature Importance)

        st.markdown("---")
        st.subheader("🧠 Interpretabilidade do Modelo")
        st.write("Peso de cada variável na construção do alerta atual (Feature Importance):")
        
        importancias = rf_model.feature_importances_
        features = ['Município', 'Mesorregião', 'Ocorrências (Mês Anterior)']
        df_importancias = pd.DataFrame({'Variável': features, 'Importância (%)': importancias * 100}).set_index('Variável')
        st.bar_chart(df_importancias, color="#ff4b4b" if predicao == 1 else "#00cc96")

        # 6. Contexto Histórico Local (Dados Reais)

        st.markdown("---")
        st.subheader("📈 Contexto Histórico Local")
        
        try:
            @st.cache_data
            def carregar_historico():
                return pd.read_csv('data/base_modelo_violencia_domestica_sc.csv') 
                
            df_historico = carregar_historico()

            # ATENÇÃO: Altere 'Municipio' para o nome exato da coluna de municípios no seu CSV
            df_mun = df_historico[df_historico['municipio'] == municipio]

            if not df_mun.empty:
                st.info(f"Ocorrências históricas registradas na base de dados para {municipio}.")
                
                # ATENÇÃO: Altere 'Mes' e 'Ocorrencias' para as colunas exatas do seu CSV
                df_grafico = df_mun[['Mes', 'Ocorrencias']].set_index('Mes')
                st.line_chart(df_grafico)
            else:
                st.warning(f"Não foram encontrados registros históricos para {municipio} na base de dados.")

        except FileNotFoundError:
            st.warning("⚠️️ Arquivo CSV de dados históricos não encontrado no repositório. O gráfico não pôde ser gerado.")
        except Exception as e:
            st.error(f"Erro ao tentar ler o dataset: {e}")
