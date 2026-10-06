import streamlit as st
import joblib
import numpy as np
import pandas as pd
import unicodedata

# 1. Configuração da Página
st.set_page_config(page_title="MVP - Violência SC", page_icon="🚨", layout="centered")

st.title("🚨 Previsão de Escalada: Violência Doméstica (SC)")
st.markdown("""
Este MVP utiliza **Machine Learning (Random Forest)** para prever a tendência de aumento das 
ocorrências de violência doméstica nos municípios de Santa Catarina para o mês subsequente.
""")

# 2. Carregamento dos Modelos e da Base de Dados (Cache)
@st.cache_resource
def load_models():
    rf_model = joblib.load('models/modelo_baseline_rf.joblib')
    le_mun = joblib.load('models/le_mun.joblib')
    le_meso = joblib.load('models/le_meso.joblib')
    scaler = joblib.load('models/scaler.joblib')
    return rf_model, le_mun, le_meso, scaler

@st.cache_data
def load_data():
    return pd.read_csv('data/base_modelo_violencia_domestica_sc.csv')

try:
    rf_model, le_mun, le_meso, scaler = load_models()
    df_historico = load_data()
except Exception as e:
    st.error(f"Erro de infraestrutura ou carregamento de arquivos: Detalhes: {e}")
    st.stop()

st.markdown("---")
st.subheader("📊 Entrada de Variáveis Preditivas")

# 3. Interface de Usuário (Apenas Município)
# Puxa os municípios diretamente do CSV para garantir consistência com os dados
municipios_disponiveis = sorted(df_historico['municipio'].dropna().unique())
municipio_selecionado = st.selectbox("Município Alvo:", municipios_disponiveis)

# Filtra o dataframe para o município escolhido
df_mun_atual = df_historico[df_historico['municipio'] == municipio_selecionado]

# --- Lógica Automática para Mesorregião ---
if 'mesoregiao' in df_historico.columns and not df_mun_atual.empty:
    mesoregiao = df_mun_atual['mesoregiao'].iloc[0]
    # Garante que a mesorregião existe nas classes do encoder
    if mesoregiao not in le_meso.classes_:
        mesoregiao = le_meso.classes_[0]
else:
    mesoregiao = le_meso.classes_[0]

# --- Lógica Automática para Período e Ocorrências ---
if not df_mun_atual.empty:
    # Pega o período mais recente disponível (assumindo formato YYYY-MM-DD)
    periodo_selecionado = df_mun_atual['periodo'].max()
    
    # Puxa o valor real de ocorrências desse último mês no CSV
    dado_filtrado = df_mun_atual[df_mun_atual['periodo'] == periodo_selecionado]
    
    # Ajuste o nome da coluna de ocorrências se no seu CSV for diferente de 'ocorrencias'
    col_ocorrencias = 'ocorrencias' if 'ocorrencias' in df_mun_atual.columns else df_mun_atual.columns[2]
    
    casos_mes_anterior = int(dado_filtrado[col_ocorrencias].values[0]) if not dado_filtrado.empty else 10
    
    st.info(f"Base para predição: Mês mais recente registrado (**{periodo_selecionado}**). "
            f"Ocorrências registradas em **{municipio_selecionado}**: **{casos_mes_anterior}** casos.")
else:
    # Fallback de segurança caso o município não tenha dados históricos
    periodo_selecionado = "N/A"
    casos_mes_anterior = 10
    st.warning(f"Não foram encontrados dados históricos suficientes para **{municipio_selecionado}**. Utilizando valor padrão de 10 ocorrências para a predição.")

# 4. Execução do Algoritmo e Resultados
if st.button("Executar Algoritmo de Predição", type="primary"):
    with st.spinner("Processando dados e consultando o modelo Random Forest..."):
        
        # Transformando entradas textuais em numéricas usando os encoders salvos
        mun_cod = le_mun.transform([municipio_selecionado])[0]
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
                st.error(f"**ALERTA PREVENTIVO PARA {municipio_selecionado.upper()}** 📈")
                st.write("O algoritmo estima uma **TENDÊNCIA DE ALTA** nas ocorrências de violência doméstica para o próximo mês. Recomenda-se alertar as redes de proteção locais e patrulhas preventivas da PM-SC.")
            else:
                st.success(f"**CENÁRIO ESTÁVEL PARA {municipio_selecionado.upper()}** 📉")
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

        # 6. Contexto Histórico Local (Série Temporal Completa)
        st.markdown("---")
        st.subheader("📈 Contexto Histórico Local")
        
        if not df_mun_atual.empty:
            st.info(f"Série temporal completa de ocorrências registradas para {municipio_selecionado}.")
            
            # Prepara o dataframe histórico do município para exibir todos os meses no gráfico
            df_grafico = df_mun_atual[['periodo', col_ocorrencias]].set_index('periodo')
            st.line_chart(df_grafico)
        else:
            st.warning(f"Não foram encontrados registros históricos suficientes para {municipio_selecionado}.")
