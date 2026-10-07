import os
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def normalize_text(text):
    if not isinstance(text, str):
        return text
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII').strip().upper()

# 2. Carregamento dos Modelos e da Base de Dados (Caminhos Absolutos)
@st.cache_resource
def load_models():
    rf_path = os.path.join(BASE_DIR, 'modelo_baseline_rf.joblib')
    if not os.path.exists(rf_path):
        rf_path = os.path.join(BASE_DIR, 'models', 'modelo_baseline_rf.joblib')
        
    le_mun_path = os.path.join(BASE_DIR, 'le_mun.joblib')
    if not os.path.exists(le_mun_path):
        le_mun_path = os.path.join(BASE_DIR, 'models', 'le_mun.joblib')

    le_meso_path = os.path.join(BASE_DIR, 'le_meso.joblib')
    if not os.path.exists(le_meso_path):
        le_meso_path = os.path.join(BASE_DIR, 'models', 'le_meso.joblib')

    scaler_path = os.path.join(BASE_DIR, 'scaler.joblib')
    if not os.path.exists(scaler_path):
        scaler_path = os.path.join(BASE_DIR, 'models', 'scaler.joblib')

    rf_model = joblib.load(rf_path)
    le_mun = joblib.load(le_mun_path)
    le_meso = joblib.load(le_meso_path)
    scaler = joblib.load(scaler_path)
    explainer = shap.TreeExplainer(rf_model)
    return rf_model, le_mun, le_meso, scaler, explainer

@st.cache_data
def load_data():
    csv_path = os.path.join(BASE_DIR, 'base_modelo_violencia_domestica_sc.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(BASE_DIR, 'data', 'base_modelo_violencia_domestica_sc.csv')
    return pd.read_csv(csv_path)

try:
    rf_model, le_mun, le_meso, scaler, explainer = load_models()
    df_historico = load_data()
except Exception as e:
    st.error(f"Erro de infraestrutura ou carregamento de arquivos: Detalhes: {e}")
    st.stop()

# --- MAPEAMENTO OFICIAL IBGE: MUNICÍPIOS DE SC -> MESORREGIÕES ---
MAPA_MESORREGIOES = {
    "Grande Florianópolis": [
        "AGUAS MORNAS", "ALFREDO WAGNER", "ANGELINA", "ANITAPOLIS", "ANTONIO CARLOS",
        "BIGUACU", "CANELINHA", "FLORIANOPOLIS", "GAROPABA", "GOVERNADOR CELSO RAMOS",
        "MAJOR GERCINO", "NOVA TRENTO", "PALHOCA", "PAULO LOPES", "RANCHO QUEIMADO",
        "SANTO AMARO DA IMPERATRIZ", "SAO BONIFACIO", "SAO JOSE", "SAO PEDRO DE ALCANTARA",
        "TIJUCAS"
    ],
    "Norte Catarinense": [
        "ARAUQUARI", "BALNEARIO BARRA DO SUL", "BELA VISTA DO TOLDO", "CAMPO ALEGRE",
        "CANOINHAS", "CORUPA", "GARUVA", "IRINEOPOLIS", "ITAIOPOLIS", "ITARARE",
        "JARAGUA DO SUL", "JOINVILLE", "MAFRA", "MAJOR VIEIRA", "MASSARANDUBA",
        "MONTE CASTELO", "PAPANDUVA", "PORTO UNIAO", "RIO NEGRINHO", "SCHROEDER",
        "SAO BENTO DO SUL", "SAO FRANCISCO DO SUL", "SAO JOAO DO ITAPERIU", "TRES BARRAS"
    ],
    "Serrana": [
        "ABDON BATISTA", "ANITA GARIBALDI", "BOM JARDIM DA SERRA", "BOM RETIRO",
        "CAMPO BELO DO SUL", "CAPAO ALTO", "CELSO RAMOS", "CORREIA PINTO", "CURITIBANOS",
        "FREI ROGERIO", "LAGES", "OTACILIO COSTA", "PAINEL", "PALMEIRA", "PONTE ALTA",
        "PONTE ALTA DO NORTE", "RIO RUFINO", "SANTA CECILIA", "SAO CRISTOVAO DO SUL",
        "SAO JOAQUIM", "SAO JOSE DO CERRITO", "URUBICI", "URUPEMA", "VARGEM", "VARGEM BONITA"
    ],
    "Sul Catarinense": [
        "ARARANGUA", "BALNEARIO ARROIO DO SILVA", "BALNEARIO GAIVOTA", "BALNEARIO RINCAO",
        "BRACO DO NORTE", "CAPIVARI DE BAIXO", "COCAL DO SUL", "CRICIUMA", "ERMO",
        "FORQUILHINHA", "GRAVATAL", "ICARA", "IMARUI", "IMBITUBA", "JACINTO MACHADO",
        "JAGUARUNA", "LAGUNA", "LAURO MULLER", "MARACAJA", "MELEIRO", "MORRO DA FUMACA",
        "MORRO GRANDE", "NOVA VENEZA", "ORLEANS", "PASSO DE TORRES", "PEDRAS GRANDES",
        "PESCARIA BRAVA", "PRAIA GRANDE", "SANGAO", "SANTA ROSA DO SUL", "SAO LUDGERO",
        "SAO MARTINHO", "SOMBRIO", "TREVISO", "TREZE DE MAIO", "TUBARAO", "TURVO", "URUSSANGA"
    ],
    "Vale do Itajaí": [
        "AGROLANDIA", "AGRONOMICA", "APIUNA", "ATALANTA", "AURORA", "BALNEARIO CAMBORIU",
        "BALNEARIO PICARRAS", "BLUMENAU", "BOTUVERA", "BRACO DO TROMBETO", "BRACO DO TROMBETAS",
        "BRUSQUE", "CAMBORIU", "CHAPADAO DO LAGEADO", "DONA EMMA", "GASPAR", "IBIRAMA",
        "ILHOTA", "INDAIAL", "ITAJAI", "ITAPEMA", "ITUPORANGA", "JOSE BOITEUX", "LAURENTINO",
        "LONTRAS", "LUIS ALVES", "MIRIM DOCE", "NAVEGANTES", "PENHA", "PETROLANDIA",
        "POUSO REDONDO", "PRESIDENTE GETULIO", "PRESIDENTE NEREU", "RIO DO CAMPO",
        "RIO DO OESTE", "RIO DO SUL", "RIO DOS CEDROS", "RODEIO", "SALETE", "SANTA TEREZINHA",
        "TAIO", "TROMBUDO CENTRAL", "VIDAL RAMOS", "VITOR MEIRELES", "WITMARSUM"
    ]
}

def obter_mesoregiao(municipio_str):
    mun_norm = normalize_text(municipio_str)
    for regiao, municipios in MAPA_MESORREGIOES.items():
        if any(normalize_text(m) == mun_norm for m in municipios):
            return regiao
    return "Oeste Catarinense"

st.markdown("---")
st.subheader("📊 Entrada de Variáveis Preditivas")

# 3. Interface de Usuário
mapa_municipios = {normalize_text(m): m for m in le_mun.classes_}
municipios_disponiveis = sorted(list(mapa_municipios.values()))

municipio_selecionado = st.selectbox("Município Alvo:", municipios_disponiveis)
municipio_norm = normalize_text(municipio_selecionado)

# Filtro no DataFrame
df_historico['municipio_norm'] = df_historico['municipio'].apply(normalize_text)
df_mun_atual = df_historico[df_historico['municipio_norm'] == municipio_norm].copy()

if not df_mun_atual.empty:
    periodo_selecionado = df_mun_atual['periodo'].max()
    dado_filtrado = df_mun_atual[df_mun_atual['periodo'] == periodo_selecionado]
    casos_mes_anterior = int(dado_filtrado['ocorrencias'].values[0]) if not dado_filtrado.empty else 10
    
    st.info(f"Base para predição: Mês mais recente registrado (**{periodo_selecionado}**). "
            f"Ocorrências registradas em **{municipio_selecionado}**: **{casos_mes_anterior}** casos.")
else:
    periodo_selecionado = "N/A"
    casos_mes_anterior = 10
    st.warning(f"Não foram encontrados dados históricos suficientes para **{municipio_selecionado}**. Utilizando valor padrão de 10 ocorrências para a predição.")

# 4. Execução do Algoritmo e Resultados
if st.button("Executar Algoritmo de Predição", type="primary"):
    with st.spinner("Processando dados e consultando o modelo Random Forest..."):
        
        mesoregiao_exata = obter_mesoregiao(municipio_selecionado)

        for classe_le in le_meso.classes_:
            if normalize_text(classe_le) == normalize_text(mesoregiao_exata):
                mesoregiao_exata = classe_le
                break

        mun_cod = le_mun.transform([municipio_selecionado])[0]
        meso_cod = le_meso.transform([mesoregiao_exata])[0]
        
        X_input = np.array([[mun_cod, meso_cod, casos_mes_anterior]])
        X_scaled = scaler.transform(X_input)
        
        predicao = rf_model.predict(X_scaled)[0]
        probabilidades = rf_model.predict_proba(X_scaled)[0]
        confianca = np.max(probabilidades) * 100
        
        if not df_mun_atual.empty and len(df_mun_atual) >= 2:
            df_ord = df_mun_atual.sort_values('periodo')
            media_historica = df_ord['ocorrencias'].mean()
            diferenca_media = casos_mes_anterior - media_historica
            pct_media = (diferenca_media / media_historica * 100) if media_historica > 0 else 0.0
        else:
            media_historica = casos_mes_anterior
            pct_media = 0.0

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

        st.markdown("### 📄 Diagnóstico")
        
        if abs(pct_media) < 5:
            relacao_media = f"praticamente alinhado à média histórica local ({media_historica:.1f} casos/mês)."
        elif pct_media > 0:
            relacao_media = f"**{abs(pct_media):.1f}% acima** da média histórica local ({media_historica:.1f} casos/mês)."
        else:
            relacao_media = f"**{abs(pct_media):.1f}% abaixo** da média histórica local ({media_historica:.1f} casos/mês)."

        if predicao == 1:
            st.markdown(f"""
            O modelo projeta uma **tendência de alta** nas ocorrências de **{municipio_selecionado}** para o próximo período.

            * **Análise do Volume:** O município registrou **{casos_mes_anterior} ocorrências** no último período, estando {relacao_media}
            * **Influência Regional:** O município pertence à mesorregião **{mesoregiao_exata}**. A dinâmica e o histórico recente dos municípios desta região contribuem para elevar a probabilidade de risco estimada pelo algoritmo.
            """)
        else:
            st.markdown(f"""
            O modelo projeta **estabilidade ou redução** nas ocorrências de **{municipio_selecionado}** para o próximo período.

            * **Análise do Volume:** O município registrou **{casos_mes_anterior} ocorrências** no último período, estando {relacao_media}
            * **Influência Regional:** O município pertence à mesorregião **{mesoregiao_exata}**. O comportamento observado nesta região indica um padrão sob controle, sustentando a projeção de estabilidade.
            """)

        # 5. Interpretabilidade com SHAP
        st.markdown("---")
        st.subheader("🧠 Por que o modelo chegou a esse resultado?")
        
        st.markdown("""
        O gráfico abaixo (*Waterfall SHAP*) explica passo a passo como o algoritmo calculou o risco para este município.
        
        **Como ler este gráfico:**
        *   **Números à esquerda (ex: Chapecó, 229):** São os valores reais das variáveis para este município.
        *   **Números nas barras (ex: +0.24):** É o impacto (peso) de cada variável na previsão final.
            *   Valores **positivos (vermelho)** aumentam a probabilidade de escalada.
            *   Valores **negativos (azul)** reduzem a probabilidade de escalada.
        *   **Base E[f(X)]:** É a probabilidade média de escalada de todos os municípios (o ponto de partida).
        *   **f(X):** É a probabilidade final calculada para este município específico.
        
        *Exemplo prático:* O fato de ser o município de **{municipio_selecionado}** contribuiu com **+0.24** (ou +24%) para a probabilidade final.
        """.replace("{municipio_selecionado}", municipio_selecionado))

        shap_values = explainer.shap_values(X_scaled)

        if isinstance(shap_values, list):
            shap_val_target = shap_values[1][0]
            expected_val = explainer.expected_value[1]
        else:
            if len(shap_values.shape) == 3:
                shap_val_target = shap_values[0, :, 1]
                expected_val = explainer.expected_value[1]
            else:
                shap_val_target = shap_values[0]
                expected_val = explainer.expected_value

        features = ['Município', 'Mesorregião', 'Ocorrências (Mês Anterior)']

        # --- CORREÇÃO: Criamos um array com os nomes legíveis para exibir no gráfico ---
        X_display = np.array([
            municipio_selecionado, 
            mesoregiao_exata, 
            str(casos_mes_anterior)
        ], dtype=object)

        fig, ax = plt.subplots(figsize=(8, 3))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_val_target,
                base_values=expected_val,
                data=X_display,  # <--- Mudança aqui: antes era X_input[0]
                feature_names=features
            ),
            show=False
        )
        st.pyplot(fig)

        # 6. Histórico Local
        st.markdown("---")
        st.subheader("📈 Contexto Histórico Local")
        
        if not df_mun_atual.empty:
            st.info(f"Série temporal completa de ocorrências registradas para {municipio_selecionado}.")
            df_grafico = df_mun_atual[['periodo', 'ocorrencias']].set_index('periodo')
            st.line_chart(df_grafico)
        else:
            st.warning(f"Não foram encontrados registros históricos suficientes para {municipio_selecionado}.")
