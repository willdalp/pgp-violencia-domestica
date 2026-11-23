import json
import os
import unicodedata

import joblib
import numpy as np
import pandas as pd
import pytest
import shap

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resolve_path(filename, subfolder):
    for candidate in [
        os.path.join(BASE_DIR, subfolder, filename),
        os.path.join(BASE_DIR, filename),
    ]:
        if os.path.exists(candidate):
            return candidate
    return os.path.join(BASE_DIR, subfolder, filename)


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


# ============================================================
# 1. TESTES UNITÁRIOS
# ============================================================
def test_normalize_text_padronizacao():
    assert normalize_text("  Chapecó  ") == "CHAPECO"
    assert normalize_text("São José do Cerrito") == "SAO JOSE DO CERRITO"
    assert normalize_text("Balneário Camboriú") == "BALNEARIO CAMBORIU"
    assert normalize_text(None) == ""
    assert normalize_text(np.nan) == ""


# ============================================================
# 2. TESTES DE INTEGRIDADE DA BASE E ARTEFATOS (INTEGRAÇÃO)
# ============================================================
def test_artefatos_sprint4_existem():
    esperados = [
        ("base_modelo_violencia_domestica_sc.csv", "data"),
        ("modelo_v2_rf.joblib", "models"),
        ("le_mun.joblib", "models"),
        ("le_meso.joblib", "models"),
        ("scaler_v2.joblib", "models"),
        ("metrics_sprint4.json", "models"),
    ]
    for filename, subfolder in esperados:
        path = resolve_path(filename, subfolder)
        assert os.path.exists(path), f"Artefato ausente: {path}"


def test_integridade_base_consolidada():
    df = pd.read_csv(resolve_path("base_modelo_violencia_domestica_sc.csv", "data"))
    with open(resolve_path("metrics_sprint4.json", "models"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert df["municipio"].nunique() == 295, "A base deve cobrir os 295 municípios de SC."
    assert df["mesoregiao"].nunique() == 6, "A base deve conter as 6 mesorregiões oficiais de SC."
    assert "Não Informada" not in df["mesoregiao"].values

    colunas_features = [
        c for c in meta["features"] if c not in ("municipio_cod", "mesoregiao_cod")
    ]
    for col in colunas_features:
        assert col in df.columns, f"Coluna ausente no CSV: {col}"
        assert df[col].isna().sum() == 0, f"Valores nulos encontrados em {col}"


def test_inferencia_ponta_a_ponta_e_shap():
    rf_model = joblib.load(resolve_path("modelo_v2_rf.joblib", "models"))
    le_mun = joblib.load(resolve_path("le_mun.joblib", "models"))
    le_meso = joblib.load(resolve_path("le_meso.joblib", "models"))
    scaler = joblib.load(resolve_path("scaler_v2.joblib", "models"))
    df = pd.read_csv(resolve_path("base_modelo_violencia_domestica_sc.csv", "data"))

    with open(resolve_path("metrics_sprint4.json", "models"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    features = meta["features"]
    assert scaler.n_features_in_ == len(features)

    amostra = df[df["municipio"] == "CHAPECO"].sort_values("periodo").iloc[-1].copy()
    amostra["municipio_cod"] = le_mun.transform([amostra["municipio"]])[0]
    amostra["mesoregiao_cod"] = le_meso.transform([amostra["mesoregiao"]])[0]

    X = np.array([[float(amostra[c]) for c in features]])
    X_scaled = scaler.transform(X)

    proba = rf_model.predict_proba(X_scaled)[0]
    assert len(proba) == 2
    assert 0.0 <= proba[1] <= 1.0
    assert np.isclose(proba.sum(), 1.0, atol=1e-5)

    explainer = shap.TreeExplainer(rf_model)
    shap_vals = explainer.shap_values(X_scaled)
    if isinstance(shap_vals, list):
        vec = shap_vals[1][0]
    elif len(np.shape(shap_vals)) == 3:
        vec = shap_vals[0, :, 1]
    else:
        vec = shap_vals[0]
    assert len(vec) == len(features)


# ============================================================
# 3. TESTES DE ACEITE (EVOLUÇÃO v1 -> v2 E CRISP-ML(Q))
# ============================================================
def test_aceite_evolucao_modelo_v2_supera_baseline():
    with open(resolve_path("metrics_sprint4.json", "models"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    bench = {item["modelo"]: item for item in meta["benchmark"]}
    v1 = bench["v1_Random_Forest_Baseline"]
    v2 = bench["v2_Random_Forest_Tunado"]

    assert v2["acuracia"] > v1["acuracia"]
    assert v2["recall"] > v1["recall"]
    assert v2["f1_score"] > v1["f1_score"]
    assert v2["roc_auc"] > v1["roc_auc"]

    # Quality gates definidos para segurança pública (prioridade para sensibilidade)
    assert v2["acuracia"] >= 0.70
    assert v2["recall"] >= 0.65
    assert v2["f1_score"] >= 0.65
