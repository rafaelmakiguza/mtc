#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Cache para armazenar previsões
@st.cache_data(show_spinner=False)
def cache_predictions():
    return {}

# Inicialização do Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate({
        "type": st.secrets["firebase"]["type"],
        "project_id": st.secrets["firebase"]["project_id"],
        "private_key_id": st.secrets["firebase"]["private_key_id"],
        "private_key": st.secrets["firebase"]["private_key"],
        "client_email": st.secrets["firebase"]["client_email"],
        "client_id": st.secrets["firebase"]["client_id"],
        "auth_uri": st.secrets["firebase"]["auth_uri"],
        "token_uri": st.secrets["firebase"]["token_uri"],
        "auth_provider_x509_cert_url": st.secrets["firebase"]["auth_provider_x509_cert_url"],
        "client_x509_cert_url": st.secrets["firebase"]["client_x509_cert_url"],
        "universe_domain": st.secrets["firebase"]["universe_domain"]
    })
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://crazytime-pedro-2022-default-rtdb.firebaseio.com/'
    })

# Referência ao nó do banco de dados
ref = db.reference("/Double_blaze2")

# Carregar o modelo XGBoost
model = xgb.Booster()
model.load_model("modelo_xgboost.json")

# Função para pré-tratar e criar features
def preprocess_data(df):
    st.write("Iniciando o pré-processamento dos dados...")
    df = df.reset_index().rename(columns={'id': 'timestamp'})
    df = pd.get_dummies(data=df, columns=['type', 'color'])
    df['payout'] = pd.to_numeric(df['payout'], errors='coerce')
    df['total_bet'] = pd.to_numeric(df['total_bet'], errors='coerce')

    # Remover colunas desnecessárias para o modelo
    df = df.drop(columns=['multiplier', 'spin_result', 'type_Special Result'], errors='ignore')
    df['online_players'] = (df['total_bet'] / 17.66).round().astype(int)

    # Feature engineering com deslocamento para evitar vazamento de dados futuros
    rolling_sum_window = 25
    df['bank_profit'] = df['total_bet'] - df['payout']
    df['rolling_sum'] = df['bank_profit'].rolling(window=rolling_sum_window).sum().shift(1)

    window_size = 5
    df['open'] = df['rolling_sum'].rolling(window=window_size).apply(lambda x: x[0], raw=True).shift(1)
    df['close'] = df['rolling_sum'].rolling(window=window_size).apply(lambda x: x[-1], raw=True).shift(1)
    df['high'] = df['rolling_sum'].rolling(window=window_size).max().shift(1)
    df['low'] = df['rolling_sum'].rolling(window=window_size).min().shift(1)

    df['velocity'] = df['rolling_sum'].diff().fillna(0)
    df['acceleration'] = df['velocity'].diff(3).fillna(0)

    dev_window = 10
    mult_bb = 2.0
    df['basis'] = df['close'].rolling(window=dev_window).mean().shift(1)
    df['dev'] = df['close'].rolling(window=dev_window).std().shift(1)
    df['upperBB'] = df['basis'] + mult_bb * df['dev']
    df['lowerBB'] = df['basis'] - mult_bb * df['dev']

    short_window_macd = 7
    long_window_macd = 25
    signal_window = 11
    df['ema_short'] = df['close'].ewm(span=short_window_macd, min_periods=1, adjust=False).mean().shift(1)
    df['ema_long'] = df['close'].ewm(span=long_window_macd, min_periods=1, adjust=False).mean().shift(1)
    df['MACD'] = df['ema_short'] - df['ema_long']
    df['Signal_Line'] = df['MACD'].ewm(span=signal_window, min_periods=1, adjust=False).mean().shift(1)
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    df['MACD_Histogram_Signal_Interaction'] = df['MACD_Histogram'] * df['Signal_Line']

    ma_long_window = 40
    slope_ma_long_window = 30
    df['MA_short'] = df['bank_profit'].rolling(window=10, min_periods=1).mean().shift(1)
    df['MA_long'] = df['bank_profit'].rolling(window=ma_long_window, min_periods=1).mean().shift(1)
    df['Slope_MA_long'] = df['MA_long'].diff().shift(1) / slope_ma_long_window
    df['MA_longer'] = df['bank_profit'].rolling(window=120, min_periods=1).mean().shift(1)

    scaler = StandardScaler()
    numerical_features = [
        'dev', 'total_bet', 'online_players', 'MACD', 'MACD_Histogram',
        'MA_short', 'MA_long', 'MA_longer', 'Signal_Line', 'acceleration', 'Slope_MA_long'
    ]
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    expected_features = model.feature_names
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_features]
    return df

# Função para realizar predição
def predict_with_model(df):
    features_for_model = df.drop(columns=['when', 'color'], errors='ignore')
    dmatrix = xgb.DMatrix(features_for_model)
    predictions = model.predict(dmatrix)
    return predictions

# Interface no Streamlit
st.title("MVP com Feature Engineering e Predição")
st.write("Clique no botão para consultar os últimos 2000 dados do Firebase, processá-los e prever no modelo.")

if st.button("Consultar e Prever"):
    st.write("Buscando dados do Firebase...")
    data = ref.order_by_key().limit_to_last(2000).get()

    if not data:
        st.error("Nenhum dado encontrado no Firebase!")
    else:
        df = pd.DataFrame.from_dict(data, orient='index')

        if 'when' in df.columns:
            df['when'] = pd.to_datetime(df['when'])
            df = df.sort_values(by='when')  # Ordena do mais antigo para o mais recente

        st.write("Dados originais do Firebase:")
        st.dataframe(df.head(30))

        cache = cache_predictions()
        new_data = df[~df['when'].isin(cache.keys())]  # Verifica pelo timestamp

        if not new_data.empty:
            new_data = new_data.sort_values(by='when')  # Garante a previsão sequencial
            processed_data = preprocess_data(new_data)
            predictions = predict_with_model(processed_data)
            for idx, pred in zip(new_data.index, predictions):
                cache[new_data.loc[idx, 'when']] = {
                    "timestamp": new_data.loc[idx, 'when'],
                    "color": new_data.loc[idx, 'color'] if 'color' in new_data.columns else None,
                    "Probabilidade": pred,
                    "Predição": int(pred > 0.8)
                }

        st.write("Resultados Previstos (mais recentes primeiro):")
        result_df = pd.DataFrame.from_dict(cache, orient='index').sort_values(by='timestamp', ascending=False)
        
        # Cálculo de precisão da classe 1
        valid_predictions = result_df.dropna(subset=['Probabilidade'])
        predicted_positives = valid_predictions['Predição'].sum()
        correct_predictions = valid_predictions.query('Predição == 1 and color == "White"').shape[0]
        actual_positives = (valid_predictions['color'] == 'White').sum()
        precision = (correct_predictions / actual_positives) if actual_positives > 0 else 0

        st.metric("Classe 1 Previsões", f"{predicted_positives}")
        st.metric("Classe 1 Corretas", f"{correct_predictions}")
        st.metric("Precisão Classe 1", f"{precision:.2%}")

        # Estilizar tabela
        def highlight_row(row):
            if row['Predição'] == 1:
                return ['background-color: darkgreen'] * len(row)
            return [''] * len(row)

        styled_df = result_df.style.apply(highlight_row, axis=1)
        st.dataframe(styled_df)
