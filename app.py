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

# Inicializar Firebase Admin
if not firebase_admin._apps:
    cred = credentials.Certificate("/Users/rafaelyano/Desktop/Python/MTC/Streamlit/crazytime-pedro-2022-firebase-adminsdk-6w9fr-a28187bdce_new.json")
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

    # Pré-tratamento inicial
    st.write("Etapa 1: Ajustando colunas iniciais...")
    df = df.reset_index().rename(columns={'id': 'timestamp'})
    df = pd.get_dummies(data=df, columns=['type', 'color'])
    df['payout'] = pd.to_numeric(df['payout'], errors='coerce')
    df['online_players'] = pd.to_numeric(df['online_players'], errors='coerce')
    df['total_bet'] = pd.to_numeric(df['total_bet'], errors='coerce')
    df['winners'] = pd.to_numeric(df['winners'], errors='coerce')

    # Remover colunas desnecessárias
    st.write("Etapa 2: Removendo colunas desnecessárias...")
    df = df.drop(columns=['when', 'multiplier', 'spin_result', 'type_Special Result'], errors='ignore')

    # Feature Engineering
    st.write("Etapa 3: Criando novas features...")
    rolling_sum_window = 25
    df['bank_profit'] = df['total_bet'] - df['payout']
    df['rolling_sum'] = df['bank_profit'].rolling(window=rolling_sum_window).sum()

    window_size = 5
    df['open'] = df['rolling_sum'].rolling(window=window_size).apply(lambda x: x[0], raw=True)
    df['close'] = df['rolling_sum'].rolling(window=window_size).apply(lambda x: x[-1], raw=True)
    df['high'] = df['rolling_sum'].rolling(window=window_size).max()
    df['low'] = df['rolling_sum'].rolling(window=window_size).min()

    df['velocity'] = df['rolling_sum'].diff().fillna(0)
    df['acceleration'] = df['velocity'].diff(3).fillna(0)

    dev_window = 10
    mult_bb = 2.0
    df['basis'] = df['close'].rolling(window=dev_window).mean()
    df['dev'] = df['close'].rolling(window=dev_window).std()
    df['upperBB'] = df['basis'] + mult_bb * df['dev']
    df['lowerBB'] = df['basis'] - mult_bb * df['dev']

    short_window_macd = 7
    long_window_macd = 25
    signal_window = 11
    df['ema_short'] = df['close'].ewm(span=short_window_macd, min_periods=1, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=long_window_macd, min_periods=1, adjust=False).mean()
    df['MACD'] = df['ema_short'] - df['ema_long']
    df['Signal_Line'] = df['MACD'].ewm(span=signal_window, min_periods=1, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    df['MACD_Histogram_Signal_Interaction'] = df['MACD_Histogram'] * df['Signal_Line']

    ma_long_window = 40
    slope_ma_long_window = 30
    df['MA_short'] = df['bank_profit'].rolling(window=10, min_periods=1).mean()
    df['MA_long'] = df['bank_profit'].rolling(window=ma_long_window, min_periods=1).mean()
    df['Slope_MA_long'] = df['MA_long'].diff() / slope_ma_long_window
    df['MA_longer'] = df['bank_profit'].rolling(window=120, min_periods=1).mean()

    # Corrigir a coluna 'color_White'
    st.write("Etapa 4: Calculando acumulado comum...")
    if 'color_White' not in df.columns:
        df['color_White'] = False  # Preencher com valores padrão

    # Calcular 'acumulado_comum'
    acumulado = 0
    acumulados = []
    for is_white in df['color_White']:
        if not is_white:
            acumulado += 1
        else:
            acumulado = 0
        acumulados.append(acumulado)
    df['acumulado_comum'] = acumulados

    # Normalização
    st.write("Etapa 5: Normalizando as features numéricas...")
    scaler = StandardScaler()
    numerical_features = [
        'dev', 'total_bet', 'online_players', 'MACD', 'MACD_Histogram',
        'MA_short', 'MA_long', 'MA_longer', 'Signal_Line', 'acceleration', 'Slope_MA_long'
    ]
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # Ajustar as colunas esperadas pelo modelo
    st.write("Etapa 6: Ajustando as features para o modelo...")
    expected_features = model.feature_names
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_features]

    st.write("Pré-processamento concluído.")
    return df

# Função para predição
def predict_with_model(df):
    st.write("Iniciando predição com o modelo...")
    dmatrix = xgb.DMatrix(df)
    predictions = model.predict(dmatrix)
    st.write("Predição concluída.")
    return predictions

# Interface no Streamlit
st.title("MVP com Feature Engineering e Predição")
st.write("Clique no botão para consultar os últimos 10 dados do Firebase, processá-los e prever no modelo.")

if st.button("Consultar e Prever"):
    st.write("Buscando dados do Firebase...")
    data = ref.order_by_key().limit_to_first(20).get()
    st.write("Dados carregados.")

    st.write("Convertendo dados para DataFrame...")
    df = pd.DataFrame.from_dict(data, orient='index')
    st.write(f"DataFrame criado com {len(df)} linhas.")

    if df.empty:
        st.error("Nenhum dado encontrado no Firebase!")
    else:
        st.write("Iniciando o processamento dos dados...")
        processed_data = preprocess_data(df)

        st.write("Realizando predições...")
        predictions = predict_with_model(processed_data)

        st.write("Adicionando predições ao DataFrame...")
        processed_data['Predição'] = (predictions > 0.5).astype(int)

        st.write("Resultados Previstos:")
        st.dataframe(processed_data[['Predição']])

