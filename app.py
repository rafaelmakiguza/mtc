#!/usr/bin/env python
# coding: utf-8

# In[ ]:

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

        # Contagem de positivos verdadeiros e totais previstos como classe 1
        true_positives = valid_predictions[
            (valid_predictions['Predição'] == 1) & (valid_predictions['color'] == 'White')
        ].shape[0]

        predicted_positives = valid_predictions['Predição'].sum()

        # Calcular a precisão
        precision = (true_positives / predicted_positives) if predicted_positives > 0 else 0

        st.metric("Classe 1 Previsões", f"{predicted_positives}")
        st.metric("Precisão Classe 1", f"{precision:.2%}")

        # Estilizar tabela
        def highlight_row(row):
            if row['Predição'] == 1:
                return ['background-color: darkgreen'] * len(row)
            return [''] * len(row)

        styled_df = result_df.style.apply(highlight_row, axis=1)
        st.dataframe(styled_df)
