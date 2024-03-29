import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import pandas_datareader as datas
from pandas_datareader import data as pdr
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from pandas_datareader.yahoo.daily import YahooDailyReader


def app():
    st.title("Modelo Regression Lineal")

    yf.pdr_override()
    start = st.date_input("Start", value=pd.to_datetime("2021-12-02"))
    end = st.date_input("End", value=pd.to_datetime("today"))

    st.title("Predicción de tendencia de acciones")

    user_input = st.text_input("Introducir cotización bursátil", "TSLA")
    
    # 1 Forma
    # df = datas.DataReader(user_input, "yahoo", start, end)
    
    # 2 Forma
    # df = yf.download(user_input, start, end)
    # df.index=df.index.strftime('%Y-%m-%d')
    # df.reset_index(inplace=True)

    # 3 Forma
    df=pdr.get_data_yahoo(user_input,start,end)

    # Describiendo los datos
    st.subheader("Datos del Diciembre - 2021 al Octubre - 2022")
    st.write(df)
    st.subheader("Descripción de la dataset")
    st.write(df.describe())

    # Visualizaciones
    st.subheader("Closing Price vs Time")
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.Close)
    st.pyplot(fig)

    # Candlestick chart
    st.subheader("Gráfico Financiero")
    candlestick = go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"]
    )

    fig = go.Figure(data=[candlestick])

    fig.update_layout(width=800, height=600, title=user_input, yaxis_title="Precio")

    st.plotly_chart(fig)

    # Añadiendo indicadores para el modelo
    df["Open-Close"] = df.Open - df.Close
    df["High-Low"] = df.High - df.Low
    df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)

    # Modelo RL

    ## Variables predictoras
    X = df[["Open-Close", "High-Low"]]
    ## Variable objetivo
    y = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
    ## División data de entrenamiento y prueba
    split_percentage = 0.8
    split = int(split_percentage * len(df))
    ## Entrenando el dataset
    X_train = X[:split]
    y_train = y[:split]
    ## Testeando el dataset
    X_test = X[split:]
    y_test = y[split:]
    ## Creación del modelo
    modelo = LinearRegression().fit(X_train, y_train)
    ## Predicción del test
    y_pred = modelo.predict(X_test)

    # Ecuación
    st.subheader("Ecuación de Regresión Lineal")
    st.write("Coefficients: " + str(modelo.coef_))
    st.write("Intercept: " + str(modelo.intercept_))

    # Señal de predicción

    df["Predicted_Signal"] = modelo.predict(X)
    ## Añadiendo columna condicional
    conditionlist = [(df["Predicted_Signal"] == 1), (df["Predicted_Signal"] == 0)]
    choicelist = ["Comprar", "Vender"]
    df["Decision"] = np.select(conditionlist, choicelist)
    st.subheader("Predicción de Señal de compra o venta")
    st.write(df)

    # Señal de compra o venta Original vs Predecido
    st.subheader("Señal de compra o venta Original vs Predecido")
    st.write(df[["Target", "Predicted_Signal"]])

    # Evaluación del modelo

    st.title("Evaluación del Modelo Regresion Lineal")
    ## Matriz de confusión
    # cm = pd.DataFrame(confusion_matrix(y_test, y_pred))
    # st.subheader('Matriz de confusión')
    # st.write(cm)

    ## Métricas
    MAE = metrics.mean_absolute_error(y_test, y_pred)
    MSE = metrics.mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

    metricas = {
        "metricas": [
            "Mean Absolute Error",
            "Mean Squared Error",
            "Root Mean Squared Error",
        ],
        "valor": [MAE, MSE, RMSE],
    }

    metricas = pd.DataFrame(metricas)
    ### Gráfica de las métricas
    st.subheader(" Grafico Métricas de rendimiento")

    st.write("MAE: " + str(MAE))
    st.write("MSE: " + str(MSE))
    st.write("RMSE: " + str(RMSE))

    fig = px.bar(
        metricas,
        x="metricas",
        y="valor",
        title="Métricas de Regresion Lineal",
        color="metricas",
    )
    st.plotly_chart(fig)
