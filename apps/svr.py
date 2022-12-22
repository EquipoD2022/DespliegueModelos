import streamlit as st
from sklearn.svm import SVR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as datas
import yfinance as yf
import plotly.express as px


def app():

    st.title("Modelo SVR")
    st.title("Predicción de tendencia de acciones del Mes usando SVR")
    start = st.date_input(
        "Inicio (Start)", value=pd.to_datetime("2022-12-01")
    )  # 2022-12-01
    end = st.date_input("Fin (End)", value=pd.to_datetime("today"))
    user_input = st.text_input("Introducir cotización bursátil", "AVGO")

    # df = datas.DataReader(user_input, "yahoo", start, end)
    df = yf.download(user_input, start, end)
    df.index = df.index.strftime('%Y-%m-%d')
    df.reset_index(inplace=True)

    # Describiendo los datos
    st.subheader("Datos del Diciembre - 2022")
    st.write(df)
    st.subheader("Descripción de la dataset")
    st.write(df.describe())

    # Visualizaciones
    st.subheader("Closing Price vs Time")
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.Close)
    st.pyplot(fig)
    # se obtiene el numero de filas y columnas
    df.shape
    st.subheader("Filas y columnas")
   # st.subheader(df.shape)
    st.write(df.shape)
    actual_prices = df.tail(1)
    actual_prices
    st.subheader(actual_prices)
# Se crea las listas vacías para almacenar los datos independientes y dependientes

    days = list()
    adj_close_prices = list()
# Se obtiener las fechas y los precios de cierre ajustados
    df_days = df.loc[:, 'Date']
    df_adj_close = df.loc[:, 'Adj Close']
# se crea datos independientes de los dias

    for day in df_days:
        days.append([int(day.split('-')[2])])
# se crea datos independientes del precio de las acciones
    for adj_close_price in df_adj_close:
        adj_close_prices.append(float(adj_close_price))

    st.subheader("Dias separados encontrados del mes")
    # st.subheader(days)
    st.write(days)

    st.subheader("precio separados por dias")
    # st.subheader(adj_close_prices)
    st.write(adj_close_prices)
# creando los 3 vectores

    st.subheader("Creacion de modelos")
# Se crea y entrena un modelo SVR usando kernel lineal

    lin_svr = SVR(kernel='linear', C=1000.0)
    lin_svr.fit(days, adj_close_prices)

# Se crea y entrena un modelo SVR usando kernel polinomial
    poly_svr = SVR(kernel='poly', C=1000.0, degree=2)
    poly_svr.fit(days, adj_close_prices)

# Se crea y entrena un modelo SVR usando kernel rbf
    rbf_svr = SVR(kernel='rbf', C=1000.0, gamma=0.15)
    rbf_svr.fit(days, adj_close_prices)
    st.write("Prediccion de capital diario segun el modelo")
    st.write("Se crea y entrena un modelo SVR usando kernel Lineal",
             lin_svr.predict(days))

    st.write("Se crea y entrena un modelo SVR usando kernel polinomial",
             poly_svr.predict(days))

    st.write("Se crea y entrena un modelo SVR usando kernel rbf",
             rbf_svr.predict(days))

    st.subheader("Traza de comparacion de  los 3 modelos")
    fig1 = plt.figure(figsize=(16, 8))
    plt.scatter(days, adj_close_prices, color='red', label='data')
    plt.plot(days, rbf_svr.predict(days), color='green', label='Modelo RBF')
    plt.plot(days, poly_svr.predict(days),
             color='orange', label='Modelo Polinomeal')
    plt.plot(days, lin_svr.predict(days), color='blue', label='Modelo Lineal')
    plt.title('Comparacion de modelos')
    st.pyplot(fig1)


# Se muestra el precio previsto para el día dado
    st.write('Se muestra el precio previsto para el día dado')
    st.write('day = [[31]]')
    day = [[31]]
    st.subheader("RBF SVR prediccion")
    st.subheader(rbf_svr.predict(day))

    st.subheader("Lineal SVR prediccion: ")
    st.subheader(lin_svr.predict(day))

    st.subheader("Polynomial SVR prediccion")
    st.subheader(poly_svr.predict(day))
