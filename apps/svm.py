import streamlit as st
import pandas_datareader as datas
# Machine learning
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# For data manipulation
import pandas as pd
import numpy as np

# To plot
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

# To ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Libreria de yahoo financie
import yfinance as yf


def app():

    st.title("Model 8 - SVM Model")
    st.title("Predicción de tendencia de acciones del Mes usando SVM")
    start = st.date_input(
        "Inicio (Start)", value=pd.to_datetime("2022-12-01")
    )  # 2021-01-01
    end = st.date_input("Fin (End)", value=pd.to_datetime("today"))
    user_input = st.text_input("Introducir cotización bursátil", "NFLX")
    # df = datas.DataReader(user_input, "yahoo", start, end)
    df = yf.download(user_input, start, end)
    df.index=df.index.strftime('%Y-%m-%d')
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
    st.subheader("filas y columnas")
    st.subheader(df.shape)

    actual_prices = df.tail(1)
    actual_prices
    st.subheader(actual_prices)

    # Crear variables predictoras
    df['Open-Close'] = df.Open - df.Close
    df['High-Low'] = df.High - df.Low
  
    # Almacenar todas las variables predictoras en una variable X
    X = df[['Open-Close', 'High-Low']]
    
    # Variables objetivo
    y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

    #Dividiendo los datos para entrenamiento y prueba
    split_percentage = 0.8
    split = int(split_percentage*len(df))
  
    # Train data set
    X_train = X[:split]
    y_train = y[:split]
  
    # Test data set
    X_test = X[split:]
    y_test = y[split:]

    # Support vector classifier
    cls = SVC().fit(X_train, y_train)

    # Vamos a predecir la señal (compra o venta) utilizando la función cls.predict().
    df['Predicted_Signal'] = cls.predict(X)

    # Calcular daily Return
    df['Return'] = df.Close.pct_change()

    # Calcular Strategy Return
    df['Strategy_Return'] = df.Return *df.Predicted_Signal.shift(1)

    # Calculate Cumulutive returns
    df['Cum_Ret'] = df['Return'].cumsum()

    # Plot Strategy Cumulative returns 
    df['Cum_Strategy'] = df['Strategy_Return'].cumsum()

    # Graficamos Plot Strategy vs Original Returns
    st.subheader("Plot Strategy vs Original Retuns")
    fig1 = plt.figure(figsize=(16, 8))
    plt.plot(df['Cum_Ret'],color='red')
    plt.plot(df['Cum_Strategy'],color='blue')
    st.pyplot(fig1)
