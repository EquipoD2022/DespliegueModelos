import streamlit as st
import numpy as np
import pandas as pd
import pandas_datareader as datas
import plotly.express as px
import keras
import datetime

from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
import yfinance as yf

# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers import Dropout
# from keras.layers import DropoutWrapper


def app():

    st.title("Model 3 - LSTM Model")

    st.title("Predicción de tendencia de acciones usando LSTM")

    start = st.date_input(
        "Inicio (Start)", value=pd.to_datetime("2021-01-01")
    )  # 2021-01-01
    end = st.date_input("Fin (End)", value=pd.to_datetime("today"))

    user_input = st.text_input("Introducir cotización bursátil", "GLD")
    
    # df = datas.DataReader(user_input, "yahoo", start, end)
    df = yf.download(user_input, start, end)
    df.index=df.index.strftime('%Y-%m-%d')
    df.reset_index(inplace=True)
    
    st.subheader("Acerca de la empresa")
    st.write(datas.get_quote_yahoo(user_input))
    st.subheader("Datos de la empresa")
    st.write(df)

    column_names = df.columns.values
    seleccion = st.selectbox(
        label="Seleccione el campo que desea predecir",
        options=column_names,
        index=3,
    )

    # Arreglamos la fecha
    df.reset_index(inplace=True)
    df.set_index("Date", inplace=True)

    st.subheader(
        "Serie de tiempo de la cotización bursátil de "
        + user_input
        + " vs "
        + seleccion
    )
    fig = px.line(
        df,
        x=df.index,
        y=seleccion,
        title="Precio de " + seleccion + " de " + user_input,
    )
    st.plotly_chart(fig)

    # mostrar un grafico de la media movil de la serie de tiempo
    st.subheader(
        "Media movil de la cotización bursátil de " + user_input + " vs " + seleccion
    )
    fig = px.line(
        df,
        x=df.index,
        y=seleccion,
        title="Precio de " + seleccion + " de " + user_input,
    )
    fig.add_scatter(
        x=df.index,
        y=df[seleccion].rolling(30).mean(),
        mode="lines",
        name="Media movil",
    )
    st.plotly_chart(fig)

    # Número de filas que tiene la data
    length_data = len(df)
    # %70 entrenamiento + %30 validacion
    split_ratio = 0.7
    length_train = round(length_data * split_ratio)

    # Número de tiempo de pasos (agrupación)
    time_step = 50

    # Dividimos los valores de la data
    # Para entrenamiento
    train_data = df.filter([seleccion])[:length_train]
    train_data_values = train_data.values

    # Para validación
    validation_data = df.filter([seleccion])[length_train:]
    validation_data_values = df.filter([seleccion])[length_train - time_step :].values

    # Escalarizamos los valores en un rango de 0 y 1
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Para los datos de entrenamiento
    scaled_dataset_train = scaler.fit_transform(train_data_values.reshape(-1, 1))

    # Para los datos de validación
    scaled_dataset_validation = scaler.fit_transform(
        validation_data_values.reshape(-1, 1)
    )

    # Creando X_train y y_train de la data de entrenamiento
    # Listas vacias
    X_train = []
    y_train = []

    # Agregamos valores a las listas
    for i in range(time_step, len(scaled_dataset_train)):
        X_train.append(scaled_dataset_train[i - time_step : i, 0])
        y_train.append(scaled_dataset_train[i, 0])

    # Convertimos las listas a array
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Transformamos la dimensión de X_train (array 3D)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Creando X_test de la data de validación
    X_test = []

    # Agregamos valores a las listas
    for i in range(time_step, len(scaled_dataset_validation)):
        X_test.append(scaled_dataset_validation[i - time_step : i, 0])

    # Convertimos la lista a array
    X_test = np.array(X_test)

    # Transformamos la dimensión de X_test (array 3D)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Modelo secuencial que consta de una pila lineal de capas.
    model = keras.Sequential()

    # Agregar una capa LSTM dándole 100 unidades de red.
    # Establecer return_sequence en verdadero para que la salida de la capa sea otra secuencia de la misma longitud.
    model.add(
        layers.LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1))
    )

    # Agregare otra capa LSTM con también 100 unidades de red.
    # Configuramos return_sequence en falso por esta vez para devolver solo la última salida en la secuencia de salida.
    model.add(layers.LSTM(100, return_sequences=False))

    # Agregar una capa de red neuronal densamente conectada con 25 unidades de red
    model.add(layers.Dense(25))

    # Agregar una capa densamente conectada que especifique la salida de 1 unidad de red
    model.add(layers.Dense(1))

    # Adoptar el optimizador "adam" y establezca el error cuadrático medio como función de pérdida.
    model.compile(loss="mean_squared_error", optimizer="adam")

    # Entrenar el modelo ajustándolo con el conjunto de entrenamiento.
    # Podemos probar con batch_size de 1 y ejecutar el entrenamiento durante 15 épocas.
    model.fit(X_train, y_train, batch_size=1, epochs=15)

    # Predecir los precios de las acciones en función del conjunto de prueba.
    predictions = model.predict(X_test)

    # Usar el método de transformación_inversa para desnormalizar los precios de las acciones pronosticados.
    predictions = scaler.inverse_transform(predictions)

    # Agregando la columna Predictions a validation_data
    validation_data["Predictions"] = predictions

    # Visualización
    st.subheader(
        "Gráfico de predicción de precios de " + seleccion + " de " + user_input
    )
    fig = px.line(
        df,
        x=df.index,
        y=seleccion,
        title="Precio de " + seleccion + " de " + user_input,
    )
    fig.add_scatter(
        x=df.index[length_train:],
        y=validation_data[seleccion],
        mode="lines",
        name="Real " + seleccion,
    )
    fig.add_scatter(
        x=df.index[length_train:],
        y=validation_data["Predictions"],
        mode="lines",
        name="Predictions " + seleccion,
    )
    st.plotly_chart(fig)

    # Predecir una fecha

    date_tomorrow = df.index[-1]

    # Obteniendo las 50 últimas filas
    X_input = df.iloc[-time_step:].Close.values

    # Lo convertimos en 2D array y escalamos los valores
    X_input = scaler.fit_transform(X_input.reshape(-1, 1))

    # Remodelamos el array a 3D
    X_input = np.reshape(X_input, (1, time_step, 1))

    # Mostramos los resultados de la predicción
    LSTM_prediction = scaler.inverse_transform(model.predict(X_input))

    st.subheader(
        "Predicción de " + seleccion + " para la fecha de " + str(date_tomorrow.date())
    )
    st.write(
        "Se estima que el precio de "
        + seleccion
        + " será de "
        + str(LSTM_prediction[0, 0])
    )
