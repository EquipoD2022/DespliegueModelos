import streamlit as st
#importamos librerias
import xgboost as xgb
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas_datareader as datas
import matplotlib.pyplot as plt


def app():
    #start = '2004-08-18'
    #end = '2022-01-20'
    start = st.date_input('Inicio(Start)',value=pd.to_datetime('2010-01-01'))
    end = st.date_input('Fin' , value=pd.to_datetime('today'))
    
    st.title('Predicción de tendencia de acciones')
    st.write('Este modelo utiliza el algoritmo XGBoost para predecir la tendencia de las acciones')
    #describir el modelo y sus caracteristicas
    st.subheader('Descripción del modelo')

    user_input = st.text_input('Introducir cotización bursátil' , 'DOGE-EUR')
    startStr = start.strftime('%Y-%m-%d')
    endStr = end.strftime('%Y-%m-%d')
    df = datas.DataReader(user_input, 'yahoo', start, end)
    #renombra la columna de datetimne como Date
    df = df.reset_index()
    df = df.rename(columns={'Date':'Date'})
    #escribir un poco acerca de la empresa introducida en user_input
    # con la libreria de pandas_datareader podemos obtener informacion de la empresa
    st.subheader('Acerca de la empresa')
    st.write(datas.get_quote_yahoo(user_input))

    #hacemos un grafico de la serie de tiempo
    st.subheader('Serie de tiempo de la cotización bursátil de '+user_input)
    fig = px.line(df, x="Date", y="Close", title='Precio de cierre de '+ user_input)
    st.plotly_chart(fig)

    ## correlacion entre las variables en un mapa de calor
    st.subheader('Correlación entre las variables')
    fig = px.imshow(df.corr())
    st.plotly_chart(fig)

    # añadir columna Dayli Return%
    df['Daily Return %'] = (df['Adj Close'] / df['Adj Close'].shift(1)) - 1
    #mostramos los datos
    st.subheader('Datos del '+ startStr +' al '+endStr)
    st.write(df)

    # Separar Date en Year, Month, Day
    df['Year'] = pd.DatetimeIndex(df['Date']).year
    df['Month'] = pd.DatetimeIndex(df['Date']).month
    df['Day'] = pd.DatetimeIndex(df['Date']).day

    #separamos los datos en train y test
    train_split = 0.9
    # Set the date at which to split train and eval data
    # Of the unique dates available, pick the split between train and eval dates
    dates_avail = df["Date"].unique()
    split_date_index = int(dates_avail.shape[0] * train_split)
    split_date = dates_avail[split_date_index]
    # Train data is on or before the split date
    train_df = df.query("Date <= @split_date")
    # And eval data is after
    eval_df = df.query("Date > @split_date")

    features = ["Year", "Month", "Day", "High", "Low", "Open", "Volume", "Daily Return %"]
    label = ["Adj Close"]
    x_train = train_df[features]
    y_train = train_df[label]
    x_eval = eval_df[features]
    y_eval = eval_df[label]
    # intentar modificar los parametros del modelo
    #input de los parametros del modelo
    st.subheader('Parámetros del modelo')
    n_estimators = st.number_input('Número de estimadores', min_value=100, max_value=10000, value=300, step=100)
    max_depth = st.number_input('Profundidad máxima', min_value=1, max_value=20, value=5, step=1)
    min_child_weight = st.number_input('Peso mínimo del hijo', min_value=1, max_value=10, value=2, step=1)
    learning_rate = st.number_input('Tasa de aprendizaje', min_value=0.01, max_value=1.0, value=0.3, step=0.01)
    #hacer una barra de progreso
    bar = st.progress(0)
    # crear el modelo
    model = xgb.XGBRegressor(
        n_estimators = n_estimators,
        max_depth = max_depth,
        min_child_weight = min_child_weight,
        learning_rate = learning_rate,
    )
    bar.progress(10)

    model.fit(
        x_train,
        y_train,
        eval_set = [(x_train, y_train), (x_eval, y_eval)],
        early_stopping_rounds = 20,
        verbose = False
    )
    bar.progress(50)

    #mostramos la importancia de las variables
    st.subheader('Importancia de las variables')
    #imprimir como una tabla la importancia de las variables
    st.write(pd.DataFrame(model.feature_importances_, index=x_train.columns, columns=["Importancia"]).sort_values(by="Importancia", ascending=False))

    bar.progress(100)

    # evaluar el modelo
    # Crear un dataframe final para verificar las predicciones
    df_pred = x_eval.copy()

    # Recrear una columna para la fecha completa y ponerla como primer columna
    date_columns = ["Year", "Month", "Day"]
    df_pred["Date"] = df_pred[date_columns].apply(
        lambda x: "-".join(x.values.astype(str)),
        axis = 1
    )

    # Predecir datos para el conjunto de datos de evaluación y 
    # guardar el Adj Close predicho como una nueva columna
    df_pred["Adj Close_Pred"] = model.predict(x_eval)
    # pasar la columa Date a la primera posicion




    # Usamos merge para combinar los datos de evaluación con los datos de predicción
    df_pred = df_pred.merge(
        y_eval, 
        how = "inner", 
        left_index = True, 
        right_index = True
    )

    # Mostrar los datos de predicción
    st.subheader('Datos de predicción')
    st.write(df_pred)

    # Graficar los datos de predicción
    st.subheader('Gráfico de predicción')
    # Graficar los datos de predicción vs los datos de evaluación
    title = "Predicción de precios de cierre de "+ user_input
    fig = px.line(
        df_pred,
        x = "Date",
        y = ["Adj Close", "Adj Close_Pred"],
        title = title
    )
    st.plotly_chart(fig)

    # evaluar el modelo
    mse = mean_squared_error(
    y_true = df_pred["Adj Close"],
    y_pred = df_pred["Adj Close_Pred"]
    )
    mae = mean_absolute_error(
        y_true = df_pred["Adj Close"],
        y_pred = df_pred["Adj Close_Pred"]
    )
    # Mean Absolute Percentage Error is the percentage of how off the predicted values are
    mape = (
        np.abs(df_pred["Adj Close"] - df_pred["Adj Close_Pred"]) / df_pred["Adj Close"]
    ).mean() * 100

    # Mostrar los errores en un gráfico 
    st.subheader('Gráfico de errores')
    # Graficar los errores
    title = "Errores de predicción de precios de cierre de "+ user_input
    fig = px.bar(
        x = ["MSE", "MAE", "MAPE"],
        y = [mse, mae, mape],
        title = title
    )
    st.plotly_chart(fig)

    # Mostrar los errores en una tabla de streamlit
    st.subheader('Metricas de error')
    st.write(pd.DataFrame(
        data = [mse, mae, mape],
        index = ["MSE", "MAE", "MAPE"],
        columns = ["Error"]
    ))
    #explicando que es cada error y un poco de teoria
    st.write('''
    * **MSE** (Mean Squared Error): Es la media de los errores al cuadrado. Es una medida de la varianza de los errores. Cuanto menor sea el MSE, mejor será el modelo.
    * **MAE** (Mean Absolute Error): Es la media de los valores absolutos de los errores. Es una medida de la dispersión de los errores. Cuanto menor sea el MAE, mejor será el modelo.
    * **MAPE** (Mean Absolute Percentage Error): Es la media de los porcentajes de los errores. Es una medida de la dispersión de los errores. Cuanto menor sea el MAPE, mejor será el modelo.
    ''')

