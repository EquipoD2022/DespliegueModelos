import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import pandas_datareader as datas
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from sklearn import metrics

def app():
    st.title('Random Forest Prediccion del Doge coin')

    start = st.date_input('Inicio',value=pd.to_datetime('2017-11-28'))
    end = st.date_input('Fin' , value=pd.to_datetime('today'))

    startStr = start.strftime('%Y-%m-%d')
    endStr = end.strftime('%Y-%m-%d')

    user_input = st.text_input('Introducir cotización bursátil' , 'DOGE-USD')
   
    df = datas.DataReader(user_input, 'yahoo', start, end)

    #mostra los datos
    st.write(df)

    st.title('Predicción de tendencia de acciones')

    # Visualizaciones
    st.subheader('Precio de cierre ajustado')
    fig = px.line(df,y='Adj Close')
    st.plotly_chart(fig)

    st.subheader('Patron de cambio diario')
    fig = plt.figure(figsize = (12,6))
    df['Adj Close'].pct_change().plot.hist(bins=50)
    plt.xlabel("Adjusted close 1 day percent change")
    st.pyplot(fig)

    ## correlacion entre las variables en un mapa de calor
    st.subheader('Correlación entre las variables')
    fig = px.imshow(df.corr())
    st.plotly_chart(fig)

    st.subheader('Feature engineering')

    #Creamos una lista de caracteristicas
    feature_names = ['Open','High','Low','Close','Adj Close','Volume']

    df['Volume_1d_change'] = df['Volume'].pct_change()

    volume_features = ['Volume_1d_change']
    feature_names.extend(volume_features)

    df['5d_future_close'] = df['Adj Close'].shift(-5)
    df['5d_close_future_pct'] = df['5d_future_close'].pct_change(5)

    df.dropna(inplace=True)

    st.write(df)

    X = df[feature_names]
    y = df['5d_close_future_pct']

    train_size = int(0.85 * y.shape[0])

    #Definiendo al conjunto de entrenamiento con el 85%
    X_train = X[:train_size]
    y_train = y[:train_size]
    #Definiendo al conjunto de entrenamiento con el 15% restante
    X_test = X[train_size:]
    y_test = y[train_size:]

    #hacer input de los parametros
    st.subheader('Parametros del modelo')
    n_estimators = st.number_input('Numero de arboles', min_value=1, max_value=1000, value=100, step=1)
    max_depth = st.number_input('Profundidad maxima', min_value=1, max_value=100, value=5, step=1)
    random_state = st.number_input('Semilla', min_value=1, max_value=100, value=2, step=1)

    #definimos el modelo
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        random_state=random_state
    )

    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    # crear un dataframe con los valores reales , los predichos y la fecha
    st.subheader('Valores reales vs predichos')
    df_pred = pd.DataFrame({'y_test':y_test, 'y_pred':y_pred}, index=y_test.index)
    st.write(df_pred)


    #hacer un grafico de la prediccion vs el valor real
    st.subheader('Prediccion con Random Forest')
    fig = px.line(df_pred, title='Prediccion con Random Forest')
    st.plotly_chart(fig)


    ## Métricas
    MAE=metrics.mean_absolute_error(y_test, y_pred)
    MSE=metrics.mean_squared_error(y_test, y_pred)
    RMSE=np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    metricas = {
        'metrica' : ['Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error'],
        'valor': [MAE, MSE, RMSE]
    }
    metricas = pd.DataFrame(metricas)

    ### Gráfica de las métricas
    st.subheader('Métricas de rendimiento')
    
    st.write(pd.DataFrame(
        data = [MSE, MAE, RMSE],
        index = ["MSE", "MAE", "MAPE"],
        columns = ["Error"]
    ))

    fig = px.bar(        
        metricas,
        x = "metrica",
        y = "valor",
        title = "Métricas del Modelo Random Forest Regressor",
        color="metrica"
    )
    st.plotly_chart(fig)

    
    
    
    