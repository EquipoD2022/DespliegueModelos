import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout,GRU
from keras import optimizers 
import pandas_datareader as datas
from itertools import cycle
import plotly.express as px
import yfinance as yf

def app():
    st.title('Modelo GRU')
    start = st.date_input('Start' , value=pd.to_datetime('2020-12-01'))
    end = st.date_input('End' , value=pd.to_datetime('today'))
    
    st.title('Predicción con el modelo de acciones')

    user_input = st.text_input('Introducir simbolo de la empresa en Yahoo Finance ' , 'AMZN')

    # df = datas.DataReader(user_input, "yahoo", start, end)
    df = yf.download(user_input, start, end)
    df.index=df.index.strftime('%Y-%m-%d')
    df.reset_index(inplace=True)

    st.subheader('Tabla de datos historica de la empresa') 
    st.write(df) #Tabla con todos los datos

    st.subheader('Valores estadisticos de la tabla') 
    st.write(df.describe()) #Tabla con valores estaditicos

    st.subheader('Modelo Gated Recurrent Unit') 
    
    # Grafica de valores vs Date
    continent_list = list(df.columns) #Valores (Open,Close,High,Low,AdjClose)
    seleccion = st.selectbox(label = "Seleccione el movimiento",options=continent_list)

    fig = px.line(
        df,
        x=df.index,
        y=seleccion,
        title="Precio de " + seleccion + " vs Date",
    )
    st.plotly_chart(fig)
    
    #Normalización de datos
    st.subheader('Precio de Cierre de las acciones')
    dataset = pd.DataFrame(df['Close'])
    dataset_norm = dataset.copy()
    dataset[['Close']]
    scaler = MinMaxScaler(feature_range=(0,1))
    dataset_norm['Close'] = scaler.fit_transform(dataset[['Close']])
    
    # Gráfica de data normalizada de Close
    fig2 = px.line(
        dataset_norm,
        x=dataset_norm.index,
        y=dataset_norm['Close'],
        title="Grafica con valores normalizados de Close",
    )
    st.plotly_chart(fig2)

    st.subheader('Partición de los Datos de valores Close')
    # Partición de datos en datos de entrenamiento, validación y prueba
    totaldata = df.values
    totaldatatrain = int(len(totaldata)*0.7)
    totaldataval = int(len(totaldata)*0.1)
    totaldatatest = int(len(totaldata)*0.2)

    # Datos en cada partición
    training_set = dataset_norm[0:totaldatatrain]
    val_set = dataset_norm[totaldatatrain:totaldatatrain+totaldataval]
    test_set = dataset_norm[totaldatatrain+totaldataval:]
    
    #Datos Entrenados
    st.subheader("1) Datos Entrenados")
    st.write(training_set) # imprime la tabla
    fig3 = px.line(
        training_set,
        x=training_set.index,
        y=training_set['Close'],
        title="Datos Entrenados (70%) ",
    )
    st.plotly_chart(fig3)

    #Datos Validacion
    st.subheader("2) Datos de Validacion")
    st.write(val_set)# imprime la tabla
    fig4 = px.line(
        val_set,
        x=val_set.index,
        y=val_set['Close'],
        title="Datos Validacion (10%) ",
    )
    st.plotly_chart(fig4)

    #Datos Prueba
    st.subheader("2) Datos de Prueba")
    st.write(test_set)# imprime la tabla
    fig5 = px.line(
        test_set,
        x=test_set.index,
        y=test_set['Close'],
        title="Datos Prueba (20%) ",
    )
    st.plotly_chart(fig5)

    #Sliding Windows
    # Iniciamos la variable lag
    lag = 2
    def create_sliding_windows(data,len_data,lag):
        x=[]
        y=[]
        for i in range(lag,len_data):
            x.append(data[i-lag:i,0])
            y.append(data[i,0]) 
        return np.array(x),np.array(y)

    array_training_set = np.array(training_set)
    array_val_set = np.array(val_set)
    array_test_set = np.array(test_set)
    
    x_train, y_train = create_sliding_windows(array_training_set,len(array_training_set), lag)
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

    x_val,y_val = create_sliding_windows(array_val_set,len(array_val_set),lag)
    x_val = np.reshape(x_val, (x_val.shape[0],x_val.shape[1],1))

    x_test,y_test = create_sliding_windows(array_test_set,len(array_test_set),lag)
    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
    
    
    # Modelo GRU (Gated Recurrent Unit
    # Parametros para el modelos GRU
    hidden_unit = 32
    batch_size= 32
    epoch = 100
    #Modelo GRU
    regressorGRU = Sequential()
    
    # Primer capa GRU
    regressorGRU.add(GRU(units=hidden_unit, return_sequences=True, input_shape=(x_train.shape[1],1), activation = 'tanh'))

    # Segundo capa GRU
    regressorGRU.add(GRU(units=hidden_unit, return_sequences=True, activation = 'tanh'))

    # Tercer capa GRU con Dropout
    regressorGRU.add(GRU(units=hidden_unit, return_sequences=False, activation = 'tanh'))
    regressorGRU.add(Dropout(0.20))
    
    # Dense
    regressorGRU.add(Dense(units=1))
    
    # Compilando el modelo Gated Recurrent Unit
    # regressorGRU.compile(optimizer=optimizers.Adam(lr=learning_rate),loss='mean_squared_error')
    regressorGRU.compile(loss='mean_squared_error',optimizer='adam')
    
    # Ajustar la data de entrenamiento y la data de validación 
    pred = regressorGRU.fit(x_train, y_train, validation_data=(x_val,y_val), batch_size=batch_size, epochs=epoch)
    
    st.subheader("Grafica de perdida (loss) de entrenamiento y validacion")
    fig6 = plt.figure(figsize=(10, 4))
    plt.plot(pred.history['loss'],'r', label='Perdida de Entrenamiento')
    plt.plot(pred.history['val_loss'],'b', label='Perdida de Validacion')
    plt.title('Pérdida de entrenamiento y validación')
    plt.ylabel('Perdida')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    st.pyplot(fig6)
    
    y_pred_test = regressorGRU.predict(x_test)

    y_pred_invert_norm = scaler.inverse_transform(y_pred_test)
    
    st.subheader('Comparación de los datos de prueba con los resultados predichos')
    datacompare = pd.DataFrame()
    datatest=np.array(dataset['Close'][totaldatatrain+totaldataval+lag:])
    datapred= y_pred_invert_norm

    datacompare['Data Test'] = datatest
    datacompare['Resultados predichos'] = datapred
    st.write(datacompare)
        
    st.subheader('Evaluación de los resultados predichos')    

    def rmse(datatest, datapred):
        return np.round(np.sqrt(np.mean((datapred - datatest) ** 2)), 4)
    
    def mape(datatest, datapred): 
        return np.round(np.mean(np.abs((datatest - datapred) / datatest) * 100), 4)
    
    st.write('Métricas de evaluación RMSE y MAPE')
    st.write(' -> Resultados RMSE Prediccion Modelo :',rmse(datatest, datapred))
    st.write(' -> Resultados MAPE Prediccion Modelo : ', mape(datatest, datapred), '%')
    
    st.header("Gráfico de los datos de prueba y los datos de predicción")
    fig7=plt.figure(num=None, figsize=(10, 4), dpi=80,facecolor='w', edgecolor='k')
    plt.title('Gráfico comparativo Data Actual y Data Predicha con el modelo GRU')
    plt.plot(datacompare['Data Test'], color='red',label='Test de Datos',linewidth=2)
    plt.plot(datacompare['Resultados predichos'], color='blue',label='Resultados del Modelo Predichos',linewidth=2)
    plt.xlabel('Fecha ')
    plt.ylabel('Precio')
    plt.legend()
    st.pyplot(fig7)
    st.write('Gracias :D')
