import streamlit as st
from multiapp import MultiApp

# import los modelos aqui
from apps import lsmt, home, regresionlineal, modeloGRU, svr, svm, EyA_Twitter

app = MultiApp()

st.markdown(
    """
Actividad Semana n°13- Equipo D - Inteligencia de Negocios
"""
)
# Add all your application here
app.add_app("Home", home.app)
# Albert
app.add_app("Modelo GRU", modeloGRU.app)  # piero
app.add_app("Modelo LSTM", lsmt.app)  # Sebas
# Rodrigo
# Juan
# Edward
app.add_app("Modelo SVR", svr.app)  # Fernando
app.add_app("Modelo SVM", svm.app)# Angel
app.add_app("Modelo Regresión Lineal", regresionlineal.app)  # Vivian
# Aldair

# Twitter
app.add_app("Extracción de Datos y Analisis de Sentimientos - Twitter", EyA_Twitter.app) # Vivian
# The main app
app.run()
