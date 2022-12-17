import streamlit as st
from multiapp import MultiApp

# import los modelos aqui
from apps import lsmt, home, regresionlineal, modeloGRU

app = MultiApp()

st.markdown(
    """
Actividad Semana n°13- Equipo D - Inteligencia de Negocios
"""
)
# Add all your application here
app.add_app("Home", home.app)
# Albert
app.add_app("Modelo GRU", modeloGRU.app) #piero
app.add_app("Modelo LSTM", lsmt.app)  # Sebas
# Rodrigo
# Juan
# Edward
# Fernando
# Angel
app.add_app("Modelo Regresión Lineal", regresionlineal.app)  # Vivian
# Aldair


# The main app
app.run()

