
import streamlit as st
from multiapp import MultiApp
# import los modelos aqui
from apps import model9,model10, home,model12, model11

app = MultiApp()

st.markdown("""
Actividad Semana n°13- Equipo D - Inteligencia de Negocios
""")
# Add all your application here
app.add_app("Home", home.app)
app.add_app("Modelo XGBoost", model10.app)
app.add_app("Modelo LSTM", model9.app)
app.add_app("Modelo Random Forest Regressor", model11.app)
app.add_app("PCA and Hierarchical Portfolio Optimisation", model12.app)


# The main app
app.run()