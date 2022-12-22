import re
from wordcloud import WordCloud, STOPWORDS
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import snscrape.modules.twitter as sntwitter
import nltk
import warnings

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
#from nltk.corpus import stopwords


def app():
    st.title("Extracción de Datos y Analisis de Sentimientos - Twitter")

    st.write(
    """Escribir el usuario de tu preferencia""" )
    st.write(
    """Por ejemplo:""")
    
    st.write("""- @BillGates --> BillGates""")
    st.write("""- @mayoredlee --> mayoredlee""")
    st.write("""- @elonmusk --> elonmusk""")

    ticker = st.text_input('Usuario en inglés a buscar:', 'elonmusk')
    st.write('Se buscara tweets del usuario:', ticker)

    st.subheader("Extracción de Datos")
    # Se crea un arreglo donde se guarda los twetts del personaje
    attributes_container = []
    maxTweets = 300
    # Se usa TwitterSearchScraper para realizar scrapping y obtener los tweets
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper('from:'+ticker).get_items()):
        if i>maxTweets:
            break
        attributes_container.append([tweet.user.username, tweet.date, tweet.likeCount, tweet.sourceLabel, tweet.content])

    # Creamos un dataframe para la lista de tweets obtenidos en el paso anterior
    tweets_df = pd.DataFrame(attributes_container, columns=["Usuario","Fecha de Creacion", "Numero de (♡) ", "Fuente de Tweet", "Tweets"])
    st.write("Observamos los datos obtenidos")
    st.write(tweets_df)


    # Distribución temporal de los tweets
    # ==============================================================================
    
    st.subheader("Número de tweets publicados - Mes")
    fig, axs = plt.subplots(figsize=(9,4))
    
    for Usuario in tweets_df.Usuario.unique():
        df_temp = tweets_df[tweets_df['Usuario'] == Usuario].copy()
        df_temp['Fecha de Creacion'] = pd.to_datetime(df_temp['Fecha de Creacion'].dt.strftime('%Y-%m'))
        df_temp = df_temp.groupby(df_temp['Fecha de Creacion']).size()
        df_temp.plot(label=Usuario, ax=axs)

    axs.set_title('Número de tweets publicados por mes')
    axs.legend()
    st.pyplot(fig)  
    
    # **Limpieza y Tokenizacion**"""
    st.subheader("Limpieza y Tokenización de Data")
    
    def limpiar_tokenizar(nuevo_texto):
        # Esta función limpia y tokeniza el texto en palabras individuales.
        # El orden en el que se va limpiando el texto no es arbitrario.
        # El listado de signos de puntuación se ha obtenido de: print(string.punctuation)
        # y re.escape(string.punctuation)"""
        
        # Se convierte todo el texto a minúsculas
        nuevo_texto = nuevo_texto.lower()
        
        # Eliminación de páginas web (palabras que empiezan por "http")
        nuevo_texto = re.sub('http\S+', ' ', nuevo_texto)
        # Eliminación de números
        nuevo_texto = re.sub("\d+", ' ', nuevo_texto)
        # Eliminación de espacios en blanco múltiples
        nuevo_texto = re.sub("\\s+", ' ', nuevo_texto)
        #Remueve @menciones
        nuevo_texto = re.sub('@[A-Za-z0–9]+', '', nuevo_texto) 
        # Remueve '#' hash tag
        nuevo_texto = re.sub('#', '', nuevo_texto) 
        # Remueve RT (retuiteado)
        nuevo_texto = re.sub('RT[\s]+', '', nuevo_texto)
         # Remueve hipervínculos
        nuevo_texto = re.sub('https?:\/\/\S+', '', nuevo_texto)

        #Tokenización
        # Eliminación de signos de puntuación
        regex = '[\\!\\"\\#\\$\\%\\&\\\'\\(\\)\\*\\+\\,\\-\\.\\/\\:\\;\\<\\=\\>\\?\\@\\[\\\\\\]\\^_\\`\\{\\|\\}\\~]'
        nuevo_texto = re.sub(regex , ' ', nuevo_texto)
        # Tokenización por palabras individuales
        nuevo_texto = nuevo_texto.split(sep = ' ')
        # Eliminación de tokens con una longitud < 2
        nuevo_texto = [token for token in nuevo_texto if len(token) > 1]

        return (nuevo_texto)

    # Se aplica la función de limpieza y tokenización a cada tweet
    # ==============================================================================
    tweets_df["Tweets-Tokenizado"] = tweets_df["Tweets"].apply(lambda x: limpiar_tokenizar(x))
    
    st.write("Tweets limpios y tokenizados")
    st.write(tweets_df[['Tweets', 'Tweets-Tokenizado']])

    st.write("Conjunto de Datos limpios y tokenizados")
    st.write(tweets_df)

    st.subheader("Analisis Exploratorio")
    # Unnest de la columna texto_tokenizado
    # ==============================================================================
    st.write("Se agrega columna con Tweets limpios y tokenizados y se quita columna Tweets")
    tweets_tidy = tweets_df.explode(column='Tweets-Tokenizado')
    tweets_tidy = tweets_tidy.drop(columns='Tweets')
    tweets_tidy = tweets_tidy.rename(columns={'Tweets-Tokenizado':'token'})
    st.write(tweets_tidy.head(10))

    # Top 5 palabras más utilizadas por el autor
    # ==============================================================================
    st.write("Top 5 palabras más utilizadas por el autor")
    st.write(tweets_tidy.groupby(['Usuario','token'])['token'] \
    .count() \
    .reset_index(name='count') \
    .groupby('Usuario') \
    .apply(lambda x: x.sort_values('count', ascending=False).head(5)))


    st.subheader("Stop Words")
    # Obtención de listado de stopwords del inglés
    st.write("Obtención de listado de stopwords del inglés y Eliminarlos")
    from nltk.corpus import stopwords
    # ==============================================================================
    stop_words = list(stopwords.words('english'))
    # Se añade la stoprword: amp, ax, ex
    stop_words.extend(("amp", "xa", "xe"))
    st.write(stop_words[:10])

    # Filtrado para excluir stopwords
    # ==============================================================================
    tweets_tidy = tweets_tidy[~(tweets_tidy["token"].isin(stop_words))]

    # Top 10 palabras por autor (sin stopwords)
    # ==============================================================================
    st.write("Top 10 palabras del autor (sin stopwords)")
    fig, axs = plt.subplots(figsize=(6, 7))
    for i, Usuario in enumerate(tweets_tidy.Usuario.unique()):
        df_temp = tweets_tidy[tweets_tidy.Usuario == Usuario]
        counts  = df_temp['token'].value_counts(ascending=False).head(10)
        counts.plot(kind='barh', color='firebrick', ax=axs)
        axs.invert_yaxis()
        axs.set_title(Usuario)

    fig.tight_layout()
    st.pyplot(fig)

    # Visualización en graficos
    st.subheader("WordCloud")
    # =======================================================

    #Obtenemos solo los tweets
    only_tweets = tweets_df.iloc[:, 4].values

    #Asignamos esa lista de tweets a un dataframe
    tweets_t = pd.DataFrame({'Tweets': only_tweets})

    nltk.download('vader_lexicon')
    # Iniciamos el SentimentIntensityAnalyzer.
    vader = SentimentIntensityAnalyzer()

    # Apply lambda function to get compound scores. Aplicamos una función lambda para obtener el puntaje compuesto
    function = lambda texto: vader.polarity_scores(texto)['compound']
    tweets_t['compound'] = tweets_t['Tweets'].apply(function)

    # Realizamos su visualización en un WordCloud
    import seaborn as sns

    allWords = ' '.join([twts for twts in tweets_t['Tweets']])
    wordCloud = WordCloud(width=500, height=300, random_state=21, max_font_size=110).generate(allWords)
    st.write("Grafico de palabras mas usadas en WordCloud")
    fig1 = plt.figure()
    plt.imshow(wordCloud, interpolation="bilinear")
    plt.axis('off')
    st.pyplot(fig1)

    st.write("Se observa tanto como en el grafico de Stop Words y WordCloud se obtiene datos similares")

    #Empieza sección de analisis de sentimientos
    st.subheader("Analisis de Sentimiento")

    # Descarga lexicon sentimientos
    # ==============================================================================
    lexicon = pd.read_table(
                'https://raw.githubusercontent.com/fnielsen/afinn/master/afinn/data/AFINN-en-165.txt',
                names = ['termino', 'sentimiento']
            )
    
    # Sentimiento promedio de cada tweet
    # ==============================================================================
    tweets_sentimientos = pd.merge(
                                left     = tweets_tidy,
                                right    = lexicon,
                                left_on  = "token", 
                                right_on = "termino",
                                how      = "inner"
                        )

    tweets_sentimientos = tweets_sentimientos.drop(columns = "termino")

    # Se suman los sentimientos de las palabras que forman cada tweet.
    tweets_sentimientos = tweets_sentimientos[["Usuario","Fecha de Creacion","sentimiento"]] \
                        .groupby(["Usuario", "Fecha de Creacion"])\
                        .sum().reset_index()
    st.write(tweets_sentimientos.head())

    
    st.write("Puntaje de Analisis de Sentimiento")
    def perfil_sentimientos(df):
        st.write("Usuario: ",Usuario)
        st.write("=" * 12)
        st.write(f"  Positivos: {round(100 * np.mean(df.sentimiento > 0), 2)}"," %")
        st.write(f"  Negativos: {round(100 * np.mean(df.sentimiento < 0), 2)}"," %")
        st.write(f"  Neutros  : {round(100 * np.mean(df.sentimiento == 0), 2)}"," %")
        st.write(" ")

    for Usuario, df in tweets_sentimientos.groupby("Usuario"):
        st.write(perfil_sentimientos(df))
    
    def score_sentimiento(score_sent):
        if  score_sent< 0:
            round(100 * np.mean(df.sentimiento < 0), 2)
            return 'Negativo'
        elif score_sent == 0:
            round(100 * np.mean(score_sent == 0), 2)
            return 'Neutral'
        else:
            round(100 * np.mean(df.sentimiento > 0), 2)
            return 'Positivo'
            
    # GRAFICO DE BARRAS
    st.subheader("Grafico de Analisis de Sentimiento")
    
    tweets_sentimientos['sentimiento'] = tweets_sentimientos['sentimiento'].apply(score_sentimiento)
    st.write(tweets_sentimientos['sentimiento'].value_counts()) 

    fig2 = plt.figure()
    plt.title('Grafico - Análisis de sentimiento')
    plt.xlabel('Sentimiento')
    plt.ylabel('Puntaje')
    tweets_sentimientos['sentimiento'].value_counts().plot(kind = 'bar', color='Purple')
    st.pyplot(fig2)

    #SENTIMIENTO PROMEDIO DE LOS TWEETS POR MES
    
    # fig3, axss = plt.subplots(figsize=(7, 4)) 

    # for Usuario in tweets_sentimientos.Usuario.unique():
    #    df = tweets_sentimientos[tweets_sentimientos.Usuario == Usuario].copy()
    #    df = df.set_index("Fecha de Creacion")
    #    df = df[['sentimiento']].resample('1M').mean()
    #    axss.plot(df.index, df.sentimiento , label=Usuario)

    #axss.set_title("Sentimiento promedio de los tweets por mes")
    #axss.legend()
    #st.pyplot(fig3)