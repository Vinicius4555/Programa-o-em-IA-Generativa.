# ==========================================================
# SUPER APP DE IA COM STREAMLIT + PANDAS
# ----------------------------------------------------------
# Projetos incluídos:
#
# 1. 🌧️ Previsão de Chuva com TensorFlow
# 2. 📩 Filtro Inteligente de Mensagens
# 3. 🏍️ Financiamento de Moto
# 4. 🎵 Recomendador Spotify IA
# 5. 🌱 Monitoramento de Plantas
#
# Tecnologias:
# - Streamlit
# - Pandas
# - TensorFlow
# - Scikit-Learn
# - Matplotlib
# ==========================================================

# ----------------------------------------------------------
# IMPORTAÇÃO DAS BIBLIOTECAS
# ----------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier

# ----------------------------------------------------------
# CONFIGURAÇÃO
# ----------------------------------------------------------

st.set_page_config(
    page_title="Super IA Didática",
    page_icon="🧠",
    layout="wide"
)

# ----------------------------------------------------------
# TÍTULO
# ----------------------------------------------------------

st.title("🧠 SUPER APP DE INTELIGÊNCIA ARTIFICIAL")

st.caption("""
Aplicação completa utilizando:
- Streamlit
- Pandas
- TensorFlow
- Machine Learning
""")

# ----------------------------------------------------------
# MENU LATERAL
# ----------------------------------------------------------

menu = st.sidebar.selectbox(

    "Escolha o sistema:",

    [

        "🌧️ Previsão de Chuva",
        "📩 Filtro de Mensagens",
        "🏍️ Financiamento de Moto",
        "🎵 Spotify IA",
        "🌱 Monitoramento de Plantas"

    ]

)

# ==========================================================
# 1. PREVISÃO DE CHUVA
# ==========================================================

if menu == "🌧️ Previsão de Chuva":

    st.header("🌧️ PREVISÃO DE CHUVA COM TENSORFLOW")

    # ------------------------------------------------------
    # DADOS
    # ------------------------------------------------------

    X_treino = np.array([

        [30, 40],
        [25, 80],
        [28, 30],
        [20, 90],
        [32, 20],
        [18, 85],
        [27, 75],
        [35, 15],
        [22, 88],
        [31, 25]

    ], dtype=float)

    y_treino = np.array([
        0, 1, 0, 1, 0,
        1, 1, 0, 1, 0
    ], dtype=float)

    # ------------------------------------------------------
    # MODELO
    # ------------------------------------------------------

    modelo = keras.Sequential([

        keras.layers.Dense(
            4,
            activation='relu',
            input_shape=(2,)
        ),

        keras.layers.Dense(
            1,
            activation='sigmoid'
        )

    ])

    modelo.compile(

        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']

    )

    # ------------------------------------------------------
    # TREINAMENTO
    # ------------------------------------------------------

    historico = modelo.fit(

        X_treino,
        y_treino,
        epochs=100,
        verbose=0

    )

    # ------------------------------------------------------
    # INPUTS
    # ------------------------------------------------------

    temperatura = st.slider(
        "Temperatura (°C)",
        0,
        45,
        25
    )

    umidade = st.slider(
        "Umidade (%)",
        0,
        100,
        60
    )

    # ------------------------------------------------------
    # PREVISÃO
    # ------------------------------------------------------

    previsao = modelo.predict(
        np.array([[temperatura, umidade]]),
        verbose=0
    )

    probabilidade = previsao[0][0]

    # ------------------------------------------------------
    # RESULTADO
    # ------------------------------------------------------

    if probabilidade > 0.5:

        st.success(
            f"🌧️ Chance de chuva: {probabilidade:.0%}"
        )

    else:

        st.warning(
            f"☀️ Pouca chance de chuva: {probabilidade:.0%}"
        )

    # ------------------------------------------------------
    # GRÁFICOS
    # ------------------------------------------------------

    fig, ax = plt.subplots()

    ax.plot(historico.history['accuracy'])

    ax.set_title('Acurácia da IA')

    ax.set_xlabel('Épocas')

    ax.set_ylabel('Acurácia')

    st.pyplot(fig)

# ==========================================================
# 2. FILTRO DE MENSAGENS
# ==========================================================

elif menu == "📩 Filtro de Mensagens":

    st.header("📩 FILTRO INTELIGENTE")

    # ------------------------------------------------------
    # BASE
    # ------------------------------------------------------

    dados = pd.DataFrame({

        'mensagem': [

            'Mensagem urgente da família',
            'Proposta comercial',
            'Oi tudo bem',
            'Ganhe dinheiro rápido'

        ],

        'categoria': [

            'Importante',
            'Relevante',
            'Desnecessária',
            'Spam'

        ]

    })

    # ------------------------------------------------------
    # IA
    # ------------------------------------------------------

    modelo = Pipeline([

        ('vetorizador', CountVectorizer()),

        ('classificador', MultinomialNB())

    ])

    modelo.fit(
        dados['mensagem'],
        dados['categoria']
    )

    # ------------------------------------------------------
    # INPUT
    # ------------------------------------------------------

    mensagem = st.text_area(
        "Digite uma mensagem:"
    )

    # ------------------------------------------------------
    # RESULTADO
    # ------------------------------------------------------

    if st.button("Analisar"):

        categoria = modelo.predict([mensagem])[0]

        st.subheader(
            f"Categoria: {categoria}"
        )

        if categoria == 'Spam':

            st.error("🚫 Spam detectado")

        elif categoria == 'Importante':

            st.success("⭐ Alta prioridade")

        elif categoria == 'Relevante':

            st.info("💼 Mensagem comercial")

        else:

            st.warning("📦 Mensagem comum")

# ==========================================================
# 3. FINANCIAMENTO
# ==========================================================

elif menu == "🏍️ Financiamento de Moto":

    st.header("🏍️ IA DE FINANCIAMENTO")

    # ------------------------------------------------------
    # BASE
    # ------------------------------------------------------

    dados = pd.DataFrame({

        'renda': [
            1500,
            3000,
            7000,
            15000
        ],

        'fidelidade': [
            2,
            5,
            8,
            10
        ],

        'aprovado': [
            0,
            1,
            1,
            1
        ]

    })

    X = dados[['renda', 'fidelidade']]

    y = dados['aprovado']

    modelo = DecisionTreeClassifier()

    modelo.fit(X, y)

    # ------------------------------------------------------
    # INPUTS
    # ------------------------------------------------------

    renda = st.number_input(
        "Renda Mensal",
        0
    )

    fidelidade = st.slider(
        "Fidelidade",
        1,
        10,
        5
    )

    # ------------------------------------------------------
    # PREVISÃO
    # ------------------------------------------------------

    resultado = modelo.predict([[
        renda,
        fidelidade
    ]])

    # ------------------------------------------------------
    # VERBA
    # ------------------------------------------------------

    verba = renda * fidelidade

    # ------------------------------------------------------
    # RESULTADO
    # ------------------------------------------------------

    if resultado[0] == 1:

        st.success("✅ Financiamento aprovado")

        st.metric(
            "Verba Liberada",
            f"R$ {verba:,.2f}"
        )

    else:

        st.error("❌ Financiamento recusado")

# ==========================================================
# 4. SPOTIFY IA
# ==========================================================

elif menu == "🎵 Spotify IA":

    st.header("🎵 RECOMENDADOR DE PLAYLIST")

    # ------------------------------------------------------
    # BASE
    # ------------------------------------------------------

    dados = pd.DataFrame({

        'frase': [

            'Estou feliz',
            'Estou triste',
            'Estou irritado',
            'Quero descansar'

        ],

        'humor': [

            'Alegre',
            'Depressivo',
            'Raivoso',
            'Preguiçoso'

        ]

    })

    modelo = Pipeline([

        ('vetorizador', CountVectorizer()),

        ('classificador', MultinomialNB())

    ])

    modelo.fit(
        dados['frase'],
        dados['humor']
    )

    # ------------------------------------------------------
    # PLAYLISTS
    # ------------------------------------------------------

    playlists = {

        'Alegre': 'Pop Hits',
        'Depressivo': 'Lofi Sad',
        'Raivoso': 'Rock Rage',
        'Preguiçoso': 'Chill Lofi'

    }

    # ------------------------------------------------------
    # INPUT
    # ------------------------------------------------------

    texto = st.text_input(
        "Como você está hoje?"
    )

    # ------------------------------------------------------
    # RESULTADO
    # ------------------------------------------------------

    if st.button("Gerar Playlist"):

        humor = modelo.predict([texto])[0]

        playlist = playlists[humor]

        st.success(
            f"🎧 Playlist: {playlist}"
        )

        felicidade = {

            'Alegre': 100,
            'Depressivo': 20,
            'Raivoso': 40,
            'Preguiçoso': 50

        }

        st.progress(
            felicidade[humor] / 100
        )

# ==========================================================
# 5. MONITORAMENTO DE PLANTAS
# ==========================================================

elif menu == "🌱 Monitoramento de Plantas":

    st.header("🌱 IA DE PLANTAS")

    # ------------------------------------------------------
    # BASE
    # ------------------------------------------------------

    dados = pd.DataFrame({

        'folhas': [
            1,
            0,
            2
        ],

        'agua': [
            600,
            100,
            1500
        ],

        'sol': [
            7,
            2,
            12
        ],

        'estado': [

            'Bom Estado',
            'Desidratada',
            'Morta'

        ]

    })

    X = dados[['folhas', 'agua', 'sol']]

    y = dados['estado']

    modelo = DecisionTreeClassifier()

    modelo.fit(X, y)

    # ------------------------------------------------------
    # INPUTS
    # ------------------------------------------------------

    folha = st.selectbox(

        "Cor das folhas",

        [

            'Verdes',
            'Amarelas',
            'Marrons'

        ]

    )

    if folha == 'Verdes':

        folha_num = 1

    elif folha == 'Amarelas':

        folha_num = 0

    else:

        folha_num = 2

    agua = st.slider(
        "Quantidade de água",
        0,
        2000,
        500
    )

    sol = st.slider(
        "Horas de sol",
        0,
        15,
        5
    )

    # ------------------------------------------------------
    # PREVISÃO
    # ------------------------------------------------------

    estado = modelo.predict([[

        folha_num,
        agua,
        sol

    ]])[0]

    # ------------------------------------------------------
    # RESULTADO
    # ------------------------------------------------------

    st.subheader(
        f"🌿 Estado da planta: {estado}"
    )

    if estado == 'Bom Estado':

        st.success(
            "Planta saudável"
        )

    elif estado == 'Desidratada':

        st.warning(
            "Pouca água"
        )

    else:

        st.error(
            "Planta em risco"
        )

    # ------------------------------------------------------
    # GRÁFICO
    # ------------------------------------------------------

    grafico = pd.DataFrame({

        'Água': [100, 500, 1000, 1500],
        'Saúde': [20, 80, 50, 10]

    })

    st.line_chart(
        grafico,
        x='Água',
        y='Saúde'
    )

# ==========================================================
# FINAL
# ==========================================================

st.markdown("---")

st.caption("""
Projeto educacional utilizando:
Python + Pandas + Streamlit + TensorFlow + Machine Learning
""")