    # ---------------------------------------------
# IA Simples com Scikit-Learn + Streamlit
# Regressão Linear: Horas de estudo x Notas
# ---------------------------------------------

# IMPORTANDO BIBLIOTECAS
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ---------------------------------------------
# TÍTULO DA PÁGINA
# ---------------------------------------------

st.title('📚 Inteligência Artificial para Iniciantes')
st.caption('Previsão de notas baseada nas horas de estudo.')

# ---------------------------------------------
# CRIANDO O DATAFRAME
# ---------------------------------------------

estudos = pd.DataFrame({
    'notas': [1, 2, 4, 6, 8, 10],
    'horas': [2, 4, 5, 7, 9, 10]
})

# MOSTRANDO OS DADOS
st.subheader('📊 Dados Utilizados')
st.dataframe(estudos)

# ---------------------------------------------
# SEPARANDO DADOS
# X = horas de estudo
# y = notas
# ---------------------------------------------

X = estudos[['horas']]
y = estudos['notas']

# ---------------------------------------------
# CRIANDO E TREINANDO O MODELO
# ---------------------------------------------

modelo = LinearRegression()

# TREINANDO A IA
modelo.fit(X, y)

# ---------------------------------------------
# SLIDER INTERATIVO
# ---------------------------------------------

st.subheader('⏰ Escolha quantas horas estudar')

horas_estudo = st.slider(
    'Horas de estudo',
    min_value=1,
    max_value=15,
    value=5
)

# ---------------------------------------------
# FAZENDO PREVISÃO
# ---------------------------------------------

previsao = modelo.predict([[horas_estudo]])

# MOSTRANDO RESULTADO
st.success(
    f'📈 Nota prevista para {horas_estudo} horas de estudo: {previsao[0]:.2f}'
)

# ---------------------------------------------
# GERANDO LINHA DE REGRESSÃO
# ---------------------------------------------

predicoes = modelo.predict(X)

# ---------------------------------------------
# CRIANDO O GRÁFICO
# ---------------------------------------------

fig, ax = plt.subplots()

# PONTOS REAIS
ax.scatter(
    estudos['horas'],
    estudos['notas']
)

# LINHA DA IA
ax.plot(
    estudos['horas'],
    predicoes
)

# PONTO DA PREVISÃO ATUAL
ax.scatter(
    horas_estudo,
    previsao[0],
    s=200
)

# PERSONALIZAÇÃO
ax.set_title('Regressão Linear')
ax.set_xlabel('Horas de Estudo')
ax.set_ylabel('Notas')

# MOSTRANDO GRÁFICO
st.pyplot(fig)

# ---------------------------------------------
# EXPLICAÇÃO FINAL
# ---------------------------------------------

st.subheader('🧠 O que a IA fez?')

st.write('''
A Inteligência Artificial analisou os dados de:
- horas de estudo
- notas obtidas

Depois disso, ela criou uma linha matemática chamada
**Regressão Linear**.

Essa linha tenta prever qual nota uma pessoa pode tirar
com base nas horas estudadas.
''')