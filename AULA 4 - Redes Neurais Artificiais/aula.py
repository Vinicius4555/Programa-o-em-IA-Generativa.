# ==========================================================
# PREVISÃO DE CANSAÇO EM GAMERS
# Utilizando Scikit-Learn + Streamlit
# Sem Matplotlib
# ==========================================================

# --------------------------
# IMPORTAÇÃO DAS BIBLIOTECAS
# --------------------------

import streamlit as st
import pandas as pd

from sklearn.linear_model import LinearRegression

# --------------------------
# TÍTULO DO APP
# --------------------------

st.title('Detecção de Cansaço em Gamers')
st.caption('Previsão de cansaço baseada nas horas jogadas')

# --------------------------
# BASE DE DADOS
# --------------------------

gamer = pd.DataFrame({
    'horas_jogo': [1, 2, 4, 6, 8, 10],
    'cansaco': [1, 2, 3, 5, 8, 10]
})

# Exibir dados
st.subheader('Base de Dados')
st.dataframe(gamer)

# --------------------------
# PREPARAÇÃO DOS DADOS
# --------------------------

X = gamer[['horas_jogo']]
y = gamer['cansaco']

# --------------------------
# MODELO DE IA
# --------------------------

modelo = LinearRegression()

modelo.fit(X, y)

# --------------------------
# ENTRADA DO USUÁRIO
# --------------------------

st.subheader('Simulação')

horas = st.slider(
    'Horas jogadas:',
    min_value=1,
    max_value=15,
    value=5
)

# --------------------------
# PREVISÃO
# --------------------------

entrada = pd.DataFrame({
    'horas_jogo': [horas]
})

previsao = modelo.predict(entrada)

# --------------------------
# RESULTADO
# --------------------------

st.success(
    f'Nível de cansaço previsto: {previsao[0]:.2f}'
)

# --------------------------
# ALERTAS
# --------------------------

if previsao[0] >= 8:
    st.warning('Alto nível de cansaço detectado.')
    st.info('Recomenda-se descanso e redução do tempo de jogo.')

elif previsao[0] >= 5:
    st.info('Cansaço moderado detectado.')
    st.write('Faça pausas durante as partidas.')

else:
    st.success('Nível de cansaço baixo.')

# --------------------------
# GRÁFICO NATIVO DO STREAMLIT
# --------------------------

st.subheader('Gráfico de Dados')

grafico = gamer.set_index('horas_jogo')

st.line_chart(grafico)

# --------------------------
# EQUAÇÃO DO MODELO
# --------------------------

st.subheader('Equação da Regressão Linear')

st.code(
    f'cansaco = ({modelo.coef_[0]:.2f} * horas_jogo) + {modelo.intercept_:.2f}'
)

# --------------------------
# INFORMAÇÕES DO MODELO
# --------------------------

st.subheader('Informações do Modelo')

st.write(f'Coeficiente Angular: {modelo.coef_[0]:.2f}')
st.write(f'Intercepto: {modelo.intercept_:.2f}')

# ==========================================================
# COMO EXECUTAR
# ==========================================================
#
# Salve como:
# app.py
#
# Execute no terminal:
#
# streamlit run app.py
#
# ==========================================================

# ==========================================================
# PREVISÃO DE VENDAS DE SORVETES
# Utilizando Python + Pandas + Streamlit
# ==========================================================

# --------------------------
# IMPORTAÇÃO DAS BIBLIOTECAS
# --------------------------

import streamlit as st
import pandas as pd

from sklearn.linear_model import LinearRegression

# --------------------------
# TÍTULO DO APLICATIVO
# --------------------------

st.title('Máquina de Previsão de Sorvetes')
st.caption('Previsão de vendas baseada na temperatura do dia')

# --------------------------
# BASE DE DADOS
# --------------------------

sorvete = pd.DataFrame({
    'temperatura': [18, 20, 24, 27, 30, 35],
    'vendas': [20, 25, 40, 55, 70, 100]
})

# --------------------------
# EXIBIÇÃO DOS DADOS
# --------------------------

st.subheader('Base de Dados')

st.dataframe(sorvete)

# --------------------------
# PREPARAÇÃO DOS DADOS
# --------------------------

X = sorvete[['temperatura']]
y = sorvete['vendas']

# --------------------------
# CRIAÇÃO DO MODELO DE IA
# --------------------------

modelo = LinearRegression()

modelo.fit(X, y)

# --------------------------
# ENTRADA DO USUÁRIO
# --------------------------

st.subheader('Simulação de Temperatura')

temperatura = st.slider(
    'Escolha a temperatura:',
    min_value=15,
    max_value=40,
    value=25
)

# --------------------------
# PREVISÃO
# --------------------------

entrada = pd.DataFrame({
    'temperatura': [temperatura]
})

previsao = modelo.predict(entrada)

# --------------------------
# RESULTADO
# --------------------------

st.success(
    f'Quantidade prevista de vendas: {previsao[0]:.0f} sorvetes'
)

# --------------------------
# MENSAGENS AUTOMÁTICAS
# --------------------------

if previsao[0] >= 80:
    st.warning('Dia muito quente. Alta procura por sorvetes.')

elif previsao[0] >= 50:
    st.info('Boa previsão de vendas para o dia.')

else:
    st.success('Movimento moderado previsto.')

# --------------------------
# REPRESENTAÇÃO GRÁFICA
# --------------------------

st.subheader('Gráfico de Vendas')

grafico = sorvete.set_index('temperatura')

st.line_chart(grafico)

# --------------------------
# EQUAÇÃO DA IA
# --------------------------

st.subheader('Equação da Regressão Linear')

st.code(
    f'vendas = ({modelo.coef_[0]:.2f} * temperatura) + {modelo.intercept_:.2f}'
)

# --------------------------
# INFORMAÇÕES DO MODELO
# --------------------------

st.subheader('Informações do Modelo')

st.write(f'Coeficiente Angular: {modelo.coef_[0]:.2f}')
st.write(f'Intercepto: {modelo.intercept_:.2f}')

# ==========================================================
# COMO EXECUTAR
# ==========================================================
#
# Salve o arquivo como:
# app.py
#
# Execute no terminal:
#
# streamlit run app.py
#
# ==========================================================

# ==========================================================
# ACADEMIA NINJA - APROVAÇÃO DE ALUNOS
# Utilizando Python + Pandas + Streamlit
# Modelo: Logistic Regression
# ==========================================================

# --------------------------
# IMPORTAÇÃO DAS BIBLIOTECAS
# --------------------------

import streamlit as st
import pandas as pd

from sklearn.linear_model import LogisticRegression

# --------------------------
# TÍTULO DO APP
# --------------------------

st.title('Academia Ninja')
st.caption('Previsão de aprovação baseada na quantidade de faltas')

# --------------------------
# BASE DE DADOS
# --------------------------

alunos = pd.DataFrame({
    'faltas': [0, 1, 2, 5, 7, 10],
    'resultado': [1, 1, 1, 0, 0, 0]
})

# --------------------------
# EXIBIR DADOS
# --------------------------

st.subheader('Base de Dados')

st.dataframe(alunos)

# --------------------------
# PREPARAÇÃO DOS DADOS
# --------------------------

X = alunos[['faltas']]
y = alunos['resultado']

# --------------------------
# CRIAÇÃO DO MODELO
# --------------------------

modelo = LogisticRegression()

modelo.fit(X, y)

# --------------------------
# ENTRADA DO USUÁRIO
# --------------------------

st.subheader('Simulação')

faltas = st.slider(
    'Quantidade de faltas:',
    min_value=0,
    max_value=15,
    value=3
)

# --------------------------
# PREVISÃO
# --------------------------

entrada = pd.DataFrame({
    'faltas': [faltas]
})

previsao = modelo.predict(entrada)

probabilidade = modelo.predict_proba(entrada)

# --------------------------
# RESULTADO
# --------------------------

if previsao[0] == 1:
    st.success('Resultado previsto: APROVADO')
else:
    st.error('Resultado previsto: REPROVADO')

# --------------------------
# PROBABILIDADES
# --------------------------

st.subheader('Probabilidades')

st.write(
    f'Chance de Aprovação: {probabilidade[0][1] * 100:.2f}%'
)

st.write(
    f'Chance de Reprovação: {probabilidade[0][0] * 100:.2f}%'
)

# --------------------------
# REPRESENTAÇÃO GRÁFICA
# --------------------------

st.subheader('Gráfico de Aprovação')

grafico = alunos.set_index('faltas')

st.line_chart(grafico)

# --------------------------
# LEGENDA DOS RESULTADOS
# --------------------------

st.subheader('Legenda')

st.write('1 = Aprovado')
st.write('0 = Reprovado')

# --------------------------
# INFORMAÇÕES DO MODELO
# --------------------------

st.subheader('Informações do Modelo')

st.write(f'Coeficiente: {modelo.coef_[0][0]:.2f}')
st.write(f'Intercepto: {modelo.intercept_[0]:.2f}')

# ==========================================================
# COMO EXECUTAR
# ==========================================================
#
# Salve como:
# app.py
#
# Execute no terminal:
#
# streamlit run app.py
#
# ==========================================================    

# ==========================================================
# PREVISÃO DE FELICIDADE DE UM CACHORRO
# Utilizando Python + Pandas + Streamlit
# ==========================================================

# --------------------------
# IMPORTAÇÃO DAS BIBLIOTECAS
# --------------------------

import streamlit as st
import pandas as pd

from sklearn.linear_model import LinearRegression

# --------------------------
# TÍTULO DO APP
# --------------------------

st.title('Felicidade dos Pets')
st.caption('Previsão de felicidade baseada na quantidade de passeios')

# --------------------------
# BASE DE DADOS
# --------------------------

pets = pd.DataFrame({
    'passeios': [1, 2, 3, 4, 5],
    'felicidade': [2, 4, 5, 8, 10]
})

# --------------------------
# EXIBIÇÃO DOS DADOS
# --------------------------

st.subheader('Base de Dados')

st.dataframe(pets)

# --------------------------
# PREPARAÇÃO DOS DADOS
# --------------------------

X = pets[['passeios']]
y = pets['felicidade']

# --------------------------
# CRIAÇÃO DO MODELO
# --------------------------

modelo = LinearRegression()

modelo.fit(X, y)

# --------------------------
# ENTRADA DO USUÁRIO
# --------------------------

st.subheader('Simulação de Passeios')

passeios = st.slider(
    'Quantidade de passeios:',
    min_value=1,
    max_value=10,
    value=3
)

# --------------------------
# PREVISÃO
# --------------------------

entrada = pd.DataFrame({
    'passeios': [passeios]
})

previsao = modelo.predict(entrada)

# --------------------------
# RESULTADO
# --------------------------

st.success(
    f'Nível previsto de felicidade: {previsao[0]:.2f}'
)

# --------------------------
# MENSAGENS AUTOMÁTICAS
# --------------------------

if previsao[0] >= 8:
    st.success('O pet está muito feliz.')

elif previsao[0] >= 5:
    st.info('O pet possui um bom nível de felicidade.')

else:
    st.warning('O pet pode precisar de mais atenção e passeios.')

# --------------------------
# REPRESENTAÇÃO GRÁFICA
# --------------------------

st.subheader('Gráfico de Felicidade')

grafico = pets.set_index('passeios')

st.line_chart(grafico)

# --------------------------
# EQUAÇÃO DA REGRESSÃO
# --------------------------

st.subheader('Equação da Regressão Linear')

st.code(
    f'felicidade = ({modelo.coef_[0]:.2f} * passeios) + {modelo.intercept_:.2f}'
)

# --------------------------
# INFORMAÇÕES DO MODELO
# --------------------------

st.subheader('Informações do Modelo')

st.write(f'Coeficiente Angular: {modelo.coef_[0]:.2f}')
st.write(f'Intercepto: {modelo.intercept_:.2f}')

# ==========================================================
# COMO EXECUTAR
# ==========================================================
#
# Salve o arquivo como:
# app.py
#
# Execute no terminal:
#
# streamlit run app.py
#
# ==========================================================

# ==========================================================
# PREVISÃO DE QUALIDADE DE FILMES
# Utilizando Python + Pandas + Streamlit
# ==========================================================

# --------------------------
# IMPORTAÇÃO DAS BIBLIOTECAS
# --------------------------

import streamlit as st
import pandas as pd

from sklearn.linear_model import LinearRegression

# --------------------------
# TÍTULO DO APP
# --------------------------

st.title('Previsão de Qualidade de Filmes')
st.caption('Previsão de nota baseada na duração do filme')

# --------------------------
# BASE DE DADOS
# --------------------------

filmes = pd.DataFrame({
    'duracao': [80, 90, 100, 110, 120],
    'nota': [4, 5, 7, 8, 9]
})

# --------------------------
# EXIBIÇÃO DOS DADOS
# --------------------------

st.subheader('Base de Dados')

st.dataframe(filmes)

# --------------------------
# PREPARAÇÃO DOS DADOS
# --------------------------

X = filmes[['duracao']]
y = filmes['nota']

# --------------------------
# CRIAÇÃO DO MODELO
# --------------------------

modelo = LinearRegression()

modelo.fit(X, y)

# --------------------------
# ENTRADA DO USUÁRIO
# --------------------------

st.subheader('Simulação de Filme')

duracao = st.slider(
    'Duração do filme (em minutos):',
    min_value=60,
    max_value=180,
    value=100
)

# --------------------------
# PREVISÃO
# --------------------------

entrada = pd.DataFrame({
    'duracao': [duracao]
})

previsao = modelo.predict(entrada)

# --------------------------
# RESULTADO
# --------------------------

st.success(
    f'Nota prevista para o filme: {previsao[0]:.2f}'
)

# --------------------------
# ANÁLISE AUTOMÁTICA
# --------------------------

if previsao[0] >= 8:
    st.success('Filme com excelente previsão de qualidade.')

elif previsao[0] >= 6:
    st.info('Filme com boa previsão de qualidade.')

else:
    st.warning('Filme com previsão de qualidade moderada.')

# --------------------------
# REPRESENTAÇÃO GRÁFICA
# --------------------------

st.subheader('Gráfico de Qualidade')

grafico = filmes.set_index('duracao')

st.line_chart(grafico)

# --------------------------
# EQUAÇÃO DA REGRESSÃO
# --------------------------

st.subheader('Equação da Regressão Linear')

st.code(
    f'nota = ({modelo.coef_[0]:.2f} * duracao) + {modelo.intercept_:.2f}'
)

# --------------------------
# INFORMAÇÕES DO MODELO
# --------------------------

st.subheader('Informações do Modelo')

st.write(f'Coeficiente Angular: {modelo.coef_[0]:.2f}')
st.write(f'Intercepto: {modelo.intercept_:.2f}')

# ==========================================================
# COMO EXECUTAR
# ==========================================================
#
# Salve o arquivo como:
# app.py
#
# Execute no terminal:
#
# streamlit run app.py
#
# ==========================================================

# ==========================================================
# PREVISÃO DE PREÇO DE PIZZAS PELO TAMANHO
# Utilizando Python + Pandas + Streamlit
# ==========================================================

# --------------------------
# IMPORTAÇÃO DAS BIBLIOTECAS
# --------------------------

import streamlit as st
import pandas as pd

from sklearn.linear_model import LinearRegression

# --------------------------
# TÍTULO DO APP
# --------------------------

st.title('Máquina Previsora de Pizzas')
st.caption('Previsão de preço baseada no tamanho da pizza')

# --------------------------
# BASE DE DADOS
# --------------------------

pizza = pd.DataFrame({
    'tamanho': [20, 25, 30, 35, 40],
    'preco': [20, 30, 40, 50, 60]
})

# --------------------------
# EXIBIÇÃO DOS DADOS
# --------------------------

st.subheader('Base de Dados')

st.dataframe(pizza)

# --------------------------
# PREPARAÇÃO DOS DADOS
# --------------------------

X = pizza[['tamanho']]
y = pizza['preco']

# --------------------------
# CRIAÇÃO DO MODELO
# --------------------------

modelo = LinearRegression()

modelo.fit(X, y)

# --------------------------
# ENTRADA DO USUÁRIO
# --------------------------

st.subheader('Simulação de Pizza')

tamanho = st.slider(
    'Escolha o tamanho da pizza:',
    min_value=15,
    max_value=50,
    value=30
)

# --------------------------
# PREVISÃO
# --------------------------

entrada = pd.DataFrame({
    'tamanho': [tamanho]
})

previsao = modelo.predict(entrada)

# --------------------------
# RESULTADO
# --------------------------

st.success(
    f'Preço previsto da pizza: R$ {previsao[0]:.2f}'
)

# --------------------------
# ANÁLISE AUTOMÁTICA
# --------------------------

if previsao[0] >= 50:
    st.warning('Pizza de tamanho grande detectada.')

elif previsao[0] >= 35:
    st.info('Pizza de tamanho médio.')

else:
    st.success('Pizza de tamanho pequeno.')

# --------------------------
# REPRESENTAÇÃO GRÁFICA
# --------------------------

st.subheader('Gráfico de Preços')

grafico = pizza.set_index('tamanho')

st.line_chart(grafico)

# --------------------------
# EQUAÇÃO DA REGRESSÃO
# --------------------------

st.subheader('Equação da Regressão Linear')

st.code(
    f'preco = ({modelo.coef_[0]:.2f} * tamanho) + {modelo.intercept_:.2f}'
)

# --------------------------
# INFORMAÇÕES DO MODELO
# --------------------------

st.subheader('Informações do Modelo')

st.write(f'Coeficiente Angular: {modelo.coef_[0]:.2f}')
st.write(f'Intercepto: {modelo.intercept_:.2f}')

# ==========================================================
# COMO EXECUTAR
# ==========================================================
#
# Salve como:
# app.py
#
# Execute no terminal:
#
# streamlit run app.py
#
# ==========================================================