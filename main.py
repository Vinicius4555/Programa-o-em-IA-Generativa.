import streamlit as st
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

st.title('Ensinar a Máquina Previsão do Futuro')
st.write('Previsão para o campeão da copa do mundo.')

st.header('Opções de campeão...')

dados = pd.DataFrame({
'gols':[12,15,10,18,14,11,16],
'ranking': [1,3,2,1,4,10,2],
'pais':['Brasil', 'Argentina', 'França', 'Brasil', 'França', 'Argentina', 'Brasil']
})

modelo_copa = DecisionTreeClassifier()
modelo_copa.fit(dados[['gols', 'ranking']], dados['pais'])

gols_input = st.number_input('Quantos gols o time fez?', 0,30,15)
rank_input = st.number_input('Qual a posição', 1,100,1)

if st.button('Prever'):
    resultado_copa = modelo_copa.predict([[gols_input, rank_input]])
    st.success(f'O provavél campeão é {resultado_copa}')

# _____________________________________________________________________________________________

st.header('Análise de notas - Previsão')

estudos = pd.DataFrame({
    'notas':[1,2,4,5,8,10],
    'horas':[2,4,5,7,9,10]
})

st.scatter_chart(estudos, x='horas', y='notas')

modelo_escola = LinearRegression()
modelo_escola.fit(estudos[['horas']], estudos['notas'])

h_estudo = st.slider('horas de estudo', 0,12,5)
nota_final = modelo_escola.predict([[h_estudo]])

st.metric (f'Sua nota seria', f'{ min(nota_final[0], 10.0):.1f}')

# ___________________________________________________________________________________________

st.title('Previsão de Vendas')
st.write('Análise se suas vendas serão boas ou ruins.')

dados_vendas = pd.DataFrame({
    'Investimento': [100, 200, 300, 400, 500, 600],
    'faturamento': [1200, 2500, 3200, 4800, 5100, 6300]
})

st.bar_chart(dados_vendas, x='Investimento', y='faturamento')

modelo_gráfico = LinearRegression()
modelo_gráfico.fit(dados_vendas[['Investimento']], dados_vendas['faturamento'])

Total = st.slider('Total Retorno', 0,600,1000)
Previsão_total = modelo_gráfico.predict([[Total]])

st.metric (f'O valor final seria', f'{ min(Previsão_total[0], 10000.0):.1f}')

