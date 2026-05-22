import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator

# Baixa o léxico do VADER (necessário para a análise rodar)
nltk.download('vader_lexicon')

# Inicializa o analisador de sentimentos
sia = SentimentIntensityAnalyzer()

# Configuração simples da página do Streamlit
st.title("Analisador de Sentimentos - NLTK")
st.write("Digite uma frase em português para analisar o sentimento.")

# Interface do Usuário: Prompt, Botão e Output
user_input = st.text_input("Digite o seu texto aqui:", placeholder="Ex: Eu adoro aprender Python e IA!")
btn_analisar = st.button("Analisar Sentimento")

# Lógica ao clicar no botão
if btn_analisar:
    if user_input.strip() == "":
        st.warning("Por favor, digite algo antes de rodar.")
    else:
        try:
            # Traduz o texto de português (pt) para inglês (en) em segundo plano
            texto_traduzido = GoogleTranslator(source='pt', target='en').translate(user_input)
            
            # O VADER calcula o score baseado na frase traduzida
            scores = sia.polarity_scores(texto_traduzido)
            compound_score = scores['compound']
            
            # Define a classificação com base no score composto
            if compound_score >= 0.05:
                sentimento = "😊 Positivo"
            elif compound_score <= -0.05:
                sentimento = "😡 Negativo"
            else:
                sentimento = "😐 Neutro"
                
            # Exibe o resultado de forma simples na tela
            st.write(f"O sentimento detectado foi: **{sentimento}**")
            st.write(f"Score Composto: `{compound_score}`")
            
            # Mostra como a frase ficou em inglês caso o usuário queira auditar
            with st.expander("Ver tradução interna do modelo"):
                st.write(f"_*Tradução utilizada:* {texto_traduzido}_")
                
        except Exception as e:
            st.error("Ocorreu um erro ao traduzir o texto. Tente novamente.")