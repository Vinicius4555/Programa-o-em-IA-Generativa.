    # ============================================================
    #  🌧️  PREVISÃO DE CHUVA COM TENSORFLOW — Introdução prática
    # ============================================================

    # --- 1. IMPORTS -------------------------------------------
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    import matplotlib.pyplot as plt
    import streamlit as st

    print("TensorFlow versão:", tf.__version__)


    # --- 2. DADOS ---------------------------------------------
    # [temperatura (°C), umidade (%)]  →  0=não choveu / 1=choveu
    X_treino = np.array([
        [30, 40], [25, 80], [28, 30], [20, 90], [32, 20],
        [18, 85], [27, 75], [35, 15], [22, 88], [31, 25],
    ], dtype=float)

    y_treino = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype=float)

    print("\n📋 Dados criados! Shape X:", X_treino.shape)


    # --- 3. MODELO --------------------------------------------
    # Rede com 2 entradas → 4 neurônios ocultos → 1 saída (0 a 1)
    modelo = keras.Sequential([
        keras.layers.Dense(4, activation='relu', input_shape=(2,)),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    modelo.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    modelo.summary()


    # --- 4. TREINAMENTO ---------------------------------------
    # epochs=100: o modelo estuda os dados 100 vezes seguidas
    historico = modelo.fit(X_treino, y_treino, epochs=100, verbose=0)

    print(f"\n✅ Treinamento concluído! Acurácia final: {historico.history['accuracy'][-1]:.0%}")


    # --- 5. PREVISÃO ------------------------------------------
    novos_dias = np.array([
        [33, 20],  # quente e seco   → esperamos: não chover
        [19, 92],  # frio e úmido    → esperamos: chover
        [26, 60],  # ameno/moderado  → incerto...
    ])

    previsoes = modelo.predict(novos_dias)

    print("\n🔮 Previsões:")
    for i, (dia, prob) in enumerate(zip(novos_dias, previsoes)):
        resultado = "🌧️  VAI CHOVER" if prob[0] > 0.5 else "☀️  NÃO vai chover"
        print(f"  Dia {i+1} | {dia[0]:.0f}°C / {dia[1]:.0f}% umidade → {resultado} ({prob[0]:.0%})")


    # --- 6. GRÁFICO -------------------------------------------
    # Visualiza como o erro caiu e a acurácia subiu durante o treino
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(historico.history['loss'], color='#D85A30')
    ax1.set_title('Erro ao longo do tempo')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Erro (loss)')

    ax2.plot(historico.history['accuracy'], color='#1D9E75')
    ax2.set_title('Acurácia ao longo do tempo')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Acurácia')

    plt.tight_layout()
    plt.show()

    # ==========================================================
    # IA PARA FILTRAR MENSAGENS DE INFLUENCIADORES
    # ----------------------------------------------------------
    # Objetivo:
    # Separar mensagens em:
    #
    # 1. Importantes
    # 2. Relevantes
    # 3. Desnecessárias
    # 4. Spam
    #
    # Tecnologias:
    # - Pandas
    # - Scikit-Learn
    #
    # Estratégia de IA:
    # Utilização de:
    # - CountVectorizer
    # - Machine Learning
    # - Naive Bayes
    # ==========================================================

    # ----------------------------------------------------------
    # IMPORTAÇÃO DAS BIBLIOTECAS
    # ----------------------------------------------------------

    import pandas as pd

    from sklearn.feature_extraction.text import CountVectorizer

    from sklearn.naive_bayes import MultinomialNB

    from sklearn.pipeline import Pipeline

    # ----------------------------------------------------------
    # BASE DE TREINAMENTO
    # ----------------------------------------------------------
    # Exemplos simulados de mensagens recebidas.

    dados = pd.DataFrame({

        'mensagem': [

            # IMPORTANTES
            'Filho precisamos falar urgente',
            'Sua mãe tentou entrar em contato',
            'Me liga quando puder',
            'Problema familiar urgente',

            # RELEVANTES
            'Gostaria de fechar uma parceria',
            'Temos uma proposta comercial',
            'Empresa interessada em publicidade',
            'Convite para campanha patrocinada',

            # DESNECESSÁRIAS
            'Bom dia',
            'kkkkkkkk',
            'Oi tudo bem',
            'Me responde',

            # SPAM
            'Ganhe dinheiro rápido clicando aqui',
            'PIX imediato sem esforço',
            'Clique nesse link agora',
            'Investimento garantido com lucro'
        ],

        'categoria': [

            'Importante',
            'Importante',
            'Importante',
            'Importante',

            'Relevante',
            'Relevante',
            'Relevante',
            'Relevante',

            'Desnecessária',
            'Desnecessária',
            'Desnecessária',
            'Desnecessária',

            'Spam',
            'Spam',
            'Spam',
            'Spam'
        ]
    })

    # ----------------------------------------------------------
    # EXIBIÇÃO DA BASE
    # ----------------------------------------------------------

    print('\nBASE DE TREINAMENTO:\n')

    print(dados)

    # ----------------------------------------------------------
    # VARIÁVEIS
    # ----------------------------------------------------------

    X = dados['mensagem']

    y = dados['categoria']

    # ----------------------------------------------------------
    # PIPELINE DE IA
    # ----------------------------------------------------------
    # CountVectorizer:
    # transforma texto em números.
    #
    # MultinomialNB:
    # algoritmo de classificação textual.

    modelo = Pipeline([

        ('vetorizador', CountVectorizer()),

        ('classificador', MultinomialNB())

    ])

    # ----------------------------------------------------------
    # TREINAMENTO DO MODELO
    # ----------------------------------------------------------

    modelo.fit(X, y)

    # ----------------------------------------------------------
    # FUNÇÃO DE CLASSIFICAÇÃO
    # ----------------------------------------------------------

    def analisar_mensagem(mensagem):

        categoria = modelo.predict([mensagem])[0]

        # ------------------------------------------------------
        # DEFINIÇÃO DAS AÇÕES
        # ------------------------------------------------------

        if categoria == 'Spam':

            acao = 'BLOQUEAR MENSAGEM'
            prioridade = 'BAIXA'

        elif categoria == 'Importante':

            acao = 'ENVIAR PARA CAIXA PRIORITÁRIA'
            prioridade = 'ALTA'

        elif categoria == 'Relevante':

            acao = 'ENVIAR PARA NEGÓCIOS'
            prioridade = 'MÉDIA'

        else:

            acao = 'ARQUIVAR'
            prioridade = 'BAIXA'

        # ------------------------------------------------------
        # RESULTADO
        # ------------------------------------------------------

        print('\n================================================')

        print(f'Mensagem: {mensagem}')

        print(f'Categoria Identificada: {categoria}')

        print(f'Prioridade: {prioridade}')

        print(f'Ação Recomendada: {acao}')

        print('================================================')

    # ----------------------------------------------------------
    # TESTES DA IA
    # ----------------------------------------------------------

    mensagens_teste = [

        'Gostaria de fazer uma parceria para divulgar nossa marca',

        'Oi tudo bem',

        'Clique aqui e ganhe dinheiro rápido',

        'Sua família está tentando falar com você',

        'Temos interesse em publicidade no Instagram',

        'kkkkkkkkkkkk',

        'PIX imediato para começar hoje'

    ]

    # ----------------------------------------------------------
    # EXECUÇÃO DOS TESTES
    # ----------------------------------------------------------

    for mensagem in mensagens_teste:

        analisar_mensagem(mensagem)

    # ----------------------------------------------------------
    # SIMULAÇÃO DE CAIXAS
    # ----------------------------------------------------------

    print('\n\nSIMULAÇÃO DAS CAIXAS INTELIGENTES:\n')

    caixa_importantes = []
    caixa_relevantes = []
    caixa_desnecessarias = []
    caixa_spam = []

    # ----------------------------------------------------------
    # ORGANIZAÇÃO AUTOMÁTICA
    # ----------------------------------------------------------

    for mensagem in mensagens_teste:

        categoria = modelo.predict([mensagem])[0]

        if categoria == 'Importante':

            caixa_importantes.append(mensagem)

        elif categoria == 'Relevante':

            caixa_relevantes.append(mensagem)

        elif categoria == 'Desnecessária':

            caixa_desnecessarias.append(mensagem)

        else:

            caixa_spam.append(mensagem)

    # ----------------------------------------------------------
    # RESULTADO FINAL
    # ----------------------------------------------------------

    print('📌 IMPORTANTES:')
    print(caixa_importantes)

    print('\n💼 RELEVANTES:')
    print(caixa_relevantes)

    print('\n🗂️ DESNECESSÁRIAS:')
    print(caixa_desnecessarias)

    print('\n🚫 SPAM:')
    print(caixa_spam)

    # ==========================================================
    # FIM DO PROJETO
    # ==========================================================

    # ==========================================================
    # SISTEMA DE ANÁLISE DE FINANCIAMENTO DE MOTO
    # ----------------------------------------------------------
    # Objetivo:
    # Simular a análise de crédito de um banco
    # para aprovar ou recusar financiamento.
    #
    # O sistema analisa:
    # - Renda
    # - Tempo de conta no banco
    # - Score de fidelidade
    # - Histórico de pagamentos
    #
    # Público:
    # - Classe baixa
    # - Classe média
    # - Classe alta
    #
    # Tecnologias:
    # - Python
    # - Pandas
    # - Scikit-Learn
    # ==========================================================

    # ----------------------------------------------------------
    # IMPORTAÇÃO DAS BIBLIOTECAS
    # ----------------------------------------------------------

    import pandas as pd

    from sklearn.tree import DecisionTreeClassifier

    # ----------------------------------------------------------
    # BASE DE DADOS
    # ----------------------------------------------------------
    # Dados fictícios simulando clientes do banco.

    dados = pd.DataFrame({

        'renda': [

            1500, 1800, 2200,
            3000, 4500, 5500,
            7000, 9000, 12000,
            15000, 20000, 25000

        ],

        'tempo_conta': [

            1, 2, 1,
            3, 4, 5,
            6, 7, 8,
            10, 12, 15

        ],

        'fidelidade': [

            2, 3, 2,
            5, 6, 7,
            8, 9, 9,
            10, 10, 10

        ],

        'historico_pagamento': [

            0, 0, 1,
            1, 1, 1,
            1, 1, 1,
            1, 1, 1

        ],

        # 1 = aprovado
        # 0 = recusado

        'aprovado': [

            0, 0, 0,
            1, 1, 1,
            1, 1, 1,
            1, 1, 1

        ]
    })

    # ----------------------------------------------------------
    # EXIBIÇÃO DA BASE
    # ----------------------------------------------------------

    print('\nBASE DE CLIENTES:\n')

    print(dados)

    # ----------------------------------------------------------
    # PREPARAÇÃO DOS DADOS
    # ----------------------------------------------------------

    X = dados[[
        'renda',
        'tempo_conta',
        'fidelidade',
        'historico_pagamento'
    ]]

    y = dados['aprovado']

    # ----------------------------------------------------------
    # CRIAÇÃO DA IA
    # ----------------------------------------------------------
    # DecisionTree:
    # algoritmo de tomada de decisão.

    modelo = DecisionTreeClassifier()

    # ----------------------------------------------------------
    # TREINAMENTO
    # ----------------------------------------------------------

    modelo.fit(X, y)

    # ----------------------------------------------------------
    # ENTRADA DO USUÁRIO
    # ----------------------------------------------------------

    print('\n======================================')
    print('SIMULADOR DE FINANCIAMENTO DE MOTO')
    print('======================================')

    renda = float(input('Digite sua renda mensal: R$ '))

    tempo_conta = int(input(
        'Há quantos anos possui conta no banco? '
    ))

    fidelidade = int(input(
        'Nota de fidelidade ao banco (1 a 10): '
    ))

    historico = int(input(
        'Possui bom histórico de pagamentos? (1=SIM / 0=NÃO): '
    ))

    # ----------------------------------------------------------
    # DADOS DO CLIENTE
    # ----------------------------------------------------------

    cliente = [[

        renda,
        tempo_conta,
        fidelidade,
        historico

    ]]

    # ----------------------------------------------------------
    # PREVISÃO DA IA
    # ----------------------------------------------------------

    resultado = modelo.predict(cliente)

    # ----------------------------------------------------------
    # DEFINIÇÃO DA VERBA LIBERADA
    # ----------------------------------------------------------

    # Quanto maior a fidelidade,
    # maior o limite liberado.

    if fidelidade <= 3:

        verba = renda * 3

    elif fidelidade <= 6:

        verba = renda * 5

    elif fidelidade <= 8:

        verba = renda * 8

    else:

        verba = renda * 12

    # ----------------------------------------------------------
    # CLASSIFICAÇÃO SOCIAL
    # ----------------------------------------------------------

    if renda <= 2500:

        classe = 'Classe Baixa'

    elif renda <= 10000:

        classe = 'Classe Média'

    else:

        classe = 'Classe Alta'

    # ----------------------------------------------------------
    # RESULTADO FINAL
    # ----------------------------------------------------------

    print('\n======================================')
    print('RESULTADO DA ANÁLISE')
    print('======================================')

    print(f'Classe Social: {classe}')

    print(f'Verba Máxima Liberada: R$ {verba:.2f}')

    # ----------------------------------------------------------
    # APROVAÇÃO
    # ----------------------------------------------------------

    if resultado[0] == 1:

        print('\n✅ FINANCIAMENTO APROVADO')

        # ----------------------------------------------
        # Mensagens personalizadas
        # ----------------------------------------------

        if fidelidade >= 8:

            print(
                'Cliente altamente fiel ao banco.'
            )

            print(
                'Taxas de juros reduzidas liberadas.'
            )

        elif fidelidade >= 5:

            print(
                'Cliente confiável.'
            )

            print(
                'Financiamento liberado com condições moderadas.'
            )

        else:

            print(
                'Financiamento aprovado com análise básica.'
            )

    else:

        print('\n❌ FINANCIAMENTO RECUSADO')

        print(
            'Motivos possíveis:'
        )

        print('- Baixa renda')
        print('- Pouco tempo de conta')
        print('- Fidelidade insuficiente')
        print('- Histórico de pagamento ruim')

    # ----------------------------------------------------------
    # RELATÓRIO FINAL
    # ----------------------------------------------------------

    print('\n======================================')
    print('RELATÓRIO BANCÁRIO')
    print('======================================')

    print(f'Renda Mensal: R$ {renda:.2f}')
    print(f'Tempo de Conta: {tempo_conta} anos')
    print(f'Fidelidade: {fidelidade}/10')
    print(f'Histórico Positivo: {historico}')

    print('\nSistema encerrado.')

    # ==========================================================
    # FIM DO PROJETO
    # ==========================================================

    # ==========================================================
    # IA DE RECOMENDAÇÃO DE PLAYLISTS - SPOTIFY STYLE
    # ----------------------------------------------------------
    # Objetivo:
    # Detectar o humor do usuário e recomendar
    # playlists automaticamente.
    #
    # Gêneros:
    # - Lofi
    # - Rock
    # - Pop
    # - Sertanejo
    # - Rap
    #
    # Tecnologias:
    # - Python
    # - Pandas
    # - Scikit-Learn
    #
    # Sistema inspirado no estilo de barras do Spotify.
    # ==========================================================

    # ----------------------------------------------------------
    # IMPORTAÇÃO DAS BIBLIOTECAS
    # ----------------------------------------------------------

    import pandas as pd

    from sklearn.feature_extraction.text import CountVectorizer

    from sklearn.naive_bayes import MultinomialNB

    from sklearn.pipeline import Pipeline

    # ----------------------------------------------------------
    # BASE DE TREINAMENTO
    # ----------------------------------------------------------
    # Frases associadas aos humores.

    dados = pd.DataFrame({

        'frase': [

            # POSITIVO
            'Hoje estou muito motivado',
            'Estou feliz e animado',
            'Meu dia está incrível',
            'Quero aproveitar a vida',

            # NEGATIVO
            'Nada está dando certo',
            'Estou desanimado',
            'Hoje foi horrível',
            'Estou muito frustrado',

            # DEPRESSIVO
            'Não quero falar com ninguém',
            'Me sinto vazio',
            'Estou muito triste',
            'Quero ficar sozinho',

            # ALEGRE
            'Quero dançar hoje',
            'Estou sorrindo muito',
            'Estou cheio de energia',
            'Hoje é dia de festa',

            # NEUTRO
            'Dia comum',
            'Tudo normal hoje',
            'Sem muitas emoções',
            'Apenas relaxando',

            # RAIVOSO
            'Estou com muita raiva',
            'Quero quebrar tudo',
            'Hoje estou irritado',
            'Estou furioso',

            # PREGUIÇOSO
            'Só quero dormir',
            'Estou cansado',
            'Quero descansar',
            'Sem energia hoje',

            # TEDIOSO
            'Estou entediado',
            'Nada interessante acontecendo',
            'Dia parado',
            'Tudo muito monótono'

        ],

        'humor': [

            'Positivo',
            'Positivo',
            'Positivo',
            'Positivo',

            'Negativo',
            'Negativo',
            'Negativo',
            'Negativo',

            'Depressivo',
            'Depressivo',
            'Depressivo',
            'Depressivo',

            'Alegre',
            'Alegre',
            'Alegre',
            'Alegre',

            'Neutro',
            'Neutro',
            'Neutro',
            'Neutro',

            'Raivoso',
            'Raivoso',
            'Raivoso',
            'Raivoso',

            'Preguiçoso',
            'Preguiçoso',
            'Preguiçoso',
            'Preguiçoso',

            'Tedioso',
            'Tedioso',
            'Tedioso',
            'Tedioso'

        ]
    })

    # ----------------------------------------------------------
    # CRIAÇÃO DA IA
    # ----------------------------------------------------------

    modelo = Pipeline([

        ('vetorizador', CountVectorizer()),

        ('classificador', MultinomialNB())

    ])

    # ----------------------------------------------------------
    # TREINAMENTO
    # ----------------------------------------------------------

    modelo.fit(

        dados['frase'],
        dados['humor']

    )

    # ----------------------------------------------------------
    # PLAYLISTS POR HUMOR
    # ----------------------------------------------------------

    playlists = {

        'Positivo': {

            'gênero': 'Pop',
            'playlist': 'Pop Energy Mix'
        },

        'Negativo': {

            'gênero': 'Lofi',
            'playlist': 'Lofi Sad Nights'
        },

        'Depressivo': {

            'gênero': 'Lofi',
            'playlist': 'Deep Emotional Lofi'
        },

        'Alegre': {

            'gênero': 'Pop',
            'playlist': 'Happy Dance Hits'
        },

        'Neutro': {

            'gênero': 'Sertanejo',
            'playlist': 'Relax Sertanejo'
        },

        'Raivoso': {

            'gênero': 'Rock',
            'playlist': 'Heavy Rock Rage'
        },

        'Preguiçoso': {

            'gênero': 'Lofi',
            'playlist': 'Sleepy Chill Beats'
        },

        'Tedioso': {

            'gênero': 'Rap',
            'playlist': 'Hype Rap Sessions'
        }

    }

    # ----------------------------------------------------------
    # MEDIDOR DE FELICIDADE
    # ----------------------------------------------------------
    # Valores simulados para o estilo Spotify.

    felicidade = {

        'Positivo': 90,
        'Negativo': 30,
        'Depressivo': 10,
        'Alegre': 100,
        'Neutro': 50,
        'Raivoso': 20,
        'Preguiçoso': 40,
        'Tedioso': 35

    }

    energia = {

        'Positivo': 80,
        'Negativo': 25,
        'Depressivo': 10,
        'Alegre': 95,
        'Neutro': 50,
        'Raivoso': 90,
        'Preguiçoso': 15,
        'Tedioso': 45

    }

    # ----------------------------------------------------------
    # ENTRADA DO USUÁRIO
    # ----------------------------------------------------------

    print('\n========================================')
    print('SPOTIFY IA - RECOMENDADOR DE PLAYLIST')
    print('========================================')

    texto = input(
        '\nDescreva como você está se sentindo hoje:\n'
    )

    # ----------------------------------------------------------
    # PREVISÃO DA IA
    # ----------------------------------------------------------

    humor_previsto = modelo.predict([texto])[0]

    # ----------------------------------------------------------
    # BUSCANDO PLAYLIST
    # ----------------------------------------------------------

    dados_playlist = playlists[humor_previsto]

    # ----------------------------------------------------------
    # FUNÇÃO DAS BARRAS
    # ----------------------------------------------------------
    # Simulando o visual do Spotify.

    def barra(valor):

        barras = int(valor / 10)

        return '█' * barras + '░' * (10 - barras)

    # ----------------------------------------------------------
    # RESULTADO
    # ----------------------------------------------------------

    print('\n========================================')
    print('ANÁLISE DE HUMOR')
    print('========================================')

    print(f'\nHumor identificado: {humor_previsto}')

    print(f'\nPlaylist Recomendada:')
    print(f'{dados_playlist["playlist"]}')

    print(f'\nGênero:')
    print(f'{dados_playlist["gênero"]}')

    # ----------------------------------------------------------
    # MEDIDORES
    # ----------------------------------------------------------

    print('\n========================================')
    print('MEDIDOR SPOTIFY DE HUMOR')
    print('========================================')

    valor_felicidade = felicidade[humor_previsto]

    valor_energia = energia[humor_previsto]

    print(
        f'\nFelicidade: '
        f'[{barra(valor_felicidade)}] '
        f'{valor_felicidade}%'
    )

    print(
        f'Energia:    '
        f'[{barra(valor_energia)}] '
        f'{valor_energia}%'
    )

    # ----------------------------------------------------------
    # RECOMENDAÇÕES EXTRAS
    # ----------------------------------------------------------

    print('\n========================================')
    print('RECOMENDAÇÕES EXTRAS')
    print('========================================')

    if humor_previsto == 'Raivoso':

        print(
            '\n🔥 Recomendação: '
            'músicas intensas para liberar energia.'
        )

    elif humor_previsto == 'Depressivo':

        print(
            '\n🌙 Recomendação: '
            'sons calmos e ambientes relaxantes.'
        )

    elif humor_previsto == 'Alegre':

        print(
            '\n🎉 Recomendação: '
            'playlists dançantes e animadas.'
        )

    elif humor_previsto == 'Preguiçoso':

        print(
            '\n😴 Recomendação: '
            'músicas leves e tranquilas.'
        )

    else:

        print(
            '\n🎵 Recomendação personalizada '
            'baseada no seu humor atual.'
        )

    # ----------------------------------------------------------
    # FINALIZAÇÃO
    # ----------------------------------------------------------

    print('\n========================================')
    print('OBRIGADO POR UTILIZAR O SPOTIFY IA')
    print('========================================')

    # ==========================================================
    # FIM DO PROJETO
    # ==========================================================

    # ==========================================================
    # APP STREAMLIT - IA DE MONITORAMENTO DE PLANTAS
    # ----------------------------------------------------------
    # Objetivo:
    # Analisar o estado de uma planta utilizando:
    #
    # - Cor das folhas
    # - Quantidade de água
    # - Incidência solar
    #
    # A IA identifica:
    # - Bom estado
    # - Desidratada
    # - Pouco sol
    # - Excesso de água
    # - Mal cuidada
    # - Morta
    #
    # Público:
    # - Pesquisadores
    # - Estudantes
    # - Uso didático
    #
    # Tecnologias:
    # - Streamlit
    # - Pandas
    # - Scikit-Learn
    # ==========================================================

    # ----------------------------------------------------------
    # IMPORTAÇÃO DAS BIBLIOTECAS
    # ----------------------------------------------------------

    import streamlit as st

    import pandas as pd

    from sklearn.tree import DecisionTreeClassifier

    # ----------------------------------------------------------
    # CONFIGURAÇÃO DA PÁGINA
    # ----------------------------------------------------------

    st.set_page_config(

        page_title='IA Monitoramento de Plantas',
        page_icon='🌱',
        layout='wide'

    )

    # ----------------------------------------------------------
    # TÍTULO
    # ----------------------------------------------------------

    st.title('🌱 Inteligência Artificial para Plantas')

    st.caption(
        'Sistema didático para análise de saúde vegetal.'
    )

    # ----------------------------------------------------------
    # EXPLICAÇÃO
    # ----------------------------------------------------------

    st.markdown("""
    # 📚 Como o sistema funciona?

    A Inteligência Artificial avalia:

    - Cor das folhas
    - Quantidade de água
    - Incidência solar

    Com base nesses fatores, a IA identifica
    o estado atual da planta.

    O sistema é útil para:

    - Pesquisas acadêmicas
    - Estudos botânicos
    - Agricultura doméstica
    - Aprendizado de IA
    """)

    # ----------------------------------------------------------
    # BASE DE TREINAMENTO
    # ----------------------------------------------------------
    # Valores simulados para aprendizado.

    dados = pd.DataFrame({

        # 0 = amarelas
        # 1 = verdes
        # 2 = marrons

        'folhas': [

            1, 1, 1,
            0, 0, 0,
            2, 2, 2,
            0, 2, 1

        ],

        # Água em ml por dia

        'agua': [

            500, 600, 700,
            100, 150, 200,
            1000, 1200, 1500,
            300, 900, 400

        ],

        # Horas de sol

        'sol': [

            6, 7, 8,
            1, 2, 3,
            10, 12, 14,
            2, 11, 4

        ],

        'estado': [

            'Bom Estado',
            'Bom Estado',
            'Bom Estado',

            'Desidratada',
            'Pouco Sol',
            'Mal Cuidada',

            'Morta',
            'Excesso de Água',
            'Morta',

            'Pouco Sol',
            'Morta',
            'Mal Cuidada'
        ]

    })

    # ----------------------------------------------------------
    # EXIBIÇÃO DA BASE
    # ----------------------------------------------------------

    st.markdown('# 📊 Base de Dados')

    st.dataframe(dados)

    # ----------------------------------------------------------
    # PREPARAÇÃO DOS DADOS
    # ----------------------------------------------------------

    X = dados[['folhas', 'agua', 'sol']]

    y = dados['estado']

    # ----------------------------------------------------------
    # IA
    # ----------------------------------------------------------

    modelo = DecisionTreeClassifier()

    modelo.fit(X, y)

    # ----------------------------------------------------------
    # MENU LATERAL
    # ----------------------------------------------------------

    st.sidebar.title('🌿 Dados da Planta')

    # ----------------------------------------------------------
    # ESCOLHA DAS FOLHAS
    # ----------------------------------------------------------

    folha_texto = st.sidebar.selectbox(

        'Cor das folhas:',

        [

            'Verdes',
            'Amarelas',
            'Marrons'

        ]

    )

    # ----------------------------------------------------------
    # CONVERSÃO
    # ----------------------------------------------------------

    if folha_texto == 'Amarelas':

        folha = 0

    elif folha_texto == 'Verdes':

        folha = 1

    else:

        folha = 2

    # ----------------------------------------------------------
    # ÁGUA
    # ----------------------------------------------------------

    agua = st.sidebar.slider(

        'Quantidade de água (ml por dia)',

        min_value=0,
        max_value=2000,
        value=500

    )

    # ----------------------------------------------------------
    # SOL
    # ----------------------------------------------------------

    sol = st.sidebar.slider(

        'Horas de incidência solar',

        min_value=0,
        max_value=15,
        value=5

    )

    # ----------------------------------------------------------
    # PREVISÃO
    # ----------------------------------------------------------

    resultado = modelo.predict([[

        folha,
        agua,
        sol

    ]])

    estado = resultado[0]

    # ----------------------------------------------------------
    # RESULTADO PRINCIPAL
    # ----------------------------------------------------------

    st.markdown('# 🧠 Resultado da IA')

    # ----------------------------------------------------------
    # ALERTAS VISUAIS
    # ----------------------------------------------------------

    if estado == 'Bom Estado':

        st.success(
            '✅ A planta está em BOM ESTADO.'
        )

    elif estado == 'Desidratada':

        st.warning(
            '💧 A planta está DESIDRATADA.'
        )

    elif estado == 'Pouco Sol':

        st.warning(
            '☀️ A planta está recebendo POUCO SOL.'
        )

    elif estado == 'Excesso de Água':

        st.error(
            '🚨 A planta está com EXCESSO DE ÁGUA.'
        )

    elif estado == 'Mal Cuidada':

        st.error(
            '⚠️ A planta aparenta estar MAL CUIDADA.'
        )

    else:

        st.error(
            '☠️ A planta aparenta estar MORTA.'
        )

    # ----------------------------------------------------------
    # GRANDES BLOCOS INFORMATIVOS
    # ----------------------------------------------------------

    st.markdown('---')

    st.markdown('# 🌞 Relatório de Incidência Solar')

    if sol < 3:

        st.info("""
        A incidência solar está muito baixa.

        Consequências possíveis:
        - Folhas amareladas
        - Crescimento lento
        - Baixa fotossíntese

        Recomendação:
        Posicionar a planta próxima a janelas
        ou áreas iluminadas.
        """)

    elif sol <= 8:

        st.success("""
        A incidência solar está equilibrada.

        Benefícios:
        - Fotossíntese adequada
        - Crescimento saudável
        - Melhor coloração das folhas
        """)

    else:

        st.warning("""
        A incidência solar está excessiva.

        Riscos:
        - Queimaduras nas folhas
        - Ressecamento
        - Morte da planta
        """)

    # ----------------------------------------------------------
    # BLOCO SOBRE ÁGUA
    # ----------------------------------------------------------

    st.markdown('---')

    st.markdown('# 💧 Relatório Hídrico')

    if agua < 250:

        st.warning("""
        Quantidade de água insuficiente.

        Sintomas:
        - Folhas secas
        - Desidratação
        - Enfraquecimento estrutural
        """)

    elif agua <= 800:

        st.success("""
        Quantidade de água adequada.

        Benefícios:
        - Boa hidratação
        - Crescimento saudável
        - Maior resistência
        """)

    else:

        st.error("""
        Quantidade de água excessiva.

        Riscos:
        - Apodrecimento das raízes
        - Fungos
        - Morte da planta
        """)

    # ----------------------------------------------------------
    # BLOCO SOBRE AS FOLHAS
    # ----------------------------------------------------------

    st.markdown('---')

    st.markdown('# 🍂 Relatório das Folhas')

    if folha_texto == 'Verdes':

        st.success("""
        Folhas verdes indicam:
        - Boa fotossíntese
        - Planta saudável
        - Nutrientes equilibrados
        """)

    elif folha_texto == 'Amarelas':

        st.warning("""
        Folhas amarelas podem indicar:
        - Pouco sol
        - Falta de nutrientes
        - Excesso de água
        """)

    else:

        st.error("""
        Folhas marrons podem indicar:
        - Planta muito debilitada
        - Queimadura solar
        - Desidratação severa
        """)

    # ----------------------------------------------------------
    # CONCLUSÃO
    # ----------------------------------------------------------

    st.markdown('---')

    st.markdown('# 📖 Conclusão Científica')

    st.write(f"""
    A Inteligência Artificial concluiu que a planta está:

    ### {estado}

    A análise foi realizada considerando:
    - Coloração das folhas
    - Quantidade de água
    - Exposição solar

    Este sistema demonstra como algoritmos de
    aprendizado de máquina podem auxiliar em
    pesquisas ambientais e monitoramento vegetal.
    """)

    # ==========================================================
    # FIM DO PROJETO
    # ==========================================================