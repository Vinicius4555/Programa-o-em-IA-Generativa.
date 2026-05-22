# Instalação (rodar 1 vez no terminal)
# pip install nltk

import nltk                          # biblioteca de processamento de linguagem natural
from nltk.tokenize import word_tokenize, sent_tokenize

# Baixa os recursos necessários do NLTK (só na primeira vez)
nltk.download('punkt')            # modelo de tokenização por pontuação
nltk.download('punkt_tab')        # versão tabular do punkt

def tokenizar_mensagem(texto: str) -> dict:
    """
    Recebe um texto e retorna tokens individuais e por sentenças.
    Útil para separar mensagens grandes em unidades menores.
    """

    # ── Tokenização por PALAVRAS ──────────────────────────────
    # word_tokenize divide o texto em palavras e pontuações
    # Ex: "Olá, mundo!" → ['Olá', ',', 'mundo', '!']
    tokens_palavras = word_tokenize(texto, language='portuguese')

    # ── Tokenização por SENTENÇAS ─────────────────────────────
    # sent_tokenize divide o texto em sentenças completas
    # Ex: "Oi. Tudo bem?" → ['Oi.', 'Tudo bem?']
    tokens_sentencas = sent_tokenize(texto, language='portuguese')

    # ── Filtrando apenas palavras reais (sem pontuação) ───────
    # isalnum() retorna True se o token for letra ou número
    so_palavras = [t for t in tokens_palavras if t.isalnum()]

    # Retorna um dicionário com os 3 resultados
    return {
        "total_tokens"    : len(tokens_palavras),
        "tokens_palavras" : tokens_palavras,
        "tokens_sentencas": tokens_sentencas,
        "so_palavras"     : so_palavras,
    }

# ── Texto de exemplo ─────────────────────────────────────
mensagem = "Mensagens de interesse são importantes. Propostas relevantes precisam de atenção especial!"

# Chama a função e guarda o resultado
resultado = tokenizar_mensagem(mensagem)

# ── Exibindo os resultados ────────────────────────────────
print(f"Total de tokens : {resultado['total_tokens']}")
print(f"Tokens (palavras): {resultado['tokens_palavras']}")
print(f"Sentenças       : {resultado['tokens_sentencas']}")
print(f"Só palavras     : {resultado['so_palavras']}")

#----------------------------------------------------------------------------------------------------------------

import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist   # classe que conta frequência automaticamente

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)   # palavras sem valor ("de", "o", "e"...)

from nltk.corpus import stopwords

def contar_frequencia(texto: str, top_n: int = 5) -> list:
    """
    Recebe uma avaliação de cliente e retorna as palavras
    mais frequentes, ignorando palavras sem significado.
    """

    # 1. Tudo em minúsculo → "Bom" e "bom" viram a mesma palavra
    texto = texto.lower()

    # 2. Tokeniza: divide o texto em palavras individuais
    tokens = word_tokenize(texto, language='portuguese')

    # 3. Remove stopwords e pontuação
    #    stopwords = palavras que não revelam padrão ("de", "o", "que"...)
    parar = set(stopwords.words('portuguese'))
    palavras = [t for t in tokens if t.isalpha() and t not in parar]

    # 4. FreqDist conta quantas vezes cada palavra aparece
    freq = FreqDist(palavras)

    # 5. Retorna as top_n palavras mais comuns como lista de tuplas
    #    Ex: [('bom', 3), ('produto', 2), ('entrega', 1)]
    return freq.most_common(top_n)

# Simulação de avaliação real de cliente
avaliacao = """
    Produto muito bom, entrega rápida e produto bem embalado.
    Atendimento bom, voltarei a comprar. Produto recomendo!
"""

resultado = contar_frequencia(avaliacao, top_n=5)

print("Palavras mais frequentes na avaliação:")
print("-" * 35)

# Exibe cada palavra com sua contagem de forma legível
for palavra, contagem in resultado:
    print(f"  {palavra:<20} → {contagem}x")

#__________________________________________________________________________________
    
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ── Dicionário de palavras por sentimento ─────────────────
# Cada lista representa sinais que o suporte deve reconhecer
PALAVRAS_NEGATIVAS = {"ruim", "péssimo", "erro", "horrível", "problema", "defeito", "decepcionante"}
PALAVRAS_POSITIVAS = {"ótimo", "excelente", "bom", "perfeito", "adorei", "recomendo", "parabéns"}

# Negações invertem o sentimento da palavra seguinte
# Ex: "não foi ruim" → a negação protege contra falso positivo
NEGACOES = {"não", "nunca", "jamais", "nem"}

def classificar_mensagem(texto: str) -> dict:
    """
    Classifica a mensagem do cliente como NEGATIVA, POSITIVA ou NEUTRA.
    Considera negações para evitar erros de classificação.
    """

    # 1. Normaliza o texto para minúsculo
    tokens = word_tokenize(texto.lower(), language='portuguese')

    pontos_neg = 0
    pontos_pos = 0
    palavras_encontradas = []

    # 2. Percorre os tokens verificando negações + sentimento
    for i, token in enumerate(tokens):

        # Verifica se o token anterior foi uma negação
        tem_negacao = i > 0 and tokens[i - 1] in NEGACOES

        if token in PALAVRAS_NEGATIVAS:
            if tem_negacao:
                pontos_pos += 1          # "não foi ruim" → vira positivo
            else:
                pontos_neg += 1          # "foi ruim" → negativo
            palavras_encontradas.append(token)

        elif token in PALAVRAS_POSITIVAS:
            if tem_negacao:
                pontos_neg += 1          # "não foi bom" → vira negativo
            else:
                pontos_pos += 1          # "foi bom" → positivo
            palavras_encontradas.append(token)

    # 3. Regra condicional de classificação final
    if pontos_neg > pontos_pos:
        classificacao = "🔴 NEGATIVA — acionar suporte"
    elif pontos_pos > pontos_neg:
        classificacao = "🟢 POSITIVA — cliente satisfeito"
    else:
        classificacao = "🟡 NEUTRA — monitorar"

    return {
        "classificacao"       : classificacao,
        "pontos_negativos"    : pontos_neg,
        "pontos_positivos"    : pontos_pos,
        "palavras_detectadas" : palavras_encontradas,
    }

avaliacoes = [
    "O produto foi péssimo, tive um erro na entrega e está com defeito.",
    "Atendimento excelente! O produto é ótimo, recomendo muito.",
    "Não foi ruim, mas também não foi bom. Entrega normal.",
]

for msg in avaliacoes:
    r = classificar_mensagem(msg)
    print(f"Mensagem : {msg}")
    print(f"Resultado: {r['classificacao']}")
    print(f"Palavras : {r['palavras_detectadas']}")
    print("-" * 60)

    # =====================================================================================

    import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus   import stopwords   # lista pronta de palavras sem valor analítico

nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

def remover_stopwords(texto: str) -> dict:
    """
    Remove stopwords de um texto em português.
    Retorna o texto limpo e um comparativo antes/depois.
    """

    # 1. Carrega o conjunto de stopwords do português
    #    Usar set() torna a busca muito mais rápida que uma lista
    lista_stop = set(stopwords.words('portuguese'))

    # 2. Normaliza e tokeniza o texto
    tokens = word_tokenize(texto.lower(), language='portuguese')

    # 3. Regra condicional: mantém apenas palavras que NÃO
    #    estejam na lista de stopwords e sejam letras puras
    palavras_limpas = [
        token for token in tokens
        if token.isalpha()           # remove pontuação e números
        and token not in lista_stop  # remove as stopwords
    ]

    # 4. Recalcula o quanto o texto foi reduzido
    total_antes  = len([t for t in tokens if t.isalpha()])
    total_depois = len(palavras_limpas)
    reducao      = round((1 - total_depois / total_antes) * 100)

    return {
        "texto_limpo"   : " ".join(palavras_limpas),
        "palavras_antes": total_antes,
        "palavras_depois": total_depois,
        "reducao_pct"   : reducao,
    }

avaliacao = "O produto chegou com defeito e a embalagem estava danificada para o transporte."

r = remover_stopwords(avaliacao)

print("Texto original :", avaliacao)
print("Texto limpo    :", r['texto_limpo'])
print(f"Palavras        : {r['palavras_antes']} → {r['palavras_depois']} (-{r['reducao_pct']}%)")

# -----------------------------------------------------------------------------------------------------

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus   import stopwords

nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# ── Palavras-chave com PESO por intensidade ───────────────
# Peso 2 = sinal forte  |  Peso 1 = sinal moderado
POSITIVAS = {
    "excelente"  : 2, "perfeito"  : 2, "adorei"     : 2,
    "ótimo"      : 2, "recomendo" : 2, "incrível"   : 2,
    "bom"        : 1, "rápido"    : 1, "satisfeito" : 1,
    "gostei"     : 1, "eficiente" : 1, "prático"    : 1,
}

NEGATIVAS = {
    "péssimo"    : 2, "horrível"  : 2, "defeito"    : 2,
    "terrível"   : 2, "decepcionei": 2, "lamentável": 2,
    "ruim"       : 1, "problema"  : 1, "demora"     : 1,
    "erro"       : 1, "quebrado"  : 1, "insatisfeito":1,
}

NEGACOES = {"não", "nunca", "jamais", "nem"}

def classificar_sentimento(comentario: str) -> dict:
    """
    Classifica o sentimento de um comentário de cliente.
    Usa sistema de pontos com peso para maior precisão.
    """

    # 1. Remove stopwords e tokeniza o texto limpo
    lista_stop = set(stopwords.words('portuguese'))
    tokens     = word_tokenize(comentario.lower(), language='portuguese')

    score_pos, score_neg = 0, 0
    achados = {"positivas": [], "negativas": []}

    # 2. Percorre tokens aplicando pesos e detectando negações
    for i, token in enumerate(tokens):
        negado = i > 0 and tokens[i - 1] in NEGACOES

        if token in POSITIVAS:
            peso = POSITIVAS[token]
            if negado:
                score_neg += peso           # "não foi ótimo" → penaliza
            else:
                score_pos += peso
                achados["positivas"].append(f"{token}(+{peso})")

        elif token in NEGATIVAS:
            peso = NEGATIVAS[token]
            if negado:
                score_pos += peso           # "sem defeito" → beneficia
            else:
                score_neg += peso
                achados["negativas"].append(f"{token}(-{peso})")

    # 3. Regra condicional com 5 níveis de sentimento
    saldo = score_pos - score_neg

    if   saldo >=  3: sentimento = "🟢 MUITO POSITIVO"
    elif saldo >=  1: sentimento = "🟩 POSITIVO"
    elif saldo ==  0: sentimento = "🟡 NEUTRO / MISTO"
    elif saldo >= -2: sentimento = "🟧 NEGATIVO"
    else:            sentimento = "🔴 MUITO NEGATIVO"

    return {
        "sentimento" : sentimento,
        "saldo"      : saldo,
        "score_pos"  : score_pos,
        "score_neg"  : score_neg,
        "achados"    : achados,
    }

comentarios = [
    "Produto excelente, entrega rápida e atendimento perfeito!",
    "Entrega rápida mas o produto veio com defeito e quebrado.",
    "Não gostei, péssimo atendimento e muita demora na entrega.",
    "Produto sem defeito, gostei bastante, recomendo a loja.",
    "Entrega normal, sem problemas. Nada de especial.",
]

print("=" * 55)
print("   RELATÓRIO DE SENTIMENTOS — EQUIPE DE MARKETING")
print("=" * 55)

for i, comentario in enumerate(comentarios, 1):
    r = classificar_sentimento(comentario)
    print(f"\n[{i}] {comentario}")
    print(f"    Sentimento : {r['sentimento']}")
    print(f"    Saldo      : {r['score_pos']}(+) vs {r['score_neg']}(-) = {r['saldo']:+d}")
    print(f"    Detectadas : pos={r['achados']['positivas']} | neg={r['achados']['negativas']}")

    # __________________________________________________________________________________________________

    import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus   import stopwords

nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# ── Mapa de setores: cada setor tem palavras-chave e prioridade ──
# Prioridade: quanto MENOR o número, mais urgente o setor
SETORES = {
    "Cancelamento": {
        "prioridade" : 1,
        "palavras"   : {"cancelar", "cancelamento", "desistir", "devolver", "reembolso"},
        "resposta"   : "Entendido! Direcionando para Cancelamentos.",
    },
    "Suporte Técnico": {
        "prioridade" : 2,
        "palavras"   : {"erro", "bug", "travou", "falha", "problema", "não funciona"},
        "resposta"   : "Identificamos um problema técnico. Conectando ao Suporte.",
    },
    "Financeiro": {
        "prioridade" : 3,
        "palavras"   : {"pagamento", "cobrança", "fatura", "pix", "boleto", "cartão"},
        "resposta"   : "Assunto financeiro detectado. Indo para o Financeiro.",
    },
    "Entregas": {
        "prioridade" : 4,
        "palavras"   : {"entrega", "prazo", "rastrear", "correios", "atraso", "frete"},
        "resposta"   : "Verificando informações de entrega para você.",
    },
}

def rotear_cliente(mensagem: str) -> dict:
    """
    Detecta palavras-chave na mensagem do cliente e
    direciona para o setor correto com base em prioridade.
    Múltiplos setores acionados → o de maior prioridade vence.
    """

    # 1. Tokeniza e normaliza a mensagem
    tokens = set(word_tokenize(mensagem.lower(), language='portuguese'))

    setores_acionados = []  # guarda todos os setores detectados

    # 2. Verifica quais setores têm palavras-chave na mensagem
    for nome_setor, dados in SETORES.items():

        # intersecção: palavras da mensagem ∩ palavras do setor
        gatilhos = tokens & dados["palavras"]   # operador & em sets = interseção

        if gatilhos:                              # se encontrou ao menos 1 palavra
            setores_acionados.append({
                "setor"      : nome_setor,
                "prioridade" : dados["prioridade"],
                "resposta"   : dados["resposta"],
                "gatilhos"   : gatilhos,
            })

    # 3. Regra condicional: sem setor → atendimento geral
    if not setores_acionados:
        return {
            "setor_principal" : "Atendimento Geral",
            "resposta"        : "Olá! Como posso te ajudar hoje?",
            "outros_setores"  : [],
            "gatilhos"        : [],
        }

    # 4. Ordena por prioridade (menor número = mais urgente)
    setores_acionados.sort(key=lambda s: s["prioridade"])
    principal = setores_acionados[0]   # setor de maior prioridade

    return {
        "setor_principal" : principal["setor"],
        "resposta"        : principal["resposta"],
        "gatilhos"        : list(principal["gatilhos"]),
        "outros_setores"  : [s["setor"] for s in setores_acionados[1:]],
    }

mensagens = [
    "Quero cancelar meu pedido",
    "Tive um erro no pagamento com cartão",
    "Minha entrega está atrasada e quero rastrear",
    "Quero cancelar meu pagamento com erro",   # aciona 3 setores!
    "Boa tarde, tudo bem?",                    # sem palavras-chave
]

print("=" * 52)
print("       CHATBOT — ROTEADOR DE ATENDIMENTO")
print("=" * 52)

for msg in mensagens:
    r = rotear_cliente(msg)
    print(f"\n Cliente  : {msg}")
    print(f"  Chatbot  : {r['resposta']}")
    print(f"  Setor    : {r['setor_principal']}")
    if r["outros_setores"]:
        print(f"  Também   : {r['outros_setores']}")
    print(f"  Gatilhos : {r['gatilhos']}")

    # ===============================================================================

    import nltk
from nltk.tokenize   import word_tokenize
from nltk.probability import FreqDist   # conta frequência automaticamente
from nltk.corpus     import stopwords

nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# Lista simulando reclamações reais de clientes
# Em produção, viria de um banco de dados ou planilha
reclamacoes = [
    "Produto chegou com defeito, a embalagem estava danificada.",
    "Entrega atrasada, prazo não foi cumprido e sem resposta.",
    "O produto parou de funcionar após dois dias, defeito grave.",
    "Cobrado duas vezes no cartão, erro no pagamento absurdo.",
    "Entrega demorou muito além do prazo, produto danificado.",
    "Atendimento péssimo, ninguém resolveu meu problema.",
    "Defeito no produto desde o primeiro uso, inaceitável.",
    "Prazo de entrega descumprido, produto chegou errado.",
    "Erro no sistema de pagamento, fui cobrado indevidamente.",
    "Produto com defeito, embalagem rasgada e prazo estourado.",
]

def analisar_reclamacoes(lista_reclamacoes: list, top_n: int = 8) -> dict:
    """
    Recebe uma lista de reclamações e identifica as palavras
    mais frequentes, revelando os principais motivos de queixa.
    """

    stop    = set(stopwords.words('portuguese'))
    palavras_gerais = []   # acumula tokens de TODAS as reclamações

    # 1. Processa cada reclamação individualmente
    for texto in lista_reclamacoes:
        tokens = word_tokenize(texto.lower(), language='portuguese')

        # Filtra: apenas letras, sem stopwords
        limpos = [
            t for t in tokens
            if t.isalpha() and t not in stop
        ]
        palavras_gerais.extend(limpos)  # extend: adiciona item a item na lista

    # 2. FreqDist conta quantas vezes cada palavra aparece no total
    freq = FreqDist(palavras_gerais)
    top  = freq.most_common(top_n)

    # 3. Calcula em quantas reclamações distintas cada palavra aparece
    #    Isso evita distorção por uma única reclamação repetitiva
    presenca = {}
    for palavra, _ in top:
        presenca[palavra] = sum(
            1 for r in lista_reclamacoes if palavra in r.lower()
        )

    return {
        "total_reclamacoes" : len(lista_reclamacoes),
        "total_palavras"    : len(palavras_gerais),
        "ranking"           : top,
        "presenca"          : presenca,
    }

r = analisar_reclamacoes(reclamacoes, top_n=8)

print("=" * 52)
print("   RELATÓRIO — PALAVRAS-CHAVE DE RECLAMAÇÕES")
print("=" * 52)
print(f"  Reclamações analisadas : {r['total_reclamacoes']}")
print(f"  Total de tokens        : {r['total_palavras']}")
print("─" * 52)
print(f"  {'PALAVRA':<18} {'OCORRÊNCIAS':>11}  {'EM RECLAMAÇÕES':>14}")
print("─" * 52)

for i, (palavra, contagem) in enumerate(r['ranking'], 1):
    em_quantas = r['presenca'][palavra]
    pct        = round(em_quantas / r['total_reclamacoes'] * 100)
    barra      = "█" * contagem          # barra visual proporcional

    print(f"  {i}. {palavra:<16} {contagem:>5}x   {em_quantas}/{r['total_reclamacoes']} recl. ({pct}%)")
    print(f"     {barra}")

print("=" * 52)

# -------------------------------------------------------------------------------------------------------------

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus   import stopwords

nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# ── Regras de classificação por departamento ──────────────
# Cada departamento tem palavras-chave e um peso por palavra
# Peso maior = sinal mais forte para aquele departamento
DEPARTAMENTOS = {
    "Suporte Técnico": {
        "icone"   : "🔧",
        "palavras": {
            "erro"      : 2, "bug"       : 2, "travou"   : 2,
            "falha"     : 2, "sistema"   : 1, "instalar" : 1,
            "lento"     : 1, "atualizar" : 1, "reiniciar": 1,
            "configurar": 1, "acesso"    : 1, "senha"    : 1,
        }
    },
    "Financeiro": {
        "icone"   : "💰",
        "palavras": {
            "boleto"    : 2, "cobrança"  : 2, "reembolso": 2,
            "fatura"    : 2, "pagamento" : 2, "pix"      : 2,
            "cartão"    : 1, "desconto"  : 1, "preço"    : 1,
            "cobrado"   : 1, "valor"     : 1, "nota"     : 1,
        }
    },
}

def classificar_mensagem(mensagem: str) -> dict:
    """
    Classifica uma mensagem em Suporte Técnico ou Financeiro.
    Em caso de empate ou ambiguidade, retorna os dois departamentos.
    """

    # 1. Tokeniza e normaliza
    tokens  = word_tokenize(mensagem.lower(), language='portuguese')
    scores  = {}   # pontuação acumulada por departamento
    achados = {}   # palavras detectadas por departamento

    # 2. Para cada departamento, soma os pesos das palavras encontradas
    for nome, dados in DEPARTAMENTOS.items():
        score       = 0
        encontradas = []

        for token in tokens:
            if token in dados["palavras"]:
                peso  = dados["palavras"][token]
                score += peso
                encontradas.append(f"{token}(+{peso})")

        scores [nome] = score
        achados[nome] = encontradas

    score_tec = scores["Suporte Técnico"]
    score_fin = scores["Financeiro"]

    # 3. Regras condicionais de classificação ─────────────────
    if score_tec == 0 and score_fin == 0:
        # Nenhuma palavra reconhecida → atendimento geral
        classificacao = ["Atendimento Geral 📋"]
        status        = "sem_match"

    elif score_tec > 0 and score_fin > 0:
        # Ambos acionados → mensagem ambígua, envia para os dois
        vencedor      = "Suporte Técnico" if score_tec >= score_fin else "Financeiro"
        outro         = "Financeiro"      if vencedor == "Suporte Técnico" else "Suporte Técnico"
        icone_v       = DEPARTAMENTOS[vencedor]["icone"]
        icone_o       = DEPARTAMENTOS[outro]["icone"]
        classificacao = [f"{icone_v} {vencedor} (principal)", f"{icone_o} {outro} (cópia)"]
        status        = "ambiguo"

    elif score_tec > 0:
        # Apenas técnico foi acionado
        icone         = DEPARTAMENTOS["Suporte Técnico"]["icone"]
        classificacao = [f"{icone} Suporte Técnico"]
        status        = "claro"

    else:
        # Apenas financeiro foi acionado
        icone         = DEPARTAMENTOS["Financeiro"]["icone"]
        classificacao = [f"{icone} Financeiro"]
        status        = "claro"

    return {
        "classificacao" : classificacao,
        "status"        : status,
        "score_tecnico" : score_tec,
        "score_financ"  : score_fin,
        "achados"       : achados,
    }

mensagens = [
    "O sistema travou e está dando erro na tela de acesso.",
    "Preciso do boleto atualizado e nota fiscal do pagamento.",
    "O boleto não abre, dá erro toda vez que tento pagar.",  # ambígua!
    "Minha senha não funciona e fui cobrado duas vezes.",    # ambígua!
    "Boa tarde, gostaria de falar com alguém.",              # sem match
]

print("=" * 56)
print("     CLASSIFICADOR AUTOMÁTICO DE MENSAGENS")
print("=" * 56)

for i, msg in enumerate(mensagens, 1):
    r = classificar_mensagem(msg)

    print(f"\n[{i}] {msg}")
    print(f"     Destino  : {' + '.join(r['classificacao'])}")
    print(f"     Scores   : 🔧 {r['score_tecnico']}pts  💰 {r['score_financ']}pts")
    print(f"     Gatilhos : tec={r['achados']['Suporte Técnico']} | fin={r['achados']['Financeiro']}")
    print(f"     Status   : {r['status']}")

# __________________________________________________________________________________________________________
    
    import re                               # módulo de expressões regulares (regex)
import unicodedata                      # remove acentos via unicode
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus   import stopwords

nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# ── Etapa 1: minúsculas ───────────────────────────────────
# "Produto" == "produto" == "PRODUTO" após lower()
def para_minusculo(texto: str) -> str:
    return texto.lower()


# ── Etapa 2: remover acentos ──────────────────────────────
# "café" → "cafe" | garante que "ótimo" e "otimo" virem o mesmo token
def remover_acentos(texto: str) -> str:
    # NFD decompõe letra+acento em dois caracteres separados
    # encode/decode remove os caracteres de acento (categoria Mn)
    nfd   = unicodedata.normalize('NFD', texto)
    return "".join(c for c in nfd
                   if unicodedata.category(c) != 'Mn')


# ── Etapa 3: remover pontuação e caracteres especiais ─────
# regex [^a-z0-9\s] = "tudo que NÃO for letra, número ou espaço"
def remover_pontuacao(texto: str) -> str:
    return re.sub(r'[^a-z0-9\s]', '', texto)


# ── Etapa 4: remover espaços extras ───────────────────────
# strip() remove bordas | \s+ colapsa múltiplos espaços em um
def remover_espacos(texto: str) -> str:
    return re.sub(r'\s+', ' ', texto).strip()


# ── Etapa 5: remover stopwords ────────────────────────────
# Remove "de", "o", "para" — palavras sem valor analítico
def remover_stopwords(tokens: list) -> list:
    stop = set(stopwords.words('portuguese'))
    return [t for t in tokens if t not in stop]

def limpar_texto(texto: str, remover_stop: bool = True) -> dict:
    """
    Pipeline completo de limpeza e normalização.
    Cada etapa é aplicada em sequência sobre o resultado anterior.
    Parâmetro remover_stop controla se stopwords são removidas.
    """

    # Sequência de transformações — ordem importa!
    passo1 = para_minusculo (texto)   # "Produto!"   → "produto!"
    passo2 = remover_acentos(passo1)   # "ótimo"      → "otimo"
    passo3 = remover_pontuacao(passo2) # "produto!"   → "produto "
    passo4 = remover_espacos (passo3)  # "produto  "  → "produto"

    # Tokeniza o texto já limpo
    tokens = word_tokenize(passo4, language='portuguese')

    # Etapa 5 opcional — depende do uso posterior
    # Para chatbot: manter stopwords  | Para análise: remover
    tokens_finais = remover_stopwords(tokens) if remover_stop else tokens

    return {
        "original"      : texto,
        "limpo"         : passo4,
        "tokens"        : tokens_finais,
        "total_antes"   : len(texto.split()),
        "total_depois"  : len(tokens_finais),
        "reducao_pct"   : round((1 - len(tokens_finais) /
                           len(texto.split())) * 100),
    }

textos = [
    "Produto EXCELENTE!!! Entrega rápida, super recomendo ;)",
    "O sistema deu ERRO novamente... já é a 3ª vez essa semana!",
    "Cobrado 2x no cartão — ABSURDO! Quero reembolso URGENTE.",
]

print("=" * 56)
print("      PIPELINE DE LIMPEZA E NORMALIZAÇÃO")
print("=" * 56)

for i, texto in enumerate(textos, 1):
    r = limpar_texto(texto)
    print(f"\n[{i}] Original  : {r['original']}")
    print(f"     Limpo     : {r['limpo']}")
    print(f"     Tokens    : {r['tokens']}")
    print(f"     Redução   : {r['total_antes']} → {r['total_depois']} palavras (-{r['reducao_pct']}%)")

    # ==========================================================================================================

    
