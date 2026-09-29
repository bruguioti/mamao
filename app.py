import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import json

# Configuração da página
st.set_page_config(
    page_title="Diagnóstico de Pragas no Mamoeiro",
    page_icon="🌿",
    layout="centered"
)

# Título do Aplicativo
st.title(" Diagnóstico Inteligente de Pragas em Mamoeiros")
st.write("Tire uma foto da folha do mamoeiro usando a câmera ou envie uma imagem para identificar a doença e ver as recomendações de controle.")

# Dicionário ampliado com todas as doenças da base de dados do Kaggle
RECOMENDACOES = {
    "Healthy": {
        "status": "Planta Saudável ✅",
        "descricao": "Nenhum sinal crítico de praga ou doença detectado.",
        "acao": "Continue com o monitoramento periódico e o manejo nutricional adequado."
    },
    "Anthracnose": {
        "status": "Antracnose (Fungo: *Colletotrichum*) ⚠️",
        "descricao": "Presença de manchas encharcadas que evoluem para lesões escuras afundadas.",
        "acao": "Remova frutos e folhas severamente infectados. Aplique fungicidas recomendados para a cultura e evite irrigação excessiva por aspersão."
    },
    "BacterialSpot": {
        "status": "Mancha Bacteriana (Bactéria) ⚠️",
        "descricao": "Presença de pequenas manchas encharcadas nas folhas que podem escurecer.",
        "acao": "Evite transitar no pomar com folhas molhadas e utilize bactericidas à base de cobre recomendados por um agrônomo."
    },
    "Curl": {
        "status": "Encarquilhamento das Folhas 🚨",
        "descricao": "Deformação e enrolamento das folhas, prejudicando a fotossíntese.",
        "acao": "Monitore a presença de insetos sugadores que atuam como vetores e remova partes severamente afetadas."
    },
    "Mealybug": {
        "status": "Infestação de Cochoninha (Mealybug) 🕷️",
        "descricao": "Presença de insetos sugadores cobertos por uma secreção cerosa branca (semelhante a algodão).",
        "acao": "Utilize óleos naturais ou inseticidas específicos para controle de cochoninhas, além de favorecer predadores naturais."
    },
    "Mite disease": {
        "status": "Infestação de Ácaros 🕷️",
        "descricao": "Folhas com aspecto bronzeado, prateado ou com teias na parte inferior.",
        "acao": "Utilize acaricidas registrados para a cultura e aumente a umidade do solo se o ambiente estiver muito seco."
    },
    "Mosaic": {
        "status": "Vírus do Mosaico do Mamoeiro 🚨",
        "descricao": "Folhas com áreas cloróticas alternadas com áreas verde-escuras, deformações e anéis nos frutos.",
        "acao": "Elimine plantas severamente doentes do pomar (rogueing) para evitar o contágio. Controle vetores como pulgões com defensivos específicos."
    },
    "Ringspot": {
        "status": "Mancha Anular do Mamoeiro (Papaya Ringspot Virus - PRSV) 🚨",
        "descricao": "Surgimento de anéis concêntricos cloróticos nas folhas e frutos, com deformações severas.",
        "acao": "Medida preventiva rigorosa com eliminação de plantas infectadas e barreiras de milho ao redor do pomar contra pulgões vetores."
    }
}

# Carregar o modelo e o arquivo de classes oficial gerado no treinamento
@st.cache_resource
def carregar_modelo_e_classes():
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    modelo_path = os.path.join(diretorio_atual, "modelo_pragas_mamao.keras")
    classes_path = os.path.join(diretorio_atual, "classes.json")
    
    model = None
    classes = []
    
    if os.path.exists(modelo_path):
        try:
            model = tf.keras.models.load_model(modelo_path, compile=False)
        except Exception as e:
            st.error(f"Erro ao carregar o modelo: {e}")
            
    if os.path.exists(classes_path):
        with open(classes_path, "r") as f:
            classes = json.load(f)
            
    return model, classes

model, class_names = carregar_modelo_e_classes()

# Opção de entrada: Câmera ou Upload de Arquivo
modo_entrada = st.radio("Escolha como deseja enviar a imagem:", ("Usar Câmera do Aparelho", "Enviar Imagem do Arquivo"))

imagem_file = None

if modo_entrada == "Usar Câmera do Aparelho":
    imagem_file = st.camera_input("Tire uma foto nítida da folha")
else:
    imagem_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

# Processamento e Predição
if imagem_file is not None:
    image = Image.open(imagem_file)
    st.image(image, caption="Imagem Analisada", use_column_width=True)
    
    if model is None or not class_names:
        st.error(" O modelo ou o arquivo de classes ('classes.json') não foram encontrados. Certifique-se de rodar o main.py atualizado primeiro.")
    else:
        with st.spinner("Analisando a folha com IA..."):
            image_resized = image.resize((224, 224))
            img_array = tf.keras.utils.img_to_array(image_resized)
            img_array = np.expand_dims(img_array, axis=0)
            
            predictions = model.predict(img_array)
            predicted_class_idx = np.argmax(predictions[0])
            confianca = float(np.max(predictions[0])) * 100
            
            # Mapeamento dinâmico baseado na ordem oficial do classes.json
            if predicted_class_idx < len(class_names):
                nome_classe = class_names[predicted_class_idx]
                info_doenca = RECOMENDACOES.get(nome_classe, {
                    "status": f"Condição: {nome_classe}",
                    "descricao": "A IA identificou esta categoria no dataset.",
                    "acao": "Consulte um agrônomo para orientações específicas sobre esta ocorrência."
                })
            else:
                info_doenca = {
                    "status": "Condição Identificada ",
                    "descricao": "A IA detectou um padrão na imagem.",
                    "acao": "Consulte um agrônomo para uma avaliação detalhada."
                }
            
            # Exibição do Resultado Dinâmico
            st.markdown("---")
            st.subheader(" Resultado do Diagnóstico")
            st.success(f"**Condição:** {info_doenca['status']} *(Confiança: {confianca:.2f}%)*")
            st.write(f"**Descrição:** {info_doenca['descricao']}")
            
            st.markdown("### 🛠️ O que fazer para mudar (Manejo Recomendado):")
            st.info(f"**Ações Corretivas Sugeridas:**\n\n{info_doenca['acao']}")