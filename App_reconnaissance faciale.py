import streamlit as st 
import numpy as np 
from PIL import Image
from joblib import load 

##importation des fonctions 
from my_fonction import get_embedding, extract_face

##Affichage du dashboard 

st.title('Démo de reconnaissance faciale')

##chargement du modèle

model = load('model.joblib')

## Chargement du fichier

uploaded_file = st.file_uploader("Télécharger une image", type = ['jpeg', 'jpg', 'png'])

if uploaded_file is not None : 
    image = Image.open(uploaded_file).convert("RGB")
    # Sauvegarde dans un fichier temporaire
    temp_path = "temp.jpg"
    image.save(temp_path)
    st.image(image, caption = "Image téléchargée", use_column_width= True)


    st.write("Traitement en cours....")
    test_face = extract_face(temp_path)
    test_emb = get_embedding(test_face)
    pred = model.predict([test_emb])

    with st.spinner('Prédiction en cours...'):
        prediction = model.predict(img_vector)
        st.success(f"La personne reconnue est : {prediction[0]}")








