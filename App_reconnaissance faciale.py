import streamlit as st 
import numpy as np 
from PIL import Image
from joblib import load 

##Affichage du dashboard 

st.title('Démo de reconnaissance faciale')

##chargement du modèle

model = load('model.joblib')

## Chargement du fichier

uploaded_file = st.file_uploader("Télécharger une image", type = ['jpeg', 'jpg', 'png'])

if uploaded_file is not None : 
    image = Image.open(uploaded_file)
    st.image(image, caption = "Image téléchargée", use_column_width= True)


    st.write("Traitement en cours....")
    # prétraitement en vecteur compatible KNN
    img_array = np.array(image.resize((64, 64)))
    img_vector = img_array.flatten().reshape(1, -1)

    with st.spinner('Prédiction en cours...'):
        prediction = model.predict(img_vector)
        st.success(f"La personne reconnue est : {prediction[0]}")


