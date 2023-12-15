# app.py

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Charger le modèle CNN
model = tf.keras.models.load_model('pomme de terre.h5')
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']


# Fonction pour prétraiter l'image
def preprocess_image(image):
    img = Image.open(image).convert('RGB')
    img = img.resize((256, 256)) 
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  
    return img_array

def classify_image(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)

    # Obtenez l'indice de la classe prédite
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_names[predicted_class_index]

    return predicted_class_name, predictions

# Ajouter une image d'arrière-plan à l'application Streamlit


# Interface utilisateur Streamlit
st.title('Application de détection des maladies à partir des images des feuilles de pommes de terre')
uploaded_image = st.file_uploader("Choisissez une image...", type="jpg")

if uploaded_image is not None:
    st.image(uploaded_image, caption='Image téléchargée.', use_column_width=True)
    st.write("")
    st.write("RESULTATS : ")

    # Classification de l'image
    predicted_class, predictions = classify_image(uploaded_image)
   

    # Affichage des résultats
    st.write("Classe prédite :")
    st.write(predicted_class)
    st.write("")
    st.write("Niveau de confiance :")
    for i, prob in enumerate(predictions[0]):
        st.write(f"{class_names[i]}: {prob:.2%}")
