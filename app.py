import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras import models, layers

# Charger le modèle CNN préalablement sauvegardé
model_CNN = models.load_model('pomme_de_terre_model.h5')

# Définir la taille des images
IMAGE_SIZE = 256

def predict(model, img, class_names):
    img_array = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img_array)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidences = predictions[0]

    return predicted_class, confidences

# Charger les class_names depuis votre modèle ou définissez-les ici
class_names = ['Potato___Early_blight', 'Potato___Healthy', 'Potato___Late_blight']


# Titre de l'application
st.title("DETECTION DE MALADIE DE POMME DE TERRE")

# Sélectionner une image à prédire
uploaded_file = st.file_uploader("Choisissez une image...", type="jpg")

if uploaded_file is not None:
    # Charger l'image et l'afficher
    image = tf.image.decode_image(uploaded_file.read(), channels=3)
    st.image(image.numpy(), caption="Image téléchargée.", use_column_width=True)

    # Prédire la classe de l'image
    predicted_class, confidences = predict(model_CNN, image, class_names)

    # Afficher le résultat
    st.write("CLASSE PREDITE :")
    st.success(predicted_class)  

    st.write("CONFIANCE PAR CLASSE :")
    for i, class_name in enumerate(class_names):
        st.success(f"{class_name}: {confidences[i] * 100:.2f}%")
