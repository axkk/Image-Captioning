# import tensorflow as tf
# tf.config.run_functions_eagerly(True)
import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from keras.models import Model
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from keras.optimizers import Adam


# Load the mappings between word and index
with open("word_to_idx.pkl", 'rb') as file:
    word_to_index = pd.read_pickle(file)

with open("idx_to_word.pkl", 'rb') as file:
    index_to_word = pd.read_pickle(file)



model = load_model('model_14.h5', compile=False)

# Recompile the model with a compatible optimizer (e.g., Adam)
model.compile(optimizer=Adam(), loss='categorical_crossentropy')

# Load ResNet50 for image feature extraction
resnet50_model = ResNet50(weights='imagenet', input_shape=(224, 224, 3))
resnet50_model = Model(resnet50_model.input, resnet50_model.layers[-2].output)

# Function to preprocess an image
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)  
    img = preprocess_input(img)
    return img

# Function to encode image to feature vector
def encode_image(img):
    img = preprocess_image(img)
    feature_vector = resnet50_model.predict(img)
    return feature_vector

# Function to predict caption for an image
def predict_caption(photo):
    inp_text = "startseq"
    for i in range(80):  # Generate caption up to max length
        sequence = [word_to_index.get(w, 0) for w in inp_text.split() if w in word_to_index]
        sequence = pad_sequences([sequence], maxlen=80, padding='post')
        ypred = model.predict([photo, sequence])
        ypred = ypred.argmax()
        word = index_to_word.get(ypred, '')
        inp_text += (' ' + word)
        if word == 'endseq':
            break
    final_caption = inp_text.split()[1:-1]  # Remove startseq and endseq
    return ' '.join(final_caption)

# Streamlit UI
st.title("Image Caption Generator")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Encode the image and predict caption
    st.write("Generating caption...")
    photo = encode_image(img).reshape((1, 2048))
    caption = predict_caption(photo)
    
    st.write(f"**Caption:** {caption}")

