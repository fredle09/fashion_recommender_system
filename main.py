import os
from tabnanny import verbose
import streamlit as st
from PIL import Image
import numpy as np
import pickle
import tensorflow
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

UPLOAD_FOLDER = "_uploads"

feature_list = np.array(pickle.load(open("embeddings.pkl", "rb")))
filenames = pickle.load(open("filenames.pkl", "rb"))

csv_data = pd.read_csv(
    "images.csv"
)  # Assuming your CSV has 'filename' and 'link' columns


model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([model, GlobalMaxPooling2D()])

st.title("Fashion Recommender System")


def save_uploaded_file(uploaded_file):
    try:
        # check if the folder exists
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
        with open(os.path.join(UPLOAD_FOLDER, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img, verbose=0).flatten()
    normalized_result = result / norm(result)

    return normalized_result


def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm="brute", metric="euclidean")
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices


# steps
# file upload -> save
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        # feature extract
        features = feature_extraction(
            os.path.join(UPLOAD_FOLDER, uploaded_file.name), model
        )
        # st.text(features)
        # recommendention
        [indices] = recommend(features, feature_list)
        # show
        col1, col2, col3, col4, col5 = st.columns(5)

        filename_list = [filenames[indice][7:] for indice in list(indices)]
        link_list = [
            csv_data.loc[csv_data["filename"] == filename, "link"].values[0]
            for filename in filename_list
        ]

        with col1:
            st.image(link_list[0], use_column_width=True)
        with col2:
            st.image(link_list[1], use_column_width=True)
        with col3:
            st.image(link_list[2], use_column_width=True)
        with col4:
            st.image(link_list[3], use_column_width=True)
        with col5:
            st.image(link_list[4], use_column_width=True)
    else:
        st.header("Some error occured in file upload")
