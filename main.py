import streamlit as st
import tensorflow as tf
import os
import cv2
import numpy as np
import random
import base64
import requests
import json
import time
import jwt
import logging
import pickle
from PIL import Image
from typing import Optional, Dict, Any, Union, Tuple
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors

# Initialize ResNet50 model
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = Sequential([model, GlobalMaxPooling2D()])

# Load pre-computed features
features_list = pickle.load(open("image_features_embedding.pkl", "rb"))
img_files_list = pickle.load(open("img_files.pkl", "rb"))

class KlingAIClient:
    def __init__(self, access_key: str, secret_key: str, base_url: str):
        self.access_key = "8848da2d9445405283f262c16b3173e7"
        self.secret_key = "22bd1c413003442786e970df07525db4"
        self.base_url = "https://api.klingai.com"
        self.logger = logging.getLogger(__name__)

    def _generate_jwt_token(self) -> str:
        headers = {
            "alg": "HS256",
            "typ": "JWT"
        }
        payload = {
            "iss": self.access_key,
            "exp": int(time.time()) + 1800,
            "nbf": int(time.time()) - 5
        }
        return jwt.encode(payload, self.secret_key, headers=headers)
    
    def _get_headers(self) -> Dict[str, str]:
        return {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {self._generate_jwt_token()}"
        }

    def try_on(self, person_img: np.ndarray, garment_img: np.ndarray, seed: int) -> Tuple[np.ndarray, str]:
        if person_img is None or garment_img is None:
            raise ValueError("Empty image")
            
        encoded_person = cv2.imencode('.jpg', cv2.cvtColor(person_img, cv2.COLOR_RGB2BGR))[1]
        encoded_person = base64.b64encode(encoded_person.tobytes()).decode('utf-8')
        
        encoded_garment = cv2.imencode('.jpg', cv2.cvtColor(garment_img, cv2.COLOR_RGB2BGR))[1]
        encoded_garment = base64.b64encode(encoded_garment.tobytes()).decode('utf-8')

        url = f"{self.base_url}/v1/images/kolors-virtual-try-on"
        data = {
            "model_name": "kolors-virtual-try-on-v1",
            "cloth_image": encoded_garment,
            "human_image": encoded_person,
            "seed": seed
        }

        try:
            response = requests.post(
                url, 
                headers=self._get_headers(),
                json=data,
                timeout=50
            )
            response.raise_for_status()
            
            result = response.json()
            task_id = result['data']['task_id']
            
            time.sleep(4)
            
            for attempt in range(12):
                try:
                    url = f"{self.base_url}/v1/images/kolors-virtual-try-on/{task_id}"
                    response = requests.get(url, headers=self._get_headers(), timeout=20)
                    response.raise_for_status()
                    
                    result = response.json()
                    status = result['data']['task_status']
                    
                    if status == "succeed":
                        output_url = result['data']['task_result']['images'][0]['url']
                        img_response = requests.get(output_url)
                        img_response.raise_for_status()
                        
                        nparr = np.frombuffer(img_response.content, np.uint8)
                        result_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                        return result_img, "Success"
                    elif status == "failed":
                        return None, f"Error: {result['data']['task_status_msg']}"
                        
                except requests.exceptions.ReadTimeout:
                    if attempt == 11:
                        return None, "Request timed out"
                        
                time.sleep(1)
                
            return None, "Processing took too long"
            
        except Exception as e:
            self.logger.error(f"Error in try_on: {str(e)}")
            return None, f"Error: {str(e)}"

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expand_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expand_img)
    result_to_resnet = model.predict(preprocessed_img)
    flatten_result = result_to_resnet.flatten()
    result_normalized = flatten_result / norm(flatten_result)
    return result_normalized

def recommend(features, features_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(features_list)
    distance, indices = neighbors.kneighbors([features])
    return indices

def save_uploaded_file(uploaded_file):
    try:
        if not os.path.exists("uploader"):
            os.makedirs("uploader")
        with open(os.path.join("uploader", uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return False

def main():
    st.title('Fashion Recommendation System with Virtual Try-On')
    
    tab1, tab2 = st.tabs(["Recommendation System", "Virtual Try-On"])
    
    with tab1:
        st.header("Get Similar Fashion Recommendations")
        uploaded_file = st.file_uploader("Choose your fashion image", key="fashion_image")
        
        if uploaded_file is not None:
            if save_uploaded_file(uploaded_file):
                show_image = Image.open(uploaded_file)
                resized_image = show_image.resize((400, 400))
                st.image(resized_image)
                
                features = extract_features(os.path.join("uploader", uploaded_file.name), model)
                indices = recommend(features, features_list)
                
                cols = st.columns(5)
                for idx, col in enumerate(cols):
                    with col:
                        st.header(f"#{idx+1}")
                        img_idx = indices[0][idx]
                        img_path = img_files_list[img_idx]["image_path"]
                        st.image(img_path)
                        st.write(f"[View Product]({img_files_list[img_idx]['product_url']})")
    
    with tab2:
        st.header("Virtual Try-On")
        client = KlingAIClient(
            access_key="YOUR_ACCESS_KEY",
            secret_key="YOUR_SECRET_KEY",
            base_url="https://api.klingai.com"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            person_image = st.file_uploader("Upload Person Image", type=['jpg', 'jpeg', 'png'], key="person_image")
        with col2:
            garment_image = st.file_uploader("Upload Garment Image", type=['jpg', 'jpeg', 'png'], key="garment_image")
            
        if person_image and garment_image:
            seed = random.randint(0, 999999)
            if st.button("Try On"):
                person_img = np.array(Image.open(person_image))
                garment_img = np.array(Image.open(garment_image))
                
                result_img, status = client.try_on(person_img, garment_img, seed)
                if result_img is not None:
                    st.image(result_img, caption="Virtual Try-On Result")
                else:
                    st.error(status)

if __name__ == "__main__":
    main()

