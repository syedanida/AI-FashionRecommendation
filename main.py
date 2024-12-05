import streamlit as st
import os
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.models import Sequential
from numpy.linalg import norm
import requests
from config import API_KEY, CSE_ID

# Initialize the model
model = ResNet50(weights="imagenet", include_top=True)  # Top layer included for classification
model.trainable = False

st.title('Clothing Recommender System')

# Fashion-related terms to prioritize search results
fashion_keywords = [
    "shirt", "pants", "shoes", "dress", "jacket", "jeans", "t-shirt", "blouse", "sweater", "coat", "shorts", "skirt",
    "sneakers", "heels", "boots", "bag", "scarf", "suit", "hoodie", "sweatshirt", "activewear", "fashion"
]

# Known e-commerce platforms or product page indicators
product_page_indicators = [
    "product", "buy", "shop", "add-to-cart", "item", "sale",
    "amazon", "ebay", "walmart", "etsy", "target", "asos", "zara", "nike", "adidas"
]

def save_file(uploaded_file):
    try:
        os.makedirs("uploader", exist_ok=True)
        with open(os.path.join("uploader", uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return True
    except OSError as e:
        st.error(f"An error occurred while saving the file: {e}")
        return False

def extract_img_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expand_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expand_img)
    result_to_resnet = model.predict(preprocessed_img)
    # Decode predictions to get human-readable labels
    decoded_predictions = decode_predictions(result_to_resnet, top=3)[0]  # Top 3 predictions
    return decoded_predictions

def search_similar_products(query):
    search_url = f"https://www.googleapis.com/customsearch/v1"
    params = {
        'q': query,
        'cx': CSE_ID,
        'key': API_KEY,
        'searchType': 'image',
        'num': 10  # Retrieve more results to ensure we can filter down to 5
    }
    response = requests.get(search_url, params=params)
    return response.json()

def filter_product_links(search_results):
    product_links = []
    fallback_links = []  # To store non-product links as backup
    if 'items' in search_results:
        for item in search_results['items']:
            link = item['image']['contextLink']
            if any(indicator in link.lower() for indicator in product_page_indicators):
                product_links.append((item['link'], link))
            else:
                fallback_links.append((item['link'], link))
    # Ensure at least 5 suggestions; use fallback links if necessary
    while len(product_links) < 5 and fallback_links:
        product_links.append(fallback_links.pop(0))
    return product_links[:5]  # Return exactly 5 suggestions

uploaded_file = st.file_uploader("Choose your image", type=["jpg", "jpeg", "png", "bmp", "gif", "tiff"])
if uploaded_file is not None:
    if save_file(uploaded_file):
        show_images = Image.open(uploaded_file)
        size = (400, 400)
        resized_im = show_images.resize(size)
        st.image(resized_im)
        
        # Extract features of uploaded image
        predictions = extract_img_features(os.path.join("uploader", uploaded_file.name), model)
        
        # Dynamically create a search query based on top prediction
        top_prediction = predictions[0][1]  # Using the top predicted label
        st.write(f"Predicted label: {top_prediction}")
        
        # Ensure the query focuses on fashion-related items
        query = top_prediction
        for keyword in fashion_keywords:
            if keyword in top_prediction.lower():
                query = top_prediction
                break
        else:
            # If no fashion-related keyword is detected, append a general fashion term
            query = f"{top_prediction} fashion"
        
        st.write(f"Searching for similar fashion items to: {query}")

        # Search for similar products using the API
        search_results = search_similar_products(query)
        
        # Filter and display at least 5 product links
        product_links = filter_product_links(search_results)
        if product_links:
            for image_link, product_link in product_links:
                st.image(image_link, use_container_width=True)
                st.markdown(f"[Buy Product]({product_link})")
        else:
            st.error("No product pages found.")
