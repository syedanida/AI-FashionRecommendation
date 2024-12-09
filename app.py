from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

# Initialize ResNet50 model
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = Sequential([model, GlobalMaxPooling2D()])

# Function to extract features from image
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expand_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expand_img)
    result_to_resnet = model.predict(preprocessed_img)
    flatten_result = result_to_resnet.flatten()
    # Normalizing
    result_normalized = flatten_result / norm(flatten_result)

    return result_normalized

# Define or replace with actual logic to fetch the product page URL
def get_product_page_url(image_filename):
    # Dummy implementation - Replace this with your actual logic
    return f"https://www.example.com/products/{image_filename}"

# List of image files
img_files = []

# Assuming the images are in 'fashion_small/images' directory
for fashion_image in os.listdir('fashion_small/images'):
    images_path = os.path.join('fashion_small/images', fashion_image)
    product_page_url = get_product_page_url(fashion_image)  # Function to fetch or generate product page URL
    img_files.append({"image_path": images_path, "product_url": product_page_url})

# Extract features for all images
image_features = []

for files in tqdm(img_files):
    features_list = extract_features(files["image_path"], model)  # Pass the image path instead of the entire dictionary
    image_features.append(features_list)

# Save the features and image metadata using pickle
pickle.dump(image_features, open("image_features_embedding.pkl", "wb"))
pickle.dump(img_files, open("img_files.pkl", "wb"))
