from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
from evaluation import (plot_feature_distribution, 
                       compute_similarity_matrix,
                       plot_similarity_heatmap,
                       calculate_clustering_quality)

# Existing model setup code remains same
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = Sequential([model, GlobalMaxPooling2D()])

# Extract features and generate evaluation metrics
image_features = []
for files in tqdm(img_files):
    features_list = extract_features(files, model)
    image_features.append(features_list)

# Generate and save evaluation metrics
plot_feature_distribution(image_features)
similarity_matrix = compute_similarity_matrix(image_features)
plot_similarity_heatmap(similarity_matrix)
clustering_score = calculate_clustering_quality(image_features)

# Save metrics to file
metrics = {
    'clustering_score': clustering_score,
    'feature_dim': len(image_features[0]),
    'total_samples': len(image_features)
}
with open('evaluation/metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)

# Save features
pickle.dump(image_features, open("image_features_embedding.pkl", "wb"))
pickle.dump(img_files, open("img_files.pkl", "wb"))
