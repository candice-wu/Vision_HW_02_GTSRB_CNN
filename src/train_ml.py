import os
import torch
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from dataset import get_dataloaders
import time
import json

def extract_features(dataloader):
    features = []
    labels_list = []
    
    print(f"Extracting features from {len(dataloader.dataset)} samples...")
    for images, labels in dataloader:
        # Flatten images: (batch_size, 3, 32, 32) -> (batch_size, 3072)
        images = images.view(images.size(0), -1)
        features.append(images.numpy())
        labels_list.append(labels.numpy())
        
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return features, labels

def train_ml():
    data_dir = 'data/raw'
    save_dir = 'models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # We use a larger batch size for faster feature extraction
    train_loader, val_loader, _ = get_dataloaders(data_dir, batch_size=256)
    
    # 3.3 Extract Features
    print("--- 1. Feature Extraction ---")
    X_train, y_train = extract_features(train_loader)
    
    # 3.4 PCA Dimensionality Reduction
    print("--- 2. PCA Dimensionality Reduction ---")
    print(f"Original feature shape: {X_train.shape}")
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    print(f"Reduced feature shape: {X_train_pca.shape} (retained 95% variance)")
    
    joblib.dump(pca, os.path.join(save_dir, 'pca_transformer.joblib'))
    
    # 3.5 Train Traditional ML Models
    print("--- 3. Training ML Models ---")
    models = {
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42),
        'NN': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42),
        'KMeans': KMeans(n_clusters=43, random_state=42) # Note: KMeans is unsupervised, labels are just for mapping later if needed
    }
    
    times_file = 'reports/training_times.json'
    if os.path.exists(times_file):
        with open(times_file, 'r') as f:
            times = json.load(f)
    else:
        times = {}
        
    for name, model in models.items():
        print(f"Training {name}...")
        start_t = time.time()
        model.fit(X_train_pca, y_train)
        t = time.time() - start_t
        times[name.upper()] = t
        
        # 3.6 Save Models
        model_path = os.path.join(save_dir, f'{name.lower()}_model.joblib')
        joblib.dump(model, model_path)
        print(f"Saved {name} to {model_path} (Took {t:.2f}s)")
        
    with open(times_file, 'w') as f:
        json.dump(times, f)
        
    print("All ML models trained and saved successfully!")

if __name__ == '__main__':
    train_ml()
