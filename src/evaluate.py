import os
import torch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from dataset import get_dataloaders
from model import GTSRB_CNN

# Set font for matplotlib (to avoid font issues)
import json
plt.rcParams['figure.figsize'] = (10, 6)

def load_models(device):
    models = {}
    
    # 1. Load CNN
    cnn = GTSRB_CNN(num_classes=43).to(device)
    cnn.load_state_dict(torch.load('models/gtsrb_cnn.pth', map_location=device))
    cnn.eval()
    models['CNN'] = cnn
    
    # 2. Load ML Models
    ml_names = ['svm', 'randomforest', 'knn', 'adaboost', 'nn', 'kmeans']
    for name in ml_names:
        model_path = f'models/{name}_model.joblib'
        if os.path.exists(model_path):
            models[name.upper()] = joblib.load(model_path)
            
    # Load PCA
    pca = joblib.load('models/pca_transformer.joblib')
    
    return models, pca

def evaluate():
    print("Starting evaluation...")
    data_dir = 'data/raw'
    save_dir = 'reports/figures/2nd_backup'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    _, _, test_loader = get_dataloaders(data_dir, batch_size=256)
    
    models, pca = load_models(device)
    
    # 1. Extract test features and labels
    print("Extracting test data features...")
    X_test_img = []
    y_true = []
    
    # For CNN predictions
    cnn_preds = []
    cnn_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images_dev = images.to(device)
            # CNN inference
            outputs = models['CNN'](images_dev)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            cnn_preds.append(predicted.cpu().numpy())
            cnn_probs.append(probs.cpu().numpy())
            
            # Flatten for ML
            X_test_img.append(images.view(images.size(0), -1).numpy())
            y_true.append(labels.numpy())
            
    X_test_flat = np.concatenate(X_test_img, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    
    results = {}
    
    # Store CNN results
    results['CNN'] = {
        'preds': np.concatenate(cnn_preds, axis=0),
        'probs': np.concatenate(cnn_probs, axis=0)
    }
    
    # 2. PCA Scree Plot
    print("Generating PCA Scree Plot...")
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Scree Plot (Retaining 95% Variance)')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'pca_scree_plot.png'))
    plt.close()
    
    # Transform test data for ML models
    X_test_pca = pca.transform(X_test_flat)
    
    # Evaluate ML Models
    for name in ['SVM', 'RANDOMFOREST', 'KNN', 'ADABOOST', 'NN', 'KMEANS']:
        if name in models:
            print(f"Evaluating {name}...")
            preds = models[name].predict(X_test_pca)
            
            if name == 'KMEANS':
                # K-means outputs cluster IDs, not true labels.
                # We use Majority Voting to map each cluster ID to the most frequent true label.
                mapped_preds = np.zeros_like(preds)
                for i in range(43):
                    mask = (preds == i)
                    if np.sum(mask) > 0:
                        # Find the most frequent true label in this cluster
                        most_common = np.bincount(y_true[mask]).argmax()
                        mapped_preds[mask] = most_common
                preds = mapped_preds
                probs = None
            else:
                try:
                    probs = models[name].predict_proba(X_test_pca)
                except:
                    probs = None # KNN might not have predict_proba depending on setup
                    
            results[name] = {'preds': preds, 'probs': probs}
            
    # 3. Accuracy Bar Chart
    print("Generating Accuracy Bar Chart...")
    ordered_names = ['CNN', 'NN', 'SVM', 'RANDOMFOREST', 'KNN', 'ADABOOST', 'KMEANS']
    accuracies = {name: accuracy_score(y_true, results[name]['preds']) for name in ordered_names if name in results}
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), hue=list(accuracies.keys()), palette='viridis', legend=False)
    plt.title('Cross-Model Accuracy Comparison', fontsize=16)
    plt.ylabel('Accuracy', fontsize=14)
    plt.ylim(0, 1.1)
    for i, v in enumerate(accuracies.values()):
        plt.text(i, v + 0.02, f"{v*100:.2f}%", ha='center', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_comparison.png'))
    plt.close()
    
    # 4. Confusion Matrix (only for CNN to avoid clutter, or best model)
    print("Generating Confusion Matrix for CNN...")
    cm = confusion_matrix(y_true, results['CNN']['preds'])
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=False, cmap='Blues')
    plt.title('CNN Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(save_dir, 'cnn_confusion_matrix.png'))
    plt.close()
    
    # 5. ROC Curve and AUC (Macro average for CNN)
    print("Generating ROC Curve...")
    y_true_bin = label_binarize(y_true, classes=range(43))
    
    macro_aucs = {}
    plt.figure()
    for name in ordered_names:
        if name in results and results[name]['probs'] is not None:
            res = results[name]
            # Calculate macro ROC AUC
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(43):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], res['probs'][:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                
            # Aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(43)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(43):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= 43
            
            macro_auc = auc(all_fpr, mean_tpr)
            macro_aucs[name] = float(macro_auc)
            plt.plot(all_fpr, mean_tpr, label=f'{name} (macro AUC = {macro_auc:.3f})')
            
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve (Macro Average)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(save_dir, 'roc_curve_comparison.png'))
    plt.close()
    
    # 6. Loss Curves
    hist_path = 'reports/training_history.csv'
    if os.path.exists(hist_path):
        print("Generating Loss Curves...")
        history = pd.read_csv(hist_path)
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['epoch'], history['train_loss'], label='Train Loss')
        plt.plot(history['epoch'], history['val_loss'], label='Val Loss')
        plt.title('CNN Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['epoch'], history['train_acc'], label='Train Acc')
        plt.plot(history['epoch'], history['val_acc'], label='Val Acc')
        plt.title('CNN Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
        plt.close()
        
    # 7. Training Time Comparison
    times_path = 'reports/training_times.json'
    if os.path.exists(times_path):
        print("Generating Training Time Comparison...")
        with open(times_path, 'r') as f:
            times = json.load(f)
        ordered_times = {name: times[name] for name in ordered_names if name in times}
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(ordered_times.keys()), y=list(ordered_times.values()), hue=list(ordered_times.keys()), palette='magma', legend=False)
        plt.title('Training Time Comparison (Seconds)', fontsize=16)
        plt.ylabel('Time (s)', fontsize=14)
        max_t = max(ordered_times.values())
        plt.ylim(0, max_t * 1.15)
        for i, v in enumerate(ordered_times.values()):
            plt.text(i, v + (max_t * 0.02), f"{v:.2f}s", ha='center', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_time_comparison.png'))
        plt.close()
        
    # 8. Save Metrics JSON
    metrics_path = os.path.join(save_dir, 'metrics.json')
    metrics_dict = {
        'accuracies': {name: float(accuracies[name]) for name in accuracies},
        'macro_aucs': macro_aucs
    }
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"Metrics saved to {metrics_path}")
    
    print("Evaluation complete! Visualizations saved in 'reports/figures/'.")

if __name__ == '__main__':
    evaluate()
