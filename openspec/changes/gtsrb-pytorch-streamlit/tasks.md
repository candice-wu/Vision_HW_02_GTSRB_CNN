## 1. Business Understanding & Setup

- [ ] 1.1 Install required Python packages (PyTorch, scikit-learn, streamlit, pandas, matplotlib, kaggle)
- [ ] 1.2 Configure Kaggle API credentials (for dataset download)
- [ ] 1.3 Initialize Git repository and set remote to `https://github.com/candice-wu/Vision_HW_02_GTSRB_CNN`

## 2. Data Understanding & Preparation

- [ ] 2.1 Create script to automatically download the Kaggle GTSRB dataset into `data/raw/` and extract it to `data/extracted/`
- [ ] 2.2 Implement PyTorch `Dataset` class for GTSRB (handle image loading and 32x32 resizing)
- [ ] 2.3 Set up train/val/test `DataLoader` instances

## 3. Modeling

- [ ] 3.1 (CNN+STN) Implement the Spatial Transformer Network (STN) module
- [ ] 3.2 (CNN+STN) Define the main CNN architecture incorporating the STN and train it
- [ ] 3.3 (ML Models) Create feature extraction script to flatten images (3072 dimensions)
- [ ] 3.4 (ML Models) Implement PCA for dimensionality reduction (retaining 95% variance)
- [ ] 3.5 (ML Models) Train SVM, Random Forest, KNN, AdaBoost, and K-means using `scikit-learn`
- [ ] 3.6 Save all trained models (CNN `.pth`, ML models `.pkl`/`.joblib`)

## 4. Evaluation

- [ ] 4.1 Evaluate CNN and all ML models on the test dataset
- [ ] 4.2 Generate PCA Scree Plot to analyze the required number of principal components
- [ ] 4.3 Generate visualizations (Confusion Matrix, ROC Curve, AUC, Accuracy Bar Charts, PR Curve, Loss Curves) for cross-model comparison
- [ ] 4.4 Draft `HW2_Report.md` and `README.md` (structured via CRISP-DM)

## 5. Deployment

- [ ] 5.1 Set up basic Streamlit application layout (`gtsrb.py`)
- [ ] 5.2 Implement random image sampling feature from the test dataset
- [ ] 5.3 Integrate all models (CNN & ML) for real-time inference on the UI
- [ ] 5.4 Push the complete project code, reports, and UI to GitHub
