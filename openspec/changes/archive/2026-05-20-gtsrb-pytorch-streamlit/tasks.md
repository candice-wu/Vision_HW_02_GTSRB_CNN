## 1. Business Understanding & Setup

- [x] 1.1 Install required Python packages (PyTorch, scikit-learn, streamlit, pandas, matplotlib, seaborn, kaggle, joblib)
- [x] 1.2 Configure Kaggle API credentials (for dataset download)
- [x] 1.3 Initialize Git repository and set remote to `https://github.com/candice-wu/Vision_HW_02_GTSRB_CNN`

## 2. Data Understanding & Preparation

- [x] 2.1 Create script to automatically download the Kaggle GTSRB dataset into `data/raw/` and extract it to `data/extracted/`
- [x] 2.2 Implement PyTorch `Dataset` class for GTSRB (handle image loading and 32x32 resizing)
- [x] 2.3 Set up train/val/test `DataLoader` instances with stratified splitting (80/20)

## 3. Modeling

- [x] 3.1 (CNN+STN) Implement the Spatial Transformer Network (STN) module
- [x] 3.2 (CNN+STN) Define the main CNN architecture incorporating the STN and train it (with Epoch history logging)
- [x] 3.3 (ML Models) Create feature extraction script to flatten images (3072 dimensions)
- [x] 3.4 (ML Models) Implement PCA for dimensionality reduction (retaining 95% variance)
- [x] 3.5 (ML Models) Train NN (MLP), SVM, Random Forest, KNN, AdaBoost, and K-means using `scikit-learn` (with Training Time tracking)
- [x] 3.6 Save all trained models (CNN `.pth`, ML models `.joblib`)

## 4. Evaluation

- [x] 4.1 Evaluate CNN and all ML models on the test dataset (including K-means Majority Voting Mapping)
- [x] 4.2 Generate PCA Scree Plot to analyze the required number of principal components
- [x] 4.3 Generate visualizations (Confusion Matrix, ROC Curve, Accuracy Bar Charts, Loss Curves, Training Time Charts) with Seaborn palettes and sorted models
- [x] 4.4 Draft `HW2_Report.md`, `README.md`, and `CHANGELOG.md` (structured via CRISP-DM)

## 5. Deployment

- [x] 5.1 Set up basic Streamlit application layout (`gtsrb.py`) with Left Sidebar metadata
- [x] 5.2 Implement random image sampling from the test dataset AND local Image Upload functionality
- [x] 5.3 Integrate all models (CNN & ML) for real-time inference on the UI
- [x] 5.4 Push the complete project code, reports, and UI to GitHub
