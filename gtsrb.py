import streamlit as st
import os
import torch
import joblib
import pandas as pd
from PIL import Image
from torchvision import transforms
from src.model import GTSRB_CNN

# Page configuration
st.set_page_config(page_title="GTSRB Traffic Sign Recognition", page_icon="🚦", layout="wide")

@st.cache_resource
def load_all_models():
    device = torch.device("cpu")
    # CNN
    cnn = GTSRB_CNN(num_classes=43).to(device)
    cnn.load_state_dict(torch.load('models/gtsrb_cnn.pth', map_location=device))
    cnn.eval()
    
    # ML Models & PCA
    pca = joblib.load('models/pca_transformer.joblib')
    ml_models = {}
    for name in ['svm', 'randomforest', 'knn', 'adaboost', 'nn']:
        path = f'models/{name}_model.joblib'
        if os.path.exists(path):
            ml_models[name.upper()] = joblib.load(path)
            
    return cnn, ml_models, pca, device

try:
    cnn, ml_models, pca, device = load_all_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
])

# 1. & 2. 左側邊欄更新
st.sidebar.markdown("### 📃 資料狀態")
st.sidebar.markdown("- **資料來源:** [Kaggle GTSRB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)")
st.sidebar.markdown("- **訓練集筆數:** 39,209")
st.sidebar.markdown("- **測試集筆數:** 12,630")
st.sidebar.markdown("- **類別總數:** 43 classes")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ 系統資訊")
st.sidebar.markdown("- **核心模型:** CNN + STN")
st.sidebar.markdown("- **關鍵技術:** PyTorch, scikit-learn, PCA")
st.sidebar.markdown("- **GitHub:** [點此前往](https://github.com/candice-wu/Vision_HW_02_GTSRB_CNN)")

# 7. 現有分頁命名 & 6. 新增評估指標分頁
tab1, tab2 = st.tabs(["🚥 GTSRB 交通號誌辨識", "📊 Evaluation Metrics"])

with tab1:
    st.title("🚦 GTSRB 交通號誌辨識系統")
    st.markdown("本系統比較了 **深度學習 (CNN + STN)** 與多種 **傳統機器學習模型 (PCA + SVM/RF/KNN/AdaBoost)** 的辨識能力。")
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("### 📸 即時影像辨識 (Real-time Inference)")
    
    upload_col, random_col = st.columns(2)
    
    with upload_col:
        uploaded_file = st.file_uploader("📂 自行上傳號誌影像 (Upload Image)", type=["png", "jpg", "jpeg", "ppm"])
        
    with random_col:
        st.markdown("<br>", unsafe_allow_html=True)
        random_clicked = st.button("🎲 隨機抽樣測試集 (Random Sample)", use_container_width=True)
        
    img_to_infer = None
    true_class = None
    
    if uploaded_file is not None:
        img_to_infer = Image.open(uploaded_file).convert('RGB')
        true_class = "未知 (自行上傳)"
    elif random_clicked:
        test_df = pd.read_csv('data/raw/Test.csv')
        random_row = test_df.sample(1).iloc[0]
        img_path = os.path.join('data/raw', random_row['Path'])
        true_class = random_row['ClassId']
        img_to_infer = Image.open(img_path).convert('RGB')
        
    if img_to_infer is not None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("輸入影像 (Input Image)")
            st.image(img_to_infer, use_container_width=True, caption=f"真實類別: {true_class}")
            
        with col2:
            st.subheader("模型預測結果 (Model Predictions)")
            
            img_tensor = transform(img_to_infer).unsqueeze(0).to(device)
            img_flat = img_tensor.view(1, -1).numpy()
            img_pca = pca.transform(img_flat)
            
            with torch.no_grad():
                outputs = cnn(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                cnn_conf, cnn_pred = torch.max(probs, 1)
            
            # 4. 增大 class 類別的數值字體
            st.markdown(f"**CNN (STN):** <span style='font-size: 28px; color: #ff4b4b; font-weight: bold;'>Class {cnn_pred.item()}</span> (Confidence: {cnn_conf.item()*100:.2f}%)", unsafe_allow_html=True)
            
            for name, model in ml_models.items():
                pred = model.predict(img_pca)[0]
                st.markdown(f"**{name}:** <span style='font-size: 22px; color: #1f77b4; font-weight: bold;'>Class {pred}</span>", unsafe_allow_html=True)
    else:
        st.info("👈 請在上傳區選擇影像，或是點擊隨機抽樣按鈕進行即時辨識！")

    # 3. 右主頁最下方資訊
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: gray;'>🎓 國立中興大學 (National Chung Hsing University)<br>系統版本：v1.1.1 (Phase 1) | 最後更新：2026-05-20</div>", unsafe_allow_html=True)

with tab2:
    st.title("📊 模型評估指標 (Evaluation Metrics)")
    st.markdown("<br>", unsafe_allow_html=True)
    
    colA, colB = st.columns(2)
    
    # 6. 每一個圖表上方要有標題，下方要有動態結語
    with colA:
        st.subheader("🏆 跨模型準確率比較")
        if os.path.exists('reports/figures/accuracy_comparison.png'):
            st.image('reports/figures/accuracy_comparison.png', use_container_width=True)
            st.info("💡 **結語:** CNN 憑藉其強大的空間特徵提取與 STN 空間對齊能力，在準確率上顯著超越了所有將影像打平降維的傳統機器學習模型。")
            
        st.markdown("<br>", unsafe_allow_html=True)
            
        st.subheader("📉 CNN 訓練損失與準確率 (Loss & Acc)")
        if os.path.exists('reports/figures/loss_curve.png'):
            st.image('reports/figures/loss_curve.png', use_container_width=True)
            st.info("💡 **結語:** 隨著訓練週期 (Epoch) 增加，Loss 快速收斂，而 Validation Accuracy 穩步上升至將近 99.6%，顯示模型學習效率極高且無嚴重 Overfitting。")
            
        st.markdown("<br>", unsafe_allow_html=True)
            
        st.subheader("🔍 CNN 混淆矩陣")
        if os.path.exists('reports/figures/cnn_confusion_matrix.png'):
            st.image('reports/figures/cnn_confusion_matrix.png', use_container_width=True)
            st.info("💡 **結語:** CNN 模型在絕大部份類別皆能精確預測，對角線呈現極深的顏色。矩陣中的微小雜訊通常來自於極度相似的號誌 (如速限 30 與 50)。")
            
    with colB:
        st.subheader("⏱️ 跨模型訓練時間比較")
        if os.path.exists('reports/figures/training_time_comparison.png'):
            st.image('reports/figures/training_time_comparison.png', use_container_width=True)
            st.info("💡 **結語:** 傳統 ML 在降維後通常能快速完成訓練；但像 SVM 若資料量龐大時，即使降維也會耗費大量時間。反觀使用 GPU 的 CNN，在龐大資料集中仍能保持極高的訓練效率。")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.subheader("📉 PCA 陡坡圖 (Scree Plot)")
        if os.path.exists('reports/figures/pca_scree_plot.png'):
            st.image('reports/figures/pca_scree_plot.png', use_container_width=True)
            st.info("💡 **結語:** 透過 PCA 降維，我們發現只要保留 86 個主成份即可涵蓋原始 3072 維度中 95% 的變異量，這成功減輕了 SVM 等傳統模型的運算負擔。")
            
        st.markdown("<br>", unsafe_allow_html=True)
            
        st.subheader("📈 跨模型 ROC 曲線")
        if os.path.exists('reports/figures/roc_curve_comparison.png'):
            st.image('reports/figures/roc_curve_comparison.png', use_container_width=True)
            st.info("💡 **結語:** ROC 曲線顯示 CNN (Macro AUC 接近 1.0) 幾乎完美貼合左上角，代表其綜合分類能力最佳；而 Random Forest 亦展現了不錯的 AUC 面積。")

    # 3. 右主頁最下方資訊
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: gray;'>🎓 國立中興大學 (National Chung Hsing University)<br>系統版本：v1.1.1 (Phase 1) | 最後更新：2026-05-20</div>", unsafe_allow_html=True)
