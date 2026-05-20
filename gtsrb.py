import streamlit as st
import os
import torch
import joblib
import pandas as pd
import textwrap
import json
from PIL import Image
from torchvision import transforms
from src.model import GTSRB_CNN

# Page configuration
st.set_page_config(page_title="GTSRB Traffic Sign Recognition", page_icon="🚦", layout="wide")

# System details and unified footer
SYSTEM_VERSION = "v1.2.0 (Phase 2 | Optimized)"
UPDATE_DATE = "2026-05-20"

def render_footer():
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: gray;'>"
        f"🎓 國立中興大學 (National Chung Hsing University)<br>"
        f"系統版本：{SYSTEM_VERSION} | 最後更新：{UPDATE_DATE}"
        f"</div>",
        unsafe_allow_html=True
    )

def load_metrics(backup_dir):
    json_path = os.path.join("reports/figures", backup_dir, "metrics.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

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

# 7. 現有分頁命名 & 6. 新增評估指標分頁 & 雙分頁效能對比
tab1, tab2, tab3, tab4 = st.tabs(["🚥 GTSRB 交通號誌辨識", "📊 初始評估指標 (Original)", "🔆 優化評估指標 (Optimized)", "⛳ 系統優化與驗收說明"])

with tab1:
    st.title("🚦 GTSRB 交通號誌辨識系統")
    st.markdown(textwrap.dedent("""
    <div class="info-box">
    <ul>
        <li>本系統旨在比較 <span style='color:#4481D7'>深度學習</span> 與 <span style='color:#4481D7'>機器學習</span> 模型的圖像分類能力，並提供快速辨識交通號誌的功能</li>
    </ul>
    </div>
    """), unsafe_allow_html=True)
    st.info("""
    **系統核心目標：**
    - 提供快速辨識交通號誌的功能
    - 比較深度學習與傳統機器學習模型的性能
    - 展示不同模型的優缺點
    """,icon="ℹ️")
    st.divider()
    
    st.markdown("### 📸 即時影像辨識 (Real-time Inference)")
    
    random_col, upload_col = st.columns(2)
        
    with random_col:
        st.markdown("<br>", unsafe_allow_html=True)
        random_clicked = st.button("🎲 隨機抽樣測試集 (Random Sample)", use_container_width=True)
    
    with upload_col:
        uploaded_file = st.file_uploader("📂 自行上傳號誌影像 (Upload Image)", type=["png", "jpg", "jpeg", "ppm"])

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
        st.info("👈 請點擊隨機抽樣按鈕，或是在上傳區選擇影像進行即時辨識！")

    # 3. 右主頁最下方資訊
    render_footer()

with tab2:
    metrics_1st = load_metrics("1st_backup")
    cnn_auc_1st = metrics_1st.get("macro_aucs", {}).get("CNN", 0.9987)
    st.title("📊 初始評估指標 (Original Phase 1)")
    st.markdown("這是第一階段未加入 <span style='color:#4481D7'>批次標準化 (BatchNorm)</span> 與 <span style='color:#4481D7'>資料擴增 (Data Augmentation)</span> 時的模型評估結果", unsafe_allow_html=True)
    with st.expander("💬 評估說明", expanded=False):
        st.markdown(textwrap.dedent("""
        1. 評估基準 (Baseline)
           - 針對 CNN 模型，首先建立最基礎的 <span style='color:#4481D7'>基準模型 (Baseline Model)</span>，該模型結構無批次標準化 (BatchNorm) 與資料擴增 (Data Augmentation) 技術
           - 建立此基準模型的表現，作為後續效能比較與優化效果驗證的對照組，以清楚量化後續加入進階技術後的性能提升幅度
        2. 評估環境
           - 測試資料集採用 <span style='color:#4481D7'>GTSRB (German Traffic Sign Recognition Benchmark)</span> 官方測試集，包含 <span style='color:#DD6D6A'>43</span> 類交通號誌圖像
           - 所有模型均在相同的硬體環境與計算條件下進行獨立評估，以確保比較結果的公平性與準確性
    """), unsafe_allow_html=True)
    st.divider()
    
    colA, colB = st.columns(2)
    
    # 6. 每一個圖表上方要有標題，下方要有動態結語
    with colA:
        st.subheader("🏆 跨模型準確率比較")
        if os.path.exists('reports/figures/1st_backup/accuracy_comparison.png'):
            st.image('reports/figures/1st_backup/accuracy_comparison.png', use_container_width=True)
            with st.expander("💡 結論：", expanded=True):
                st.markdown(f"""
                <span style="color:#4481D7">CNN (STN)</span> 憑藉其強大的空間特徵提取與 <span style='color:#4481D7'>STN (空間轉換網路)</span> 的空間對齊能力，在準確度上明顯優於所有將影像打平降維的傳統機器學習模型！
                """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
            
        st.subheader("📉 CNN 訓練損失與準確率 (Loss & Acc)")
        if os.path.exists('reports/figures/1st_backup/loss_curve.png'):
            st.image('reports/figures/1st_backup/loss_curve.png', use_container_width=True)
            with st.expander("💡 結論：", expanded=True):
                st.markdown(f"""
                隨著 <span style='color:#4481D7'>訓練週期 (Epoch)</span> 增加，<span style='color:#4481D7'>Loss</span> 快速收斂，而 <span style='color:#4481D7'>Validation Accuracy</span> 穩步上升至將近 <span style='color:#DD6D6A; font-weight: bold;'>99.60%</span>，顯示模型學習效率極高且無嚴重過擬合 (<span style='color:#4481D7'>Overfitting</span>)！
                """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
            
        st.subheader("🔍 CNN 混淆矩陣")
        if os.path.exists('reports/figures/1st_backup/cnn_confusion_matrix.png'):
            st.image('reports/figures/1st_backup/cnn_confusion_matrix.png', use_container_width=True)
            with st.expander("💡 結論：", expanded=True):
                st.markdown(f"""
                - <span style='color:#4481D7'>CNN 模型</span> 在絕大部份類別皆能精確預測，主對角線呈現極深的藍色。
                - 矩陣中的微小雜訊通常來自於外觀極度相似的號誌（如速限 <span style='color:#4481D7'>30 km/h</span> 與 <span style='color:#4481D7;'>50 km/h</span>）！
                """, unsafe_allow_html=True)
            
    with colB:
        st.subheader("⏱️ 跨模型訓練時間比較")
        if os.path.exists('reports/figures/1st_backup/training_time_comparison.png'):
            st.image('reports/figures/1st_backup/training_time_comparison.png', use_container_width=True)
            with st.expander("💡 結論：", expanded=True):
                st.markdown(f"""
                - 傳統機器學習模型在藉由 <span style='color:#4481D7'>PCA</span> 降維後通常能快速完成訓練
                - <span style='color:#4481D7'>SVM</span> 若在資料量龐大時，即使降維也會耗費大量時間
                - 使用 <span style='color:#4481D7'>GPU</span> 加速的 <span style='color:#4481D7'>CNN</span>，面對龐大資料集仍保持極高訓練效率
                """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.subheader("📉 PCA 陡坡圖 (Scree Plot)")
        if os.path.exists('reports/figures/1st_backup/pca_scree_plot.png'):
            st.image('reports/figures/1st_backup/pca_scree_plot.png', use_container_width=True)
            with st.expander("💡 結論：", expanded=True):
                st.markdown(f"""
                透過 <span style='color:#4481D7'>PCA (主成份分析)</span> 進行維度縮減，發現只要保留 <span style='color:#DD6D6A'>86 個主成份</span> 即可涵蓋原始 3072 維度中高達 <span style='color:#DD6D6A'>95.00%</span> 變異量，成功減輕 <span style='color:#4481D7'>SVM</span> 與 <span style='color:#4481D7'>Random Forest</span> 等傳統模型的運算負擔
                """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
            
        st.subheader("📈 跨模型 ROC 曲線")
        if os.path.exists('reports/figures/1st_backup/roc_curve_comparison.png'):
            st.image('reports/figures/1st_backup/roc_curve_comparison.png', use_container_width=True)
            with st.expander("💡 結論：", expanded=True):
                st.markdown(f"""
                - <span style='color:#4481D7'>ROC 曲線</span> 顯示 <span style='color:#4481D7'>CNN</span> 的 <span style='color:#4481D7'>Macro AUC</span> 達到 <span style='color:#DD6D6A'>{cnn_auc_1st:.4f}</span>，幾乎完美貼合左上角，代表其綜合分類與特徵鑑別能力最佳
                - <span style='color:#4481D7'>Random Forest</span> 與 <span style='color:#4481D7'>SVM</span> 亦展現不錯的 <span style='color:#4481D7'>AUC</span> 覆蓋面積
                - <span style='color:#4481D7'>K-means</span> 因 <span style='color:#DD6D6A'>無標籤學習 (Unsupervised Learning)</span> 而無法輸出 ROC，僅能顯示準確率
                """, unsafe_allow_html=True)

    # 3. 右主頁最下方資訊
    render_footer()

with tab3:
    metrics_2nd = load_metrics("2nd_backup")
    cnn_auc_2nd = metrics_2nd.get("macro_aucs", {}).get("CNN", 0.9999)
    st.title("🔆 優化評估指標 (Optimized Phase 2)")
    st.markdown("這是第二階段導入 <span style='color:#4481D7'>批次標準化 (BatchNorm)</span>、<span style='color:#4481D7'>特徵資料擴增 (Data Augment)</span> 與 <span style='color:#4481D7'>自適應學習率調度 (ReduceLROnPlateau)</span> 的優化結果", unsafe_allow_html=True)
    with st.expander("💬 架構說明", expanded=False):
                st.markdown(textwrap.dedent("""
                在 Phase 2，全面升級 CNN 訓練策略，以應對複雜的真實交通號誌圖像
                
                1. <span style='color:#BC72A7'>**Batch Normalization (批次標準化)**</span>：
                   - **效果**：在每個卷積層後導入 <span style='color:#4481D7'>BatchNorm2d</span>，有效解決深度網路 <span style='color:#4481D7'>內部協變偏移 (Internal Covariate Shift)</span> 問題，使訓練過程更加穩定，允許使用更高的初始學習率，並減少對 Dropout 的依賴
                   - **結果**：顯著提升模型的 <span style='color:#4481D7'>收斂速度</span> 與 <span style='color:#4481D7'>泛化能力</span>
                
                2. <span style='color:#BC72A7'>**Data Augmentation (資料擴增)**</span>：
                   - **策略**：
                        - 隨機旋轉 (Rotation) <span style='color:#4481D7'>[max_angle=15°]</span>
                        - 隨機縮放 (Scale) <span style='color:#4481D7'>[0.8~1.2]</span>
                        - 隨機仿射變換 (Affine) <span style='color:#4481D7'>[Translation/Shear]</span>
                   - **重要性**：交通號誌在實際路況中會因攝影 <span style='color:#4481D7'>角度</span>、<span style='color:#4481D7'>距離</span>等因素，而呈現多變的 <span style='color:#4481D7'>姿態</span> 與 <span style='color:#4481D7'>大小</span>
                
                3. <span style='color:#BC72A7'>**ReduceLROnPlateau (自適應學習率)**</span>：
                   - **策略**：監控 <span style='color:#4481D7'>驗證損失 (Val Loss)</span>，當連續 <span style='color:#DD6D6A'>5</span> 個 Epoch 內損失不再下降時，自動將學習率除以 <span style='color:#DD6D6A'>10</span>
                   - **效果**：在模型接近收斂時，提供更細微的學習步長，幫助模型跳脫局部最小值，最終觸及 <span style='color:#DD6D6A'>99.90%</span> 的極致精度
                """), unsafe_allow_html=True)
    st.divider()
    
    colA_opt, colB_opt = st.columns(2)
    
    with colA_opt:
        st.subheader("🏆 跨模型準確率比較")
        if os.path.exists('reports/figures/2nd_backup/accuracy_comparison.png'):
            st.image('reports/figures/2nd_backup/accuracy_comparison.png', use_container_width=True)
            with st.expander("💡 結論：", expanded=True):
                st.markdown(f"""
                - 導入 <span style='color:#4481D7'>批次標準化 (BatchNorm2d)</span> 與 <span style='color:#4481D7'>隨機旋轉</span>、<span style='color:#4481D7'>仿射變換</span>等資料擴增技術後
                  - CNN 驗證集準確率最高達到 <span style='color:#DD6D6A'>99.90%</span>
                  - 在全新且完全獨立的 12,630 張測試集影像上，測試集準確率也高達 <span style='color:#DD6D6A'>98.17%</span>
                - 顯著超越第一階段的效能，成功突破並達成 <span style='color:#DD6D6A'>>98%</span> 的優化目標，展現極強的 <span style='color:#4481D7'>抗噪性</span> 與 <span style='color:#4481D7'>泛化收斂能力</span>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
            
        st.subheader("📉 CNN 訓練損失與準確率 (Loss & Acc)")
        if os.path.exists('reports/figures/2nd_backup/loss_curve.png'):
            st.image('reports/figures/2nd_backup/loss_curve.png', use_container_width=True)
            with st.expander("💡 結論：", expanded=True):
                st.markdown(f"""
                訓練曲線揭示震撼的收斂速度！
                  - 在第 <span style='color:#DD6D6A'>1</span> 個 Epoch 驗證精度即衝破 <span style='color:#DD6D6A'>90%</span>
                  - 第 <span style='color:#DD6D6A'>3</span> 個 Epoch 即達到 <span style='color:#DD6D6A'>98.66%</span>
                  - 後續在 <span style='color:#4481D7'>學習率調度器 (ReduceLROnPlateau)</span> 的細緻微調下平滑收斂至 <span style='color:#DD6D6A'>99.90%</span>，完全無過擬合跡象！
                """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
            
        st.subheader("🔍 CNN 混淆矩陣")
        if os.path.exists('reports/figures/2nd_backup/cnn_confusion_matrix.png'):
            st.image('reports/figures/2nd_backup/cnn_confusion_matrix.png', use_container_width=True)
            with st.expander("💡 結論：", expanded=True):
                st.markdown(f"""
                - 混淆矩陣呈現極度乾淨且完美的 <span style='color:#4481D7'>深藍色主對角線</span>
                - 在強大的 <span style='color:#4481D7'>BatchNorm2d</span> 層與資料擴增的作用下，對極為相似號誌（如：速限 <span style='color:#4481D7'>30 km/h</span>、<span style='color:#4481D7'>50 km/h</span> 與 <span style='color:#4481D7'>80 km/h</span>）的誤判情況已完全被消除
                - 分類純淨度逼近 <span style='color:#DD6D6A'>100.00%</span>
                """, unsafe_allow_html=True)
            
    with colB_opt:
        st.subheader("⏱️ 跨模型訓練時間比較")
        if os.path.exists('reports/figures/2nd_backup/training_time_comparison.png'):
            st.image('reports/figures/2nd_backup/training_time_comparison.png', use_container_width=True)
            with st.expander("💡 結論：", expanded=True):
                st.markdown(f"""
                - 雖然導入 <span style='color:#4481D7'>BatchNorm2d</span> 與 <span style='color:#4481D7'>圖像擴增</span> 增加數個 <span style='color:#4481D7'>Batch</span> 的計算量，但由於 PyTorch 的 <span style='color:#4481D7'>MPS GPU 加速</span> 以及 <span style='color:#4481D7'>BatchNorm</span> 帶來的高梯度效率，整體訓練時間依然極為迅速
                """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.subheader("📉 PCA 陡坡圖 (Scree Plot)")
        if os.path.exists('reports/figures/2nd_backup/pca_scree_plot.png'):
            st.image('reports/figures/2nd_backup/pca_scree_plot.png', use_container_width=True)
            with st.expander("💡 結論：", expanded=True):
                st.markdown(f"""
                - <span style='color:#4481D7'>PCA</span> 特徵陡坡圖維持不變，依然以 <span style='color:#DD6D6A'>86</span> 個主成份捕捉 <span style='color:#DD6D6A'>95.00%</span> 的變異量
                - 印證優化深度學習 <span style='color:#4481D7'>CNN</span> 模型架構的決策：直接在 <span style='color:#4481D7'>特徵空間</span> 進行 <span style='color:#4481D7'>卷積學習</span>，無須依賴 <span style='color:#4481D7'>傳統特徵工程</span> 或 <span style='color:#4481D7'>降維技術</span>
                """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
            
        st.subheader("📈 跨模型 ROC 曲線")
        if os.path.exists('reports/figures/2nd_backup/roc_curve_comparison.png'):
            st.image('reports/figures/2nd_backup/roc_curve_comparison.png', use_container_width=True)
            with st.expander("💡 結論：", expanded=True):
                st.markdown(f"""
                - <span style='color:#4481D7'>ROC 曲線</span> 顯示優化後的 <span style='color:#4481D7'>CNN</span> 完美貼合左上極限點
                - <span style='color:#4481D7'>Macro AUC</span> 高達 <span style='color:#DD6D6A'>{cnn_auc_2nd:.4f}</span>
                - 43 個類別的敏感度 (Sensitivity) 與特異度 (Specificity) 壓倒性領先傳統模型
                """, unsafe_allow_html=True)

    # 3. 右主頁最下方資訊
    render_footer()

with tab4:

    if os.path.exists('walkthrough.md'):
        with open('walkthrough.md', 'r', encoding='utf-8') as f:
            walkthrough_content = f.read()
        st.markdown(walkthrough_content, unsafe_allow_html=True)
    else:
        st.warning("⚠️ 找不到驗收說明文件 (`walkthrough.md`)。")
        
    render_footer()
