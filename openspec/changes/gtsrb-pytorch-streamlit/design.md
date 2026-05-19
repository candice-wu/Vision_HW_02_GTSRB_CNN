## Context

為完成「交通號誌辨識」課程作業，需參考既有的 `GTSRB_Keras_STN` 架構，實作並比較不同技術方案。原本使用 TensorFlow/Keras 框架，因環境相容性問題，現改為使用 PyTorch。作業的核心要求包含建置深度卷積神經網路（CNN）搭配空間轉換網路（STN），以及建立多種傳統機器學習模型作為基準比較，並透過 Streamlit 開發即時互動式介面。

## Goals / Non-Goals

**Goals:**
- 以 PyTorch 實作帶有 STN (Spatial Transformer Network) 的 CNN 模型，並重現或超越參考文獻的精確度。
- 使用 `scikit-learn` 實作多個分類模型 (SVM, Random Forest, K-means, KNN, AdaBoost) 進行跨模型比較。
- 引入 PCA (Principal Component Analysis) 作為降維工具，輔助傳統機器學習模型從高維度影像特徵中提取關鍵資訊。
- 透過 Streamlit 開發互動介面，允許隨機抽取測試集影像並即時呼叫各模型進行推理 (Inference)。
- 自動化產出 Loss / Accuracy 圖表，幫助撰寫最終作業報告。

**Non-Goals:**
- 開發可用於實際生產環境 (Production) 或邊緣裝置 (Edge devices) 的超輕量模型。
- 設計過於複雜的資料擴增管線 (Data Augmentation)，主要驗證 STN 學習空間不變性的能力。

## Decisions

1. **框架選擇: PyTorch vs TensorFlow**
   - **Rationale**: 由於使用者的電腦與 TensorFlow/Keras 存在相容性問題，選擇在學術界廣泛使用且除錯方便的 PyTorch。STN 在 PyTorch 中可透過 `torch.nn.functional.affine_grid` 與 `grid_sample` 原生支援。
2. **機器學習比較工具: scikit-learn**
   - **Rationale**: `scikit-learn` 具備完整的 SVM、Random Forest、K-means、KNN、AdaBoost 與 PCA 實作。針對影像（32x32x3 = 3072 維），直接放入傳統分類器可能會遭遇「維度災難」(Curse of Dimensionality)，因此先透過 PCA 將特徵壓縮 (例如保留 95% 變異量) 再放入模型，可大幅增進訓練效率。
3. **互動介面: Streamlit**
   - **Rationale**: 利用最少的程式碼建立資料科學應用的首選。能迅速整合 matplotlib/Plotly 圖表以及 PyTorch 模型預測。

## Risks / Trade-offs

- **Risk: 傳統機器學習模型對於高維度影像的訓練時間過長。**
  - **Mitigation**: 使用 PCA 將影像特徵降維，不僅能加速訓練，還能減輕過度擬合 (Overfitting) 的風險。
- **Risk: CNN 模型訓練可能較耗時（若無強大 GPU）。**
  - **Mitigation**: PyTorch 可使用 MPS (Mac 的 Metal Performance Shaders) 加速。此外，使用預先縮放好 (32x32) 的資料集可減少即時前處理的負擔。
