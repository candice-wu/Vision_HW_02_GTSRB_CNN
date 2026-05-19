## Why

本專案旨在完成「交通號誌辨識」的系統實作與作業報告。我們需要建立一個具備 CNN 深度學習（含 STN 空間轉換網路）與傳統機器學習（SVM, Random Forest, K-means, KNN, AdaBoost）的跨模型比較系統，並使用 PCA 作降維的一個工具來輔助機器學習模型。為了讓展示更有趣、互動性更高，我們將透過 Streamlit 實作一個前端介面，讓使用者能即時抽樣預測。

## What Changes

- 使用 PyTorch 框架取代原有的 TensorFlow/Keras，以避免電腦規格的依賴衝突。
- 引入 Kaggle 的 GTSRB 原始資料集，並建置 PyTorch Dataset 與 DataLoader 以進行訓練與驗證。
- 實作帶有 Spatial Transformer Network (STN) 的卷積神經網路 (CNN)。
- 實作多種傳統機器學習分類模型 (NN baseline, SVM, Random Forest, K-means, KNN, AdaBoost)，並搭配 PCA 降維工具，用以進行基準比較。
- 建立一個 Streamlit App 作為互動介面，可隨機抽取圖片並顯示各模型的即時判斷結果。
- 產生後續所需的實驗數據 (Accuracy / Loss) 圖表，以滿足作業報告對於系統設計、跨模型分析、技術討論等三大章節的要求。

## Capabilities

### New Capabilities
- `dataset-pipeline`: 下載與處理 Kaggle GTSRB 資料集，提供影像前處理與特徵抽取。
- `cnn-stn-model`: 基於 PyTorch 實作之深度卷積神經網路與空間轉換網路 (STN)。
- `ml-models`: 傳統機器學習分類模型群 (SVM, RF, KNN, K-means, AdaBoost) 及其輔助的 PCA 降維處理管線。
- `streamlit-ui`: 即時互動式前端介面，提供隨機圖片抽取與跨模型預測展示。
- `evaluation-report`: 模型評估、表現分析、與作業報告素材生成模組。

### Modified Capabilities
<!-- No existing capabilities are changing since this is a new project -->

## Impact

- **Dependencies**: 需要安裝 PyTorch (`torch`, `torchvision`), `scikit-learn`, `streamlit`, `pandas`, `matplotlib`, `kaggle` 等套件。
- **Code**: 從零開始建構訓練腳本、推論模組與 Streamlit UI 應用程式。
- **Systems**: 訓練將使用 CPU 或 MPS (Mac GPU) 加速，並且會在本機運行 Streamlit Server 進行即時展示。
