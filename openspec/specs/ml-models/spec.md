# 目的 (Purpose)
本規格定義了傳統機器學習管線的建置。包含將影像特徵轉換為一維向量、實施主成分分析 (PCA) 進行降維，並將降維後的特徵送入多個 scikit-learn 機器學習模型中進行訓練，作為深度學習 CNN 的基準對照組。

## 需求 (Requirements)

### 需求：傳統機器學習分類器 (Traditional ML Classifiers)
系統「必須」使用 `scikit-learn` 實作多個傳統機器學習與類神經網路分類器 (NN, SVM, Random Forest, K-means, KNN, AdaBoost)，以進行 GTSRB 影像分類任務。

#### 情境：訓練 ML 模型 (Training ML models)
- **當 (WHEN)** 將訓練資料提供給機器學習管線時
- **則 (THEN)** 所有指定的 ML 模型皆會進行數據擬合 (fit)，並產生可供呼叫的預測模型。

### 需求：PCA 降維處理 (PCA Dimensionality Reduction)
系統「必須」在使用傳統 ML 模型訓練前，先透過主成分分析 (PCA) 來降低已攤平 (flattened) 影像向量的維度。

#### 情境：套用 PCA (Applying PCA)
- **當 (WHEN)** 將大小為 3072 (32x32x3) 的攤平影像傳入 PCA 模組時
- **則 (THEN)** 該模組會輸出保留 95% 原始變異數的降維特徵向量，供後續機器學習模型使用，藉此大幅減少訓練時間並防止維度災難。
