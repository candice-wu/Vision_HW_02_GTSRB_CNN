# 更新日誌 (Changelog)

本專案的所有重要變更皆會記錄於此檔案中。

格式基於 [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) 規範，
且本專案遵循 [語意化版本控制 (Semantic Versioning)](https://semver.org/spec/v2.0.0.html)。

## [1.1.1] - 2026-05-20 (Phase 1 最終版)

### 新增 (Added)
- **類神經網路 (NN)**：在機器學習管線中加入了 `MLPClassifier`，作為基礎神經網路的對照基準。
- **K-means 群集**：在評估管線中加入了非監督式 K-means 模型，並實作了「多數決映射 (Majority Voting Mapping)」邏輯，使其準確率 (20.61%) 能與其他模型進行比較。
- **圖片上傳介面 (Image Uploader UI)**：在 Streamlit 應用程式 (`gtsrb.py`) 中啟用互動式圖片推論功能，支援多模型同時預測。
- **訓練歷史追蹤 (History Tracking)**：在 `train_cnn.py` 中新增 Epoch 歷史追蹤 (Train/Val Loss & Acc)，並將結果儲存至 `reports/training_history.csv`。
- **訓練時間追蹤 (Training Time Tracking)**：嵌入執行計時器以記錄 CNN 與所有機器學習模型的訓練時間，並儲存至 `reports/training_times.json`。

### 變更 (Changed)
- **介面強化 (UI Enhancements)**：翻新 Streamlit UI 介面設計（左側邊欄顯示 metadata，主頁面負責分類與評估指標展示）。
- **視覺化顏色與字體 (Visualization)**：將評估圖表的顏色遷移至 Seaborn 的 `viridis` 與 `magma` 色盤，放大數字字體（保留 2 位小數），並修復所有警告訊息。
- **模型顯示順序 (Model Display Order)**：重新排序評估圖表的顯示邏輯，嚴格遵循順序：CNN ➔ NN ➔ SVM ➔ Random Forest ➔ KNN ➔ AdaBoost ➔ K-means。

### 修復 (Fixed)
- 修復了 `src/evaluate.py` 繪圖時產生的 Seaborn `FutureWarning` 警告（透過同時套用 `hue` 與 `palette` 參數解決）。

## [1.0.0] - 2026-05-19

### 新增 (Added)
- 實作結合 Spatial Transformer Networks (STN) 的基準 CNN 架構。
- 實作特徵擷取與標準機器學習管線 (SVM, RF, KNN, AdaBoost)。
- 為機器學習模型實作 PCA 降維處理 (保留 95% 變異數)。
- 建立基礎 Streamlit 應用程式，用以展示本機測試集圖片抽樣與評估圖表。
