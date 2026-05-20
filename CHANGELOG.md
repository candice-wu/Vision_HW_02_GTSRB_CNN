# 更新日誌 (Changelog)

本專案的所有重要變更皆會記錄於此檔案中。

格式基於 [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) 規範，
且本專案遵循 [語意化版本控制 (Semantic Versioning)](https://semver.org/spec/v2.0.0.html)。

## [1.2.0] - 2026-05-20 (Phase 2 優化完成版)

### 新增 (Added)
- **全新四頁籤佈局 (Four-Tab UI Layout)**：Streamlit 升級為四分頁架構，新增 `tab4` 負責動態解析與即時渲染 `walkthrough.md`。
- **動態指標數據載入 (Data-Driven Metrics)**：開發 `load_metrics()`，從 JSON 檔動態擷取最新的 Macro ROC AUC 與模型準確率，拒絕 hardcode，確保圖表與文字結論精準同步。
- **學術風格結論收摺器 (Academic Expander Conclusions)**：重構 `tab2` 與 `tab3` 的所有圖表評估結論，統一使用高階 `st.expander` 搭配藍色/紅橘色 HSL 語意高亮顯示。
- **全域統一頁尾模組 (Modular Global Footer)**：封裝 `render_footer()`，統一管理全域版更號與發佈日期，實現極致的 DRY 原則。

### 變更 (Changed)
- **卷積神經網路效能躍升 (CNN Performance)**：CNN 核心層新增 `nn.BatchNorm2d` 並增加 Dropout 控制過擬合；測試精度躍升至不可思議的 **`98.17%`** (驗證集達 **`99.90%`**)，ROC AUC 達到完美的 **`1.000` (0.9999)**。
- **穩健資料擴增管道 (Robust Data Augmentation)**：為訓練集引入旋轉、色彩擾動與隨機仿射等影像擴增，驗證與測試集則保持標準前處理，大幅提升真實場景泛化能力。
- **自適應學習率調度 (Adaptive LR Scheduler)**：整合 `ReduceLROnPlateau` 優化器退火機制，於驗證 Loss 停滯時自動調降學習率，配合 25 Epoch 進行精細收斂。
- **評估資料夾分層隔離 (Isolated Backup Folders)**：將原始第一階段數據完整備份至 `1st_backup/`，優化後的第二階段數據備份至 `2nd_backup/`，並在前端分別調用展示。

### 修復 (Fixed)
- 徹底修正跨模型比較結論中「圖文不一致」的 Bug，使文字描述中的數值完美貼合 JSON 指標。

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
