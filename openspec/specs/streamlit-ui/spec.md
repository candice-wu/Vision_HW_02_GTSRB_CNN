## Purpose
本規格定義了基於 Streamlit 框架建立的 Web 應用程式介面，讓使用者可以透過隨機抽樣或自行上傳圖片的方式，與訓練好的分類模型進行互動，達到即時推論與展示系統能力的目的。

## Requirements

### 需求：隨機圖片抽樣與上傳介面 (Random Image Sampling and Upload UI)
系統「必須」提供一個 Streamlit UI 按鈕允許使用者從測試集中隨機抽取交通號誌影像，同時「必須」提供檔案上傳器，允許使用者自行上傳本機圖片。

#### 情境：點擊「隨機抽樣」或「上傳圖片」 (Clicking Random Image or Uploading)
- **當 (WHEN)** 使用者點擊隨機抽樣按鈕，或成功上傳一張交通號誌圖片時
- **則 (THEN)** 該圖片會被顯示在畫面上，且系統會為其準備對應的真實標籤 (若為隨機抽樣)。

### Requirement: 即時模型推論 (Real-time Model Inference)
系統 MUST 利用展示在畫面上的圖片，即時呼叫 CNN 模型與傳統 ML 模型進行推論，並將預測結果顯示給使用者。如果模型的最高預測信賴度（機率）低於 0.75 門檻，系統 MUST 攔截該預測結果並改為顯示「這是非德國交通號誌」。

#### Scenario: 檢視預測結果 (Viewing predictions)
- **當 (WHEN)** 圖片載入完成後，且模型的最高預測信賴度不低於 0.75 時
- **則 (THEN)** UI 介面會以清晰的版面結構，展示出 CNN, NN, SVM, RF, KNN, 與 AdaBoost 模型對該圖片的預測類別，以及相對應的信心分數 (Confidence Scores，若該模型支援)。

#### Scenario: 攔截低信賴度非德國號誌 (Intercepting low confidence non-German signs)
- **當 (WHEN)** 圖片載入並完成模型推論後，且該模型的最高預測信賴度低於 0.75 時
- **則 (THEN)** UI 介面 MUST 攔截原預測類別輸出，並改為顯示「這是非德國交通號誌」的警告字樣與攔截提示。

### 需求：四頁籤佈局 (Four-Tab UI Layout)
系統的 `st.tabs` 呼叫「必須」改為四個分頁：
1. `st.tabs[0]` / `tab1`："🚥 GTSRB 交通號誌辨識" (隨機抽樣/即時辨識主頁)
2. `st.tabs[1]` / `tab2`："📊 初始評估指標 (Original)" (第一階段評估結果展示)
3. `st.tabs[2]` / `tab3`："🔆 優化評估指標 (Optimized)" (第二階段優化後評估結果展示)
4. `st.tabs[3]` / `tab4`："⛳ 系統優化與驗收說明" (讀取並呈現專案目錄下 walkthrough.md 的詳細成果驗收)

#### 情境：切換網頁分頁 (Switching tabs in the application)
- **當 (WHEN)** 使用者在網頁上點擊不同的分頁按鈕時
- **則 (THEN)** 系統「必須」流暢切換至對應的頁面，且維持版尾資訊渲染以及模型快取載入。

### 需求：分層圖表資料夾隔離與讀取 (Isolated Metrics Folders Reading)
系統「必須」實作圖表資料夾的隔離與獨立讀取，以在不同的分頁呈現不同優化階段的模型指標。

#### 情境：讀取初始評估圖表 (Reading Original Metrics)
- **當 (WHEN)** 使用者進入 `tab2` 時
- **則 (THEN)** 系統「必須」從 `reports/figures/1st_backup/` 目錄中載入並呈現 accuracy_comparison, loss_curve, cnn_confusion_matrix, training_time_comparison, pca_scree_plot 以及 roc_curve_comparison 六張圖表，並搭配原始結語說明。

#### 情境：讀取優化評估圖表 (Reading Optimized Metrics)
- **當 (WHEN)** 使用者進入 `tab3` 時
- **則 (THEN)** 系統「必須」從 `reports/figures/2nd_backup/` 目錄中載入並呈現相同命名的六張優化後圖表，並搭配針對 BatchNorm 與 Data Augment 優化後的特有結語。
