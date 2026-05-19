# 目的 (Purpose)
本規格定義了基於 Streamlit 框架建立的 Web 應用程式介面，讓使用者可以透過隨機抽樣或自行上傳圖片的方式，與訓練好的分類模型進行互動，達到即時推論與展示系統能力的目的。

## 需求 (Requirements)

### 需求：隨機圖片抽樣與上傳介面 (Random Image Sampling and Upload UI)
系統「必須」提供一個 Streamlit UI 按鈕允許使用者從測試集中隨機抽取交通號誌影像，同時「必須」提供檔案上傳器，允許使用者自行上傳本機圖片。

#### 情境：點擊「隨機抽樣」或「上傳圖片」 (Clicking Random Image or Uploading)
- **當 (WHEN)** 使用者點擊隨機抽樣按鈕，或成功上傳一張交通號誌圖片時
- **則 (THEN)** 該圖片會被顯示在畫面上，且系統會為其準備對應的真實標籤 (若為隨機抽樣)。

### 需求：即時模型推論 (Real-time Model Inference)
系統「必須」利用展示在畫面上的圖片，即時呼叫 CNN 模型與傳統 ML 模型進行推論，並將預測結果顯示給使用者。

#### 情境：檢視預測結果 (Viewing predictions)
- **當 (WHEN)** 圖片載入完成後
- **則 (THEN)** UI 介面會以清晰的版面結構，展示出 CNN, NN, SVM, RF, KNN, 與 AdaBoost 模型對該圖片的預測類別，以及相對應的信心分數 (Confidence Scores，若該模型支援)。
