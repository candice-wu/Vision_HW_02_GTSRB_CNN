## MODIFIED Requirements

### Requirement: 即時模型推論 (Real-time Model Inference)
系統 MUST 利用展示在畫面上的圖片，即時呼叫 CNN 模型與傳統 ML 模型進行推論，並將預測結果顯示給使用者。如果模型的最高預測信賴度（機率）低於 0.75 門檻，系統 MUST 攔截該預測結果並改為顯示「這是非德國交通號誌」。

#### Scenario: 檢視預測結果 (Viewing predictions)
- **當 (WHEN)** 圖片載入完成後，且模型的最高預測信賴度不低於 0.75 時
- **則 (THEN)** UI 介面會以清晰的版面結構，展示出 CNN, NN, SVM, RF, KNN, 與 AdaBoost 模型對該圖片的預測類別，以及相對應的信心分數 (Confidence Scores，若該模型支援)。

#### Scenario: 攔截低信賴度非德國號誌 (Intercepting low confidence non-German signs)
- **當 (WHEN)** 圖片載入並完成模型推論後，且該模型的最高預測信賴度低於 0.75 時
- **則 (THEN)** UI 介面 MUST 攔截原預測類別輸出，並改為顯示「這是非德國交通號誌」的警告字樣與攔截提示。
