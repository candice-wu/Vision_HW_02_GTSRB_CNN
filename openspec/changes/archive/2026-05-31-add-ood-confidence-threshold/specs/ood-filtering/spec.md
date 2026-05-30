## ADDED Requirements

### Requirement: 信賴度安全門檻過濾 (Confidence Threshold OOD Filtering)
系統 MUST 在推理階段實作一個 0.75 信賴度安全門檻過濾模組，用於動態檢查所有支援機率輸出的模型的最高置信機率。當最高置信機率小於 0.75 時，系統 MUST 將預測狀態標記為「非已知號誌」(OOD)。

#### Scenario: CNN置信機率高於門檻放行
- **當 (WHEN)** 輸入一張交通號誌影像，且 CNN 預測的最大 Softmax 機率為 0.92 (>= 0.75) 時
- **則 (THEN)** 系統 MUST 允許正常輸出該預測類別。

#### Scenario: CNN置信機率低於門檻攔截
- **當 (WHEN)** 輸入一張非分佈內或干擾影像，且 CNN 預測的最大 Softmax 機率為 0.61 (< 0.75) 時
- **則 (THEN)** 系統 MUST 將其判定為 OOD，並拒絕輸出正常類別。

#### Scenario: 傳統ML模型機率提取
- **當 (WHEN)** 執行 Random Forest、KNN 或 AdaBoost 等傳統 ML 模型推論時
- **則 (THEN)** 系統 MUST 調用其 `predict_proba()` 取得預測機率，並比照 0.75 安全門檻進行攔截判定。

#### Scenario: 支持向量機決策邊界處理
- **當 (WHEN)** 執行 SVM 模型推論且未啟用原生機率校準時
- **則 (THEN)** 系統 MUST 以適當的決策邊界距離 (decision_function) 評估其信賴度，並於低於安全限制時判定為未知號誌。
