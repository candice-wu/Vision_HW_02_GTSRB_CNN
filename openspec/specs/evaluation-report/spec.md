# 目的 (Purpose)
本規格定義了模型效能的評估機制與視覺化產出邏輯。包含紀錄 CNN 的訓練歷史 (Epoch Loss/Accuracy)、追蹤所有模型的訓練時間，並彙整各模型的預測結果以繪製跨模型比較的圖表 (如準確率比較、ROC 曲線、混淆矩陣等)。

## 需求 (Requirements)

### 需求：模型準確率與損失指標 (Model Accuracy and Loss Metrics)
系統「必須」能在訓練與評估階段，計算並記錄 CNN 模型的準確率與損失值。

#### 情境：CNN 評估 (CNN Evaluation)
- **當 (WHEN)** 在測試集上執行評估迴圈時
- **則 (THEN)** 系統會回報整體的預測準確率與分類交叉熵損失 (categorical loss)，並繪製學習曲線。

### 需求：跨模型比較指標 (Cross-Model Comparison Metrics)
系統「必須」彙整 CNN 與所有機器學習模型的測試集準確率與訓練時間，以產生視覺化的比較圖表。

#### 情境：產生比較報告 (Generating Comparison Report)
- **當 (WHEN)** 所有模型完成評估後
- **則 (THEN)** 系統會依序 (CNN, NN, SVM, RF, KNN, AdaBoost, K-means) 產出長條圖與相關圖表，比較各模型的最終準確率表現與訓練耗時，並為 K-means 實作多數決映射 (Majority Voting Mapping) 以加入比較行列。
