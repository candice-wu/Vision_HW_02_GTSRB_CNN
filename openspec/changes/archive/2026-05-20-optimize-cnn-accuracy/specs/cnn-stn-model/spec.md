## MODIFIED Requirements

### Requirement: CNN Architecture Definition
系統 MUST 使用 PyTorch 定義一個卷積神經網路 (CNN)，用於分類 43 種 GTSRB 交通號誌類別。為了確保在多分類問題上的穩定收斂性與高維特徵表達能力，每個卷積層 (Conv2d) 後 MUST 緊接一個批次標準化層 (BatchNorm2d)。

#### Scenario: Forward pass of CNN
- **當 (WHEN)** 輸入形狀為 (N, 3, 32, 32) 的影像批次進入 CNN 時
- **則 (THEN)** 模型在經過空間變換與包含批次標準化的卷積計算後，輸出形狀為 (N, 43) 的對數機率 (logits)，且在訓練時每個 Epoch 收斂表現應顯著優於無標準化之版本。
