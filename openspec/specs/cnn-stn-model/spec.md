# 目的 (Purpose)
本規格定義了卷積神經網路 (CNN) 與空間轉換網路 (STN) 的架構與預期行為。STN 負責處理交通號誌影像的空間變換 (如平移、縮放)，以增強 CNN 特徵擷取的魯棒性。

## 需求 (Requirements)

### 需求：CNN 架構定義 (CNN Architecture Definition)
系統「必須」使用 PyTorch 定義一個卷積神經網路 (CNN)，用於分類 43 種 GTSRB 交通號誌類別。

#### 情境：CNN 的正向傳播 (Forward pass of CNN)
- **當 (WHEN)** 輸入形狀為 (N, 3, 32, 32) 的影像批次進入 CNN 時
- **則 (THEN)** 模型輸出形狀為 (N, 43) 的對數機率 (logits)。

### 需求：空間轉換網路模組 (Spatial Transformer Network Module)
系統「必須」包含一個 STN 模組（使用 `affine_grid` 與 `grid_sample`），作為 CNN 的第一層組件，允許對影像進行空間變換。

#### 情境：STN 轉換 (STN transformation)
- **當 (WHEN)** 影像通過 STN 時
- **則 (THEN)** 網路能學習並輸出經過空間校正 (對齊/裁切) 的影像，再傳遞給後續的卷積層。
