## MODIFIED Requirements

### Requirement: PyTorch Dataset and DataLoader
系統 MUST 實作 PyTorch 的 Dataset 類別，用以讀取 GTSRB 影像並套用前處理。系統 MUST 針對訓練集與驗證/測試集分離前處理管道 (Transforms)：訓練集前處理 MUST 套用資料擴增 (Data Augmentation) 技術（包含隨機旋轉、亮度/對比度隨機調整及隨機仿射變換），而驗證與測試集 MUST 僅套用標準縮放 (32x32) 與歸一化前處理，最終透過 DataLoader 進行批次供檔。

#### Scenario: Fetching a batch for training
- **當 (WHEN)** 訓練 DataLoader 進行迭代時
- **則 (THEN)** 會產出一個套用了資料擴增（旋轉、色彩抖動、仿射）的影像張量批次以及對應的真實分類標籤，以提升模型泛化能力。

#### Scenario: Fetching a batch for validation or testing
- **當 (WHEN)** 驗證或測試 DataLoader 進行迭代時
- **則 (THEN)** 會產出一個僅經過乾淨縮放 (32x32) 與標準化的影像張量批次，用以公正評估模型精度。
