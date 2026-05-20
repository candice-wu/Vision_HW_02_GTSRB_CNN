# 目的 (Purpose)
本規格定義了 GTSRB 原始資料集的下載、解析以及 PyTorch 訓練資料管線的建置流程，確保後續模型能接收正確格式與維度的訓練影像。

## 需求 (Requirements)

### 需求：Kaggle 資料集下載與解析 (Kaggle Dataset Downloading and Parsing)
系統「必須」具備從 Kaggle 下載 GTSRB 資料集並解析其目錄結構 (train/test 目錄、CSV 標籤) 的能力。

#### 情境：下載原始資料 (Downloading raw data)
- **當 (WHEN)** 資料集初始化腳本執行時
- **則 (THEN)** 系統會驗證本機端是否已存在資料集，若不存在，則自動從 Kaggle 下載並解壓縮。

### 需求：PyTorch Dataset 與 DataLoader (PyTorch Dataset and DataLoader)
系統「必須」實作 PyTorch 的 Dataset 類別，用以讀取 GTSRB 影像並套用前處理。系統「必須」針對訓練集與驗證/測試集分離前處理管道 (Transforms)：訓練集前處理「必須」套用資料擴增 (Data Augmentation) 技術（包含隨機旋轉、亮度/對比度隨機調整及隨機仿射變換），而驗證與測試集「必須」僅套用標準縮放 (32x32) 與歸一化前處理，最終透過 DataLoader 進行批次供檔。

#### 情境：提取訓練批次 (Fetching a batch for training)
- **當 (WHEN)** 訓練 DataLoader 進行迭代時
- **則 (THEN)** 會產出一個套用了資料擴增（旋轉、色彩抖動、仿射）的影像張量批次以及對應的真實分類標籤，以提升模型泛化能力。

#### 情境：提取驗證或測試批次 (Fetching a batch for validation or testing)
- **當 (WHEN)** 驗證或測試 DataLoader 進行迭代時
- **則 (THEN)** 會產出一個僅經過乾淨縮放 (32x32) 與標準化的影像張量批次，用以公正評估模型精度。
