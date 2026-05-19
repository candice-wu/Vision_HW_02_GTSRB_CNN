# 目的 (Purpose)
本規格定義了 GTSRB 原始資料集的下載、解析以及 PyTorch 訓練資料管線的建置流程，確保後續模型能接收正確格式與維度的訓練影像。

## 需求 (Requirements)

### 需求：Kaggle 資料集下載與解析 (Kaggle Dataset Downloading and Parsing)
系統「必須」具備從 Kaggle 下載 GTSRB 資料集並解析其目錄結構 (train/test 目錄、CSV 標籤) 的能力。

#### 情境：下載原始資料 (Downloading raw data)
- **當 (WHEN)** 資料集初始化腳本執行時
- **則 (THEN)** 系統會驗證本機端是否已存在資料集，若不存在，則自動從 Kaggle 下載並解壓縮。

### 需求：PyTorch Dataset 與 DataLoader (PyTorch Dataset and DataLoader)
系統「必須」實作 PyTorch 的 Dataset 類別，用以讀取 GTSRB 影像並進行必要的前處理轉換 (如縮放至 32x32 像素)，最終透過 DataLoader 進行批次供檔。

#### 情境：提取訓練批次 (Fetching a batch for training)
- **當 (WHEN)** DataLoader 進行迭代 (iterate) 時
- **則 (THEN)** 會產出一個經過縮放的影像張量批次 (tensors) 以及對應的真實分類標籤。
