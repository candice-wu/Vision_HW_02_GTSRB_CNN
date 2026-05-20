## Context

當前實作的交通號誌辨識卷積網路架構 (`GTSRB_CNN`) 包含一個空間轉換網路 (STN) 以及三個標準卷積層 (Conv2d)。在第一階段的實驗中，模型在不包含任何標準化機制與資料擴增的情況下完成了 15 個 Epochs 的訓練，學習率固定在 `0.001`，其驗證集與測試集精度大約落在 90% ~ 95% 之間，未能穩定突破 98% ~ 99% 的水準。

為了解決收斂效率低、泛化能力弱的問題，需要引入批次標準化、特徵擴增與自適應學習率退火策略。

## Goals / Non-Goals

**Goals:**
- **引入 Batch Normalization**：在每個卷積層的輸出後加入二維批次標準化層 (`nn.BatchNorm2d`)，穩定特徵分佈。
- **實作 Transform 分離與資料擴增**：只對訓練集 Dataloader 套用隨機旋轉 (±15°)、隨機色彩抖動 (亮度/對比度)、與隨機仿射平移，而驗證集/測試集保持純淨。
- **導入 ReduceLROnPlateau 學習率調度器**：動態監控驗證準確率 (`val_acc`)，在精度陷入停滯時主動退火（調降學習率），細緻優化權重。
- **將最大訓練輪數擴展至 25~30 Epochs**，以配合資料擴增後的特徵多樣性，確保模型最終正確率穩定突破 98% 或 99%。
- **實作前後版本效能圖表直觀對比 UI**：透過備份目錄隔離第一階段原圖 (`1st_backup/`) 與本次優化新圖 (`2nd_backup/`)，並在 Streamlit 中以雙分頁模式 (`tab2` 與 `tab3`) 同時呈現，以利直觀對比優化成果。

**Non-Goals:**
- 不替換更深的經典骨幹網路（如 ResNet, EfficientNet 等預訓練大型網路），維持原生自製 CNN 與 STN 網路主體架構。
- 不修改傳統機器學習模型（SVM, Random Forest, KNN 等）的特徵降維與建模邏輯。

## Decisions

### 決策 1：在 CNN 卷積層後引進 `nn.BatchNorm2d`
- **方案選擇**：在每一個 `nn.Conv2d` 後緊接 `nn.BatchNorm2d`，再送入 `F.relu` 與 `F.max_pool2d`。
- **考量點**：批次標準化能有效解決梯度消失/爆炸問題，並顯著平滑 Loss 空間。
- **備選替代方案**：`nn.LayerNorm` 或不加標準化。相較之下，`BatchNorm2d` 是卷積影像特徵最標準且實證效果最佳的對齊工具。

### 決策 2：在 Dataloader 管道中分離 Train/Val 數據前處理
- **方案選擇**：修改 `src/dataset.py` 中的 `get_dataloaders`。將原本單一的 `transform` 拆分為 `train_transform` 與 `val_transform`：
  - `train_transform` 添加 `RandomRotation(15)`、`ColorJitter(brightness=0.2, contrast=0.2)` 及 `RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))`。
  - `val_transform` 僅保留 `Resize((32, 32))` 與 `Normalize`。
- **考量點**：資料擴增能增加訓練樣本的多樣性，強迫模型學習不受光照、形變與角度干擾的穩健特徵。驗證集保持不變，可確保評估指標的無偏差性。
- **備選替代方案**：使用 `Albumentations` 等第三方擴增庫。雖然功能強大，但 PyTorch `torchvision.transforms` 對於此類簡單圖像變換已極為高效，能避免引入新的第三方套件依賴。

### 決策 3：使用 `ReduceLROnPlateau` 作為動態學習率衰減器
- **方案選擇**：在訓練迴圈 `train_cnn.py` 中引入 `optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)`。在每個 Epoch 的驗證完畢後呼叫 `scheduler.step(val_acc)`。
- **考量點**：相較於固定輪數衰減的 `StepLR`，`ReduceLROnPlateau` 具備「適應性」。它能敏感地察覺驗證集精度是否不再爬升，並在最適當時機進行「學習率退火」，幫助模型在梯度極小值區域進行極為精細的權重收斂。
- **備選替代方案**：`CosineAnnealingLR` (餘弦退火) 或固定不變。固定學習率在後期容易震盪，而餘弦退火的週期難以與提前收斂進行動態調控。

### 決策 4：四分頁效能對比與驗收前端設計 (Four-Tab UI Layout)
- **方案選擇**：
  1. 隔離兩階段數據圖表：
     * 初始階段圖表置於 `reports/figures/1st_backup/`。
     * 本次優化階段圖表置於 `reports/figures/2nd_backup/`，藉由修改 `src/evaluate.py` 中的 `save_dir = 'reports/figures/2nd_backup'` 實現自動保存。
  2. Streamlit 前端變更：
     * 將原 `tab2 = st.tabs(["🚥 GTSRB 交通號誌辨識", "📊 Evaluation Metrics"])` 改為四個分頁：`tab1, tab2, tab3, tab4 = st.tabs(["🚥 GTSRB 交通號誌辨識", "📊 初始評估指標 (Original)", "📈 優化評估指標 (Optimized)", "📄 系統優化與驗收說明"])`。
     * `tab2` 中的所有 `os.path.exists` 及 `st.image` 尋找路徑皆加上前綴 `1st_backup/`。
     * `tab3` 複製並微調 `tab2` 的佈局結構，但專門讀取並呈現來自 `2nd_backup/` 下的圖表，以呈現高達 99.90% 驗證集精度、98.17% 測試集精度與極速收斂的 Loss 曲線。
     * 新增 `tab4` 用以在 Streamlit 介面內直接讀取並動態渲染專案根目錄下的 `walkthrough.md` 文件，使系統驗收成果更加視覺化且方便閱讀。

## Risks / Trade-offs

- **[Risk 1] 資料擴增強度過大導致模型欠擬合 (Underfitting)**
  - **緩解措施**：限制旋轉幅度在合理範圍 (±15度)，色彩調整限制在較輕微的 0.2，平移在 (0.1, 0.1) 以內，避免過度破壞圖像的標籤語意（如箭頭方向）。
- **[Risk 2] 訓練 Epochs 增加與加入標準化層導致訓練時間微幅上升**
  - **緩解措施**：雖然單一 Epoch 運算量微增，但 `BatchNorm2d` 帶來了極高的梯度效率。通常模型在 10~15 Epoch 內就能達到先前 15 Epoch 都達不到的超高精準度，後續的 Epochs 僅是用於微調收斂，整體訓練耗時依然完全可控。
