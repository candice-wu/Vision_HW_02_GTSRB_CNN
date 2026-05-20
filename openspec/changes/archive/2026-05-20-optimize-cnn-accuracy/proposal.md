## Why

當前實作的 CNN + STN 交通號誌辨識模型，雖然整合了空間轉換網路，但由於缺乏**批次標準化 (Batch Normalization)**、**資料擴增 (Data Augmentation)** 以及**動態學習率調度器 (Learning Rate Scheduler)**，導致模型的收斂上限受到限制，驗證與測試集準確率無法突破 98% ~ 99% 的水準。

為了進一步提升交通號誌辨識系統的精度，並展現出符合學術期末發表或實務部署的高水準效能，迫切需要對現有的深度學習管道進行全方位的優化。

## What Changes

本變更旨在對深度學習模型架構與資料處理管道實施以下核心修改：
1. **引進批次標準化 (Batch Normalization)**：在 `GTSRB_CNN` 的每個卷積層 (Conv2d) 之後，ReLU 激活函數與最大池化層之前，正式插入 `nn.BatchNorm2d` 層，以穩定內部協變量偏移 (Internal Covariate Shift)，加快收斂速度。
2. **分離並優化資料前處理 (Transform Separation)**：
   * **訓練集 (Train)**：引進隨機旋轉 (`RandomRotation`)、亮度與對比度調節 (`ColorJitter`) 以及隨機仿射變換 (`RandomAffine`) 作為資料擴增技術，提升模型泛化能力。
   * **驗證集與測試集 (Val/Test)**：僅保持縮放與標準化，不使用任何隨機擴增，確保評估基準的準確性。
3. **加入學習率調度器 (ReduceLROnPlateau)**：在 `train_cnn.py` 的 Adam 優化器後加入 PyTorch 動態學習率衰減調度器，當驗證集準確率在連續數個 Epochs 停滯時自動調降學習率，促使模型細緻微調權重。
4. **拉長訓練 Epochs**：將最大訓練輪數從 15 輪提升至 25~30 輪，以配合資料擴增與學習率衰減的完整訓練週期。
5. **引進模型效能對比機制 (Model Performance Comparison UI)**：
   * 將第一階段 (1st Run) 原生的 6 張評估圖表移置 `reports/figures/1st_backup/` 目錄。
   * 調整 `gtsrb.py` 的 `tab2` (原評估分頁) 從 `1st_backup/` 讀取並呈現初始版本的模型評估指標。
   * 將本次優化階段 (2nd Run) 產生的 6 張高精度模型評估圖表儲存於 `reports/figures/2nd_backup/` 目錄。
   * 在 Streamlit UI 中新增第 3 個分頁 `tab3` (📊 Optimized Metrics (BatchNorm + Aug))，動態讀取 `2nd_backup/` 圖表以進行前後版本效能的直觀可視化對比。

## Capabilities

### New Capabilities
*(無新增之系統核心功能)*

### Modified Capabilities
- `cnn-stn-model`: 於卷積神經網路架構中整合批次標準化層 (`nn.BatchNorm2d`)，藉以提升收斂速度與模型特徵學習上限。
- `dataset-pipeline`: 分離訓練管道與驗證/測試管道之前處理，對訓練集獨佔引入圖像旋轉、顏色擾動及隨機仿射等資料擴增機制。
- `streamlit-ui`: 新增效能對比分頁 `tab3`，展示加入批次標準化與資料擴增優化後的指標，並將 `tab2` 轉為展示原始第一階段版本評估指標。

## Impact

- **`src/model.py`**：`GTSRB_CNN` 模型的類別定義，需新增三個 `nn.BatchNorm2d` 屬性，並更新 `forward` 傳播邏輯。
- **`src/dataset.py`**：`get_dataloaders` 函式，需拆分為兩個不同的 `transforms` 物件（`train_transform` 與 `val_transform`），並分別套用至訓練與驗證 Dataset。
- **`src/train_cnn.py`**：訓練主腳本，需將 `epochs` 上調至 25~30，並在訓練迴圈中加入 `scheduler.step(val_acc)` 的動態學習率調整。
- **`src/evaluate.py`**：評估與圖表產生腳本，需將輸出路徑 `save_dir` 改為 `reports/figures/2nd_backup`。
- **`gtsrb.py`**：Streamlit 前端應用介面，將 `tab2` 的圖表讀取路徑改為 `reports/figures/1st_backup/`，並加入 `tab3` 的宣告以讀取並呈現 `reports/figures/2nd_backup/` 下優化後的圖表。
