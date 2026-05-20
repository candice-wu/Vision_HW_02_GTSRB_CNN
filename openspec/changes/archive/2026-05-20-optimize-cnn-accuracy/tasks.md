## 1. 資料前處理與 DataLoader 優化

- [x] 1.1 修改 `src/dataset.py` 分離訓練與驗證/測試資料集的 `transform` 前處理管道。
- [x] 1.2 在訓練資料集管道中，獨佔套用隨機旋轉 (`RandomRotation(15)`)、隨機色彩對比抖動 (`ColorJitter`) 以及隨機仿射變換 (`RandomAffine`) 等資料擴增技術。
- [x] 1.3 確保驗證與測試資料集的前處理管道僅包含標準縮放 (`Resize((32, 32))`) 與歸一化 (`Normalize`)。

## 2. CNN 模型架構升級

- [x] 2.1 修改 `src/model.py` 的 `GTSRB_CNN` 類別，在三個卷積層後各宣告一個批次標準化層 (`nn.BatchNorm2d`)。
- [x] 2.2 更新 `GTSRB_CNN` 的 `forward` 前向傳播函數，使特徵圖在通過卷積層後先執行批次標準化，再進行激活與最大池化。

## 3. 訓練腳本優化與 LR 調度器整合

- [x] 3.1 修改 `src/train_cnn.py` 將預設最大訓練輪數 `epochs` 調整至 25 ~ 30 輪。
- [x] 3.2 在 Adam 優化器宣告後，正式引進 PyTorch 動態學習率衰減調度器 `optim.lr_scheduler.ReduceLROnPlateau`，設定 `mode='max'`, `factor=0.1`, `patience=3`。
- [x] 3.3 在 Epoch 的 validation 評估迴圈完成並計算出 `val_acc` 後，加入 `scheduler.step(val_acc)` 邏輯以進行自適應學習率退火。
- [x] 3.4 修改 `src/evaluate.py`，將評估圖表之輸出存檔路徑 `save_dir` 改寫為 `'reports/figures/2nd_backup'`，以進行第二階段結果備份隔離。
- [x] 3.5 修改 `gtsrb.py` 的前端頁籤結構，調整為三個分頁：將原 `tab2` 的讀取路徑改為 `'reports/figures/1st_backup/'`；新增 `tab3` 並使其讀取 `'reports/figures/2nd_backup/'` 以展示 BatchNorm + Aug 性能優化指標。

## 4. 模型重訓、評估與效能驗證

- [x] 4.1 使用專案虛擬環境 `/Users/candicewu/virtualenvs/Vision_HW_02` 執行 `python src/train_cnn.py` 進行重訓，並儲存最佳權重。
- [x] 4.2 執行 `python src/evaluate.py` 重新計算各項機器學習與深度學習模型之效能指標，並輸出全新的比較圖表。
- [x] 4.3 驗證優化後的 CNN+STN 模型在驗證集與測試集上的 Accuracy 是否成功超越 98% ~ 99% 的頂尖目標，並確認 Streamlit web app 的正常運作與載入。
