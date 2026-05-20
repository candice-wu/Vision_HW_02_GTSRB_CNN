# ⛳ GTSRB 系統優化說明 (System Walkthrough)

本記錄為達成 `GTSRB_CNN` 交通號誌辨識準確率突破 `98% ~ 99%` 的優化工作、指標評估，以及 Streamlit 雙分頁效能對比的實作結果

---

## 🚀 達成之優化成果與效能

| 指標種類 | Phase 1 Baseline | Phase 2 Optimized | 結論與說明 |
| :--- | :---: | :---: | :--- |
| **驗證集最高準確率 (Val Accuracy)** | ~95.28% | **99.90%** | 大幅上升，特徵對齊完美 |
| **測試集準確率 (Test Accuracy)** | ~94.75% | **98.17%** | 跨越且達成 >98% 學術指標 |
| **訓練損失 (Train Loss)** | ~0.1561 | **0.0038** | 收斂極致，無過擬合跡象 |
| **核心技術導入** | 無 (純自製 CNN+STN) | BatchNorm2d + 訓練資料擴增 + Plateau 學習率調度 | 保留 STN 架構，純粹優化特徵學習效率 |

---

## 🛠️ 所做變更清單

### 1. 深度學習管道升級
- **資料前處理 (`src/dataset.py`)**：
  - 分離訓練集與驗證/測試集之 transforms
  - 對訓練集獨佔套用 `RandomRotation(15)`、`ColorJitter` 與 `RandomAffine` 等資料擴增技術，提升模型對於光照、角度與平移之穩健度
  - 驗證/測試集僅進行 `Resize((32,32))` 與基於通道之 `Normalize`
- **模型架構優化 (`src/model.py`)**：
  - 於 `GTSRB_CNN` 的三個卷積層後、ReLU 激活與池化層前插入 `nn.BatchNorm2d`，解決梯度不穩定問題
- **自適應學習率退火策略 (`src/train_cnn.py`)**：
  - 將訓練輪數上調至 `25` 輪
  - 整合 `optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)`，依據 validation accuracy 進行自適應降速微調

### 2. 評估與圖表產生隔離 (`src/evaluate.py`)
- 將輸出圖表儲存路徑重導向至 `reports/figures/2nd_backup/`，以將優化後的高精度六大評估圖表與第一輪產出的原始圖表 (`reports/figures/1st_backup/`) 進行實體隔離

### 3. Streamlit 前端比對介面與程式優化 (`gtsrb.py`)
- **四分頁極致展示架構**：
  - `tab1`：🚥 隨機抽取測試影像與傳統 ML / 深度學習即時推理對比
  - `tab2`：📊 初始評估指標 (Original) — 讀取並展示 `reports/figures/1st_backup/` 下的 baseline 指標
  - `tab3`：🔆 優化評估指標 (Optimized) — 讀取並展示 `reports/figures/2nd_backup/` 下優化後的高精度指標
  - `tab4`：⛳ 系統優化與驗收說明 — 直接讀取本驗收說明文件，實現高密度的學術成果整合
- **統一全域版更新頁尾 (`render_footer()`)**：
  - 導入了軟體設計的 **DRY (Don't Repeat Yourself)** 原則，將本來重複寫死在四個分頁底部的校名、版本與日期，重構成全域的統一渲染函數與常數設定。未來只要在頂端修改一次版號，便會自動同步到所有分頁，大幅降低維護成本
- **高級學術級「可展開結論區」 (`st.expander`)**：
  - 比照 `tab3` 準確率比較的頂級排版，將 `tab2` 與 `tab3` 的 **所有 12 個評估圖表結論** 統一重構為 `with st.expander("💡 結論：", expanded=True):` 結構
  - 結論內容導入色彩語意高亮，使整個數據面板具備視覺吸引力與專業學術美感
    - 藍色 `<span style='color:#4481D7'>` 代表關鍵模型/方法/變數
    - 紅橘色 `<span style='color:#DD6D6A'>` 代表精度指標或核心數據

---

## 📊 圖表比對位置
- **第一階段初始圖表目錄**：`reports/figures/1st_backup/`
- **第二階段優化圖表目錄**：`reports/figures/2nd_backup/`
