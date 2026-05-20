# 目的 (Purpose)
本規格定義了優化階段的 Streamlit 可視化評估介面變更，為使用者提供原始版本（第一階段）與優化版本（第二階段）之間的對比功能。

## MODIFIED Requirements

### Requirement: Four-Tab UI Layout
系統的 `st.tabs` 呼叫 MUST 改為四個分頁：
1. `st.tabs[0]` / `tab1`："🚥 GTSRB 交通號誌辨識" (隨機抽樣/即時辨識主頁)
2. `st.tabs[1]` / `tab2`："📊 初始評估指標 (Original)" (第一階段評估結果展示)
3. `st.tabs[2]` / `tab3`："📈 優化評估指標 (Optimized)" (第二階段優化後評估結果展示)
4. `st.tabs[3]` / `tab4`："📄 系統優化與驗收說明" (讀取並呈現專案目錄下 walkthrough.md 的詳細成果驗收)

#### Scenario: Switching tabs in the application
- **當 (WHEN)** 使用者在網頁上點擊不同的分頁按鈕時
- **則 (THEN)** 系統 MUST 流暢切換至對應的頁面，且維持版尾資訊渲染以及模型快取載入。

### Requirement: Isolated Metrics Folders Reading
系統 MUST 實作圖表資料夾的隔離與獨立讀取，以在不同的分頁呈現不同優化階段的模型指標。

#### Scenario: Reading Original Metrics
- **當 (WHEN)** 使用者進入 `tab2` 時
- **則 (THEN)** 系統 MUST 從 `reports/figures/1st_backup/` 目錄中載入並呈現 accuracy_comparison, loss_curve, cnn_confusion_matrix, training_time_comparison, pca_scree_plot 以及 roc_curve_comparison 六張圖表，並搭配原始結語說明。

#### Scenario: Reading Optimized Metrics
- **當 (WHEN)** 使用者進入 `tab3` 時
- **則 (THEN)** 系統 MUST 從 `reports/figures/2nd_backup/` 目錄中載入並呈現相同命名的六張優化後圖表，並搭配針對 BatchNorm 與 Data Augment 優化後的特有結語。
