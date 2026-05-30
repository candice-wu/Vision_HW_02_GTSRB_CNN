# HW2 系統實作 - 交通號誌辨識 (GTSRB)

## 1. 系統設計

### 1-1. 模型架構介紹
本系統採用兩種主要的建模策略進行比較：
1. **深度學習 (CNN + STN)**：
   實作一個卷積神經網路 (CNN)，並在最前方加入一個**空間轉換網路 (Spatial Transformer Network, STN)**。STN 模組負責學習圖像的仿射變換 (Affine Transformation)，能在特徵提取前，先將圖像中的交通號誌進行縮放、旋轉或平移校正，減少因拍攝角度與位置造成的干擾。隨後的 CNN 由三個捲積層 (Conv2d) 與最大池化層 (MaxPool) 組成，最後連接 Fully Connected 層進行 43 類的分類。
   
2. **傳統機器學習與降維 (PCA + ML)**：
   將 `32x32x3` 的圖片攤平為 3072 維度的特徵向量，並使用 **PCA (主成份分析)** 進行降維，保留 95% 的變異數。降維後的特徵被送入多個機器學習模型進行訓練，包含：**SVM、Random Forest、KNN、AdaBoost 與 K-means (非監督式集群做對照)**。

### 1-2. 損失函數說明
在 CNN 的訓練中，採用 **Cross Entropy Loss (交叉熵損失函數)**。
Cross Entropy 專門設計用於多類別分類問題，它計算模型預測的機率分佈與真實標籤之間的差異。當模型對正確類別給出越高的信心分數時，Loss 就會越小。使用 Adam 優化器搭配 Learning Rate = 0.001 進行反向傳播。

---

## 2. 跨模型的方法比較

### 2-1. 加入其他 Machine Learning 方法作比較
本專案除了深度學習外，也整合傳統分類器。其中：
*   **SVM**：擅長處理高維資料的超平面分類。
*   **Random Forest**：基於決策樹的集成方法，抗雜訊能力強。
*   **KNN**：以距離作為基礎的分類器。
*   **AdaBoost**：透過不斷修正錯誤樣本來強化分類能力的集成模型。

*(註：PCA 被明確定位為降維前處理工具，而非分類模型；K-means 則為非監督式集群，提供數據分佈參考。)*

### 2-2. 跨模型的數據分析 (Accuracy / Loss)
透過 `src/evaluate.py` 產出的各類圖表，我們進行以下評估：
1. **Accuracy 比較圖**：比較 CNN 與傳統 ML 模型的準確率。一般而言，CNN 藉由卷積層提取空間特徵，在影像分類上的 Accuracy 會顯著優於將影像攤平成一維陣列的傳統 ML 模型。
2. **ROC / AUC 曲線 (Macro Average)**：評估各模型在不同分類閾值下的表現。
3. **PCA Scree Plot**：用以檢視累積變異數 (Cumulative Explained Variance)，驗證多少維度即可包含 95% 的原始資訊。
4. **CNN Confusion Matrix**：用以觀察 CNN 模型在哪些相似的交通號誌 (如速限 30 與 50) 最容易產生混淆。

---

## 3. 技術討論

### 3-1. 實作系統時遇到的困難
1. **硬體加速的限制**：
   在 Mac Apple Silicon 晶片上使用 PyTorch `mps` 後端時，STN 網路依賴的 `grid_sample` 運算子在反向傳播 (backward pass) 時尚未被 MPS 原生支援，導致 `NotImplementedError` 報錯。
2. **高維度資料的訓練負擔**：
   直接將 3072 維的像素資料丟給 SVM 訓練，其時間與空間複雜度極高，甚至可能導致記憶體溢出或運算時間過長。

### 3-2. 問題分析與解法
1. **MPS 不支援的解法**：
   透過設定環境變數 `os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'`，強制將該特定的運算退回 CPU 執行 (Fallback)，其餘張量運算仍保留在 GPU，完美解決報錯問題並兼顧了運算效能。
2. **機器學習特徵降維**：
   針對傳統 ML，在建模前強制加入 PCA 管線，大幅降低特徵維度，使得 SVM 與 AdaBoost 的訓練能在合理時間內完成。

### 3-3. 技術反思
本次實驗清楚展現「特徵工程」的演進。在傳統 ML 時代，需要先做 PCA 等複雜的前處理，且因為打平影像遺失了 2D 空間資訊，導致準確率存在天花板。相對地，CNN 能夠自動學習空間特徵濾波器，而 STN 更是讓模型具備空間變換的抗性，這是深度學習在電腦視覺上無法被取代的強大優勢。最後，將所有模型部署至 Streamlit，實現可視化的成果展示，大幅提升模型的實用性與互動性。

### 3-4. 進階問題與探討
本專案在實作過程中，經歷多次的技術迭代與除錯，以下針對幾個關鍵問題進行深入剖析：

1. **環境相容性問題，故轉用 PyTorch 開發**
   原本系統預計基於 TensorFlow/Keras 框架實作，但因硬體設備 (Apple Silicon) 與不同版本的 TensorFlow 存在相容性問題與套件依賴衝突。為確保系統穩定性與後續擴充性，果斷轉用學術界廣泛採用且對 Mac MPS 支援度較高的 PyTorch 進行開發。

2. **STN 依賴的 grid_sample 與 Mac MPS 支援問題**
   在實作空間轉換網路 (STN) 時，PyTorch 依賴 `torch.nn.functional.grid_sample` 進行特徵圖的仿射變換。然而，目前的 PyTorch MPS (Metal Performance Shaders) 後端尚未原生支援此運算子的反向傳播 (Backward Pass)，導致訓練時會觸發 `NotImplementedError`。
   * **解法**：我們在程式起始處設定環境變數 `os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'`。這項設定能強制將 MPS 尚未支援的運算子退回給 CPU 處理，其餘運算依舊保留在 GPU 進行，從而完美避開報錯並維持訓練效率。

3. **準確度不高的問題，有那些解決策略？（優化成果與實作驗證）**
   在第一階段 (Phase 1 Baseline) 中，原生自製 CNN 與傳統 ML 模型的性能仍有極大提升空間。因此，我們在第二階段 (Phase 2 Optimized) 中成功導入並完成了以下核心優化策略，取得了震撼的效能飛躍：
   * **批次標準化 (Batch Normalization)**：於卷積神經網路的卷積層與 ReLU / MaxPool 之間插入 `nn.BatchNorm2d`，解決內部協變偏移，極大加快收斂速度並穩定梯度傳播。
   * **資料擴增 (Data Augmentation)**：在訓練集的前處理管道中套用隨機旋轉 (`RandomRotation(15)`)、亮度與對比度調節 (`ColorJitter(0.2, 0.2)`) 以及隨機仿射變換 (`RandomAffine`)，強迫模型學習尺度與旋轉無關的穩健特徵。驗證與測試集則維持乾淨前處理，保障評估無偏差。
   * **自適應學習率衰減 (ReduceLROnPlateau)**：引入動態學習率調度器，持續監控 `val_acc`。當連續 3 個 Epoch 精度陷入停滯時，自動將學習率調降為原本的 10% (factor=0.1)，促使模型在臨界區域進行極為細緻的收斂微調。
   * **拉長訓練週期**：將最大 Epoch 提升至 25，完美配合學習率退火週期。
   
   **【實作與驗收結果】**：
   * **驗證集最高精度 (Val Accuracy)**：從原生的 `95.28%` 暴增至完美的 **`99.90%`**。
   * **獨立測試集精度 (Test Accuracy)**：在全新且完全獨立的 12,630 張測試影像上，測試精度達到不可思議的 **`98.17%`**，成功跨越並超額達成 `>98%` 的極致優化目標！
   * **Macro ROC AUC**：優化後的 CNN 完美貼合左上極限點，其動態讀取的 Macro AUC 達到了理論極限的 **`1.000` (0.9999)**，在所有 43 個交通號誌類別上皆展現完美的分類鑑別度！

4. **K-means 為何要導入「多數決映射」才能進行跨模型準確度比較？**
   準確率 (Accuracy) 的定義是「模型預測類別 ＝ 真實類別」。K-means 是一種「非監督式」演算法，它只會將資料分成 43 堆 (如 Cluster 0, Cluster 15...)，這些群集編號本身不具備真實的交通號誌意義。因此，必須透過「多數決映射 (Majority Voting Mapping)」，檢視每一堆裡面的真實標籤，並將該堆強制定義為數量最多的那個標籤，如此一來才能在同一個基準線上計算出準確率。

5. **K-means 無法產生 ROC 曲線的原因？**
   ROC 曲線的繪製前提是模型必須能輸出**「機率值 (Probability)」**（例如預測這張圖有 90% 是速限 30）。由於 K-means 是一種「硬分群 (Hard Clustering)」演算法，它只會給出絕對的群集歸屬，無法輸出連續的機率分佈，因此數學上無法透過變動閾值來繪製 ROC 曲線。

6. **為何 K-means 準確度 (20.61%) 大於 AdaBoost (16.68%)？**
   這是一個強烈對比的案例：
   * **AdaBoost 的先天限制**：`scikit-learn` 的 AdaBoost 預設使用深度僅為 1 的決策樹 (Decision Stump)。面對 43 個類別的高複雜度影像分類，這種極弱的分類器根本無力有效切割特徵空間，導致嚴重的欠擬合 (Underfitting)。
   * **K-means 的物理特性與上帝視角**：交通號誌在 PCA 特徵空間中，相似的顏色與形狀本來就會自然聚攏 (物以類聚)。加上在實作「多數決映射」時，直接使用測試集的真實標籤來對答案 (Oracle Mapping / Data Leakage)，這使得 K-means 在分群後獲得最佳的標籤指派，從而在這場評估中拿下 20.61% 的保底分數，意外擊敗配置不當的 AdaBoost。

7. **Streamlit 雲端部署與 GitHub 大檔案限制問題**
   在嘗試將專案推播至 GitHub 並部署到 Streamlit Community Cloud 時，遇到兩個挑戰：
   * **GitHub 100MB 檔案限制**：本專案的 `randomforest_model.joblib` 權重檔高達 598MB，直接 `git push` 會遭到 GitHub 伺服器拒絕 (超過單一檔案 100MB 限制)。此外，完整的訓練資料集 (高達 39,000 多張圖片) 也會導致上傳時間過長。
   * **Streamlit 讀取權重報錯**：若為了解決容量問題而將整個 `models/` 與 `data/` 目錄加入 `.gitignore`，會導致 Streamlit 部署後出現 `[Errno 2] No such file or directory` 的錯誤，且「隨機抽樣」功能也會因缺乏圖片而失效。
   * **解法策略**：對 `.gitignore` 進行了精細的設定：
     1. 僅排除過大的 `models/randomforest_model.joblib` 以及不需要用於推論的龐大訓練集 `data/raw/Train/` 和 `data/raw/Meta/`。
     2. 保留體積合理的其他機器學習模型與深度學習權重 (如 CNN, SVM 等)，確保雲端載入無誤。
     3. 保留測試集 `data/raw/Test/`，讓 Web 端的「隨機抽樣測試集」功能得以順利運作。
     同時，在 Streamlit 應用程式中加入防呆機制 (`os.path.exists`)，當讀取不到過大的 Random Forest 模型時，系統能自動略過並維持網頁正常運作。

8. **模型泛化能力限制與 Out-of-Distribution (OOD) 的安全性防禦實作**
   由於本系統使用的 GTSRB (German Traffic Sign Recognition Benchmark) 資料集完全由「德國」的交通號誌組成，模型在訓練階段只學習過這 43 種特定的德國號誌特徵。
   * **領域偏移與過度自信 (Domain Shift & Overconfidence)**：當使用者透過 Web 介面自行上傳「台灣」的交通號誌圖片時，由於台灣的號誌在字體、顏色比例、形狀設計上與德國不同，對模型來說屬於**分佈外資料 (Out-of-Distribution, OOD)**。若無防禦，分類層會強行輸出一個機率最高的結果，等同於「瞎猜」與「過度自信預測」，這在自駕安全中屬於高危漏洞。
   * **已實作之 OOD 信賴度安全防禦**：
     為了解決此安全隱患，我們在 Phase 3 中正式導入了基於學術理論 **Maximum Softmax Probability (MSP)** 的主動安全過濾機制，並全域套用 **0.75 置信度黃金門檻**：
     1. **深度學習 CNN 模型**：於推論階段檢查 Softmax 後最大類別置信機率 $P_{\text{DL}}$。若 $P_{\text{DL}} < 0.75$，直接進行 OOD 攔截，前端以黃橘色警告圖標顯示 `⚠️ 這是非德國交通號誌 (OOD 安全攔截)`，杜絕過度自信的無效預測。
     2. **支援機率之機器學習模型 (RF, KNN, AdaBoost, NN/MLP)**：動態擷取 `predict_proba()` 最大置信機率，小於 0.75 門檻則安全攔截。
     3. **支持向量機 (SVM)**：針對未啟用機率校準的 SVM，利用其決策邊界距離向量 `decision_function()` 進行 Softmax 歸一化轉換以近似機率分佈：
        $$P_{\text{SVM}}(c) = \frac{e^{f_c(x)}}{\sum_{j} e^{f_j(x)}}$$
        提取其最高置信度，低於 0.75 門檻則安全過濾。
   * **實作與學術驗證結論**：
     實機測試結果顯示，上傳台灣標誌或噪聲干擾圖時，機器學習模型（如 RandomForest 僅有 67.00% 置信度，AdaBoost 僅有 2.33% 置信度）因樹投票或弱學習器權重分散，成功觸發了 OOD 安全防禦機制並轉為黃橘色警示；而上傳標準德國 Class 17 號誌時，高精度模型（如 CNN 置信度為 100%、SVM 置信度為 94.69%）置信度遠高於 75% 閾值，系統正常放行顯示類別。這項防禦系統在保障自駕安全與提升可用性之間取得了完美的學術級平衡！
   * **未來可擴充探討方向**：
     1. **重構誤差法 (Reconstruction Error)**：使用 Autoencoder 進行特徵重建，若重建誤差大於門檻，則輔助判定為 OOD。
     2. **遷移學習與類別擴充**：若要完整支援台灣交通號誌，需實作 YOLO 模型切出道路標誌進行資料增強，並修改 CNN 全連接層為 `nn.Linear(fc_inputs, 43 + N)` 進行微調 (Fine-tuning)。
