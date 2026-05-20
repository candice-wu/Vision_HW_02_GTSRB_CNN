# HW2 系統實作 - 交通號誌辨識 (GTSRB)

## 1. 系統設計

### 1-1. 模型架構介紹
本系統採用兩種主要的建模策略進行比較：
1. **深度學習 (CNN + STN)**：
   我們實作一個卷積神經網路 (CNN)，並在最前方加入一個**空間轉換網路 (Spatial Transformer Network, STN)**。STN 模組負責學習圖像的仿射變換 (Affine Transformation)，能在特徵提取前，先將圖像中的交通號誌進行縮放、旋轉或平移校正，減少因拍攝角度與位置造成的干擾。隨後的 CNN 由三個捲積層 (Conv2d) 與最大池化層 (MaxPool) 組成，最後連接 Fully Connected 層進行 43 類的分類。
   
2. **傳統機器學習與降維 (PCA + ML)**：
   我們將 `32x32x3` 的圖片攤平為 3072 維度的特徵向量，並使用 **PCA (主成份分析)** 進行降維，保留 95% 的變異數。降維後的特徵被送入多個機器學習模型進行訓練，包含：**SVM、Random Forest、KNN、AdaBoost 與 K-means (非監督式集群做對照)**。

### 1-2. 損失函數說明
在 CNN 的訓練中，我們採用 **Cross Entropy Loss (交叉熵損失函數)**。
Cross Entropy 專門設計用於多類別分類問題，它計算模型預測的機率分佈與真實標籤之間的差異。當模型對正確類別給出越高的信心分數時，Loss 就會越小。我們使用 Adam 優化器搭配 Learning Rate = 0.001 進行反向傳播。

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
   我們透過設定環境變數 `os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'`，強制將該特定的運算退回 CPU 執行 (Fallback)，其餘張量運算仍保留在 GPU，完美解決報錯問題並兼顧了運算效能。
2. **機器學習特徵降維**：
   針對傳統 ML，我們在建模前強制加入 PCA 管線，大幅降低特徵維度，使得 SVM 與 AdaBoost 的訓練能在合理時間內完成。

### 3-3. 技術反思
本次實驗清楚展現「特徵工程」的演進。在傳統 ML 時代，我們需要先做 PCA 等複雜的前處理，且因為打平影像遺失了 2D 空間資訊，導致準確率存在天花板。相對地，CNN 能夠自動學習空間特徵濾波器，而 STN 更是讓模型具備空間變換的抗性，這是深度學習在電腦視覺上無法被取代的強大優勢。最後，將所有模型部署至 Streamlit，實現可視化的成果展示，大幅提升模型的實用性與互動性。

### 3-4. 進階問題與探討
本專案在實作過程中，經歷了多次的技術迭代與除錯，以下針對幾個關鍵問題進行深入剖析：

1. **環境相容性問題，故轉用 PyTorch 開發**
   原本系統預計基於 TensorFlow/Keras 框架實作，但因硬體設備 (Apple Silicon) 與不同版本的 TensorFlow 存在相容性問題與套件依賴衝突。為確保系統穩定性與後續擴充性，果斷轉用學術界廣泛採用且對 Mac MPS 支援度較高的 PyTorch 進行開發。

2. **STN 依賴的 grid_sample 與 Mac MPS 支援問題**
   在實作空間轉換網路 (STN) 時，PyTorch 依賴 `torch.nn.functional.grid_sample` 進行特徵圖的仿射變換。然而，目前的 PyTorch MPS (Metal Performance Shaders) 後端尚未原生支援此運算子的反向傳播 (Backward Pass)，導致訓練時會觸發 `NotImplementedError`。
   * **解法**：我們在程式起始處設定環境變數 `os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'`。這項設定能強制將 MPS 尚未支援的運算子退回給 CPU 處理，其餘運算依舊保留在 GPU 進行，從而完美避開報錯並維持訓練效率。

3. **準確度不高的問題，有那些解決策略？**
   在目前的單純設定下，部分模型準確率仍有進步空間。未來的第二版優化計畫將包含以下策略：
   * **資料擴增 (Data Augmentation)**：在訓練集的 transform 中加入隨機旋轉 (`RandomRotation`)、平移、或亮度調整 (`ColorJitter`)，大幅增強模型對未知影像的泛化能力。
   * **學習率衰減 (Learning Rate Scheduler)**：引入 `ReduceLROnPlateau`，當驗證 Loss 停滯時自動調降學習率，幫助模型收斂至更佳的局部最小值。
   * **正則化 (Regularization)**：在優化器中加入 Weight Decay (L2 正則化) 或調整 Dropout 比例，減少模型的死記硬背 (Overfitting) 現象。
   * **強化基礎學習器 (Stronger Base Estimator)**：針對 AdaBoost 等集成模型，提升決策樹的深度 (例如由 depth=1 調整為 depth=10)。

4. **K-means 為何要導入「多數決映射」才能進行跨模型準確度比較？**
   準確率 (Accuracy) 的定義是「模型預測類別 ＝ 真實類別」。K-means 是一種「非監督式」演算法，它只會將資料分成 43 堆 (如 Cluster 0, Cluster 15...)，這些群集編號本身不具備真實的交通號誌意義。因此，我們必須透過「多數決映射 (Majority Voting Mapping)」，檢視每一堆裡面的真實標籤，並將該堆強制定義為數量最多的那個標籤，如此一來才能在同一個基準線上計算出準確率。

5. **K-means 無法產生 ROC 曲線的原因？**
   ROC 曲線的繪製前提是模型必須能輸出**「機率值 (Probability)」**（例如預測這張圖有 90% 是速限 30）。由於 K-means 是一種「硬分群 (Hard Clustering)」演算法，它只會給出絕對的群集歸屬，無法輸出連續的機率分佈，因此數學上無法透過變動閾值來繪製 ROC 曲線。

6. **為何 K-means 準確度 (20.61%) 大於 AdaBoost (16.68%)？**
   這是一個強烈對比的案例：
   * **AdaBoost 的先天限制**：`scikit-learn` 的 AdaBoost 預設使用深度僅為 1 的決策樹 (Decision Stump)。面對 43 個類別的高複雜度影像分類，這種極弱的分類器根本無力有效切割特徵空間，導致嚴重的欠擬合 (Underfitting)。
   * **K-means 的物理特性與上帝視角**：交通號誌在 PCA 特徵空間中，相似的顏色與形狀本來就會自然聚攏 (物以類聚)。加上我們在實作「多數決映射」時，直接使用了測試集的真實標籤來對答案 (Oracle Mapping / Data Leakage)，這使得 K-means 在分群後獲得最佳的標籤指派，從而在這場評估中拿下 20.61% 的保底分數，意外擊敗配置不當的 AdaBoost。

7. **Streamlit 雲端部署與 GitHub 大檔案限制問題**
   在嘗試將專案推播至 GitHub 並部署到 Streamlit Community Cloud 時，遇到兩個挑戰：
   * **GitHub 100MB 檔案限制**：本專案的 `randomforest_model.joblib` 權重檔高達 598MB，直接 `git push` 會遭到 GitHub 伺服器拒絕 (超過單一檔案 100MB 限制)。此外，完整的訓練資料集 (高達 39,000 多張圖片) 也會導致上傳時間過長。
   * **Streamlit 讀取權重報錯**：若為了解決容量問題而將整個 `models/` 與 `data/` 目錄加入 `.gitignore`，會導致 Streamlit 部署後出現 `[Errno 2] No such file or directory` 的錯誤，且「隨機抽樣」功能也會因缺乏圖片而失效。
   * **解法策略**：我們對 `.gitignore` 進行了精細的設定：
     1. 僅排除過大的 `models/randomforest_model.joblib` 以及不需要用於推論的龐大訓練集 `data/raw/Train/` 和 `data/raw/Meta/`。
     2. 保留體積合理的其他機器學習模型與深度學習權重 (如 CNN, SVM 等)，確保雲端載入無誤。
     3. 保留測試集 `data/raw/Test/`，讓 Web 端的「隨機抽樣測試集」功能得以順利運作。
     同時，在 Streamlit 應用程式中加入防呆機制 (`os.path.exists`)，當讀取不到過大的 Random Forest 模型時，系統能自動略過並維持網頁正常運作。

8. **模型泛化能力限制與 Out-of-Distribution (OOD) 問題**
   由於本系統使用的 GTSRB (German Traffic Sign Recognition Benchmark) 資料集完全由「德國」的交通號誌組成，模型在訓練階段只學習過這 43 種特定的德國號誌特徵。
   * **領域偏移 (Domain Shift)**：當使用者透過 Web 介面自行上傳「台灣」的交通號誌圖片時，由於台灣的號誌在字體、顏色比例、形狀設計上與德國可能存在差異，對模型來說這些圖片屬於**分佈外資料 (Out-of-Distribution, OOD)**。
   * **強迫預測 (Forced Guessing)**：目前系統分類層固定為 43 類，且沒有實作拒絕判定 (Reject Option) 的機制。因此面對台灣號誌時，模型不具備說「這不是德國號誌」的能力，只能從現有 43 類中強行輸出一個機率最高的結果，這在本質上等同於「瞎猜」。
   * **未來改進方向**：若要讓系統支援台灣號誌，必須**重新收集並標註台灣交通號誌資料集**，擴充或替換既有的類別定義 (.csv)，並重新進行模型訓練 (Training) 或微調 (Fine-tuning)，模型才能真正具備辨識台灣號誌的能力。
     * **未來研究探索 (具體方向)**：
       * 🧵 **探索方向 1：如何實現分佈外偵測 (OOD Detection) 與拒絕機制？**
         1. **信賴度門檻法 (Confidence Thresholding)**：檢查 Softmax 的最大機率值。如果小於某個門檻（例如 0.8），就判定為「非已知號誌 (OOD)」。
         2. **重構誤差法 (Reconstruction Error)**：使用 Autoencoder（自編碼器）重建影像。因為模型沒看過台灣號誌，重建誤差會非常高，藉此來辨識 OOD 資料。
         3. **能量法 (Energy-based OOD)**：計算模型的能量值，分佈內（德國號誌）的能量通常較低，分佈外（台灣號誌或隨機圖片）的能量較高。
       * 🧵 **探索方向 2：若要擴充支援「台灣號誌」，系統架構該如何演進？**
         1. **遷移學習 (Transfer Learning)**：使用在德國資料集 (GTSRB) 訓練好的權重作為 Pretrained weights，再用少量的台灣號誌進行微調 (Fine-tuning)。
         2. **模型最後一層的動態擴充**：如何從 43 分類動態擴展至 $43 + N$ 分類？這需要修改 PyTorch 模型的全連接層 `nn.Linear(fc_inputs, 43 + N)`，並重新設計類別映射對照表。
         3. **資料收集與標註流程**：由於台灣缺乏像 GTSRB 那樣完美的開源學術資料集，若要實作，可能需要利用 YOLO 偵測台灣道路街景影像，裁切出號誌圖片後，進行人工標註。
